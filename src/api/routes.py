"""FastAPI routes for the LoL Highlight & Analysis API"""

import os
import sys
import json
import time
import shutil
import pandas as pd
import requests
from typing import Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from datetime import datetime
import numpy as np
import xgboost as xgb
from dotenv import load_dotenv
from xgboost import XGBClassifier

from src.api.models import (
    HealthCheckResponse,
    MatchRequest,
    ProfileRequest,
    GapAnalysisRequest,
    MatchAnalysisResult,
    ProfileAnalysisResult,
    GapAnalysisResult,
    AnalysisGenerateRequest,
    AnalysisResult,
    HighlightGenerateRequest,
    HighlightResult,
)
from src.services.analyzer import MatchAnalyzer
from src.models.rule_based import RuleBasedGapAnalyzer
from src import __version__

# Load environment variables
load_dotenv()

# Import player_rating module
models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
if models_dir not in sys.path:
    sys.path.insert(0, models_dir)

from player_rating import compute_player_impact, add_derived_features

# Import LLM reporter
from src.services.llm_reporter import generate_match_report

# Initialize FastAPI app
app = FastAPI(
    title="LoL Highlight & Analysis API",
    description="AI-powered League of Legends match analysis and highlight generation",
    version=__version__,
)

# Add CORS middleware for Spring Boot integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
match_analyzer = MatchAnalyzer()
gap_analyzer = RuleBasedGapAnalyzer()

# Initialize Impact Score Model
# __file__ = /Users/.../lol_project/AI/src/api/routes.py
# We need to go up to AI/models
AI_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # AI/
MODEL_DIR = os.path.join(AI_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "player_impact_model.json")
FEAT_PATH = os.path.join(MODEL_DIR, "feature_cols.json")

# Global model variables
impact_model = None
impact_explainer = None
feature_cols = []

class ModelWrapper:
    """Wrapper for XGBoost Booster to mimic sklearn-style predict_proba"""
    def __init__(self, booster):
        self.bst = booster
        
    def predict_proba(self, X):
        # Check if X is already DMatrix, if not convert
        if not isinstance(X, xgb.DMatrix):
            dtest = xgb.DMatrix(X)
        else:
            dtest = X
        preds = self.bst.predict(dtest)
        # Binary classification: preds is probability of class 1
        return np.column_stack((1-preds, preds))
    
    def predict(self, X):
        if not isinstance(X, xgb.DMatrix):
            dtest = xgb.DMatrix(X)
        else:
            dtest = X
        preds = self.bst.predict(dtest)
        return (preds > 0.5).astype(int)

@app.on_event("startup")
async def startup_event():
    global impact_model, impact_explainer, feature_cols

    # 0. Load LSTM win-probability model
    try:
        from src.models.event_lstm import load_lstm_model
        lstm_model = load_lstm_model()
        if lstm_model is None:
            print("[WARN] LSTM model not loaded — highlight impact scores will be 0")
    except Exception as e:
        print(f"[WARN] LSTM model load failed: {e}")

    # 1. Load Feature Columns
    if os.path.exists(FEAT_PATH):
        try:
            with open(FEAT_PATH, "r") as f:
                feature_cols = json.load(f)
            print(f"[INFO] Loaded {len(feature_cols)} feature columns")
        except Exception as e:
            print(f"[ERROR] Failed to load feature columns: {e}")

    # 2. Load Model (Robust Loading)
    if os.path.exists(MODEL_PATH):
        try:
            print(f"[INFO] Loading model from {MODEL_PATH}...")
            # Try loading as XGBClassifier first (standard)
            try:
                impact_model = XGBClassifier()
                impact_model.load_model(MODEL_PATH)
                print("[INFO] Model loaded as XGBClassifier")
            except Exception as e:
                print(f"[WARN] Failed to load as XGBClassifier: {e}")
                print("[INFO] Attempting to load as raw Booster...")
                bst = xgb.Booster()
                bst.load_model(MODEL_PATH)
                impact_model = ModelWrapper(bst)
                print("[INFO] Model loaded as raw Booster (wrapped)")

        except Exception as e:
            print(f"[ERROR] Failed to load Impact Model: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"[WARN] Impact Score model not found at {MODEL_PATH}")

    # 3. Initialize SHAP Explainer
    if impact_model:
        try:
            import shap
            print(f"[INFO] Initializing SHAP explainer...")
            # If wrapped booster, access internal booster
            model_to_explain = impact_model.bst if isinstance(impact_model, ModelWrapper) else impact_model
            impact_explainer = shap.TreeExplainer(model_to_explain)
            print("[INFO] SHAP explainer initialized")
        except Exception as e:
            print(f"[WARN] Failed to initialize SHAP: {e}")
            print("Impact scores will be available, but detailed breakdown might fail.")


# Riot API configuration
RIOT_API_KEY = os.getenv("RIOT_API_KEY")
ACCOUNT_REGION = "asia"
MATCH_REGION = "asia"

riot_session = requests.Session()
if RIOT_API_KEY:
    riot_session.headers.update({"X-Riot-Token": RIOT_API_KEY})

# ============================================================================
# 실행 모드 설정
# USE_BACKEND_API=true  → 백엔드 API 사용 (프로덕션)
# USE_BACKEND_API=false → Riot API만 사용 (로컬 개발)
# ============================================================================
USE_BACKEND_API = os.getenv("USE_BACKEND_API", "false").lower() == "true"

# Backend API configuration (nexus-gg.kro.kr)
BACKEND_API_BASE_URL = os.getenv("BACKEND_API_BASE_URL", "https://nexus-gg.kro.kr")
BACKEND_API_TOKEN = os.getenv("BACKEND_API_TOKEN", "")  # JWT 토큰 (필요시 설정)

backend_api_session = requests.Session()
if BACKEND_API_TOKEN:
    backend_api_session.headers.update({"Authorization": f"Bearer {BACKEND_API_TOKEN}"})

# 시작 시 모드 출력
print(f"[CONFIG] USE_BACKEND_API = {USE_BACKEND_API}")
print(f"[CONFIG] BACKEND_API_BASE_URL = {BACKEND_API_BASE_URL if USE_BACKEND_API else 'N/A (disabled)'}")

# Directories for file uploads
UPLOAD_DIR = "uploads"
CLIPS_DIR = "clips"
STATIC_DIR = os.path.join(AI_DIR, "static")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CLIPS_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# Mount static files
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    app.mount("/clips", StaticFiles(directory=CLIPS_DIR), name="clips")
    print(f"[INFO] Static files mounted at /static from {STATIC_DIR}")


@app.get("/", response_class=HTMLResponse)
async def root():
    """
    Serve the main web UI
    """
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    else:
        # Fallback to health check if no index.html
        return {
            "status": "healthy",
            "version": __version__,
            "message": "Web UI not found. Please check /docs for API documentation."
        }


@app.get("/api", response_model=HealthCheckResponse)
async def health_check():
    """
    Health check endpoint

    Returns service status and version information
    """
    return HealthCheckResponse(
        status="healthy", version=__version__, timestamp=datetime.now()
    )


@app.get("/health", response_model=HealthCheckResponse)
async def health():
    """Alternative health check endpoint"""
    return HealthCheckResponse(
        status="healthy", version=__version__, timestamp=datetime.now()
    )


@app.post("/api/v1/analyze/match", response_model=MatchAnalysisResult)
async def analyze_match(request: MatchRequest):
    """
    Analyze a specific match for a player (POST version)

    Args:
        request: Match analysis request containing match_id, summoner_name, tag_line

    Returns:
        Complete match analysis including player stats, gap analysis, and key moments

    Raises:
        HTTPException: If match not found or analysis fails
    """
    try:
        result = match_analyzer.analyze_match(
            match_id=request.match_id,
            summoner_name=request.summoner_name,
            tag_line=request.tag_line,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.get("/api/v1/analyze/match/{match_id}", response_model=MatchAnalysisResult)
async def analyze_match_get(
    match_id: str,
    summoner_name: str,
    tag_line: str
):
    """
    Analyze a specific match for a player (GET version)

    Args:
        match_id: Match ID (path parameter)
        summoner_name: Summoner name (query parameter)
        tag_line: Tag line (query parameter)

    Returns:
        Complete match analysis including player stats, gap analysis, and key moments

    Raises:
        HTTPException: If match not found or analysis fails
    """
    try:
        result = match_analyzer.analyze_match(
            match_id=match_id,
            summoner_name=summoner_name,
            tag_line=tag_line,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/api/v1/analyze/profile", response_model=ProfileAnalysisResult)
async def analyze_profile(request: ProfileRequest):
    """
    Analyze player profile based on recent games

    Args:
        request: Profile analysis request containing summoner_name, tag_line, recent_games

    Returns:
        Complete profile analysis including average stats, gap analysis, champion pool, and trends

    Raises:
        HTTPException: If player not found or analysis fails
    """
    try:
        result = match_analyzer.analyze_profile(
            summoner_name=request.summoner_name,
            tag_line=request.tag_line,
            recent_games=request.recent_games,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/api/v1/analyze/gap", response_model=GapAnalysisResult)
async def analyze_gap(request: GapAnalysisRequest):
    """
    Perform gap analysis on player statistics from a specific match

    Args:
        request: Gap analysis request containing match_id and puuid

    Returns:
        Gap analysis result with strengths, weaknesses, and recommendations

    Raises:
        HTTPException: If match not found or analysis fails
    """
    try:
        # Get match details
        match_data = match_analyzer.riot_client.get_match_details(request.match_id)
        player_stats_raw = match_analyzer.riot_client.extract_player_stats_from_match(
            match_data, request.puuid
        )

        if not player_stats_raw:
            raise ValueError(f"Player not found in match {request.match_id}")

        # Create PlayerStats object
        player_stats = match_analyzer.create_player_stats_from_match(player_stats_raw)

        # Perform gap analysis
        result = gap_analyzer.analyze_gap(
            player_stats=player_stats,
            tier=request.tier,
            division="I",
        )
        return result

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gap analysis failed: {str(e)}")


@app.get("/api/v1/tiers")
async def get_available_tiers():
    """
    Get list of available tiers for baseline comparison

    Returns:
        List of tier names and their baseline statistics
    """
    tiers = gap_analyzer.baseline_loader.get_all_tiers()
    tier_data = {}

    for tier in tiers:
        tier_data[tier] = gap_analyzer.baseline_loader.get_baseline(tier)

    return {"tiers": tiers, "baselines": tier_data}


@app.post("/api/v1/suggest-tier")
async def suggest_tier(request: GapAnalysisRequest):
    """
    Suggest appropriate target tier based on player statistics from a match

    Args:
        request: Gap analysis request containing match_id and puuid

    Returns:
        Suggested tier and comparison with multiple tiers

    Raises:
        HTTPException: If analysis fails
    """
    try:
        # Get match details
        match_data = match_analyzer.riot_client.get_match_details(request.match_id)
        player_stats_raw = match_analyzer.riot_client.extract_player_stats_from_match(
            match_data, request.puuid
        )

        if not player_stats_raw:
            raise ValueError(f"Player not found in match {request.match_id}")

        # Create PlayerStats object
        player_stats = match_analyzer.create_player_stats_from_match(player_stats_raw)

        # Suggest tier
        suggested_tier = gap_analyzer.suggest_target_tier(player_stats)
        all_comparisons = gap_analyzer.compare_with_multiple_tiers(player_stats)

        return {
            "suggested_tier": suggested_tier,
            "current_tier": request.tier,
            "comparisons": {tier: analysis.dict() for tier, analysis in all_comparisons.items()},
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tier suggestion failed: {str(e)}")


# ============================================================================
# Helper Functions for Riot API and Feature Extraction
# ============================================================================

def riot_get(url: str, params: Optional[dict] = None) -> dict:
    """Riot API GET 요청"""
    if not RIOT_API_KEY:
        raise HTTPException(status_code=500, detail="RIOT_API_KEY not configured")

    time.sleep(0.3)  # Rate limit 방지
    resp = riot_session.get(url, params=params)

    if resp.status_code == 429:
        retry_after = int(resp.headers.get("Retry-After", 2))
        time.sleep(retry_after)
        resp = riot_session.get(url, params=params)

    if not resp.ok:
        raise HTTPException(
            status_code=resp.status_code,
            detail=f"Riot API error: {resp.status_code} - {resp.text}"
        )

    return resp.json()


def get_puuid_from_riot_id(game_name: str, tag_line: str) -> str:
    """Riot ID -> PUUID 변환"""
    url = f"https://{ACCOUNT_REGION}.api.riotgames.com/riot/account/v1/accounts/by-riot-id/{game_name}/{tag_line}"
    data = riot_get(url)
    return data["puuid"]


def get_recent_match_ids(puuid: str, count: int = 20, queue: int = 420) -> list:
    """최근 매치 ID 목록 조회 - Riot API 직접 호출 (fallback)"""
    url = f"https://{MATCH_REGION}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids"
    params = {"start": 0, "count": count, "queue": queue}
    response = riot_session.get(url, params=params)
    response.raise_for_status()
    return response.json()


def get_recent_match_ids_from_team_api(game_name: str, tag_line: str, count: int = 20) -> list:
    """최근 매치 ID 목록 조회 - 백엔드 API 사용 (nexus-gg.kro.kr)

    Args:
        game_name: 소환사 이름
        tag_line: 태그 라인
        count: 조회할 매치 수

    Returns:
        매치 ID 목록 (예: ["KR_8031304127", "KR_8031234567", ...])
    """
    try:
        from urllib.parse import quote
        encoded_game_name = quote(game_name)
        encoded_tag_line = quote(tag_line)
        url = f"{BACKEND_API_BASE_URL}/api/matches/summoner/{encoded_game_name}/{encoded_tag_line}"
        params = {"page": 0, "size": count}

        response = backend_api_session.get(url, params=params, timeout=10)

        if response.status_code == 200:
            data = response.json()
            # 백엔드 API 응답 형식: { "data": { "content": [...] } }
            if data.get("data") and data["data"].get("content"):
                match_ids = [match["matchId"] for match in data["data"]["content"]]
                print(f"[INFO] Backend API: Retrieved {len(match_ids)} matches for {game_name}#{tag_line}")
                return match_ids
            else:
                print(f"[WARN] Backend API: No matches found for {game_name}#{tag_line}")
                return []
        else:
            print(f"[WARN] Backend API returned {response.status_code}, falling back to Riot API")
            return None  # None을 반환하면 fallback 처리

    except requests.exceptions.RequestException as e:
        print(f"[WARN] Backend API request failed: {e}, falling back to Riot API")
        return None  # None을 반환하면 fallback 처리


def get_recent_matches_with_fallback(game_name: str, tag_line: str, puuid: str, count: int = 20) -> list:
    """최근 매치 ID 조회 - 백엔드 API 우선, 실패 시 Riot API fallback

    Args:
        game_name: 소환사 이름
        tag_line: 태그 라인
        puuid: 플레이어 PUUID (Riot API fallback용)
        count: 조회할 매치 수

    Returns:
        매치 ID 목록
    """
    # 로컬 모드면 바로 Riot API 사용
    if not USE_BACKEND_API:
        print(f"[INFO] Local mode: Using Riot API directly for {game_name}#{tag_line}")
        return get_recent_match_ids(puuid, count)

    # 1. 백엔드 API 시도
    match_ids = get_recent_match_ids_from_team_api(game_name, tag_line, count)

    if match_ids is not None and len(match_ids) > 0:
        return match_ids

    # 2. Fallback: Riot API 직접 호출
    print(f"[INFO] Using Riot API fallback for {game_name}#{tag_line}")
    return get_recent_match_ids(puuid, count)


# ============================================================================
# Backend API 저장 함수들 (nexus-gg.kro.kr)
# ============================================================================

def save_highlight_to_team_api(
    match_id: str,
    title: str,
    start_time: int,
    end_time: int,
    highlight_type: str = "CUSTOM",
    description: str = ""
) -> Optional[dict]:
    """하이라이트를 백엔드 API에 저장

    Args:
        match_id: 매치 ID (예: KR_8031304127)
        title: 하이라이트 제목
        start_time: 시작 시간 (초)
        end_time: 종료 시간 (초)
        highlight_type: 유형 (KILL, MULTI_KILL, PENTAKILL, BARON, DRAGON, TOWER_DESTROY, TEAM_FIGHT, CUSTOM)
        description: 설명

    Returns:
        저장된 하이라이트 정보 또는 None (실패 시)
    """
    # 로컬 모드면 저장 스킵
    if not USE_BACKEND_API:
        print(f"[INFO] Local mode: Skipping Backend API save for highlight - {title}")
        return {"local_mode": True, "title": title, "skipped": True}

    try:
        url = f"{BACKEND_API_BASE_URL}/api/highlights"
        payload = {
            "matchId": match_id,
            "title": title,
            "startTime": start_time,
            "endTime": end_time,
            "type": highlight_type,
            "description": description
        }

        response = backend_api_session.post(url, json=payload, timeout=10)

        if response.status_code in [200, 201]:
            print(f"[INFO] Backend API: Highlight saved - {title}")
            return response.json()
        else:
            print(f"[WARN] Backend API: Failed to save highlight - {response.status_code}: {response.text}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"[WARN] Backend API: Highlight save failed - {e}")
        return None


def save_analysis_to_team_api(match_id: str) -> Optional[dict]:
    """분석 결과를 백엔드 API에 저장

    Args:
        match_id: 매치 ID (예: KR_8031304127)

    Returns:
        저장된 분석 정보 또는 None (실패 시)
    """
    # 로컬 모드면 저장 스킵
    if not USE_BACKEND_API:
        print(f"[INFO] Local mode: Skipping Backend API save for analysis - {match_id}")
        return {"local_mode": True, "match_id": match_id, "skipped": True}

    try:
        url = f"{BACKEND_API_BASE_URL}/api/analyses"
        payload = {
            "matchId": match_id
        }

        response = backend_api_session.post(url, json=payload, timeout=10)

        if response.status_code in [200, 201]:
            print(f"[INFO] Backend API: Analysis saved for match {match_id}")
            return response.json()
        else:
            print(f"[WARN] Backend API: Failed to save analysis - {response.status_code}: {response.text}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"[WARN] Backend API: Analysis save failed - {e}")
        return None


def save_highlights_batch_to_team_api(match_id: str, highlights: list) -> list:
    """여러 하이라이트를 백엔드 API에 일괄 저장

    Args:
        match_id: 매치 ID
        highlights: 하이라이트 목록 (각 항목에 timestamp, type, description 등 포함)

    Returns:
        저장 성공한 하이라이트 목록
    """
    saved_highlights = []

    # 하이라이트 타입 매핑
    type_mapping = {
        "CHAMPION_KILL": "KILL",
        "CHAMPION_SPECIAL_KILL": "MULTI_KILL",
        "BUILDING_KILL": "TOWER_DESTROY",
        "ELITE_MONSTER_KILL": "DRAGON",  # 또는 BARON
        "death": "KILL",
        "kill": "KILL",
        "multi_kill": "MULTI_KILL",
        "tower": "TOWER_DESTROY",
        "dragon": "DRAGON",
        "baron": "BARON",
    }

    for h in highlights:
        # 타입 변환
        original_type = h.get("type", "CUSTOM")
        highlight_type = type_mapping.get(original_type, "CUSTOM")

        # 바론 체크
        if "baron" in h.get("description", "").lower():
            highlight_type = "BARON"
        elif "dragon" in h.get("description", "").lower() or "드래곤" in h.get("description", ""):
            highlight_type = "DRAGON"

        # 제목 생성
        timestamp = h.get("timestamp", 0)
        minutes = int(timestamp // 60)
        seconds = int(timestamp % 60)
        title = f"[{minutes}:{seconds:02d}] {h.get('description', highlight_type)}"

        # 클립 시간 (기본 앞뒤 5초)
        start_time = max(0, int(timestamp) - 5)
        end_time = int(timestamp) + 10

        result = save_highlight_to_team_api(
            match_id=match_id,
            title=title[:100],  # 제목 길이 제한
            start_time=start_time,
            end_time=end_time,
            highlight_type=highlight_type,
            description=h.get("description", "")[:500]  # 설명 길이 제한
        )

        if result:
            saved_highlights.append(result)

    print(f"[INFO] Backend API: Saved {len(saved_highlights)}/{len(highlights)} highlights")
    return saved_highlights


def get_match_data(match_id: str) -> dict:
    """매치 데이터 조회"""
    url = f"https://{MATCH_REGION}.api.riotgames.com/lol/match/v5/matches/{match_id}"
    return riot_get(url)


def get_timeline_data(match_id: str) -> dict:
    """타임라인 데이터 조회"""
    url = f"https://{MATCH_REGION}.api.riotgames.com/lol/match/v5/matches/{match_id}/timeline"
    return riot_get(url)


def extract_team_features_from_match(match_data: dict, timeline_data: dict, team_id: int) -> pd.DataFrame:
    """팀 전체의 feature를 한 번에 추출 (정규화를 위해)"""
    info = match_data["info"]
    participants = info["participants"]
    match_id = match_data["metadata"]["matchId"]

    # 게임 시간
    game_duration = info["gameDuration"]
    game_minutes = game_duration / 60.0

    # 타임라인 프레임
    frames = timeline_data["info"]["frames"]
    frame_15_idx = min(15, len(frames) - 1)
    frame_10_idx = min(10, len(frames) - 1)

    # 같은 팀 플레이어들
    team_players = [p for p in participants if p["teamId"] == team_id]

    all_rows = []
    for player in team_players:
        participant_id = player["participantId"]

        # 타임라인 스탯
        frame_15 = frames[frame_15_idx]["participantFrames"][str(participant_id)]
        cs_15 = frame_15.get("minionsKilled", 0) + frame_15.get("jungleMinionsKilled", 0)
        gold_15 = frame_15.get("totalGold", 0)
        xp_15 = frame_15.get("xp", 0)

        frame_10 = frames[frame_10_idx]["participantFrames"][str(participant_id)]
        lane_minions_10 = frame_10.get("minionsKilled", 0)
        jungle_cs_10 = frame_10.get("jungleMinionsKilled", 0)
        early_cs_total = lane_minions_10 + jungle_cs_10

        # 기본 스탯
        kills = player["kills"]
        deaths = player["deaths"]
        assists = player["assists"]
        kda = (kills + assists) / max(1, deaths)
        vision_score = player.get("visionScore", 0)
        vision_score_per_min = vision_score / max(1, game_minutes)

        # Per-minute 스탯
        gold_per_minute = player.get("challenges", {}).get("goldPerMinute", 0) or 0
        damage_per_minute = player.get("challenges", {}).get("damagePerMinute", 0) or 0
        kill_participation = player.get("challenges", {}).get("killParticipation", 0) or 0
        solo_kills = player.get("challenges", {}).get("soloKills", 0) or 0
        control_wards = player.get("challenges", {}).get("controlWardsPlaced", 0) or 0
        ward_takedowns = player.get("challenges", {}).get("wardTakedowns", 0) or 0
        ward_takedowns_before_20m = player.get("challenges", {}).get("wardTakedownsBefore20M", 0) or 0
        skillshots_dodged = player.get("challenges", {}).get("skillshotsDodged", 0) or 0
        skillshots_hit = player.get("challenges", {}).get("skillshotsHit", 0) or 0
        longest_living = player.get("longestTimeSpentLiving", 0) or 0

        row = {
            "matchId": match_id,
            "puuid": player["puuid"],
            "teamId": team_id,
            "win": 1 if player["win"] else 0,
            "champion": player["championName"],
            "role": player.get("teamPosition", "UNKNOWN"),

            "kills": kills,
            "deaths": deaths,
            "assists": assists,
            "kda": kda,

            "goldPerMinute": gold_per_minute,
            "damagePerMinute": damage_per_minute,
            "visionScore": vision_score,
            "visionScorePerMinute": vision_score_per_min,

            "killParticipation": kill_participation,
            "soloKills": solo_kills,

            "controlWardsPlaced": control_wards,
            "wardTakedowns": ward_takedowns,
            "wardTakedownsBefore20M": ward_takedowns_before_20m,

            "skillshotsDodged": skillshots_dodged,
            "skillshotsHit": skillshots_hit,
            "longestTimeSpentLiving": longest_living,

            "cs_15": cs_15,
            "gold_15": gold_15,
            "xp_15": xp_15,
            "laneMinionsFirst10Minutes": lane_minions_10,
            "jungleCsBefore10Minutes": jungle_cs_10,
            "early_cs_total": early_cs_total,
        }
        all_rows.append(row)

    df = pd.DataFrame(all_rows)
    df_with_derived = add_derived_features(df)
    return df_with_derived


def extract_features_from_match(match_data: dict, timeline_data: dict, target_puuid: str) -> pd.DataFrame:
    """매치 데이터와 타임라인에서 모델이 필요로 하는 feature를 추출"""
    info = match_data["info"]
    participants = info["participants"]

    # 타겟 플레이어 찾기
    player = None
    for p in participants:
        if p["puuid"] == target_puuid:
            player = p
            break

    if not player:
        raise HTTPException(status_code=404, detail="Player not found in this match")

    # 기본 정보
    match_id = match_data["metadata"]["matchId"]
    team_id = player["teamId"]
    win = player["win"]
    champion = player["championName"]
    role = player.get("teamPosition", "UNKNOWN")

    # 게임 시간 (초 -> 분)
    game_duration = info["gameDuration"]
    game_minutes = game_duration / 60.0

    # 팀원들
    team_players = [p for p in participants if p["teamId"] == team_id]

    # Raw 스탯 추출
    kills = player["kills"]
    deaths = player["deaths"]
    assists = player["assists"]
    kda = (kills + assists) / max(1, deaths)

    # Per-minute 스탯
    gold_per_minute = player.get("challenges", {}).get("goldPerMinute", 0) or 0
    damage_per_minute = player.get("challenges", {}).get("damagePerMinute", 0) or 0
    vision_score = player.get("visionScore", 0)
    vision_score_per_minute = vision_score / max(1, game_minutes)

    # 킬 관여율
    kill_participation = player.get("challenges", {}).get("killParticipation", 0) or 0
    solo_kills = player.get("challenges", {}).get("soloKills", 0) or 0

    # 시야 관련
    control_wards_placed = player.get("challenges", {}).get("controlWardsPlaced", 0) or 0
    ward_takedowns = player.get("challenges", {}).get("wardTakedowns", 0) or 0
    ward_takedowns_before_20m = player.get("challenges", {}).get("wardTakedownsBefore20M", 0) or 0

    # 스킬샷
    skillshots_dodged = player.get("challenges", {}).get("skillshotsDodged", 0) or 0
    skillshots_hit = player.get("challenges", {}).get("skillshotsHit", 0) or 0

    # 최장 생존 시간
    longest_time_spent_living = player.get("longestTimeSpentLiving", 0) or 0

    # 타임라인 기반 feature
    frames = timeline_data["info"]["frames"]
    participant_id = player["participantId"]

    # 15분 스탯
    frame_15_idx = min(15, len(frames) - 1)
    frame_15 = frames[frame_15_idx]["participantFrames"][str(participant_id)]

    cs_15 = frame_15.get("minionsKilled", 0) + frame_15.get("jungleMinionsKilled", 0)
    gold_15 = frame_15.get("totalGold", 0)
    xp_15 = frame_15.get("xp", 0)

    # 10분 스탯
    frame_10_idx = min(10, len(frames) - 1)
    frame_10 = frames[frame_10_idx]["participantFrames"][str(participant_id)]

    lane_minions_10 = frame_10.get("minionsKilled", 0)
    jungle_cs_10 = frame_10.get("jungleMinionsKilled", 0)
    early_cs_total = lane_minions_10 + jungle_cs_10

    # DataFrame 생성 - 타겟 플레이어만 추출 (팀원은 나중에 개별 호출)
    row_data = {
        "matchId": match_id,
        "puuid": target_puuid,
        "teamId": team_id,
        "win": 1 if win else 0,
        "champion": champion,
        "role": role,

        "kills": kills,
        "deaths": deaths,
        "assists": assists,
        "kda": kda,

        "goldPerMinute": gold_per_minute,
        "damagePerMinute": damage_per_minute,
        "visionScore": vision_score,
        "visionScorePerMinute": vision_score_per_minute,

        "killParticipation": kill_participation,
        "soloKills": solo_kills,

        "controlWardsPlaced": control_wards_placed,
        "wardTakedowns": ward_takedowns,
        "wardTakedownsBefore20M": ward_takedowns_before_20m,

        "skillshotsDodged": skillshots_dodged,
        "skillshotsHit": skillshots_hit,
        "longestTimeSpentLiving": longest_time_spent_living,

        "cs_15": cs_15,
        "gold_15": gold_15,
        "xp_15": xp_15,
        "laneMinionsFirst10Minutes": lane_minions_10,
        "jungleCsBefore10Minutes": jungle_cs_10,
        "early_cs_total": early_cs_total,
    }

    # 팀원 데이터 포함 (derived features 계산을 위해)
    all_team_data = []
    for tp in team_players:
        tp_participant_id = tp["participantId"]

        # 타임라인에서 각 팀원의 15분/10분 스탯 추출
        tp_frame_15 = frames[frame_15_idx]["participantFrames"][str(tp_participant_id)]
        tp_cs_15 = tp_frame_15.get("minionsKilled", 0) + tp_frame_15.get("jungleMinionsKilled", 0)
        tp_gold_15 = tp_frame_15.get("totalGold", 0)
        tp_xp_15 = tp_frame_15.get("xp", 0)

        tp_frame_10 = frames[frame_10_idx]["participantFrames"][str(tp_participant_id)]
        tp_lane_minions_10 = tp_frame_10.get("minionsKilled", 0)
        tp_jungle_cs_10 = tp_frame_10.get("jungleMinionsKilled", 0)
        tp_early_cs_total = tp_lane_minions_10 + tp_jungle_cs_10

        # 기본 스탯
        tp_kills = tp["kills"]
        tp_deaths = tp["deaths"]
        tp_assists = tp["assists"]
        tp_kda = (tp_kills + tp_assists) / max(1, tp_deaths)
        tp_vision = tp.get("visionScore", 0)
        tp_vision_per_min = tp_vision / max(1, game_minutes)

        # Per-minute 스탯
        tp_gold_per_minute = tp.get("challenges", {}).get("goldPerMinute", 0) or 0
        tp_damage_per_minute = tp.get("challenges", {}).get("damagePerMinute", 0) or 0
        tp_kill_participation = tp.get("challenges", {}).get("killParticipation", 0) or 0
        tp_solo_kills = tp.get("challenges", {}).get("soloKills", 0) or 0
        tp_control_wards = tp.get("challenges", {}).get("controlWardsPlaced", 0) or 0
        tp_ward_takedowns = tp.get("challenges", {}).get("wardTakedowns", 0) or 0
        tp_ward_takedowns_before_20m = tp.get("challenges", {}).get("wardTakedownsBefore20M", 0) or 0
        tp_skillshots_dodged = tp.get("challenges", {}).get("skillshotsDodged", 0) or 0
        tp_skillshots_hit = tp.get("challenges", {}).get("skillshotsHit", 0) or 0
        tp_longest_living = tp.get("longestTimeSpentLiving", 0) or 0

        tp_row = {
            "matchId": match_id,
            "puuid": tp["puuid"],
            "teamId": team_id,
            "win": 1 if tp["win"] else 0,
            "champion": tp["championName"],
            "role": tp.get("teamPosition", "UNKNOWN"),

            "kills": tp_kills,
            "deaths": tp_deaths,
            "assists": tp_assists,
            "kda": tp_kda,

            "goldPerMinute": tp_gold_per_minute,
            "damagePerMinute": tp_damage_per_minute,
            "visionScore": tp_vision,
            "visionScorePerMinute": tp_vision_per_min,

            "killParticipation": tp_kill_participation,
            "soloKills": tp_solo_kills,

            "controlWardsPlaced": tp_control_wards,
            "wardTakedowns": tp_ward_takedowns,
            "wardTakedownsBefore20M": tp_ward_takedowns_before_20m,

            "skillshotsDodged": tp_skillshots_dodged,
            "skillshotsHit": tp_skillshots_hit,
            "longestTimeSpentLiving": tp_longest_living,

            "cs_15": tp_cs_15,
            "gold_15": tp_gold_15,
            "xp_15": tp_xp_15,
            "laneMinionsFirst10Minutes": tp_lane_minions_10,
            "jungleCsBefore10Minutes": tp_jungle_cs_10,
            "early_cs_total": tp_early_cs_total,
        }
        all_team_data.append(tp_row)

    df = pd.DataFrame(all_team_data)
    df_with_derived = add_derived_features(df)
    target_df = df_with_derived[df_with_derived["puuid"] == target_puuid]

    return target_df


# ============================================================================
# New API Endpoints: Analysis & Highlights
# ============================================================================

@app.post("/api/v1/analysis/generate")
async def generate_analysis(request: AnalysisGenerateRequest):
    """
    AI 분석 생성 (Impact Score + Gap Analysis)

    Args:
        request: 분석 생성 요청 (match_id, puuid, game_name, tag_line, tier)

    Returns:
        완전한 AI 분석 결과 (Impact Score + Gap Analysis + Player Stats)
    """
    try:
        if not impact_model:
            raise HTTPException(status_code=503, detail="Impact Score model not loaded")

        # 1. 매치 데이터 조회
        match_data = get_match_data(request.match_id)
        timeline_data = get_timeline_data(request.match_id)

        # 2. Feature 추출
        df = extract_features_from_match(match_data, timeline_data, request.puuid)
        row = df.iloc[0]

        # feature_cols에 없는 컬럼은 0으로 채우기
        for col in feature_cols:
            if col not in row.index:
                row[col] = 0.0

        # 3. Impact Score 계산
        impact_report = compute_player_impact(row, impact_model, impact_explainer, feature_cols)

        # 4. Gap Analysis
        player_stats_raw = match_analyzer.riot_client.extract_player_stats_from_match(
            match_data, request.puuid
        )

        if not player_stats_raw:
            raise ValueError(f"Player not found in match {request.match_id}")

        player_stats = match_analyzer.create_player_stats_from_match(player_stats_raw)
        gap_result = gap_analyzer.analyze_gap(
            player_stats=player_stats,
            tier=request.tier,
            division="I",
        )

        # 5. 응답 구성
        top_features = []
        for feat in impact_report["features"]["top"]:
            top_features.append({
                "name": feat["name"],
                "displayName": feat["displayName"],
                "direction": feat["direction"],
                "shap": feat["shap"],
                "value": feat["value"]
            })

        result = {
            "match_id": request.match_id,
            "puuid": request.puuid,
            "game_name": request.game_name,
            "tag_line": request.tag_line,
            "champion": impact_report["champion"],
            "role": impact_report["role"],
            "win": bool(impact_report["win"]),

            "impact_score": impact_report["impactScore"],
            "baselineProba": impact_report["baselineProba"],
            "predictedProba": impact_report["predictedProba"],
            "summary": impact_report["summary"],
            "top_features": top_features,

            "gap_analysis": gap_result.model_dump(),
            "game_duration": match_data["info"]["gameDuration"],
            "player_stats": player_stats.model_dump()
        }

        return result

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/api/v1/highlights/generate")
async def generate_highlights(
    video: UploadFile = File(...),
    game_name: str = Form(...),
    tag_line: str = Form(...),
    match_id: Optional[str] = Form(None),
    top_highlights: Optional[int] = Form(None),
    top_mistakes: Optional[int] = Form(None),
    impact_threshold: Optional[float] = Form(None)
):
    """
    하이라이트 클립 생성

    Args:
        video: 게임 영상 파일
        game_name: Riot ID 이름
        tag_line: Riot ID 태그
        match_id: 매치 ID (선택사항, 없으면 최근 매치 자동 조회)
        top_highlights: 잘한 장면 개수 (impact_threshold가 없을 때만 사용, 기본값 5)
        top_mistakes: 못한 장면 개수 (impact_threshold가 없을 때만 사용, 기본값 3)
        impact_threshold: Impact Score 임계치 (절대값 기준, 예: 5.0이면 ±5 이상만 선택)

    Returns:
        생성된 클립 정보
    """
    try:
        # Import highlight extraction functions
        from src.api.highlight_extractor import (
            extract_highlights_from_timeline,
            create_clip,
            get_top_highlights
        )
        from src.api.impact_highlight_integration import (
            enrich_highlights_with_impact,
            generate_match_summary
        )

        # 1. PUUID 가져오기 (Riot API - 백엔드 API에 없음)
        puuid = get_puuid_from_riot_id(game_name, tag_line)

        # 2. 매치 ID가 없으면 최근 매치 조회 (백엔드 API 우선, fallback으로 Riot API)
        if not match_id:
            print(f"[INFO] No match_id provided, fetching recent matches for {game_name}#{tag_line}")
            match_ids = get_recent_matches_with_fallback(game_name, tag_line, puuid, count=1)
            if not match_ids:
                raise HTTPException(status_code=404, detail="No recent matches found")
            match_id = match_ids[0]
            print(f"[INFO] Using most recent match: {match_id}")

        # 3. 영상 저장
        video_filename = f"{match_id}_{puuid}.mp4"
        video_path = os.path.join(UPLOAD_DIR, video_filename)

        with open(video_path, "wb") as f:
            shutil.copyfileobj(video.file, f)

        print(f"[INFO] Saved video: {video_path}")

        # 4. 타임라인 데이터 가져오기
        timeline_data = get_timeline_data(match_id)
        match_data = get_match_data(match_id)

        # 4. 하이라이트 추출
        all_highlights = extract_highlights_from_timeline(timeline_data, puuid)

        if not all_highlights:
            raise HTTPException(status_code=404, detail="No highlights found for this player")

        # 5. Impact Score 모델과 연동
        participant_id = None
        for idx, p in enumerate(timeline_data['metadata']['participants']):
            if p == puuid:
                participant_id = idx + 1
                break

        if participant_id and impact_model:
            all_highlights = enrich_highlights_with_impact(
                all_highlights,
                timeline_data,
                match_data,
                participant_id,
                impact_model,
                feature_cols
            )

        # 6. 잘한 부분 / 못한 부분 분리
        if impact_threshold is not None:
            # Impact Score 임계치 기반 선택
            from src.api.highlight_extractor import filter_by_impact_threshold
            highlight_clips = filter_by_impact_threshold(all_highlights, threshold=impact_threshold, category='highlight')
            mistake_clips = filter_by_impact_threshold(all_highlights, threshold=impact_threshold, category='mistake')
        else:
            # 개수 기반 선택 (기본값 적용)
            _top_highlights = top_highlights if top_highlights is not None else 5
            _top_mistakes = top_mistakes if top_mistakes is not None else 3
            highlight_clips = get_top_highlights(all_highlights, top_n=_top_highlights, category='highlight')
            mistake_clips = get_top_highlights(all_highlights, top_n=_top_mistakes, category='mistake')

        # 7. 클립 생성
        created_highlights = []
        created_mistakes = []

        for h in highlight_clips:
            clip_path = create_clip(video_path, h, output_dir=os.path.join(CLIPS_DIR, match_id))
            if clip_path:
                created_highlights.append({
                    "clip_path": clip_path,
                    "timestamp": h["timestamp"],
                    "type": h["type"],
                    "base_importance": h["importance"],
                    "impact_score": h.get("impact_score", 0),
                    "combined_importance": h.get("combined_importance", h["importance"]),
                    "description": h["description"],
                    "impact_description": h.get("impact_description", ""),
                    "details": h["details"]
                })

        for h in mistake_clips:
            clip_path = create_clip(video_path, h, output_dir=os.path.join(CLIPS_DIR, match_id))
            if clip_path:
                created_mistakes.append({
                    "clip_path": clip_path,
                    "timestamp": h["timestamp"],
                    "type": h["type"],
                    "base_importance": h["importance"],
                    "impact_score": h.get("impact_score", 0),
                    "combined_importance": h.get("combined_importance", h["importance"]),
                    "description": h["description"],
                    "impact_description": h.get("impact_description", ""),
                    "details": h["details"]
                })

        # 8. 매치 정보 및 Impact Score 계산 (본인 + 팀원)
        player_info = None
        impact_result = None
        team_impact_scores = []

        # 먼저 본인 팀 찾기
        target_team_id = None
        for participant in match_data['info']['participants']:
            if participant['puuid'] == puuid:
                target_team_id = participant['teamId']
                break

        # 같은 팀의 모든 플레이어 Impact Score 계산 (한 번에 처리)
        if target_team_id and impact_model:
            try:
                # 팀 전체 feature 추출 (정규화를 위해)
                team_df = extract_team_features_from_match(match_data, timeline_data, target_team_id)

                for idx, row in team_df.iterrows():
                    player_puuid = row['puuid']

                    # feature_cols에 없는 컬럼은 0으로 채우기
                    for col in feature_cols:
                        if col not in row.index:
                            row[col] = 0.0

                    # Impact Score 계산
                    impact_report = compute_player_impact(row, impact_model, impact_explainer, feature_cols)

                    # 디버그 로그
                    print(f"[DEBUG] {row['champion']}: prob={impact_report['predictedProba']:.3f}, base={impact_report['baselineProba']:.3f}, score={impact_report['impactScore']:.1f}, grade={impact_report['grade']}")

                    # participant 정보 찾기
                    participant = None
                    for p in match_data['info']['participants']:
                        if p['puuid'] == player_puuid:
                            participant = p
                            break

                    if participant:
                        team_member = {
                            "puuid": player_puuid,
                            "champion": participant['championName'],
                            "role": participant['teamPosition'],
                            "summoner_name": participant.get('summonerName', participant.get('riotIdGameName', 'Unknown')),
                            "kills": participant['kills'],
                            "deaths": participant['deaths'],
                            "assists": participant['assists'],
                            "impact_score": impact_report["impactScore"],
                            "is_current_player": player_puuid == puuid
                        }

                        team_impact_scores.append(team_member)

                        # 본인 정보 상세 저장
                        if player_puuid == puuid:
                            # 응답용 데이터 구성
                            top_positive = []
                            top_negative = []

                            for feat in impact_report["features"]["top"]:
                                feat_data = {
                                    "name": feat["name"],
                                    "displayName": feat["displayName"],
                                    "value": feat["value"],
                                    "shap": feat["shap"]
                                }

                                # direction이 "up"이면 positive, "down"이면 negative
                                if feat["direction"] == "up":
                                    top_positive.append(feat_data)
                                else:
                                    top_negative.append(feat_data)

                            impact_result = {
                                "impactScore": impact_report["impactScore"],
                                "impact_score": impact_report["impactScore"],  # 하위 호환성
                                "grade": impact_report["grade"],
                                "baselineProba": impact_report["baselineProba"],
                                "baseline_proba": impact_report["baselineProba"],  # 하위 호환성
                                "predictedProba": impact_report["predictedProba"],
                                "predicted_proba": impact_report["predictedProba"],  # 하위 호환성
                                "summary": impact_report["summary"],
                                "matchComment": impact_report["matchComment"],
                                "scoreBreakdown": impact_report["scoreBreakdown"],
                                "features": impact_report["features"],
                                "champion": impact_report["champion"],
                                "role": impact_report["role"],
                                "win": impact_report["win"],
                                "top_positive_features": top_positive,
                                "top_negative_features": top_negative
                            }

                            player_info = {
                                "champion": participant['championName'],
                                "role": participant['teamPosition'],
                                "win": participant['win'],
                                "kills": participant['kills'],
                                "deaths": participant['deaths'],
                                "assists": participant['assists'],
                                "impact_result": impact_result
                            }

            except Exception as e:
                print(f"[WARN] Failed to calculate team impact scores: {e}")
                import traceback
                traceback.print_exc()

            # Impact Score 순으로 정렬
            team_impact_scores.sort(key=lambda x: x['impact_score'], reverse=True)

        # 9. Gap Analysis 계산
        gap_result = None
        if player_info:
            try:
                player_stats_raw = match_analyzer.riot_client.extract_player_stats_from_match(
                    match_data, puuid
                )
                if player_stats_raw:
                    player_stats = match_analyzer.create_player_stats_from_match(player_stats_raw)
                    # 기본 티어는 PLATINUM으로 설정 (추후 request에서 받을 수 있음)
                    gap_result = gap_analyzer.analyze_gap(
                        player_stats=player_stats,
                        tier="PLATINUM",
                        division="I",
                    )
                    print(type(gap_result))
                    print(gap_result)
            except Exception as e:
                print(f"[WARN] Gap Analysis 실패: {e}")

        # 10. 매치 요약 생성
        match_summary = None
        if participant_id:
            match_summary = generate_match_summary(all_highlights, match_data, participant_id)

        # 11. AI 리포트 생성 (OpenAI GPT-4)
        ai_report = None
        try:
            # 리포트 생성용 데이터 준비
            report_match_data = {
                'player': player_info,
                'game_info': {
                    'duration_seconds': match_data['info']['gameDuration'],
                    'duration_formatted': f"{match_data['info']['gameDuration'] // 60}:{match_data['info']['gameDuration'] % 60:02d}"
                }
            }

            # player_info에서 impact_result 추출
            player_impact_result = None
            if player_info and 'impact_result' in player_info:
                player_impact_result = player_info['impact_result']

            ai_report = generate_match_report(
                match_data=report_match_data,
                highlights=all_highlights,
                team_impact_scores=team_impact_scores,
                impact_result=player_impact_result,
                gap_result=gap_result
            )
        except Exception as e:
            print(f"AI 리포트 생성 실패: {e}")
            ai_report = "AI 리포트를 생성할 수 없습니다. OpenAI API 키를 확인해주세요."

        # 12. 백엔드 API에 결과 저장 (비동기적으로 실패해도 응답에 영향 없음)
        backend_api_saved = {
            "analysis": None,
            "highlights": [],
            "mistakes": []
        }

        try:
            # 분석 결과 저장
            analysis_result = save_analysis_to_team_api(match_id)
            if analysis_result:
                backend_api_saved["analysis"] = analysis_result
                print(f"[INFO] Backend API: Analysis saved for {match_id}")

            # 하이라이트 저장 (잘한 장면)
            if created_highlights:
                saved_highlights = save_highlights_batch_to_team_api(match_id, created_highlights)
                backend_api_saved["highlights"] = saved_highlights

            # 실수 클립 저장 (못한 장면)
            if created_mistakes:
                saved_mistakes = save_highlights_batch_to_team_api(match_id, created_mistakes)
                backend_api_saved["mistakes"] = saved_mistakes

        except Exception as e:
            print(f"[WARN] Backend API 저장 실패 (응답에는 영향 없음): {e}")

        return {
            "match_id": match_id,
            "player": {
                "gameName": game_name,
                "tagLine": tag_line,
                "puuid": puuid
            },
            "match_info": player_info,
            "match_summary": match_summary,
            "team_impact_scores": team_impact_scores,  # 팀원 영향력 추가
            "ai_report": ai_report,  # AI 리포트 추가
            "highlights": created_highlights,
            "mistakes": created_mistakes,
            "video_path": video_path,
            "total_clips": len(created_highlights) + len(created_mistakes),
            "backend_api_saved": backend_api_saved  # 백엔드 API 저장 결과
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Highlight generation failed: {str(e)}")


# For running with uvicorn
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
