"""FastAPI routes for the LoL Highlight & Analysis API"""

import os
import sys
import json
import time
import asyncio
import shutil
import pandas as pd
import numpy as np
import requests
from typing import Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from datetime import datetime
from dotenv import load_dotenv
from xgboost import XGBClassifier


def _to_native(obj):
    """numpy/pandas 타입을 Python native 타입으로 재귀 변환"""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_native(v) for v in obj]
    return obj

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
from src.services.gemini_coach import get_coaching
from src import __version__

# Load environment variables
load_dotenv()

# Import player_rating module
models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
if models_dir not in sys.path:
    sys.path.insert(0, models_dir)

from player_rating import compute_player_impact, add_derived_features

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

impact_model = None
feature_cols = []
impact_explainer = None

if os.path.exists(MODEL_PATH) and os.path.exists(FEAT_PATH):
    try:
        print(f"[INFO] Loading Impact Score model from {MODEL_PATH}")
        impact_model = XGBClassifier()
        impact_model.load_model(MODEL_PATH)

        with open(FEAT_PATH, "r") as f:
            feature_cols = json.load(f)

        print(f"[INFO] Impact Score model loaded with {len(feature_cols)} features")

        # Initialize SHAP explainer
        try:
            import shap
            print(f"[INFO] Initializing SHAP explainer...")
            impact_explainer = shap.TreeExplainer(impact_model)
            print(f"[INFO] SHAP explainer initialized successfully")
        except Exception as e:
            print(f"[WARN] Failed to initialize SHAP explainer: {e}")
            impact_explainer = None

    except Exception as e:
        print(f"[WARN] Failed to load Impact Score model: {e}")
        impact_model = None
else:
    print(f"[WARN] Impact Score model not found at {MODEL_PATH}")

# Riot API configuration
RIOT_API_KEY = os.getenv("RIOT_API_KEY")
ACCOUNT_REGION = "asia"
MATCH_REGION = "asia"

riot_session = requests.Session()
if RIOT_API_KEY:
    riot_session.headers.update({"X-Riot-Token": RIOT_API_KEY})

# Directories for file uploads
UPLOAD_DIR = "uploads"
CLIPS_DIR = "clips"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CLIPS_DIR, exist_ok=True)

app.mount("/clips", StaticFiles(directory=CLIPS_DIR), name="clips")


@app.get("/", response_model=HealthCheckResponse)
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


def get_match_data(match_id: str) -> dict:
    """매치 데이터 조회"""
    url = f"https://{MATCH_REGION}.api.riotgames.com/lol/match/v5/matches/{match_id}"
    return riot_get(url)


def get_timeline_data(match_id: str) -> dict:
    """타임라인 데이터 조회"""
    url = f"https://{MATCH_REGION}.api.riotgames.com/lol/match/v5/matches/{match_id}/timeline"
    return riot_get(url)


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

    # DataFrame 생성
    all_team_data = []

    for tp in team_players:
        tp_kills = tp["kills"]
        tp_deaths = tp["deaths"]
        tp_assists = tp["assists"]
        tp_kda = (tp_kills + tp_assists) / max(1, tp_deaths)
        tp_vision = tp.get("visionScore", 0)

        is_target = (tp["puuid"] == target_puuid)

        if is_target:
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
        else:
            row_data = {
                "matchId": match_id,
                "puuid": tp["puuid"],
                "teamId": team_id,
                "win": 1 if tp["win"] else 0,
                "kda": tp_kda,
                "visionScore": tp_vision,
                "deaths": tp_deaths,
            }

        all_team_data.append(row_data)

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

        gap_dump = gap_result.model_dump()
        base_strengths = gap_dump.get("strengths", [])
        base_weaknesses = gap_dump.get("weaknesses", [])
        base_recommendations = gap_dump.get("recommendations", [])

        # 6. Gemini 코칭 (실패 시 rule-based fallback)
        kda = f"{player_stats.kills}/{player_stats.deaths}/{player_stats.assists}"
        game_duration_min = match_data["info"]["gameDuration"] / 60

        participant_id = None
        summoner_spells = None
        ally_champions = []
        enemy_champions = []
        player_team_id = None
        player_role = impact_report.get("role", "")
        for idx, p in enumerate(match_data["info"]["participants"]):
            if p["puuid"] == request.puuid:
                participant_id = idx + 1
                player_team_id = p.get("teamId")
                summoner_spells = (p.get("summoner1Id", 0), p.get("summoner2Id", 0))
                if not player_role:
                    player_role = p.get("teamPosition", "")
                break

        bot_partner_pos_a = "UTILITY" if player_role == "BOTTOM" else ("BOTTOM" if player_role == "UTILITY" else None)
        bot_partner_name_a = None
        other_allies_a = []
        for p in match_data["info"]["participants"]:
            if p["puuid"] == request.puuid:
                continue
            name = p.get("championName", "")
            pos = p.get("teamPosition", "")
            if p.get("teamId") == player_team_id:
                if bot_partner_pos_a and pos == bot_partner_pos_a:
                    bot_partner_name_a = name
                else:
                    other_allies_a.append(name)
            else:
                enemy_champions.append(name)
        ally_champions = ([bot_partner_name_a] + other_allies_a) if bot_partner_name_a else other_allies_a

        llm_coaching = get_coaching(
            champion=impact_report.get("champion", ""),
            role=impact_report.get("role", ""),
            win=bool(impact_report.get("win", False)),
            tier=request.tier,
            kda=kda,
            cs_per_min=player_stats.cs_per_min,
            damage_share=player_stats.damage_share * 100,
            vision_score=player_stats.vision_score_per_min,
            gold_per_min=player_stats.gold_per_min,
            game_duration_min=game_duration_min,
            strengths=base_strengths,
            weaknesses=base_weaknesses,
            impact_score=impact_report.get("impactScore", 50.0),
            timeline_data=timeline_data,
            participant_id=participant_id,
            summoner_spells=summoner_spells,
            ally_champions=ally_champions,
            enemy_champions=enemy_champions,
        )

        if llm_coaching:
            gap_dump["strengths"] = base_strengths
            gap_dump["weaknesses"] = base_weaknesses
            gap_dump["recommendations"] = llm_coaching.get("improvements") or base_recommendations
            gap_dump["coaching_summary"]  = llm_coaching.get("summary", "")
            gap_dump["coaching_early_game"]  = llm_coaching.get("early_game", "")
            gap_dump["coaching_mid_game"]    = llm_coaching.get("mid_game", "")
            gap_dump["coaching_late_game"]   = llm_coaching.get("late_game", "")
            gap_dump["coaching_key_pattern"] = llm_coaching.get("key_pattern", "")

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

            "gap_analysis": gap_dump,
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
    match_id: str = Form(...),
    game_name: str = Form(...),
    tag_line: str = Form(...),
    top_highlights: int = Form(5),
    top_mistakes: int = Form(3),
    champion: str = Form(""),
    role: str = Form(""),
    game_start_offset: float = Form(0.0),
):
    """
    하이라이트 클립 생성

    Args:
        video: 게임 영상 파일
        match_id: 매치 ID
        game_name: Riot ID 이름
        tag_line: Riot ID 태그
        top_highlights: 잘한 장면 개수
        top_mistakes: 못한 장면 개수

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

        # 1. PUUID 가져오기
        puuid = get_puuid_from_riot_id(game_name, tag_line)

        # 2. 영상 저장
        video_filename = f"{match_id}_{puuid}.mp4"
        video_path = os.path.join(UPLOAD_DIR, video_filename)

        with open(video_path, "wb") as f:
            shutil.copyfileobj(video.file, f)

        print(f"[INFO] Saved video: {video_path}")

        # 3. 타임라인 데이터 가져오기
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
        highlight_clips = get_top_highlights(all_highlights, top_n=top_highlights, category='highlight')
        mistake_clips = get_top_highlights(all_highlights, top_n=top_mistakes, category='mistake')

        # 챔피언/포지션/소환사주문/팀 구성 match_data에서 직접 조회
        clip_champion = champion
        clip_role = role
        clip_summoner_spells = None
        clip_ally_champions = []
        clip_enemy_champions = []
        clip_lane_opponent = None
        clip_player_team_id = None
        for p in match_data["info"]["participants"]:
            if p["puuid"] == puuid:
                clip_champion = clip_champion or p.get("championName", "")
                clip_role = clip_role or p.get("teamPosition", "")
                clip_summoner_spells = (p.get("summoner1Id", 0), p.get("summoner2Id", 0))
                clip_player_team_id = p.get("teamId")
                break
        # 바텀 파트너 포지션: 플레이어가 BOTTOM이면 파트너는 UTILITY, 반대도 마찬가지
        bot_partner_pos = None
        if clip_role == "BOTTOM":
            bot_partner_pos = "UTILITY"
        elif clip_role == "UTILITY":
            bot_partner_pos = "BOTTOM"

        bot_partner_name = None
        other_allies = []
        for p in match_data["info"]["participants"]:
            if p["puuid"] == puuid:
                continue
            name = p.get("championName", "")
            pos = p.get("teamPosition", "")
            if p.get("teamId") == clip_player_team_id:
                if bot_partner_pos and pos == bot_partner_pos:
                    bot_partner_name = name  # 실제 바텀 파트너
                else:
                    other_allies.append(name)
            else:
                clip_enemy_champions.append(name)
                # 같은 포지션 = 직접 상대 라이너
                if clip_role and pos == clip_role:
                    clip_lane_opponent = name

        # 바텀 파트너를 ally_champions 맨 앞에 배치해 analyze_clip의 bot_partner 로직이 올바르게 동작하도록
        if bot_partner_name:
            clip_ally_champions = [bot_partner_name] + other_allies
        else:
            clip_ally_champions = other_allies

        # 타임라인 스노우볼 데이터 미리 계산
        moments_map = {}
        if participant_id:
            try:
                from src.services.gemini_coach import extract_game_moments
                moments = extract_game_moments(timeline_data, participant_id, top_n=20)
                for m in moments:
                    moments_map[m["time"]] = m
            except Exception as e:
                print(f"[WARN] Failed to extract moments: {e}")

        # 7. 클립 생성 (동기 — FFmpeg)
        # game_start_offset: 녹화 시작 시점의 게임 내 시간(초, 보통 음수 e.g. -85)
        # 영상 내 실제 위치 = event_timestamp - game_start_offset
        all_clip_data = []  # (h, clip_path, category)

        for h in highlight_clips:
            clip_path = create_clip(video_path, h, output_dir=os.path.join(CLIPS_DIR, match_id),
                                    game_start_offset=game_start_offset)
            if clip_path:
                all_clip_data.append((h, clip_path, "highlight"))

        for h in mistake_clips:
            clip_path = create_clip(video_path, h, output_dir=os.path.join(CLIPS_DIR, match_id),
                                    game_start_offset=game_start_offset)
            if clip_path:
                all_clip_data.append((h, clip_path, "mistake"))

        # 8. Gemini 코칭 — impact_score 기준 highlight 1개, mistake 1개만 분석
        from src.services.gemini_coach import analyze_clip_async

        async def _coaching_task(h, clip_path):
            ts = h["timestamp"]
            minute = int(ts // 60)
            second = int(ts % 60)
            time_str = f"{minute}분{second:02d}초"
            m = moments_map.get(time_str)
            try:
                return await analyze_clip_async(
                    clip_path=clip_path,
                    event_kind=h["type"],
                    time_str=time_str,
                    pre_score=m["pre_score"] if m else "?:?",
                    pre_gold_diff=m["pre_gold_diff"] if m else 0,
                    snowball_gold=m["gold_delta"] if m else 0,
                    snowball_label=m["label"] if m else "직후",
                    champion=clip_champion,
                    role=clip_role,
                    summoner_spells=clip_summoner_spells,
                    ally_champions=clip_ally_champions,
                    enemy_champions=clip_enemy_champions,
                    lane_opponent=clip_lane_opponent,
                    event_sec_in_clip=h.get("clip_event_sec"),
                )
            except Exception as e:
                print(f"[WARN] Coaching task failed: {e}")
                return None

        # impact_score > combined_importance > importance 순으로 top-1 선택
        def _top_clip(clips_with_category, category):
            candidates = [(h, cp) for h, cp, cat in clips_with_category if cat == category]
            if not candidates:
                return None
            return max(candidates, key=lambda x: (
                x[0].get("impact_score", 0),
                x[0].get("combined_importance", 0),
                x[0].get("importance", 0),
            ))

        top_highlight = _top_clip(all_clip_data, "highlight")
        top_mistake   = _top_clip(all_clip_data, "mistake")
        # id() 대신 dict에 직접 플래그 마킹 (id()는 메모리 재사용으로 충돌 가능)
        for h, _, _ in all_clip_data:
            h["_use_gemini"] = False
        if top_highlight:
            top_highlight[0]["_use_gemini"] = True
        if top_mistake:
            top_mistake[0]["_use_gemini"] = True
        print(f"[DEBUG] Gemini targets: highlight={top_highlight[0]['type'] if top_highlight else None}, "
              f"mistake={top_mistake[0]['type'] if top_mistake else None}")

        def _fallback_coaching(h, category):
            """Gemini 실패 시 이벤트 데이터 기반 기본 피드백 생성"""
            ts = h["timestamp"]
            minute = int(ts // 60)
            second = int(ts % 60)
            time_str = f"{minute}분{second:02d}초"
            m = moments_map.get(time_str)
            event_kind = h["type"]
            impact = h.get("impact_score", 0)
            snowball = m["gold_delta"] if m else 0
            gold_diff = m["pre_gold_diff"] if m else 0

            if category == "highlight":
                if snowball > 1000:
                    return f"{time_str} 킬 이후 {snowball:,}G 골드 차이를 만들어낸 핵심 플레이입니다. 이런 상황에서의 오브젝트 전환 타이밍을 유지하세요."
                elif impact > 10:
                    return f"{time_str} 승률에 {impact:.1f}% 기여한 플레이입니다. 당시 골드차 {'+' if gold_diff >= 0 else ''}{gold_diff:,}G 상황에서 좋은 결정을 내렸습니다."
                else:
                    return f"{time_str} 긍정적인 기여를 한 장면입니다. 이 패턴을 다음 게임에서도 유지하세요."
            else:
                if abs(snowball) > 1000:
                    return f"{time_str} 데스 이후 {abs(snowball):,}G 손실이 발생했습니다. 해당 시점의 포지셔닝과 시야 확보를 재검토하세요."
                elif abs(impact) > 5:
                    return f"{time_str} 데스로 인해 승률이 {abs(impact):.1f}% 감소했습니다. 당시 골드차 {'+' if gold_diff >= 0 else ''}{gold_diff:,}G 상황에서 교전 판단을 재고해보세요."
                else:
                    return f"{time_str} 불필요한 데스 장면입니다. 시야 없이 진입하거나 불리한 교전을 피하는 습관을 기르세요."

        async def _maybe_coaching(h, clip_path, category):
            if h.get("_use_gemini"):
                print(f"[DEBUG] Calling Gemini for {category}: {h['type']} at {h['timestamp']:.0f}s")
                result = await _coaching_task(h, clip_path)
                if result is not None:
                    return result
                print(f"[DEBUG] Gemini failed for {category}, falling back")
            return _fallback_coaching(h, category)

        coaching_results = await asyncio.gather(
            *[_maybe_coaching(h, cp, cat) for h, cp, cat in all_clip_data]
        )

        # 결과 조합
        created_highlights = []
        created_mistakes = []
        for i, (h, clip_path, category) in enumerate(all_clip_data):
            coaching = coaching_results[i] if not isinstance(coaching_results[i], Exception) else _fallback_coaching(h, category)
            clip_dict = {
                "clip_path": clip_path,
                "timestamp": float(h["timestamp"]),
                "type": h["type"],
                "base_importance": float(h["importance"]),
                "impact_score": float(h.get("impact_score", 0)),
                "combined_importance": float(h.get("combined_importance", h["importance"])),
                "description": h["description"],
                "impact_description": h.get("impact_description", ""),
                "details": h["details"],
                "coaching": coaching,
            }
            if category == "highlight":
                created_highlights.append(clip_dict)
            else:
                created_mistakes.append(clip_dict)

        # 8. 매치 정보 추가
        player_info = None
        for participant in match_data['info']['participants']:
            if participant['puuid'] == puuid:
                player_info = {
                    "championName": participant['championName'],
                    "teamPosition": participant['teamPosition'],
                    "win": participant['win'],
                    "kills": participant['kills'],
                    "deaths": participant['deaths'],
                    "assists": participant['assists']
                }
                break

        # 9. 매치 요약 생성
        match_summary = None
        if participant_id:
            match_summary = generate_match_summary(all_highlights, match_data, participant_id)

        result = _to_native({
            "match_id": match_id,
            "player": {
                "gameName": game_name,
                "tagLine": tag_line,
                "puuid": puuid
            },
            "match_info": player_info,
            "match_summary": match_summary,
            "highlights": created_highlights,
            "mistakes": created_mistakes,
            "video_path": video_path,
            "total_clips": len(created_highlights) + len(created_mistakes)
        })
        return JSONResponse(content=result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Highlight generation failed: {str(e)}")


# For running with uvicorn
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
