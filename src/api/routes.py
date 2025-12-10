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
from datetime import datetime
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
    match_id: str = Form(...),
    game_name: str = Form(...),
    tag_line: str = Form(...),
    top_highlights: int = Form(5),
    top_mistakes: int = Form(3)
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

        return {
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
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Highlight generation failed: {str(e)}")


# For running with uvicorn
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
