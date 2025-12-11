# AI/src/api/impact_score.py

import os
import sys
import json
import time
import pandas as pd
from typing import Dict, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import shap
from xgboost import XGBClassifier
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# player_rating.py 함수 import
# AI/src/models 디렉토리를 절대 경로로 추가
models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
if models_dir not in sys.path:
    sys.path.insert(0, models_dir)

from player_rating import compute_player_impact, add_derived_features

app = FastAPI(title="LoL Player Impact API", version="1.0.0")

# ---------------------------
# 1. 모델 로딩
# ---------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, "models")  # AI/models 폴더

MODEL_PATH = os.path.join(MODEL_DIR, "player_impact_model.json")
FEAT_PATH = os.path.join(MODEL_DIR, "feature_cols.json")

model = None
feature_cols = []

if os.path.exists(MODEL_PATH) and os.path.exists(FEAT_PATH):
    try:
        print(f"[INFO] Loading model from {MODEL_PATH}")
        model = XGBClassifier()
        model.load_model(MODEL_PATH)

        print(f"[INFO] Loading feature columns from {FEAT_PATH}")
        with open(FEAT_PATH, "r") as f:
            feature_cols = json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to load model: {e}")
        model = None
        feature_cols = []
else:
    print(f"[WARN] Model files not found at {MODEL_DIR}")

explainer = None
if model is not None:
    print(f"[INFO] Building SHAP explainer from model...")
    # SHAP explainer를 pickle로 저장/로드하는 대신 모델에서 재생성
    # TreeExplainer는 모델만 있으면 생성 가능 (훈련 데이터 불필요)
    try:
        if hasattr(model, 'get_booster'):
            import xgboost as xgb
            import tempfile

            booster = model.get_booster()

            # SHAP 버전 호환성 문제 해결: base_score 파싱 에러 우회
            # XGBoost의 새로운 형식 '[5E-1]'을 SHAP가 파싱하지 못하는 문제
            try:
                import json as json_lib
                model_dump = booster.save_config()
                model_dict = json_lib.loads(model_dump)

                # base_score가 리스트 형식인 경우 첫 번째 값만 사용
                if 'learner' in model_dict and 'learner_model_param' in model_dict['learner']:
                    base_score = model_dict['learner']['learner_model_param'].get('base_score', '0.5')
                    if isinstance(base_score, str) and base_score.startswith('['):
                        # '[5E-1]' -> '5E-1' -> 0.5
                        base_score_clean = base_score.strip('[]')
                        model_dict['learner']['learner_model_param']['base_score'] = base_score_clean

                        print(f"[INFO] Fixed base_score: {base_score} -> {base_score_clean}")

                        # 수정된 config로 완전히 새로운 Booster 생성
                        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                            temp_path = f.name
                            json_lib.dump(model_dict, f)

                        # 새 Booster 생성 및 로드
                        new_booster = xgb.Booster()
                        new_booster.load_config(json_lib.dumps(model_dict))
                        new_booster.load_model(MODEL_PATH)
                        booster = new_booster

                        os.unlink(temp_path)
            except Exception as e:
                print(f"[WARN] Could not fix base_score, trying without fix: {e}")

            explainer = shap.TreeExplainer(booster)
        else:
            explainer = shap.TreeExplainer(model)
    except Exception as e:
        print(f"[WARN] SHAP explainer build failed (XGBoost/SHAP version incompatibility): {e}")
        print("[INFO] API will work without SHAP explainer (using basic feature importance)")
        explainer = None

if model is not None:
    print(f"[INFO] Model and explainer loaded successfully with {len(feature_cols)} features")
else:
    print(f"[INFO] Running without model - some features will be unavailable")

# ---------------------------
# 2. Riot API 설정
# ---------------------------
RIOT_API_KEY = os.getenv("RIOT_API_KEY")
if not RIOT_API_KEY:
    print("[WARN] RIOT_API_KEY not set. API endpoints requiring Riot API will fail.")

ACCOUNT_REGION = "asia"
MATCH_REGION = "asia"

session = requests.Session()
if RIOT_API_KEY:
    session.headers.update({"X-Riot-Token": RIOT_API_KEY})


# ---------------------------
# 3. Pydantic Models
# ---------------------------
class RiotIdRequest(BaseModel):
    """Riot ID를 입력받아 플레이어 영향력 점수를 계산"""
    gameName: str
    tagLine: str
    match_index: int = 0  # 최근 경기 중 몇 번째 (0 = 가장 최근)


# ---------------------------
# 4. Riot API 헬퍼 함수
# ---------------------------
def riot_get(url: str, params: Optional[dict] = None) -> dict:
    """Riot API GET 요청"""
    if not RIOT_API_KEY:
        raise HTTPException(status_code=500, detail="RIOT_API_KEY not configured")

    time.sleep(0.3)  # Rate limit 방지
    resp = session.get(url, params=params)

    if resp.status_code == 429:
        retry_after = int(resp.headers.get("Retry-After", 2))
        time.sleep(retry_after)
        resp = session.get(url, params=params)

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


def get_recent_match_ids(puuid: str, count: int = 5, queue: int = 420) -> list:
    """최근 매치 ID 목록 조회 (queue 420 = 솔로랭크)"""
    url = f"https://{MATCH_REGION}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids"
    params = {"start": 0, "count": count, "queue": queue}
    return riot_get(url, params=params)


def get_match_data(match_id: str) -> dict:
    """매치 데이터 조회"""
    url = f"https://{MATCH_REGION}.api.riotgames.com/lol/match/v5/matches/{match_id}"
    return riot_get(url)


def get_timeline_data(match_id: str) -> dict:
    """타임라인 데이터 조회"""
    url = f"https://{MATCH_REGION}.api.riotgames.com/lol/match/v5/matches/{match_id}/timeline"
    return riot_get(url)


# ---------------------------
# 5. Feature 추출 함수
# ---------------------------
def extract_features_from_match(match_data: dict, timeline_data: dict, target_puuid: str) -> pd.DataFrame:
    """
    매치 데이터와 타임라인에서 모델이 필요로 하는 feature를 추출

    필요한 21개 feature:
    - cs_15, gold_15, xp_15, early_cs_total
    - laneMinionsFirst10Minutes, jungleCsBefore10Minutes
    - goldPerMinute, damagePerMinute, visionScorePerMinute
    - kda, kda_norm, killParticipation, soloKills
    - vision_norm, controlWardsPlaced, wardTakedowns, wardTakedownsBefore20M
    - skillshotsDodged, skillshotsHit, longestTimeSpentLiving
    - death_rate
    """
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

    # ---------------------------
    # Raw 스탯 추출
    # ---------------------------
    kills = player["kills"]
    deaths = player["deaths"]
    assists = player["assists"]

    # KDA 계산
    kda = (kills + assists) / max(1, deaths)

    # Per-minute 스탯
    gold_per_minute = player.get("challenges", {}).get("goldPerMinute", 0) or 0
    damage_per_minute = player.get("challenges", {}).get("damagePerMinute", 0) or 0
    vision_score = player.get("visionScore", 0)
    vision_score_per_minute = vision_score / max(1, game_minutes)

    # 킬 관여율
    kill_participation = player.get("challenges", {}).get("killParticipation", 0) or 0

    # 솔로킬
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

    # ---------------------------
    # 타임라인 기반 feature
    # ---------------------------
    frames = timeline_data["info"]["frames"]
    participant_id = player["participantId"]

    # 15분 스탯 (frame 15 또는 마지막 프레임)
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

    # ---------------------------
    # DataFrame 생성 (add_derived_features를 위해)
    # ---------------------------
    # 같은 팀 플레이어들의 데이터도 포함 (정규화를 위해)
    all_team_data = []

    for tp in team_players:
        tp_kills = tp["kills"]
        tp_deaths = tp["deaths"]
        tp_assists = tp["assists"]
        tp_kda = (tp_kills + tp_assists) / max(1, tp_deaths)
        tp_vision = tp.get("visionScore", 0)

        is_target = (tp["puuid"] == target_puuid)

        # Timeline 데이터는 target만 상세하게, 나머지는 기본값
        if is_target:
            row_data = {
                "matchId": match_id,
                "puuid": target_puuid,
                "teamId": team_id,
                "win": 1 if win else 0,
                "champion": champion,
                "role": role,

                # Raw features
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

                # Timeline features
                "cs_15": cs_15,
                "gold_15": gold_15,
                "xp_15": xp_15,
                "laneMinionsFirst10Minutes": lane_minions_10,
                "jungleCsBefore10Minutes": jungle_cs_10,
                "early_cs_total": early_cs_total,
            }
        else:
            # 팀원은 정규화용 최소 데이터만
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

    # DataFrame 생성
    df = pd.DataFrame(all_team_data)

    # add_derived_features 적용 (kda_norm, vision_norm, death_rate 계산)
    df_with_derived = add_derived_features(df)

    # target 플레이어만 반환
    target_df = df_with_derived[df_with_derived["puuid"] == target_puuid]

    return target_df


# ---------------------------
# 6. API Endpoints
# ---------------------------
@app.get("/")
def root():
    """API 정보"""
    return {
        "name": "LoL Player Impact API",
        "version": "1.0.0",
        "features": len(feature_cols),
        "endpoints": [
            "/impact/by-riot-id",
            "/health"
        ]
    }


@app.get("/health")
def health_check():
    """헬스 체크"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "explainer_loaded": explainer is not None,
        "features": len(feature_cols)
    }


@app.post("/impact/by-riot-id")
def get_impact_by_riot_id(req: RiotIdRequest):
    """
    Riot ID로 플레이어 영향력 점수 조회

    Args:
        gameName: Riot ID 이름 (예: "Hide on bush")
        tagLine: Riot ID 태그 (예: "KR1")
        match_index: 최근 경기 중 몇 번째 경기인지 (0 = 가장 최근)

    Returns:
        영향력 점수 및 상세 리포트
    """
    try:
        # 1. Riot ID -> PUUID
        puuid = get_puuid_from_riot_id(req.gameName, req.tagLine)

        # 2. 최근 매치 ID 조회
        match_ids = get_recent_match_ids(puuid, count=max(req.match_index + 1, 5))

        if not match_ids:
            raise HTTPException(status_code=404, detail="No ranked matches found")

        if req.match_index >= len(match_ids):
            raise HTTPException(
                status_code=400,
                detail=f"Match index {req.match_index} out of range. Player has {len(match_ids)} recent matches."
            )

        match_id = match_ids[req.match_index]

        # 3. 매치 데이터 조회
        match_data = get_match_data(match_id)
        timeline_data = get_timeline_data(match_id)

        # 4. Feature 추출
        df = extract_features_from_match(match_data, timeline_data, puuid)

        # 5. 모델 예측 (compute_player_impact 사용)
        row = df.iloc[0]

        # feature_cols에 없는 컬럼은 0으로 채우기
        for col in feature_cols:
            if col not in row.index:
                row[col] = 0.0

        report = compute_player_impact(row, model, explainer, feature_cols)

        # 6. 응답 포맷팅
        report["riotId"] = f"{req.gameName}#{req.tagLine}"

        return report

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    print("[INFO] Starting API server on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
