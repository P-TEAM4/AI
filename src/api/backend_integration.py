"""
백엔드 통합용 API 엔드포인트
백엔드에서 이미 수집한 매치 데이터를 받아서 분석하는 API
"""

import os
import sys
import json
import pandas as pd
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from datetime import datetime
from xgboost import XGBClassifier
import shap

# player_rating.py 함수 import
models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
if models_dir not in sys.path:
    sys.path.insert(0, models_dir)

from player_rating import compute_player_impact, add_derived_features

# ---------------------------
# Pydantic Models
# ---------------------------

class MatchData(BaseModel):
    """백엔드로부터 받는 매치 데이터 형식"""
    match_id: str = Field(description="매치 ID")
    game_duration: int = Field(description="게임 시간 (초)")
    game_creation: int = Field(description="게임 생성 시각 (timestamp)")
    queue_id: int = Field(description="큐 타입 (420=솔로랭크)")


class PlayerMatchStats(BaseModel):
    """플레이어 한 명의 매치 통계"""
    puuid: str
    summoner_name: str = Field(alias="summonerName")
    game_name: str = Field(alias="gameName")
    tag_line: str = Field(alias="tagLine")

    # 기본 정보
    champion_name: str = Field(alias="championName")
    champion_id: int = Field(alias="championId")
    team_id: int = Field(alias="teamId")
    team_position: str = Field(alias="teamPosition")
    participant_id: int = Field(alias="participantId")
    win: bool

    # KDA
    kills: int
    deaths: int
    assists: int

    # 골드/경험치
    gold_earned: int = Field(alias="goldEarned")
    gold_spent: int = Field(alias="goldSpent")

    # CS
    total_minions_killed: int = Field(alias="totalMinionsKilled")
    neutral_minions_killed: int = Field(alias="neutralMinionsKilled")

    # 데미지
    total_damage_dealt_to_champions: int = Field(alias="totalDamageDealtToChampions")
    magic_damage_dealt_to_champions: int = Field(alias="magicDamageDealtToChampions")
    physical_damage_dealt_to_champions: int = Field(alias="physicalDamageDealtToChampions")
    true_damage_dealt_to_champions: int = Field(alias="trueDamageDealtToChampions")

    # 시야
    vision_score: int = Field(alias="visionScore")
    wards_placed: int = Field(alias="wardsPlaced")
    wards_killed: int = Field(alias="wardsKilled")
    vision_wards_bought_in_game: int = Field(alias="visionWardsBoughtInGame")

    # Challenges (optional)
    challenges: Optional[Dict[str, Any]] = None

    # 생존 시간
    longest_time_spent_living: Optional[int] = Field(default=0, alias="longestTimeSpentLiving")

    class Config:
        populate_by_name = True


class TimelineFrame(BaseModel):
    """타임라인 프레임 데이터"""
    timestamp: int
    participant_id: int = Field(alias="participantId")
    level: int
    current_gold: int = Field(alias="currentGold")
    total_gold: int = Field(alias="totalGold")
    xp: int
    minions_killed: int = Field(alias="minionsKilled")
    jungle_minions_killed: int = Field(alias="jungleMinionsKilled")

    class Config:
        populate_by_name = True


class TimelineData(BaseModel):
    """타임라인 데이터 (간소화 버전)"""
    frames: Dict[str, List[TimelineFrame]] = Field(
        description="각 분(minute)별 프레임 데이터. Key는 분 단위 (예: '10', '15')"
    )


class MatchAnalysisRequest(BaseModel):
    """백엔드로부터 받는 전체 매치 분석 요청"""
    match_data: MatchData
    player_stats: PlayerMatchStats
    timeline_data: Optional[TimelineData] = None
    tier: Optional[str] = Field(default="GOLD", description="플레이어 티어 (Gap Analysis용)")


class BatchMatchAnalysisRequest(BaseModel):
    """여러 매치를 한 번에 분석하는 요청"""
    matches: List[MatchAnalysisRequest] = Field(description="분석할 매치 목록")
    aggregate: bool = Field(default=False, description="평균 지표 계산 여부")


class ImpactScoreResult(BaseModel):
    """Impact Score 분석 결과"""
    impact_score: float = Field(description="최종 영향력 점수 (-100 ~ +100)")
    baseline_proba: float = Field(alias="baselineProba", description="베이스라인 승률")
    predicted_proba: float = Field(alias="predictedProba", description="예측 승률")
    summary: str = Field(description="한 줄 요약")

    # 주요 기여 지표
    top_positive_features: List[Dict[str, Any]] = Field(alias="topPositiveFeatures")
    top_negative_features: List[Dict[str, Any]] = Field(alias="topNegativeFeatures")

    # 원본 통계
    raw_stats: Dict[str, float] = Field(alias="rawStats")

    class Config:
        populate_by_name = True


class MatchAnalysisResponse(BaseModel):
    """매치 분석 응답"""
    match_id: str
    puuid: str
    game_name: str
    tag_line: str
    champion: str
    role: str
    win: bool
    game_duration: int

    # Impact Score 결과
    impact_result: ImpactScoreResult = Field(alias="impactResult")

    # 타임스탬프
    analyzed_at: datetime = Field(alias="analyzedAt", default_factory=datetime.now)

    class Config:
        populate_by_name = True


class BatchAnalysisResponse(BaseModel):
    """일괄 분석 응답"""
    total_matches: int = Field(alias="totalMatches")
    successful: int
    failed: int
    results: List[MatchAnalysisResponse]

    # 평균 지표 (aggregate=True일 때)
    average_impact_score: Optional[float] = Field(default=None, alias="averageImpactScore")
    win_rate: Optional[float] = Field(default=None, alias="winRate")
    avg_stats: Optional[Dict[str, float]] = Field(default=None, alias="avgStats")

    class Config:
        populate_by_name = True


# ---------------------------
# Feature 추출 함수
# ---------------------------

def extract_features_from_backend_data(
    player_stats: PlayerMatchStats,
    timeline_data: Optional[TimelineData],
    game_duration: int,
    team_players: Optional[List[PlayerMatchStats]] = None
) -> pd.DataFrame:
    """
    백엔드에서 받은 데이터로부터 모델이 필요로 하는 feature 추출

    Args:
        player_stats: 플레이어 통계
        timeline_data: 타임라인 데이터 (optional)
        game_duration: 게임 시간 (초)
        team_players: 같은 팀 플레이어들 (정규화용)

    Returns:
        Feature DataFrame
    """
    game_minutes = game_duration / 60.0

    # ---------------------------
    # 기본 스탯 추출
    # ---------------------------
    kills = player_stats.kills
    deaths = player_stats.deaths
    assists = player_stats.assists
    kda = (kills + assists) / max(1, deaths)

    # Per-minute 스탯
    gold_per_minute = player_stats.gold_earned / max(1, game_minutes)
    damage_per_minute = player_stats.total_damage_dealt_to_champions / max(1, game_minutes)
    vision_score = player_stats.vision_score
    vision_score_per_minute = vision_score / max(1, game_minutes)

    # CS
    cs_total = player_stats.total_minions_killed + player_stats.neutral_minions_killed

    # Challenges에서 추출
    challenges = player_stats.challenges or {}
    kill_participation = challenges.get("killParticipation", 0) or 0
    solo_kills = challenges.get("soloKills", 0) or 0
    control_wards_placed = challenges.get("controlWardsPlaced", 0) or 0
    ward_takedowns = player_stats.wards_killed
    ward_takedowns_before_20m = challenges.get("wardTakedownsBefore20M", 0) or 0
    skillshots_dodged = challenges.get("skillshotsDodged", 0) or 0
    skillshots_hit = challenges.get("skillshotsHit", 0) or 0

    # 생존 시간
    longest_time_spent_living = player_stats.longest_time_spent_living or 0

    # ---------------------------
    # 타임라인 기반 feature (있으면)
    # ---------------------------
    cs_15 = 0
    gold_15 = 0
    xp_15 = 0
    lane_minions_10 = 0
    jungle_cs_10 = 0

    if timeline_data and timeline_data.frames:
        frames = timeline_data.frames
        pid = str(player_stats.participant_id)

        # 15분 데이터
        if "15" in frames:
            for frame in frames["15"]:
                if frame.participant_id == player_stats.participant_id:
                    cs_15 = frame.minions_killed + frame.jungle_minions_killed
                    gold_15 = frame.total_gold
                    xp_15 = frame.xp
                    break

        # 10분 데이터
        if "10" in frames:
            for frame in frames["10"]:
                if frame.participant_id == player_stats.participant_id:
                    lane_minions_10 = frame.minions_killed
                    jungle_cs_10 = frame.jungle_minions_killed
                    break

    early_cs_total = lane_minions_10 + jungle_cs_10

    # ---------------------------
    # DataFrame 생성
    # ---------------------------
    row_data = {
        "matchId": player_stats.puuid + "_temp",  # 임시
        "puuid": player_stats.puuid,
        "teamId": player_stats.team_id,
        "win": 1 if player_stats.win else 0,
        "champion": player_stats.champion_name,
        "role": player_stats.team_position,

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

    # 팀원 데이터가 있으면 함께 처리 (정규화용)
    if team_players:
        all_team_data = []

        for tp in team_players:
            tp_kda = (tp.kills + tp.assists) / max(1, tp.deaths)
            is_target = (tp.puuid == player_stats.puuid)

            if is_target:
                all_team_data.append(row_data)
            else:
                all_team_data.append({
                    "matchId": tp.puuid + "_temp",
                    "puuid": tp.puuid,
                    "teamId": tp.team_id,
                    "win": 1 if tp.win else 0,
                    "kda": tp_kda,
                    "visionScore": tp.vision_score,
                    "deaths": tp.deaths,
                })

        df = pd.DataFrame(all_team_data)
        df_with_derived = add_derived_features(df)
        target_df = df_with_derived[df_with_derived["puuid"] == player_stats.puuid]
        return target_df
    else:
        # 팀원 데이터 없으면 단독 처리
        df = pd.DataFrame([row_data])
        df_with_derived = add_derived_features(df)
        return df_with_derived


# ---------------------------
# API 함수
# ---------------------------

def create_backend_integration_app(
    model: XGBClassifier,
    explainer: Optional[shap.TreeExplainer],
    feature_cols: List[str]
) -> FastAPI:
    """
    백엔드 통합용 FastAPI 앱 생성

    Args:
        model: 학습된 XGBoost 모델
        explainer: SHAP explainer (optional)
        feature_cols: 모델이 사용하는 feature 목록

    Returns:
        FastAPI 앱
    """
    app = FastAPI(
        title="LoL Backend Integration API",
        description="백엔드에서 수집한 매치 데이터를 받아서 AI 분석을 수행하는 API",
        version="1.0.0"
    )

    @app.get("/")
    def root():
        return {
            "name": "LoL Backend Integration API",
            "version": "1.0.0",
            "description": "백엔드가 수집한 매치 데이터를 분석",
            "endpoints": [
                "/api/v1/backend/analyze/match",
                "/api/v1/backend/analyze/batch",
                "/health"
            ]
        }

    @app.get("/health")
    def health():
        return {
            "status": "healthy",
            "model_loaded": model is not None,
            "explainer_loaded": explainer is not None,
            "features": len(feature_cols)
        }

    @app.post("/api/v1/backend/analyze/match", response_model=MatchAnalysisResponse)
    def analyze_match_from_backend(request: MatchAnalysisRequest):
        """
        백엔드가 수집한 단일 매치 데이터를 분석

        Args:
            request: 매치 데이터 + 플레이어 통계 + 타임라인

        Returns:
            Impact Score 분석 결과
        """
        try:
            if not model:
                raise HTTPException(status_code=503, detail="Model not loaded")

            # Feature 추출
            df = extract_features_from_backend_data(
                player_stats=request.player_stats,
                timeline_data=request.timeline_data,
                game_duration=request.match_data.game_duration
            )

            row = df.iloc[0]

            # feature_cols에 없는 컬럼은 0으로 채우기
            for col in feature_cols:
                if col not in row.index:
                    row[col] = 0.0

            # Impact Score 계산
            impact_report = compute_player_impact(row, model, explainer, feature_cols)

            # 응답 구성
            top_positive = []
            top_negative = []

            for feat in impact_report["features"]["top"]:
                feat_data = {
                    "name": feat["name"],
                    "displayName": feat["displayName"],
                    "value": feat["value"],
                    "shap": feat["shap"]
                }

                if feat["direction"] == "positive":
                    top_positive.append(feat_data)
                else:
                    top_negative.append(feat_data)

            # Raw stats 추출
            raw_stats = {
                "kills": int(row.get("kills", 0)),
                "deaths": int(row.get("deaths", 0)),
                "assists": int(row.get("assists", 0)),
                "kda": float(row.get("kda", 0)),
                "goldPerMinute": float(row.get("goldPerMinute", 0)),
                "damagePerMinute": float(row.get("damagePerMinute", 0)),
                "visionScorePerMinute": float(row.get("visionScorePerMinute", 0)),
                "cs_15": float(row.get("cs_15", 0)),
                "gold_15": float(row.get("gold_15", 0))
            }

            impact_result = ImpactScoreResult(
                impact_score=impact_report["impactScore"],
                baselineProba=impact_report["baselineProba"],
                predictedProba=impact_report["predictedProba"],
                summary=impact_report["summary"],
                topPositiveFeatures=top_positive,
                topNegativeFeatures=top_negative,
                rawStats=raw_stats
            )

            response = MatchAnalysisResponse(
                match_id=request.match_data.match_id,
                puuid=request.player_stats.puuid,
                game_name=request.player_stats.game_name,
                tag_line=request.player_stats.tag_line,
                champion=request.player_stats.champion_name,
                role=request.player_stats.team_position,
                win=request.player_stats.win,
                game_duration=request.match_data.game_duration,
                impactResult=impact_result
            )

            return response

        except HTTPException:
            raise
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    @app.post("/api/v1/backend/analyze/batch", response_model=BatchAnalysisResponse)
    def analyze_batch_from_backend(request: BatchMatchAnalysisRequest):
        """
        여러 매치를 한 번에 분석

        Args:
            request: 매치 목록 + aggregate 옵션

        Returns:
            일괄 분석 결과 + (선택적) 평균 지표
        """
        try:
            results = []
            successful = 0
            failed = 0

            for match_req in request.matches:
                try:
                    # 각 매치 분석
                    result = analyze_match_from_backend(match_req)
                    results.append(result)
                    successful += 1
                except Exception as e:
                    print(f"[WARN] Failed to analyze match {match_req.match_data.match_id}: {e}")
                    failed += 1

            # 평균 계산 (aggregate=True일 때)
            avg_impact_score = None
            win_rate = None
            avg_stats = None

            if request.aggregate and results:
                total_impact = sum(r.impact_result.impact_score for r in results)
                avg_impact_score = total_impact / len(results)

                wins = sum(1 for r in results if r.win)
                win_rate = wins / len(results)

                # Raw stats 평균
                all_raw_stats = [r.impact_result.raw_stats for r in results]
                avg_stats = {}

                for key in all_raw_stats[0].keys():
                    values = [stats[key] for stats in all_raw_stats]
                    avg_stats[key] = sum(values) / len(values)

            return BatchAnalysisResponse(
                totalMatches=len(request.matches),
                successful=successful,
                failed=failed,
                results=results,
                averageImpactScore=avg_impact_score,
                winRate=win_rate,
                avgStats=avg_stats
            )

        except Exception as e:
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")

    return app


# ---------------------------
# 실행 예시
# ---------------------------
if __name__ == "__main__":
    # 모델 로딩
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    MODEL_DIR = os.path.join(BASE_DIR, "models")
    MODEL_PATH = os.path.join(MODEL_DIR, "player_impact_model.json")
    FEAT_PATH = os.path.join(MODEL_DIR, "feature_cols.json")

    model = None
    feature_cols = []
    explainer = None

    if os.path.exists(MODEL_PATH) and os.path.exists(FEAT_PATH):
        try:
            print(f"[INFO] Loading model from {MODEL_PATH}")
            model = XGBClassifier()
            model.load_model(MODEL_PATH)

            with open(FEAT_PATH, "r") as f:
                feature_cols = json.load(f)

            print(f"[INFO] Model loaded with {len(feature_cols)} features")

            # SHAP explainer
            try:
                explainer = shap.TreeExplainer(model)
                print(f"[INFO] SHAP explainer initialized")
            except Exception as e:
                print(f"[WARN] SHAP explainer failed: {e}")
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            exit(1)
    else:
        print(f"[ERROR] Model files not found at {MODEL_DIR}")
        exit(1)

    # FastAPI 앱 생성
    app = create_backend_integration_app(model, explainer, feature_cols)

    # 서버 실행
    import uvicorn
    print("[INFO] Starting Backend Integration API on http://0.0.0.0:8001")
    uvicorn.run(app, host="0.0.0.0", port=8001)
