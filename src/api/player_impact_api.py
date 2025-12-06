# src/api/player_impact_api.py

from typing import Dict, Optional

import joblib
import pandas as pd
#import shap
from fastapi import FastAPI
from pydantic import BaseModel

# player_rating.py 에 있는 함수 import
from src.models.player_rating import compute_player_impact

app = FastAPI(title="LoL Player Impact API")

MODEL_PATH = "models/player_win_model.joblib"
FEAT_PATH = "models/player_win_features.joblib"

model = joblib.load(MODEL_PATH)
feature_cols = joblib.load(FEAT_PATH)

explainer = None

class PlayerImpactRequest(BaseModel):
    puuid: str
    matchId: str

    # 메타 정보 (선택적으로 같이 받음)
    teamId: Optional[int] = None
    win: Optional[int] = None       # 1(승), 0(패) 모르면 None
    champion: Optional[str] = None
    role: Optional[str] = None

    # 학습에 썼던 feature 값들 (kills, deaths, dmg_share, gold_share, participation 등)
    features: Dict[str, float]

@app.post("/impact/report")
def impact_report(req: PlayerImpactRequest):
    """
    한 경기 + 한 플레이어에 대한 임팩트 점수 & SHAP 리포트 반환
    """

    # 1) row 형태로 합치기 (compute_player_impact 가 기대하는 구조 맞추기)
    row_dict: Dict[str, object] = {}

    # (1) 메타 정보
    row_dict["matchId"] = req.matchId
    row_dict["puuid"] = req.puuid
    row_dict["teamId"] = req.teamId if req.teamId is not None else -1
    row_dict["win"] = req.win if req.win is not None else -1
    row_dict["champion"] = req.champion if req.champion is not None else ""
    row_dict["role"] = req.role if req.role is not None else ""

    # (2) feature 값들
    for k, v in req.features.items():
        row_dict[k] = v

    # pandas Series 로 변환
    row = pd.Series(row_dict)

    for col in feature_cols:
        if col not in row.index:
            row[col] = 0.0

    # 2) 임팩트 리포트 계산 (점수 + shap 설명)
    report = compute_player_impact(row, model, explainer, feature_cols)

    # 3) 그대로 JSON 응답
    return report