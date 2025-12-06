import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import shap
from xgboost import XGBClassifier


PLAYER_DF_PATH = "data/processed/player_stats.csv"


def load_player_data(path: str = PLAYER_DF_PATH) -> pd.DataFrame:
    """플레이어 단위 스탯 CSV 로드.

    Expected columns (at least):
      - matchId, teamId, win, puuid, role, champion
      - dmg_share, gold_share, visionScore, kda
      - dragon_participation, baron_participation, atakan_participation,
        herald_participation, tower_participation, plate_participation
    """
    df = pd.read_csv(path)
    df = df.fillna(0)
    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """팀 기준 정규화 및 집계 feature 추가.

    - vision_norm: 같은 경기/팀 내에서 0~1 스케일로 정규화된 시야 점수
    - kda_norm   : 같은 경기/팀 내에서 0~1 스케일로 정규화된 KDA
    - obj_participation: 드래곤/바론/아타칸/전령/타워/플레이트 참여도의 평균
    """
    df = df.copy()

    team_key = ["matchId", "teamId"]

    def _norm_group(x: pd.Series) -> pd.Series:
        mn, mx = x.min(), x.max()
        if mx - mn < 1e-9:
            # 모두 같은 값이면 0으로 두기
            return pd.Series(0.0, index=x.index)
        return (x - mn) / (mx - mn)

    if "visionScore" in df.columns:
        df["vision_norm"] = df.groupby(team_key)["visionScore"].transform(_norm_group)
    else:
        df["vision_norm"] = 0.0

    if "kda" in df.columns:
        df["kda_norm"] = df.groupby(team_key)["kda"].transform(_norm_group)
    else:
        df["kda_norm"] = 0.0

    obj_cols = [
        "dragon_participation",
        "baron_participation",
        "atakan_participation",
        "herald_participation",
        "tower_participation",
        "plate_participation",
    ]
    existing_obj_cols = [c for c in obj_cols if c in df.columns]
    if existing_obj_cols:
        df["obj_participation"] = df[existing_obj_cols].mean(axis=1)
    else:
        df["obj_participation"] = 0.0

    return df


def train_win_model(df: pd.DataFrame):
    """플레이어 단위로 '이긴 팀에 속했는지(win)'를 예측하는 XGBoost 모델 학습.

    이 모델의 출력 P(win=1 | stats)를 0~100 점수로 해석하고,
    SHAP으로 어떤 스탯이 그 확률을 올렸는지/내렸는지 분석한다.
    """
    df = df.copy()

    if "win" not in df.columns:
        raise ValueError("win 컬럼이 없습니다.")

    feature_candidates = [
        "dmg_share",
        "gold_share",
        "obj_participation",
        "vision_norm",
        "kda",  
        "cs_15",
        "gold_15",
        "xp_15",
        "dragon_participation",
        "baron_participation",
        "atakan_participation",
        "herald_participation",
        "tower_participation",
        "plate_participation",
    ]
    feature_cols = [c for c in feature_candidates if c in df.columns]
    if not feature_cols:
        raise ValueError("사용 가능한 feature가 없습니다.")

    # Ensure all feature columns are numeric floats (avoid object dtypes)
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    X = df[feature_cols].to_numpy(dtype=float)
    y = df["win"].values

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)

    return model, X, feature_cols


def build_shap_explainer(
    model,
    X: np.ndarray,
    feature_cols,
    sample_size: int = 1000,
    background_size: int = 200,
):
    """XGBClassifier + SHAP 설정.

    f(x) = P(win=1 | x) 를 대상으로 SHAP 값을 계산한다.
    model 객체를 직접 넘기지 않고, predict_proba 람다를 사용해서
    버전 호환 이슈를 줄인다.
    """
    n = X.shape[0]
    sample_size = min(sample_size, n)
    idx = np.random.choice(n, size=sample_size, replace=False)
    X_sample = X[idx]

    bg_size = min(background_size, X_sample.shape[0])
    bg_idx = np.random.choice(X_sample.shape[0], size=bg_size, replace=False)
    background = X_sample[bg_idx]

    # f(x): 승리 확률만 반환하는 함수
    f = lambda data: model.predict_proba(data)[:, 1]

    explainer = shap.Explainer(f, background)
    shap_values = explainer(X_sample)

    print("\n=== SHAP Global Summary (P(win) ~ features) ===")
    # shap_values는 Explanation 객체이므로 그대로 summary_plot에 넘길 수 있다.
    shap.summary_plot(shap_values, show=False, feature_names=feature_cols)
    plt.tight_layout()
    plt.show()

    return explainer


def compute_player_impact(
    row: pd.Series,
    model,
    explainer,
    feature_cols,
) -> dict:
    match_id = row["matchId"]
    team_id = row["teamId"]
    champ = row.get("champion", "Unknown")
    role = row.get("role", "Unknown")
    win = row.get("win", 0)

    x = row[feature_cols].astype(float).to_numpy().reshape(1, -1)

    # 1) 모델이 추정한 승리 확률 및 0~100 점수
    prob_win = float(model.predict_proba(x)[0, 1])
    score_100 = prob_win * 100.0

    # 2) SHAP 관련 초기값 (explainer 가 None 인 경우를 위해 기본값 준비)
    base_value = 0.5
    shap_vals = np.zeros(len(feature_cols), dtype=float)

    if explainer is not None:
        # explainer가 있을 때만 SHAP 계산
        shap_expl = explainer(x)
        # Tree / Permutation explainer 모두 대응하도록 values를 1차원 배열로 변환
        shap_vals = np.array(shap_expl.values[0], dtype=float)

        # base_values 가 스칼라/배열 어떤 형태여도 첫 값을 가져오도록 처리
        bv = shap_expl.base_values
        try:
            base_value = float(np.ravel(bv)[0])
        except Exception:
            base_value = float(bv)

    # 3) SHAP 절대값 기준 상위 feature 선택 (explainer가 없으면 전부 0으로 처리됨)
    abs_vals = np.abs(shap_vals)
    order = np.argsort(abs_vals)[::-1]
    top_k = 5
    top_idx = order[:top_k]

    feature_kor = {
        "kda": "KDA",
        "dmg_share": "팀 내 딜 비중",
        "gold_share": "팀 내 골드 비중",
        "obj_participation": "오브젝트 전체 참여도",
        "vision_norm": "팀 내 시야 기여도",
        "cs_15": "15분 CS",
        "gold_15": "15분 골드",
        "xp_15": "15분 경험치",
        "dragon_participation": "드래곤 참여도",
        "baron_participation": "바론 참여도",
        "atakan_participation": "아타칸 참여도",
        "herald_participation": "전령 참여도",
        "tower_participation": "포탑 참여도",
        "plate_participation": "포탑 방패 참여도",
    }

    # 4) 점수 구간에 따른 요약 문구
    if score_100 >= 85:
        summary = "이번 판에서 팀 승리에 크게 기여한 경기입니다."
    elif score_100 >= 60:
        summary = "팀 승리에 의미 있는 기여를 한 경기입니다."
    elif score_100 >= 35:
        summary = "기여와 아쉬운 점이 모두 있었던 무난한 경기입니다."
    elif score_100 >= 15:
        summary = "전반적으로 아쉬운 경기였으며 개선 여지가 많습니다."
    else:
        summary = "이번 판에서는 팀 승리에 거의 기여하지 못한 경기입니다."

    # 5) 상위 feature들 정리 (API/콘솔 공용)
    top_features = []
    for i in top_idx:
        fname = feature_cols[i]
        direction = "up" if shap_vals[i] > 0 else "down"
        raw_val = float(row[fname])
        top_features.append({
            "name": fname,
            "displayName": feature_kor.get(fname, fname),
            "direction": direction,
            "shap": float(shap_vals[i]),
            "value": raw_val,
        })

    report = {
        "matchId": match_id,
        "teamId": int(team_id),
        "champion": champ,
        "role": role,
        "win": int(win),
        "impactScore": float(score_100),
        "baselineProba": float(base_value),
        "predictedProba": float(prob_win),
        "summary": summary,
        "features": {
            "top": top_features,
            "raw": {fname: float(row[fname]) for fname in feature_cols},
        },
    }
    return report


def shap_player_report(
    df: pd.DataFrame,
    model,
    explainer,
    feature_cols,
    puuid: str,
    match_id: str | None = None,
):
    """특정 유저(puuid)의 한 경기 리포트.

    - 점수: 모델이 추정한 승리 확률을 0~100 점수로 변환
    - 해설: SHAP 값 기준으로 점수를 올린/깎은 주요 스탯을 출력
    """
    player_rows = df[df["puuid"] == puuid]
    if player_rows.empty:
        print(f"[WARN] puuid={puuid} 에 해당하는 데이터가 없습니다.")
        return

    if match_id is not None:
        row_sel = player_rows[player_rows["matchId"] == match_id]
        if row_sel.empty:
            print(f"[WARN] puuid={puuid}, matchId={match_id} 조합이 없습니다.")
            return
        row = row_sel.iloc[0]
    else:
        row = player_rows.iloc[-1]
        match_id = row["matchId"]

    report = compute_player_impact(row, model, explainer, feature_cols)

    print("\n==============================")
    print(f"[플레이 리포트] {report['matchId']} – {report['champion']} ({report['role']})")
    print(f"팀: {report['teamId']}, 결과: {'승리' if report['win'] == 1 else '패배'}")
    print(f"모델 기준 승리 기여 점수: {report['impactScore']:.1f} / 100")
    print(f"(baseline={report['baselineProba']:.3f}, prob={report['predictedProba']:.3f})")
    print(f"\n요약: {report['summary']}")

    print("\n 주요 영향 지표 (상위 5개, |SHAP| 기준)")
    print(f"{'지표':20s} {'영향 방향':8s} {'SHAP':>10s} {'값':>12s}")
    print("-" * 60)
    for feat in report["features"]["top"]:
        direction_kor = "상향" if feat["direction"] == "up" else "하향"
        print(f"{feat['displayName']:20s} {direction_kor:8s} {feat['shap']:10.4f} {feat['value']:12.3f}")

    # waterfall 시각화 (옵션)
    try:
        x = row[feature_cols].astype(float).to_numpy().reshape(1, -1)
        shap_expl = explainer(x)
        shap.plots.waterfall(shap_expl[0], show=False)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"[WARN] waterfall plot 실패: {e}")

    return report


def main():
    print("[INFO] Loading player stats from:", PLAYER_DF_PATH)
    df = load_player_data(PLAYER_DF_PATH)
    print(f"[INFO] Loaded {len(df)} player-rows")

    df = add_derived_features(df)

    print("\n[INFO] Training win classifier for SHAP-based impact...")
    model, X, feature_cols = train_win_model(df)

    print("[INFO] Building SHAP explainer (global analysis)...")
    explainer = build_shap_explainer(model, X, feature_cols, sample_size=2000, background_size=200)

    unique_puuids = df["puuid"].unique()[25:30]

    for p in unique_puuids:
        print(f"\n[INFO] Sample player report for puuid={p}")
        shap_player_report(df, model, explainer, feature_cols, puuid=p)


if __name__ == "__main__":
    main()