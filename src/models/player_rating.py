import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import shap
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


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
    - kda_norm: 같은 경기/팀 내에서 0~1 스케일로 정규화된 KDA
    - early_cs_total: 초반 총 CS (라인 + 정글 CS 합산)
    - death_rate: 분당 데스 수 (낮을수록 좋음)
    """
    df = df.copy()

    team_key = ["matchId", "teamId"]

    def _norm_group(x: pd.Series) -> pd.Series:
        mn, mx = x.min(), x.max()
        if mx - mn < 1e-9:
            # 모두 같은 값이면 0으로 두기
            return pd.Series(0.0, index=x.index)
        return (x - mn) / (mx - mn)

    # 시야 점수 정규화 (팀 내)
    if "visionScore" in df.columns:
        df["vision_norm"] = df.groupby(team_key)["visionScore"].transform(_norm_group)
    else:
        df["vision_norm"] = 0.0

    # KDA 정규화 (팀 내)
    if "kda" in df.columns:
        df["kda_norm"] = df.groupby(team_key)["kda"].transform(_norm_group)
    else:
        df["kda_norm"] = 0.0

    # 초반 총 CS (라인 + 정글)
    lane_cs = df.get("laneMinionsFirst10Minutes", pd.Series(0, index=df.index))
    jungle_cs = df.get("jungleCsBefore10Minutes", pd.Series(0, index=df.index))
    df["early_cs_total"] = lane_cs.fillna(0) + jungle_cs.fillna(0)

    # 분당 데스율 (낮을수록 좋음)
    if "deaths" in df.columns and "goldPerMinute" in df.columns:
        # goldPerMinute이 있다면 게임 시간을 역산할 수 있지만, 간단히 deaths 자체를 사용
        # 실제로는 game_duration이 필요하지만, 여기서는 deaths만 사용
        df["death_rate"] = df["deaths"]
    else:
        df["death_rate"] = 0.0

    return df


def split_data_by_match(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
):
    """matchId 기준으로 데이터를 train/val/test로 분할.

    같은 경기의 플레이어들이 서로 다른 세트에 섞이지 않도록 matchId 단위로 분할.

    Args:
        df: 전체 데이터프레임
        test_size: 테스트 세트 비율 (기본 15%)
        val_size: 검증 세트 비율 (기본 15%)
        random_state: 랜덤 시드

    Returns:
        train_df, val_df, test_df
    """
    unique_matches = df["matchId"].unique()
    print(f"[INFO] 전체 매치 수: {len(unique_matches)}")

    # 먼저 train+val과 test로 분할
    train_val_matches, test_matches = train_test_split(
        unique_matches,
        test_size=test_size,
        random_state=random_state,
    )

    # train+val을 다시 train과 val로 분할
    val_ratio = val_size / (1 - test_size)  # train+val 중에서 val의 비율
    train_matches, val_matches = train_test_split(
        train_val_matches,
        test_size=val_ratio,
        random_state=random_state,
    )

    train_df = df[df["matchId"].isin(train_matches)].copy()
    val_df = df[df["matchId"].isin(val_matches)].copy()
    test_df = df[df["matchId"].isin(test_matches)].copy()

    print(f"[INFO] Train 매치: {len(train_matches)} ({len(train_df)} 플레이어)")
    print(f"[INFO] Val 매치: {len(val_matches)} ({len(val_df)} 플레이어)")
    print(f"[INFO] Test 매치: {len(test_matches)} ({len(test_df)} 플레이어)")

    # 각 세트의 승률 확인
    print(f"[INFO] Train 승률: {train_df['win'].mean():.3f}")
    print(f"[INFO] Val 승률: {val_df['win'].mean():.3f}")
    print(f"[INFO] Test 승률: {test_df['win'].mean():.3f}")

    return train_df, val_df, test_df


def train_win_model(train_df: pd.DataFrame, val_df: pd.DataFrame = None):
    """플레이어 단위로 '이긴 팀에 속했는지(win)'를 예측하는 XGBoost 모델 학습.

    데이터 누수를 최소화하기 위해 승패 독립적인 feature만 사용:
    - 초반 성장 지표 (10~15분)
    - 효율성 지표 (per minute)
    - 개인 플레이 품질 (KDA, solo kills 등)

    Args:
        train_df: 학습 데이터
        val_df: 검증 데이터 (early stopping용, optional)

    Returns:
        model, feature_cols
    """
    train_df = train_df.copy()

    if "win" not in train_df.columns:
        raise ValueError("win 컬럼이 없습니다.")

    # 안전한 feature 목록 (승패 독립적)
    feature_candidates = [
        # 초반 성장 (15분 이전)
        "cs_15",
        "gold_15",
        "xp_15",
        "early_cs_total",  # 10분 CS
        "laneMinionsFirst10Minutes",
        "jungleCsBefore10Minutes",

        # 효율성 지표 (per minute)
        "goldPerMinute",
        "damagePerMinute",
        "visionScorePerMinute",

        # 개인 플레이 품질
        "kda",
        "kda_norm",  # 팀 내 정규화된 KDA
        "killParticipation",
        "soloKills",

        # 시야 기여도
        "vision_norm",  # 팀 내 정규화된 시야
        "controlWardsPlaced",
        "wardTakedowns",
        "wardTakedownsBefore20M",

        # 플레이 품질
        "skillshotsDodged",
        "skillshotsHit",
        "longestTimeSpentLiving",

        # 생존력 (역지표)
        "death_rate",
    ]

    feature_cols = [c for c in feature_candidates if c in train_df.columns]
    if not feature_cols:
        raise ValueError("사용 가능한 feature가 없습니다.")

    print(f"[INFO] 사용 중인 feature 개수: {len(feature_cols)}")
    print(f"[INFO] Feature 목록: {feature_cols}")

    # Ensure all feature columns are numeric floats (avoid object dtypes)
    for c in feature_cols:
        train_df[c] = pd.to_numeric(train_df[c], errors="coerce").fillna(0.0)

    X_train = train_df[feature_cols].to_numpy(dtype=float)
    y_train = train_df["win"].values

    # Validation 데이터 준비 (있는 경우)
    eval_set = None
    if val_df is not None:
        val_df = val_df.copy()
        for c in feature_cols:
            val_df[c] = pd.to_numeric(val_df[c], errors="coerce").fillna(0.0)
        X_val = val_df[feature_cols].to_numpy(dtype=float)
        y_val = val_df["win"].values
        eval_set = [(X_val, y_val)]

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        base_score=0.5,  # SHAP 호환성을 위해 명시적으로 설정
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=20 if eval_set else None,
    )

    if eval_set:
        model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=False,
        )
        print(f"[INFO] Best iteration: {model.best_iteration}")
    else:
        model.fit(X_train, y_train)

    return model, feature_cols


def evaluate_model(model, df: pd.DataFrame, feature_cols: list, dataset_name: str = "Test"):
    """모델 성능 평가.

    Args:
        model: 학습된 XGBoost 모델
        df: 평가할 데이터프레임
        feature_cols: feature 컬럼 리스트
        dataset_name: 데이터셋 이름 (출력용)

    Returns:
        metrics: 성능 지표 딕셔너리
    """
    df = df.copy()

    # Feature 준비
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    X = df[feature_cols].to_numpy(dtype=float)
    y_true = df["win"].values

    # 예측
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]

    # 성능 지표 계산
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_pred_proba),
    }

    print(f"\n=== {dataset_name} Set Performance ===")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"ROC AUC:   {metrics['roc_auc']:.4f}")

    return metrics


def build_shap_explainer(
    model,
    df: pd.DataFrame,
    feature_cols: list,
    sample_size: int = 1000,
    background_size: int = 200,
):
    """XGBClassifier + SHAP 설정.

    f(x) = P(win=1 | x) 를 대상으로 SHAP 값을 계산한다.
    TreeExplainer를 사용하여 pickle 가능하도록 구현.

    Args:
        model: 학습된 XGBoost 모델
        df: SHAP 분석에 사용할 데이터프레임
        feature_cols: feature 컬럼 리스트
        sample_size: SHAP 분석 샘플 수
        background_size: SHAP background 샘플 수 (TreeExplainer는 무시됨)

    Returns:
        explainer: SHAP TreeExplainer 객체
    """
    df = df.copy()

    # Feature 준비
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    X = df[feature_cols].to_numpy(dtype=float)

    n = X.shape[0]
    sample_size = min(sample_size, n)
    idx = np.random.choice(n, size=sample_size, replace=False)
    X_sample = X[idx]

    # TreeExplainer 사용 (람다 함수 불필요, pickle 가능)
    # XGBoost sklearn wrapper의 경우 get_booster()를 사용하여 호환성 문제 해결
    if hasattr(model, 'get_booster'):
        from xgboost import Booster
        import tempfile
        import os

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

                    print(f"[INFO] Fixed base_score for SHAP: {base_score} -> {base_score_clean}")

                    # 임시 파일에 모델 저장 후 새 Booster로 로드
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                        temp_model = f.name
                        booster.save_model(temp_model)

                    # 새 Booster 생성 및 config 수정 후 모델 로드
                    new_booster = Booster()
                    new_booster.load_config(json_lib.dumps(model_dict))
                    new_booster.load_model(temp_model)
                    booster = new_booster

                    os.unlink(temp_model)
        except Exception as e:
            print(f"[WARN] Could not fix base_score: {e}")
            import traceback
            traceback.print_exc()

        explainer = shap.TreeExplainer(booster)
    else:
        explainer = shap.TreeExplainer(model)
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
        # 초반 성장
        "cs_15": "15분 CS",
        "gold_15": "15분 골드",
        "xp_15": "15분 경험치",
        "early_cs_total": "10분 총 CS",
        "laneMinionsFirst10Minutes": "10분 라인 CS",
        "jungleCsBefore10Minutes": "10분 정글 CS",

        # 효율성 지표
        "goldPerMinute": "분당 골드",
        "damagePerMinute": "분당 딜량",
        "visionScorePerMinute": "분당 시야 점수",

        # 개인 플레이 품질
        "kda": "KDA",
        "kda_norm": "팀 내 KDA 기여도",
        "killParticipation": "킬 관여율",
        "soloKills": "솔로킬",

        # 시야 기여도
        "vision_norm": "팀 내 시야 기여도",
        "controlWardsPlaced": "제어 와드 설치",
        "wardTakedowns": "와드 제거",
        "wardTakedownsBefore20M": "20분 전 와드 제거",

        # 플레이 품질
        "skillshotsDodged": "스킬샷 회피",
        "skillshotsHit": "스킬샷 적중",
        "longestTimeSpentLiving": "최장 생존 시간",

        # 생존력
        "death_rate": "데스 수",

        # 기존 (하위 호환)
        "dmg_share": "팀 내 딜 비중",
        "gold_share": "팀 내 골드 비중",
        "obj_participation": "오브젝트 전체 참여도",
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

    # 데이터 분할
    print("\n[INFO] Splitting data into train/val/test sets...")
    train_df, val_df, test_df = split_data_by_match(df, test_size=0.15, val_size=0.15, random_state=42)

    # 모델 학습
    print("\n[INFO] Training win classifier for SHAP-based impact...")
    model, feature_cols = train_win_model(train_df, val_df)

    # 성능 평가
    print("\n[INFO] Evaluating model on train/val/test sets...")
    train_metrics = evaluate_model(model, train_df, feature_cols, dataset_name="Train")
    val_metrics = evaluate_model(model, val_df, feature_cols, dataset_name="Validation")
    test_metrics = evaluate_model(model, test_df, feature_cols, dataset_name="Test")

    # SHAP 분석 (train 데이터 사용)
    print("\n[INFO] Building SHAP explainer (global analysis)...")
    explainer = build_shap_explainer(model, train_df, feature_cols, sample_size=2000, background_size=200)

    # 샘플 리포트 (test 데이터에서)
    unique_puuids = test_df["puuid"].unique()[:5]

    for p in unique_puuids:
        print(f"\n[INFO] Sample player report for puuid={p}")
        shap_player_report(test_df, model, explainer, feature_cols, puuid=p)


if __name__ == "__main__":
    main()