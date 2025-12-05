import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier


DF_PATH = "data/processed/matches_advanced.csv"


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # 혹시 모를 결측치는 0으로 (전처리에서 대부분 처리돼 있을 거라 거의 영향 없음)
    df = df.fillna(0)

    return df


def train_global_model(df: pd.DataFrame):
    print("=== [1] Global Model (duration 제거) ===")

    y = df["winner"]

    # ID/텍스트/티어/시간 관련 컬럼 제거
    drop_cols = ["matchId", "tier", "winner", "duration"]

    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols]

    print(f"- 사용 feature 개수: {len(feature_cols)}")
    print(f"- feature 목록: {feature_cols}\n")

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    # === 성능 평가 ===
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test, y_proba)
    except ValueError:
        auc = np.nan

    print(f"- Test Accuracy : {acc:.4f}")
    print(f"- Test ROC-AUC  : {auc:.4f}\n")

    # === Feature Importance 출력 ===
    importance = model.feature_importances_
    feat_list = list(zip(feature_cols, importance))
    feat_sorted = sorted(feat_list, key=lambda x: x[1], reverse=True)

    print("=== Global Feature Importance (Top 20) ===")
    for f, score in feat_sorted[:20]:
        print(f"{f:30s} {score:.5f}")
    print()

    plot_global_importance(feat_sorted, top_n=15)

    return model, feat_sorted, X

def plot_tier_importance_heatmap(tier_importances: dict, top_features: int = 10):
    """
    tier_importances: {tier: {feature_name: importance, ...}, ...}
    """

    # 1) 전체 티어에서 공통으로 자주 등장하는 feature 상위 top_features 선정
    # 먼저 각 feature의 평균 importance 계산
    feature_scores = {}
    for tier_dict in tier_importances.values():
        for f, score in tier_dict.items():
            feature_scores.setdefault(f, []).append(score)

    avg_scores = {f: np.mean(scores) for f, scores in feature_scores.items()}
    # 중요도 평균 기준 상위 feature만 사용
    selected_features = [f for f, _ in sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)[:top_features]]

    tiers = sorted(tier_importances.keys())

    # 2) (feature × tier) 매트릭스 만들기
    mat = []
    for f in selected_features:
        row = []
        for t in tiers:
            row.append(tier_importances[t].get(f, 0.0))
        mat.append(row)

    mat = np.array(mat)

    # 3) 시각화
    plt.figure(figsize=(10, 0.5 * len(selected_features) + 2))
    im = plt.imshow(mat, aspect="auto")

    plt.colorbar(im, label="Importance")
    plt.yticks(range(len(selected_features)), selected_features)
    plt.xticks(range(len(tiers)), tiers, rotation=45)

    plt.title(f"Tier-wise Feature Importance (Top {top_features} features)")
    plt.tight_layout()
    plt.show()

def shap_global_analysis(model, X, max_samples: int = 1000):
    """
    X: 학습에 사용한 feature DataFrame (전부 또는 일부)
    max_samples: SHAP 계산에 사용할 샘플 수 (너무 크면 느려져서 제한)
    """

    # 샘플 수 제한 (랜덤 샘플링)
    if len(X) > max_samples:
        X_sample = X.sample(max_samples, random_state=42)
    else:
        X_sample = X

    print(f"[SHAP] Using {len(X_sample)} samples for explanation")

    # KernelExplainer를 사용해서 xgboost/shap 버전 호환 문제를 우회한다.
    # 배경 분포(background)는 X_sample 중 일부만 사용.
    bg_size = min(100, len(X_sample))
    X_bg = X_sample.sample(bg_size, random_state=42)

    def predict_proba_fn(data):
        # data는 numpy array로 들어오므로 DataFrame으로 감싸서 컬럼 정렬을 맞춰준다.
        df_in = pd.DataFrame(data, columns=X_sample.columns)
        return model.predict_proba(df_in)[:, 1]

    explainer = shap.KernelExplainer(predict_proba_fn, X_bg.to_numpy())
    shap_values = explainer.shap_values(X_sample.to_numpy(), nsamples="auto")

    # 1) Summary plot (beeswarm)
    shap.summary_plot(shap_values, X_sample, max_display=15)

    # 2) Feature importance bar plot (SHAP value 기준)
    shap.summary_plot(shap_values, X_sample, plot_type="bar", max_display=15)

    return explainer, shap_values, X_sample

def plot_global_importance(feat_sorted, top_n: int = 10):
    # feat_sorted: [(feature_name, importance), ...] 형태
    top = feat_sorted[:top_n]
    names = [f for f, _ in top]
    scores = [s for _, s in top]

    plt.figure(figsize=(8, 4))
    # 위에서 아래로 잘 보이게 뒤집어서 그리기
    plt.barh(names[::-1], scores[::-1])
    plt.xlabel("Importance")
    plt.title(f"Global Feature Importance (Top {top_n})")
    plt.tight_layout()
    plt.show()

def train_tier_wise(df: pd.DataFrame, min_samples: int = 200):
    print("=== [2] Tier-wise Feature Importance ===")
    tiers = df["tier"].unique()

    tier_importances = {}

    for tier in sorted(tiers):
        df_t = df[df["tier"] == tier]

        if len(df_t) < min_samples:
            print(f"- {tier}: 샘플 {len(df_t)}개 (min {min_samples}) → 스킵")
            continue

        print(f"\n--- Tier: {tier} (샘플 {len(df_t)}) ---")

        y = df_t["winner"]
        drop_cols = ["matchId", "tier", "winner", "duration"]
        feature_cols = [c for c in df_t.columns if c not in drop_cols]
        X = df_t[feature_cols]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # 티어별은 데이터가 적을 수 있으니 가볍게
        model = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.07,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            n_jobs=-1,
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        try:
            auc = roc_auc_score(y_test, y_proba)
        except ValueError:
            auc = np.nan

        print(f"- Test Accuracy : {acc:.4f}")
        print(f"- Test ROC-AUC  : {auc:.4f}")

        importance = model.feature_importances_
        feat_list = list(zip(feature_cols, importance))
        feat_sorted = sorted(feat_list, key=lambda x: x[1], reverse=True)

        print(f"- Top 10 features for {tier}:")
        for f, score in feat_sorted[:10]:
            print(f"  {f:25s} {score:.5f}")

        tier_importances[tier] = dict(feat_list)

    return tier_importances


def main():
    df = load_data(DF_PATH)

    model, global_importance, X_all = train_global_model(df)
    explainer, shap_values, X_sample = shap_global_analysis(model, X_all)
    tier_importances = train_tier_wise(df)
    plot_tier_importance_heatmap(tier_importances, top_features=10)

if __name__ == "__main__":
    main()