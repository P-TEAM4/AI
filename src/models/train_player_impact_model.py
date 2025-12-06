# src/models/train_player_impact_model.py

import os
import joblib

from player_rating import (
    PLAYER_DF_PATH,
    load_player_data,
    add_derived_features,
    train_win_model,
    build_shap_explainer,
)

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

def main():
    print(f"[INFO] Loading player stats from: {PLAYER_DF_PATH}")
    df = load_player_data(PLAYER_DF_PATH)
    df = add_derived_features(df)

    print("[INFO] Training win classifier (player impact model)...")
    model, X, feature_cols = train_win_model(df)

    # 1) 모델 & 피처 저장
    model_path = os.path.join(MODELS_DIR, "player_win_model.joblib")
    feat_path = os.path.join(MODELS_DIR, "player_win_features.joblib")

    joblib.dump(model, model_path)
    joblib.dump(feature_cols, feat_path)

    print(f"[INFO] Saved model to {model_path}")
    print(f"[INFO] Saved feature_cols to {feat_path}")

    # 2) SHAP explainer는 시각화/디버깅용으로만 사용 (파일로 저장하지 않음)
    print("[INFO] Building SHAP explainer for deployment...")
    explainer = build_shap_explainer(
        model,
        X,
        feature_cols,
        sample_size=2000,
        background_size=200,
    )
    print("[INFO] SHAP explainer built (not saved to disk).")

if __name__ == "__main__":
    main()