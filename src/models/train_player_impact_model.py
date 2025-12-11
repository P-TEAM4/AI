# src/models/train_player_impact_model.py

import os
import json
import joblib

from player_rating import (
    PLAYER_DF_PATH,
    load_player_data,
    add_derived_features,
    split_data_by_match,
    train_win_model,
    evaluate_model,
    build_shap_explainer,
)

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)


def save_model_artifacts(model, feature_cols, explainer=None, metrics=None):
    """학습된 모델과 관련 메타데이터를 저장합니다.

    Args:
        model: 학습된 XGBoost 모델
        feature_cols: 사용된 feature 컬럼 리스트
        explainer: SHAP explainer (선택사항, 용량이 크므로 저장 안 할 수도 있음)
        metrics: 모델 성능 지표 딕셔너리
    """
    # 1) XGBoost 모델 저장 (JSON 형식 권장)
    model_path = os.path.join(MODELS_DIR, "player_impact_model.json")
    model.save_model(model_path)
    print(f"[INFO] Saved XGBoost model to {model_path}")

    # 2) Feature 컬럼 저장 (JSON)
    feature_path = os.path.join(MODELS_DIR, "feature_cols.json")
    with open(feature_path, "w") as f:
        json.dump(feature_cols, f, indent=2)
    print(f"[INFO] Saved feature_cols to {feature_path}")

    # 3) 성능 메트릭 저장 (JSON)
    if metrics:
        metrics_path = os.path.join(MODELS_DIR, "model_metrics.json")
        # numpy float를 일반 float로 변환
        metrics_serializable = {k: float(v) for k, v in metrics.items()}
        with open(metrics_path, "w") as f:
            json.dump(metrics_serializable, f, indent=2)
        print(f"[INFO] Saved model metrics to {metrics_path}")

    # 4) SHAP explainer 저장 (선택사항, 용량이 큼)
    if explainer:
        explainer_path = os.path.join(MODELS_DIR, "shap_explainer.joblib")
        joblib.dump(explainer, explainer_path)
        print(f"[INFO] Saved SHAP explainer to {explainer_path}")
    else:
        print("[INFO] SHAP explainer not saved (can be rebuilt from model)")


def main():
    print(f"[INFO] Loading player stats from: {PLAYER_DF_PATH}")
    df = load_player_data(PLAYER_DF_PATH)
    print(f"[INFO] Loaded {len(df)} player-rows")

    # Feature engineering
    df = add_derived_features(df)

    # 데이터 분할 (matchId 기준)
    print("\n[INFO] Splitting data into train/val/test sets...")
    train_df, val_df, test_df = split_data_by_match(
        df,
        test_size=0.15,
        val_size=0.15,
        random_state=42
    )

    # 모델 학습
    print("\n[INFO] Training win classifier (player impact model)...")
    model, feature_cols = train_win_model(train_df, val_df)

    # 성능 평가
    print("\n[INFO] Evaluating model on train/val/test sets...")
    train_metrics = evaluate_model(model, train_df, feature_cols, dataset_name="Train")
    val_metrics = evaluate_model(model, val_df, feature_cols, dataset_name="Validation")
    test_metrics = evaluate_model(model, test_df, feature_cols, dataset_name="Test")

    # SHAP explainer 생성 (선택사항)
    print("\n[INFO] Building SHAP explainer...")
    try:
        explainer = build_shap_explainer(
            model,
            train_df,
            feature_cols,
            sample_size=2000,
            background_size=200,
        )
    except Exception as e:
        print(f"[WARN] SHAP explainer build failed (XGBoost/SHAP version incompatibility): {e}")
        print("[INFO] Skipping SHAP explainer, model will still work without it")
        explainer = None

    # 모델 저장
    print("\n[INFO] Saving model artifacts...")
    save_model_artifacts(
        model=model,
        feature_cols=feature_cols,
        explainer=None,  # Explainer는 저장하지 않음 (런타임에 재생성)
        metrics=test_metrics,  # Test set 성능 저장
    )

    print("\n[SUCCESS] Model training and saving completed!")
    print(f"[INFO] Model artifacts saved to: {MODELS_DIR}/")


if __name__ == "__main__":
    main()