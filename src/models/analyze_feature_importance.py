"""Feature 중요도 및 선택 이유 분석 스크립트

이 스크립트는 다음을 생성합니다:
1. XGBoost Feature Importance 차트
2. Feature-Target 상관관계 분석
3. SHAP Global Summary
4. Feature 선택 이유 리포트
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier

from player_rating import (
    load_player_data,
    add_derived_features,
)

# 한글 폰트 설정 (Mac의 경우)
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# 프로젝트 루트 디렉토리 기준 경로 설정
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AI_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # AI 폴더
PROJECT_ROOT = os.path.dirname(AI_ROOT)  # lol_project 폴더
PLAYER_DF_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "player_stats.csv")
MODELS_DIR = os.path.join(AI_ROOT, "models")
RESULTS_DIR = os.path.join(AI_ROOT, "models", "feature_analysis")
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_trained_model():
    """저장된 모델과 feature 목록 로드"""
    model_path = os.path.join(MODELS_DIR, "player_impact_model.json")
    feature_path = os.path.join(MODELS_DIR, "feature_cols.json")

    if not os.path.exists(model_path) or not os.path.exists(feature_path):
        raise FileNotFoundError(
            f"모델 파일이 없습니다. 먼저 train_player_impact_model.py를 실행하세요."
        )

    model = XGBClassifier()
    model.load_model(model_path)

    with open(feature_path, "r") as f:
        feature_cols = json.load(f)

    print(f"[INFO] Loaded model from {model_path}")
    print(f"[INFO] Loaded {len(feature_cols)} features")

    return model, feature_cols


def plot_xgboost_importance(model, feature_cols, top_n=20):
    """XGBoost 내장 Feature Importance 시각화

    Args:
        model: 학습된 XGBoost 모델
        feature_cols: feature 이름 리스트
        top_n: 상위 몇 개 feature를 표시할지
    """
    importance = model.feature_importances_

    # Feature 이름과 중요도를 DataFrame으로 정리
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importance
    }).sort_values('importance', ascending=False)

    # 상위 N개만 선택
    top_features = importance_df.head(top_n)

    # 시각화
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top_features)), top_features['importance'].values)
    plt.yticks(range(len(top_features)), top_features['feature'].values)
    plt.xlabel('Feature Importance (Gain)', fontsize=12)
    plt.title(f'XGBoost Feature Importance (Top {top_n})', fontsize=14)
    plt.gca().invert_yaxis()
    plt.tight_layout()

    save_path = os.path.join(RESULTS_DIR, "xgboost_feature_importance.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[INFO] Saved XGBoost importance plot to {save_path}")
    plt.close()

    # CSV로도 저장
    csv_path = os.path.join(RESULTS_DIR, "feature_importance.csv")
    importance_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"[INFO] Saved importance rankings to {csv_path}")

    return importance_df


def analyze_feature_target_correlation(df, feature_cols, target='win'):
    """Feature와 Target(승패) 간 상관관계 분석"""

    # 상관관계 계산
    correlations = []
    for feat in feature_cols:
        if feat in df.columns:
            corr = df[feat].corr(df[target])
            correlations.append({
                'feature': feat,
                'correlation': corr,
                'abs_correlation': abs(corr)
            })

    corr_df = pd.DataFrame(correlations).sort_values('abs_correlation', ascending=False)

    # 시각화
    plt.figure(figsize=(10, 8))
    top_n = min(20, len(corr_df))
    top_corr = corr_df.head(top_n)

    colors = ['red' if x < 0 else 'blue' for x in top_corr['correlation'].values]
    plt.barh(range(len(top_corr)), top_corr['correlation'].values, color=colors)
    plt.yticks(range(len(top_corr)), top_corr['feature'].values)
    plt.xlabel('Correlation with Win', fontsize=12)
    plt.title(f'Feature-Target Correlation (Top {top_n})', fontsize=14)
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    plt.gca().invert_yaxis()
    plt.tight_layout()

    save_path = os.path.join(RESULTS_DIR, "feature_target_correlation.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[INFO] Saved correlation plot to {save_path}")
    plt.close()

    # CSV 저장
    csv_path = os.path.join(RESULTS_DIR, "feature_correlations.csv")
    corr_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"[INFO] Saved correlations to {csv_path}")

    return corr_df


def plot_feature_distributions(df, feature_cols, top_n=6):
    """승/패별 Feature 분포 비교 (상위 N개)"""

    # 데이터가 있는 feature만 필터링
    available_features = [f for f in feature_cols if f in df.columns][:top_n]

    n_features = len(available_features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
    axes = axes.flatten() if n_features > 1 else [axes]

    for idx, feat in enumerate(available_features):
        ax = axes[idx]

        # 승/패로 분리
        win_data = df[df['win'] == 1][feat].dropna()
        loss_data = df[df['win'] == 0][feat].dropna()

        # 박스플롯
        ax.boxplot([win_data, loss_data], labels=['Win', 'Loss'])
        ax.set_title(f'{feat}', fontsize=11)
        ax.set_ylabel('Value')
        ax.grid(axis='y', alpha=0.3)

    # 빈 subplot 숨기기
    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, "feature_distributions.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[INFO] Saved distribution plots to {save_path}")
    plt.close()


def generate_feature_selection_report(importance_df, corr_df):
    """Feature 선택 이유를 설명하는 리포트 생성"""

    # Feature 카테고리별 설명
    feature_categories = {
        "초반 성장 지표 (Early Game Growth)": [
            "cs_15", "gold_15", "xp_15", "early_cs_total",
            "laneMinionsFirst10Minutes", "jungleCsBefore10Minutes"
        ],
        "효율성 지표 (Per-Minute Efficiency)": [
            "goldPerMinute", "damagePerMinute", "visionScorePerMinute"
        ],
        "개인 플레이 품질 (Individual Performance)": [
            "kda", "kda_norm", "killParticipation", "soloKills"
        ],
        "시야 기여도 (Vision Control)": [
            "vision_norm", "controlWardsPlaced", "wardTakedowns",
            "wardTakedownsBefore20M"
        ],
        "플레이 품질 (Mechanics & Survival)": [
            "skillshotsDodged", "skillshotsHit", "longestTimeSpentLiving", "death_rate"
        ]
    }

    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("Feature 선택 이유 분석 리포트")
    report_lines.append("=" * 80)
    report_lines.append("")

    report_lines.append("## 1. Feature 선택 원칙")
    report_lines.append("")
    report_lines.append("이 모델은 '데이터 누수(Data Leakage)'를 방지하기 위해 다음 원칙을 따릅니다:")
    report_lines.append("- 승패 독립적인 지표만 사용 (게임 결과에 직접적으로 영향받지 않는 지표)")
    report_lines.append("- 개인 플레이 품질을 측정 가능한 지표")
    report_lines.append("- 초반 성장 및 효율성 지표 (15분 이전 데이터 우선)")
    report_lines.append("")

    report_lines.append("## 2. Feature 카테고리별 분류")
    report_lines.append("")

    for category, features in feature_categories.items():
        report_lines.append(f"### {category}")
        for feat in features:
            # Importance 정보
            imp_row = importance_df[importance_df['feature'] == feat]
            imp_value = imp_row['importance'].values[0] if not imp_row.empty else 0.0
            imp_rank = imp_row.index[0] + 1 if not imp_row.empty else "N/A"

            # Correlation 정보
            corr_row = corr_df[corr_df['feature'] == feat]
            corr_value = corr_row['correlation'].values[0] if not corr_row.empty else 0.0

            report_lines.append(
                f"  - {feat:30s} | Importance: {imp_value:6.4f} (#{imp_rank}) | "
                f"Win Correlation: {corr_value:+6.3f}"
            )
        report_lines.append("")

    report_lines.append("## 3. 상위 10개 중요 Feature")
    report_lines.append("")
    top_10 = importance_df.head(10)
    report_lines.append(f"{'Rank':>4} | {'Feature':30s} | {'Importance':>12} | {'Win Corr':>10}")
    report_lines.append("-" * 70)

    for idx, row in top_10.iterrows():
        feat = row['feature']
        imp = row['importance']

        corr_row = corr_df[corr_df['feature'] == feat]
        corr = corr_row['correlation'].values[0] if not corr_row.empty else 0.0

        rank = idx + 1
        report_lines.append(f"{rank:4d} | {feat:30s} | {imp:12.6f} | {corr:+10.3f}")

    report_lines.append("")
    report_lines.append("## 4. 해석")
    report_lines.append("")
    report_lines.append("- **Importance**: 모델이 예측할 때 해당 feature를 얼마나 자주/많이 사용했는지")
    report_lines.append("- **Win Correlation**: 해당 feature 값이 높을수록 승리 확률이 높은지 (+) 낮은지 (-)")
    report_lines.append("- 높은 Importance + 높은 |Correlation| = 승패 예측에 매우 유용한 지표")
    report_lines.append("")

    report_lines.append("## 5. Feature 선택의 이점")
    report_lines.append("")
    report_lines.append("✓ 데이터 누수 방지: 게임 결과가 아닌 플레이어 개인의 플레이 품질 측정")
    report_lines.append("✓ 해석 가능성: 각 feature가 무엇을 의미하는지 명확함")
    report_lines.append("✓ 공정성: 팀 전체 성과가 아닌 개인 기여도를 평가")
    report_lines.append("✓ 실용성: 플레이어가 개선 가능한 영역 제시")
    report_lines.append("")

    report_text = "\n".join(report_lines)

    # 저장
    report_path = os.path.join(RESULTS_DIR, "feature_selection_report.txt")
    with open(report_path, "w", encoding='utf-8') as f:
        f.write(report_text)

    print(f"[INFO] Saved feature selection report to {report_path}")
    print("\n" + "=" * 80)
    print(report_text)
    print("=" * 80)

    return report_text


def main():
    print("[INFO] Feature 중요도 분석 시작\n")

    # 1. 모델 및 feature 로드
    print("=" * 80)
    print("Step 1: 모델 및 Feature 로드")
    print("=" * 80)
    model, feature_cols = load_trained_model()

    # 2. 데이터 로드 (전체 데이터 사용)
    print("\n" + "=" * 80)
    print("Step 2: 데이터 로드 및 전처리")
    print("=" * 80)
    df = load_player_data(PLAYER_DF_PATH)
    df = add_derived_features(df)
    print(f"[INFO] Loaded {len(df)} player records")

    # Feature 준비 (숫자형 변환)
    for col in feature_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    # 3. XGBoost Feature Importance
    print("\n" + "=" * 80)
    print("Step 3: XGBoost Feature Importance 계산")
    print("=" * 80)
    importance_df = plot_xgboost_importance(model, feature_cols, top_n=20)

    # 4. Feature-Target Correlation
    print("\n" + "=" * 80)
    print("Step 4: Feature-Target 상관관계 분석")
    print("=" * 80)
    corr_df = analyze_feature_target_correlation(df, feature_cols)

    # 5. Feature 분포 시각화
    print("\n" + "=" * 80)
    print("Step 5: Feature 분포 시각화 (승/패 비교)")
    print("=" * 80)
    plot_feature_distributions(df, importance_df['feature'].head(9).tolist(), top_n=9)

    # 6. Feature 선택 리포트 생성
    print("\n" + "=" * 80)
    print("Step 6: Feature 선택 이유 리포트 생성")
    print("=" * 80)
    generate_feature_selection_report(importance_df, corr_df)

    print("\n" + "=" * 80)
    print(f"[SUCCESS] 분석 완료! 결과는 {RESULTS_DIR}/ 에 저장되었습니다.")
    print("=" * 80)
    print(f"\n생성된 파일:")
    print(f"  1. {RESULTS_DIR}/xgboost_feature_importance.png")
    print(f"  2. {RESULTS_DIR}/feature_target_correlation.png")
    print(f"  3. {RESULTS_DIR}/feature_distributions.png")
    print(f"  4. {RESULTS_DIR}/feature_importance.csv")
    print(f"  5. {RESULTS_DIR}/feature_correlations.csv")
    print(f"  6. {RESULTS_DIR}/feature_selection_report.txt")


if __name__ == "__main__":
    main()
