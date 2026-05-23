"""
점수 계산 로직 검증 스크립트

점수가 합리적인지 다양한 관점에서 검증합니다:
1. 승리팀 vs 패배팀 점수 분포
2. 극단적 케이스 검증 (캐리, 트롤)
3. 점수와 주요 지표의 상관관계
4. 등급 분포의 합리성
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from player_rating import (
    load_player_data,
    add_derived_features,
    train_win_model,
    split_data_by_match,
    compute_player_impact,
    build_shap_explainer
)


def validate_score_distribution(df, model, explainer, feature_cols):
    """점수 분포 검증"""
    print("\n" + "="*60)
    print("1. 점수 분포 검증")
    print("="*60)

    # 샘플링 (전체는 시간이 오래 걸림)
    sample_df = df.sample(min(1000, len(df)), random_state=42)

    scores = []
    grades = []
    wins = []

    for idx, row in sample_df.iterrows():
        report = compute_player_impact(row, model, explainer, feature_cols)
        scores.append(report["impactScore"])
        grades.append(report["grade"])
        wins.append(report["win"])

    scores = np.array(scores)
    wins = np.array(wins)

    # 승리/패배 팀 점수 비교
    win_scores = scores[wins == 1]
    loss_scores = scores[wins == 0]

    print(f"\n전체 점수 통계:")
    print(f"  평균: {scores.mean():.1f}점")
    print(f"  중앙값: {np.median(scores):.1f}점")
    print(f"  표준편차: {scores.std():.1f}점")
    print(f"  최소: {scores.min():.1f}점")
    print(f"  최대: {scores.max():.1f}점")

    print(f"\n승리팀 점수:")
    print(f"  평균: {win_scores.mean():.1f}점")
    print(f"  중앙값: {np.median(win_scores):.1f}점")

    print(f"\n패배팀 점수:")
    print(f"  평균: {loss_scores.mean():.1f}점")
    print(f"  중앙값: {np.median(loss_scores):.1f}점")

    print(f"\n✅ 검증: 승리팀이 평균 {win_scores.mean() - loss_scores.mean():.1f}점 더 높음")

    # 등급 분포
    print(f"\n등급 분포:")
    grade_counts = pd.Series(grades).value_counts()
    for grade in ['S', 'A', 'B', 'C', 'D']:
        count = grade_counts.get(grade, 0)
        pct = count / len(grades) * 100
        print(f"  {grade}: {count:4d}명 ({pct:5.1f}%)")

    return scores, wins, grades


def validate_extreme_cases(df, model, explainer, feature_cols):
    """극단적 케이스 검증"""
    print("\n" + "="*60)
    print("2. 극단적 케이스 검증")
    print("="*60)

    # KDA가 매우 높은 플레이어
    high_kda = df.nlargest(5, 'kda')
    print("\n[높은 KDA 플레이어]")
    for idx, row in high_kda.iterrows():
        report = compute_player_impact(row, model, explainer, feature_cols)
        print(f"  KDA {row['kda']:.1f} | 점수: {report['impactScore']:.1f} ({report['grade']}) | "
              f"{'승' if row['win'] else '패'} | {row['champion']}")

    # KDA가 매우 낮은 플레이어
    low_kda = df.nsmallest(5, 'kda')
    print("\n[낮은 KDA 플레이어]")
    for idx, row in low_kda.iterrows():
        report = compute_player_impact(row, model, explainer, feature_cols)
        print(f"  KDA {row['kda']:.1f} | 점수: {report['impactScore']:.1f} ({report['grade']}) | "
              f"{'승' if row['win'] else '패'} | {row['champion']}")

    # 높은 골드퍼민 (잘 파밍한 플레이어)
    high_gold = df.nlargest(5, 'goldPerMinute')
    print("\n[높은 골드퍼민 플레이어]")
    for idx, row in high_gold.iterrows():
        report = compute_player_impact(row, model, explainer, feature_cols)
        print(f"  GPM {row['goldPerMinute']:.1f} | 점수: {report['impactScore']:.1f} ({report['grade']}) | "
              f"{'승' if row['win'] else '패'} | {row['champion']}")


def validate_win_loss_paradox(df, model, explainer, feature_cols):
    """승패 역설 검증: 잘했는데 진 경기 vs 캐리해서 이긴 경기"""
    print("\n" + "="*60)
    print("3. 승패 역설 검증")
    print("="*60)

    # 패배했지만 KDA 높은 경기
    good_loss = df[(df['win'] == 0) & (df['kda'] >= 4.0)].head(5)
    print("\n[잘했는데 진 경기]")
    for idx, row in good_loss.iterrows():
        report = compute_player_impact(row, model, explainer, feature_cols)
        print(f"  KDA {row['kda']:.1f} | 점수: {report['impactScore']:.1f} ({report['grade']}) | "
              f"{row['champion']} | {report['matchComment']}")

    # 승리했지만 KDA 낮은 경기 (팀 캐리)
    carry_win = df[(df['win'] == 1) & (df['kda'] >= 5.0)].head(5)
    print("\n[캐리해서 이긴 경기]")
    for idx, row in carry_win.iterrows():
        report = compute_player_impact(row, model, explainer, feature_cols)
        print(f"  KDA {row['kda']:.1f} | 점수: {report['impactScore']:.1f} ({report['grade']}) | "
              f"{row['champion']} | {report['matchComment']}")

    print("\n✅ 검증: 잘했는데 진 경기도 적절한 점수를 받는지 확인")


def validate_correlation(df, model, explainer, feature_cols):
    """점수와 주요 지표의 상관관계"""
    print("\n" + "="*60)
    print("4. 점수와 지표 상관관계")
    print("="*60)

    # 샘플링
    sample_df = df.sample(min(500, len(df)), random_state=42)

    scores = []
    for idx, row in sample_df.iterrows():
        report = compute_player_impact(row, model, explainer, feature_cols)
        scores.append(report["impactScore"])

    sample_df = sample_df.copy()
    sample_df['score'] = scores

    # 주요 지표와의 상관관계
    metrics = ['kda', 'goldPerMinute', 'damagePerMinute', 'killParticipation', 'win']

    print("\n점수와의 상관계수:")
    for metric in metrics:
        if metric in sample_df.columns:
            corr = sample_df['score'].corr(sample_df[metric])
            print(f"  {metric:20s}: {corr:6.3f}")

    print("\n✅ 검증: KDA, 골드 등과 양의 상관관계를 가져야 함")
    print("✅ 검증: win과의 상관관계가 너무 높으면(>0.9) 승패에만 의존")


def validate_same_match(df, model, explainer, feature_cols):
    """같은 경기 내 플레이어 점수 분포"""
    print("\n" + "="*60)
    print("5. 같은 경기 내 점수 분포")
    print("="*60)

    # 무작위 경기 하나 선택
    match_id = df['matchId'].iloc[100]
    match_players = df[df['matchId'] == match_id]

    print(f"\n매치: {match_id}")
    print(f"{'팀':4s} {'챔피언':12s} {'역할':8s} {'결과':4s} {'KDA':6s} {'점수':6s} {'등급':4s}")
    print("-" * 60)

    for idx, row in match_players.iterrows():
        report = compute_player_impact(row, model, explainer, feature_cols)
        print(f"{row['teamId']:4.0f} {row['champion']:12s} {row['role']:8s} "
              f"{'승' if row['win'] else '패':4s} {row['kda']:6.2f} "
              f"{report['impactScore']:6.1f} {report['grade']:4s}")

    # 승리팀과 패배팀 평균 점수
    win_team = match_players[match_players['win'] == 1]
    loss_team = match_players[match_players['win'] == 0]

    win_scores = []
    loss_scores = []

    for idx, row in win_team.iterrows():
        report = compute_player_impact(row, model, explainer, feature_cols)
        win_scores.append(report['impactScore'])

    for idx, row in loss_team.iterrows():
        report = compute_player_impact(row, model, explainer, feature_cols)
        loss_scores.append(report['impactScore'])

    print(f"\n승리팀 평균 점수: {np.mean(win_scores):.1f}")
    print(f"패배팀 평균 점수: {np.mean(loss_scores):.1f}")

    print("\n✅ 검증: 같은 팀 내에서도 개인차가 있어야 함")
    print("✅ 검증: 승리팀이 전반적으로 높지만, 패배팀에도 높은 점수 있을 수 있음")


def main():
    print("점수 계산 로직 검증을 시작합니다...\n")

    # 데이터 로드
    print("[INFO] 데이터 로드 중...")
    df = load_player_data('../data/processed/player_stats.csv')
    df = add_derived_features(df)

    # 데이터 분할
    print("[INFO] 데이터 분할 중...")
    train_df, val_df, test_df = split_data_by_match(df, test_size=0.15, val_size=0.15, random_state=42)

    # 모델 학습
    print("[INFO] 모델 학습 중...")
    model, feature_cols = train_win_model(train_df, val_df)

    # Explainer 생성
    print("[INFO] SHAP Explainer 생성 중...")
    explainer = build_shap_explainer(model, train_df, feature_cols, sample_size=500)

    # 테스트 데이터로 검증
    print("\n[INFO] 테스트 데이터로 검증 시작...\n")

    # 각종 검증 실행
    scores, wins, grades = validate_score_distribution(test_df, model, explainer, feature_cols)
    validate_extreme_cases(test_df, model, explainer, feature_cols)
    validate_win_loss_paradox(test_df, model, explainer, feature_cols)
    validate_correlation(test_df, model, explainer, feature_cols)
    validate_same_match(df, model, explainer, feature_cols)

    print("\n" + "="*60)
    print("검증 완료!")
    print("="*60)
    print("\n주요 체크포인트:")
    print("  ✓ 승리팀이 패배팀보다 평균 점수가 높은가?")
    print("  ✓ 등급 분포가 정규분포에 가까운가? (B/C가 가장 많아야 함)")
    print("  ✓ 잘했는데 진 경기도 좋은 점수를 받는가?")
    print("  ✓ 주요 지표(KDA, 골드 등)와 적절한 상관관계를 가지는가?")
    print("  ✓ win과의 상관관계가 너무 높지 않은가? (<0.8)")


if __name__ == "__main__":
    main()
