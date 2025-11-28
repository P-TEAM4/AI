"""
가중치 평가 및 최적화 스크립트

현재 가중치가 적절한지 평가하고, 최적의 가중치를 찾습니다.

사용법:
    python scripts/evaluate_weights.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services.baseline_trainer import BaselineTrainer


class WeightEvaluator:
    """가중치 평가 및 최적화 클래스"""

    def __init__(self, data_file: str = "data/processed_match_data.json"):
        """
        초기화

        Args:
            data_file: 학습 데이터 파일 경로
        """
        self.data_file = Path(data_file)
        self.trainer = BaselineTrainer()
        self.df = None

    def load_data(self):
        """데이터 로드"""
        print("데이터 로딩 중...")
        self.df = self.trainer.load_match_data(str(self.data_file))
        self.df = self.trainer.calculate_per_minute_stats(self.df)
        print(f"총 {len(self.df)} 경기 로드 완료\n")

    def calculate_feature_importance_by_winrate(self) -> pd.DataFrame:
        """
        승률과의 상관관계로 각 지표의 중요도 분석

        Returns:
            각 지표의 중요도 데이터프레임
        """
        print("=" * 80)
        print("승률 기반 지표 중요도 분석")
        print("=" * 80)

        features = [
            "kda",
            "cs_per_min",
            "gold_per_min",
            "vision_score_per_min",
            "damage_share",
        ]

        results = []

        for feature in features:
            if feature not in self.df.columns:
                continue

            # 승리/패배 그룹별 평균
            win_avg = self.df[self.df["win"] == True][feature].mean()
            lose_avg = self.df[self.df["win"] == False][feature].mean()

            # 차이 계산
            diff = win_avg - lose_avg
            diff_percent = (diff / lose_avg * 100) if lose_avg != 0 else 0

            # 상관계수 계산
            correlation = self.df[[feature, "win"]].corr().iloc[0, 1]

            results.append(
                {
                    "지표": feature,
                    "승리_평균": round(win_avg, 2),
                    "패배_평균": round(lose_avg, 2),
                    "차이": round(diff, 2),
                    "차이율(%)": round(diff_percent, 2),
                    "상관계수": round(correlation, 3),
                }
            )

        df_result = pd.DataFrame(results)
        df_result = df_result.sort_values("상관계수", ascending=False)

        print("\n결과:")
        print(df_result.to_string(index=False))
        print("\n" + "=" * 80)

        return df_result

    def calculate_feature_variance_by_tier(self) -> pd.DataFrame:
        """
        티어별 지표의 분산도 분석
        분산이 큰 지표일수록 티어 구분에 중요

        Returns:
            티어별 분산도 데이터프레임
        """
        print("\n" + "=" * 80)
        print("티어별 분산도 분석")
        print("=" * 80)

        features = [
            "kda",
            "cs_per_min",
            "gold_per_min",
            "vision_score_per_min",
            "damage_share",
        ]

        tier_stats = self.df.groupby("tier")[features].mean()

        results = []
        for feature in features:
            if feature not in tier_stats.columns:
                continue

            # 티어 간 평균의 표준편차
            std = tier_stats[feature].std()
            mean = tier_stats[feature].mean()

            # 변동계수 (Coefficient of Variation)
            cv = (std / mean * 100) if mean != 0 else 0

            # 최대-최소 범위
            range_val = tier_stats[feature].max() - tier_stats[feature].min()
            range_percent = (range_val / mean * 100) if mean != 0 else 0

            results.append(
                {
                    "지표": feature,
                    "평균": round(mean, 2),
                    "표준편차": round(std, 2),
                    "변동계수(%)": round(cv, 2),
                    "범위": round(range_val, 2),
                    "범위비율(%)": round(range_percent, 2),
                }
            )

        df_result = pd.DataFrame(results)
        df_result = df_result.sort_values("변동계수(%)", ascending=False)

        print("\n결과:")
        print(df_result.to_string(index=False))
        print("\n해석: 변동계수가 클수록 티어 구분에 중요한 지표")
        print("=" * 80)

        return df_result

    def suggest_optimal_weights(
        self, winrate_df: pd.DataFrame, variance_df: pd.DataFrame
    ) -> Dict[str, float]:
        """
        분석 결과를 바탕으로 최적 가중치 제안

        Args:
            winrate_df: 승률 분석 결과
            variance_df: 분산도 분석 결과

        Returns:
            제안된 가중치
        """
        print("\n" + "=" * 80)
        print("최적 가중치 제안")
        print("=" * 80)

        # 승률 상관계수와 변동계수를 결합
        scores = {}

        for _, row in winrate_df.iterrows():
            feature = row["지표"]
            corr = abs(row["상관계수"])  # 절대값 사용

            # 분산도에서 변동계수 찾기
            var_row = variance_df[variance_df["지표"] == feature]
            if not var_row.empty:
                cv = var_row.iloc[0]["변동계수(%)"]
            else:
                cv = 0

            # 점수 계산 (승률 상관 60%, 분산도 40%)
            score = corr * 0.6 + (cv / 100) * 0.4
            scores[feature] = score

        # 정규화하여 합이 1이 되도록
        total_score = sum(scores.values())
        weights = {k: v / total_score for k, v in scores.items()}

        print("\n현재 가중치:")
        current_weights = {
            "kda": 0.25,
            "cs_per_min": 0.20,
            "gold_per_min": 0.20,
            "vision_score_per_min": 0.15,
            "damage_share": 0.20,
        }
        for feature, weight in current_weights.items():
            print(f"  {feature}: {weight:.2f} ({weight*100:.0f}%)")

        print("\n제안 가중치:")
        for feature, weight in sorted(
            weights.items(), key=lambda x: x[1], reverse=True
        ):
            change = weight - current_weights.get(feature, 0)
            change_str = f"({change:+.2f})" if change != 0 else ""
            print(f"  {feature}: {weight:.2f} ({weight*100:.0f}%) {change_str}")

        print("\n변경 사항:")
        for feature in weights.keys():
            current = current_weights.get(feature, 0)
            suggested = weights[feature]
            change = suggested - current

            if abs(change) > 0.01:
                direction = "증가" if change > 0 else "감소"
                print(
                    f"  - {feature}: {abs(change):.2f} {direction} ({abs(change)*100:.0f}%p)"
                )

        print("\n" + "=" * 80)

        return weights

    def generate_update_code(self, weights: Dict[str, float]):
        """
        업데이트할 Python 코드 생성

        Args:
            weights: 제안된 가중치
        """
        print("\n" + "=" * 80)
        print("src/models/rule_based.py에 적용할 코드:")
        print("=" * 80)
        print("\n# Weights for different statistics")
        print("weights = {")
        for feature, weight in sorted(
            weights.items(), key=lambda x: x[1], reverse=True
        ):
            print(f'    "{feature}": {weight:.2f},')
        print("}")
        print("\n" + "=" * 80)


def main():
    evaluator = WeightEvaluator()

    print("=" * 80)
    print("가중치 평가 시스템")
    print("=" * 80)

    # 1. 데이터 로드
    evaluator.load_data()

    # 2. 승률 기반 중요도 분석
    winrate_df = evaluator.calculate_feature_importance_by_winrate()

    # 3. 티어별 분산도 분석
    variance_df = evaluator.calculate_feature_variance_by_tier()

    # 4. 최적 가중치 제안
    weights = evaluator.suggest_optimal_weights(winrate_df, variance_df)

    # 5. 업데이트 코드 생성
    evaluator.generate_update_code(weights)

    print("\n참고:")
    print("- 제안된 가중치는 데이터 기반 분석 결과입니다")
    print("- 실제 적용 시 도메인 지식과 함께 고려하세요")
    print("- A/B 테스트를 통해 실제 성능을 검증하는 것을 권장합니다")


if __name__ == "__main__":
    main()
