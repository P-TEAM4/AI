"""
League Data Preprocessor

엑셀 파일에서 학습 데이터를 전처리하여 베이스라인 학습에 사용할 수 있는 형태로 변환합니다.

Input: league_data.xlsx (40,410 rows × 94 columns)
Output: processed_match_data.json (학습용 데이터)

사용법:
    python scripts/preprocess_league_data.py --input /path/to/league_data.xlsx
    python scripts/preprocess_league_data.py --input /path/to/league_data.xlsx --output data/custom_output.json
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import json
from typing import Dict, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class LeagueDataPreprocessor:
    """리그 오브 레전드 매치 데이터 전처리 클래스"""

    def __init__(self, filepath: str):
        """
        전처리기 초기화

        Args:
            filepath: 엑셀 파일 경로
        """
        self.filepath = Path(filepath)
        self.df = None
        self.processed_df = None

    def load_data(self):
        """엑셀 파일 로드"""
        print(f"Loading data from {self.filepath}")

        if not self.filepath.exists():
            raise FileNotFoundError(f"File not found: {self.filepath}")

        # 엑셀 파일 로드
        self.df = pd.read_excel(self.filepath)

        print(f"Loaded {len(self.df)} records")
        print(f"   Columns: {len(self.df.columns)}")
        print(f"   Shape: {self.df.shape}")

    def validate_columns(self):
        """필수 컬럼 확인"""
        required_columns = [
            "solo_tier",  # 솔로랭크 티어
            "kills",
            "deaths",
            "assists",
            "gold_earned",
            "game_duration",
            "vision_score",
            "total_damage_dealt_to_champions",
            "win",
        ]

        missing = [col for col in required_columns if col not in self.df.columns]

        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        print(f"All required columns present")

    def clean_data(self):
        """데이터 정제"""
        print("\nCleaning data...")

        original_len = len(self.df)

        # 1. 솔로랭크 티어가 있는 데이터만 (Unranked 제외)
        self.df = self.df[self.df["solo_tier"].notna()]
        print(f"   - Removed unranked players: {original_len - len(self.df)} rows")

        # 2. 정규 게임만 (queue_id == 420: Ranked Solo/Duo)
        if "queue_id" in self.df.columns:
            self.df = self.df[self.df["queue_id"] == 420]
            print(f"   - Filtered to ranked solo/duo only: {len(self.df)} rows")

        # 3. 게임 시간이 유효한 경기만 (최소 10분, 최대 60분)
        self.df = self.df[
            (self.df["game_duration"] >= 600) & (self.df["game_duration"] <= 3600)
        ]
        print(f"   - Filtered valid game duration: {len(self.df)} rows")

        # 4. 결측치 제거
        essential_cols = ["kills", "deaths", "assists", "gold_earned", "vision_score"]
        before_len = len(self.df)
        self.df = self.df.dropna(subset=essential_cols)
        print(f"   - Removed missing values: {before_len - len(self.df)} rows")

        print(f"Cleaned data: {len(self.df)} rows remaining")

    def calculate_features(self):
        """학습에 필요한 피처 계산"""
        print("\nCalculating features...")

        df = self.df.copy()

        # 1. KDA 계산
        # deaths가 0이면 kills + assists 반환
        df["kda"] = df.apply(
            lambda row: (
                (row["kills"] + row["assists"]) / row["deaths"]
                if row["deaths"] > 0
                else (row["kills"] + row["assists"])
            ),
            axis=1,
        )
        print("   ✓ KDA calculated")

        # 2. 게임 시간 (분 단위)
        df["game_minutes"] = df["game_duration"] / 60

        # 3. CS/min 계산 (미니언 킬 컬럼이 있는 경우)
        if "totalMinionsKilled" in df.columns and "neutralMinionsKilled" in df.columns:
            df["total_cs"] = (
                df["totalMinionsKilled"] + df["neutralMinionsKilled"]
            )
            df["cs_per_min"] = df["total_cs"] / df["game_minutes"]
            print("   ✓ CS/min calculated from raw data")
        elif "cs_per_min" in df.columns:
            # 이미 계산된 경우
            print("   ✓ CS/min already exists")
        else:
            # CS 데이터가 없으면 0으로 설정 (경고)
            print("   CS data not found, setting to 0")
            df["cs_per_min"] = 0

        # 4. Gold/min 계산
        df["gold_per_min"] = df["gold_earned"] / df["game_minutes"]
        print("   ✓ Gold/min calculated")

        # 5. Vision/min 계산
        df["vision_score_per_min"] = df["vision_score"] / df["game_minutes"]
        print("   ✓ Vision/min calculated")

        # 6. Damage Share 계산 (팀 내 비중)
        # game_id와 team_id로 그룹화하여 팀 전체 데미지 계산
        if "game_id" in df.columns and "team_id" in df.columns:
            # 팀별 총 데미지 계산
            team_damage = df.groupby(["game_id", "team_id"])[
                "total_damage_dealt_to_champions"
            ].transform("sum")

            # Damage Share 계산 (0으로 나누기 방지)
            df["damage_share"] = df["total_damage_dealt_to_champions"] / team_damage.replace(0, 1)
            print("   ✓ Damage share calculated")
        else:
            # 게임/팀 정보가 없으면 평균값으로 설정
            print("   Game/Team info not found, using default damage share")
            df["damage_share"] = 0.20

        # 7. 티어 표준화 (대문자로 변환)
        df["tier"] = df["solo_tier"].str.upper()

        # 8. Division 정보 추가 (있으면)
        if "solo_rank" in df.columns:
            df["division"] = df["solo_rank"]
        else:
            df["division"] = "I"  # 기본값

        self.processed_df = df
        print(f"Features calculated")

    def select_features(self) -> pd.DataFrame:
        """학습에 필요한 컬럼만 선택"""
        print("\nSelecting features for training...")

        # 학습에 필요한 컬럼
        feature_columns = [
            "tier",  # 티어
            "division",  # 디비전
            "kda",  # KDA
            "cs_per_min",  # CS/분
            "gold_earned",  # 총 골드
            "gold_per_min",  # 골드/분
            "vision_score",  # 비전 스코어
            "vision_score_per_min",  # 비전/분
            "damage_share",  # 팀 내 딜 비중
            "game_duration",  # 게임 시간 (초)
            "win",  # 승패
        ]

        # 존재하는 컬럼만 선택
        available_columns = [col for col in feature_columns if col in self.processed_df.columns]

        result_df = self.processed_df[available_columns].copy()

        print(f"Selected {len(available_columns)} features")
        print(f"   Features: {available_columns}")

        return result_df

    def get_tier_statistics(self) -> Dict:
        """티어별 통계 출력"""
        if self.processed_df is None:
            return {}

        print("\nTier Statistics:")
        print("=" * 80)

        tier_stats = self.processed_df.groupby("tier").agg(
            {
                "tier": "count",
                "kda": "mean",
                "cs_per_min": "mean",
                "gold_per_min": "mean",
                "vision_score_per_min": "mean",
                "damage_share": "mean",
                "win": "mean",
            }
        )

        tier_stats.columns = [
            "count",
            "avg_kda",
            "avg_cs_per_min",
            "avg_gold_per_min",
            "avg_vision_per_min",
            "avg_damage_share",
            "win_rate",
        ]

        tier_stats["win_rate"] = tier_stats["win_rate"] * 100
        tier_stats = tier_stats.round(2)

        print(tier_stats.to_string())
        print("=" * 80)

        return tier_stats.to_dict()

    def save_to_json(self, output_path: str):
        """처리된 데이터를 JSON으로 저장"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 학습용 데이터 선택
        training_df = self.select_features()

        # JSON으로 변환 (records 형식)
        data = training_df.to_dict(orient="records")

        # 저장
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"\nData saved to {output_path}")
        print(f"   Records: {len(data)}")
        print(f"   File size: {output_path.stat().st_size / 1024:.2f} KB")

    def process(self, output_path: str):
        """전체 전처리 파이프라인 실행"""
        print("=" * 80)
        print("LEAGUE DATA PREPROCESSING")
        print("=" * 80)

        # 1. 데이터 로드
        self.load_data()

        # 2. 컬럼 검증
        self.validate_columns()

        # 3. 데이터 정제
        self.clean_data()

        # 4. 피처 계산
        self.calculate_features()

        # 5. 티어별 통계
        self.get_tier_statistics()

        # 6. 저장
        self.save_to_json(output_path)

        print("\nPreprocessing completed successfully!")
        print(f"\nNext step:")
        print(f"   python scripts/train_baselines.py --input {output_path} --display")


def main():
    parser = argparse.ArgumentParser(description="Preprocess League of Legends match data")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Input Excel file path (league_data.xlsx)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="data/processed_match_data.json",
        help="Output JSON file path (default: data/processed_match_data.json)",
    )

    args = parser.parse_args()

    try:
        # 전처리 실행
        preprocessor = LeagueDataPreprocessor(args.input)
        preprocessor.process(args.output)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
