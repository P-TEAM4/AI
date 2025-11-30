"""
Train Tier Baselines from Downloaded Match Data

티어별 평균 통계(Baseline)를 계산하는 스크립트입니다.
collect_tier_data.py로 수집한 데이터를 사용합니다.

Usage:
    # 단일 파일에서 학습
    python scripts/train_baselines.py --input data/tier_collections/gold_tier_v15.23.json

    # 티어별 수집 데이터에서 자동으로 학습 (추천)
    python scripts/train_baselines.py --auto

    # CSV 파일에서 학습
    python scripts/train_baselines.py --input data/raw_matches.csv
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services.baseline_trainer import BaselineTrainer


def load_data_from_file(filepath: str) -> pd.DataFrame:
    """
    Load match data from various file formats

    Args:
        filepath: Path to data file (.json, .csv, .parquet)

    Returns:
        DataFrame with match data
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    print(f"Loading data from {filepath}")

    if filepath.suffix == ".json":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            # If nested by tier
            all_data = []
            for tier, matches in data.items():
                if isinstance(matches, list):
                    for match in matches:
                        match["tier"] = tier
                        all_data.append(match)
            df = pd.DataFrame(all_data)
        else:
            raise ValueError("Unsupported JSON structure")

    elif filepath.suffix == ".csv":
        df = pd.read_csv(filepath)

    elif filepath.suffix == ".parquet":
        df = pd.read_parquet(filepath)

    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")

    print(f"Loaded {len(df)} records")
    return df


def validate_data(df: pd.DataFrame) -> bool:
    """
    Validate that DataFrame has required columns

    Args:
        df: DataFrame to validate

    Returns:
        True if valid, raises ValueError otherwise
    """
    required_columns = ["tier", "kda", "game_duration"]

    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    print(f"Data validation passed")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Tiers: {df['tier'].unique()}")

    return True


def load_tier_collections_auto() -> tuple[pd.DataFrame, str]:
    """
    data/tier_collections/ 디렉토리에서 모든 티어 데이터 자동 로드

    Returns:
        (모든 티어 데이터가 합쳐진 DataFrame, 게임 버전)
    """
    collections_dir = Path("data/tier_collections")

    if not collections_dir.exists():
        raise FileNotFoundError(f"Collections directory not found: {collections_dir}")

    # 모든 티어 파일 찾기
    tier_files = list(collections_dir.glob("*_tier_*.json"))

    if not tier_files:
        raise FileNotFoundError(f"No tier data files found in {collections_dir}")

    print(f"Found {len(tier_files)} tier data files")

    all_data = []
    game_version = None

    for tier_file in tier_files:
        print(f"Loading: {tier_file.name}")

        with open(tier_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 첫 번째 파일에서 게임 버전 추출
        if game_version is None:
            metadata = data.get("metadata", {})
            game_version = metadata.get("game_version", "unknown")

        # data 필드에서 플레이어 데이터 추출
        players_data = data.get("data", [])
        all_data.extend(players_data)

        print(f"  → Loaded {len(players_data)} player records")

    print(f"\nTotal records loaded: {len(all_data)}")
    print(f"Game version: {game_version}")
    return pd.DataFrame(all_data), game_version


def main():
    parser = argparse.ArgumentParser(description="Train tier baselines from match data")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        help="Input data file (.json, .csv, .parquet)",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Auto-load all tier data from data/tier_collections/",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="data/tier_baselines.json",
        help="Output baseline file (default: data/tier_baselines.json)",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Display baselines after training",
    )

    args = parser.parse_args()

    # --input과 --auto 둘 다 없으면 에러
    if not args.input and not args.auto:
        parser.error("Either --input or --auto must be specified")

    try:
        print("=" * 80)
        print("TIER BASELINE TRAINING")
        print("=" * 80)

        # Load data
        game_version = None
        if args.auto:
            print("\n[AUTO] Auto-loading tier data from data/tier_collections/\n")
            df, game_version = load_tier_collections_auto()
        else:
            print(f"\n[LOAD] Loading data from {args.input}\n")
            df = load_data_from_file(args.input)

        # Validate data
        validate_data(df)

        # Initialize trainer
        trainer = BaselineTrainer()

        # Train baselines
        print("\n[TRAIN] Training baselines...")
        baselines = trainer.train_from_data(df)

        # Display results
        if args.display:
            trainer.display_baselines()

        # Determine output filename
        output_file = args.output
        if game_version and game_version != "unknown":
            base_name = Path(output_file).stem
            ext = Path(output_file).suffix
            parent = Path(output_file).parent
            output_file = str(parent / f"{base_name}_{game_version}{ext}")

        # Save baselines
        print(f"\nSaving baselines to {output_file}")
        trainer.save_baselines(output_file)

        # Generate settings.py code
        print("\nGenerated code for settings.py:")
        print("=" * 80)
        print(trainer.generate_settings_code())
        print("=" * 80)

        print("\nTraining completed successfully!")
        print(f"   Trained tiers: {list(baselines.keys())}")
        print(f"   Total samples: {trainer.metadata.get('total_samples', 0)}")
        print(f"\nTo use these baselines, restart your server:")
        print(f"   python main.py")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
