"""
Train Tier Baselines from Downloaded Match Data

Step 1: Use this script to train baselines from local data
Step 2: After deployment, use collect_tier_baselines.py for live updates

Usage:
    python scripts/train_baselines.py --input data/raw_matches.json
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


def main():
    parser = argparse.ArgumentParser(description="Train tier baselines from match data")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Input data file (.json, .csv, .parquet)",
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

    try:
        print("=" * 80)
        print("TIER BASELINE TRAINING")
        print("=" * 80)

        # Load data
        df = load_data_from_file(args.input)

        # Validate data
        validate_data(df)

        # Initialize trainer
        trainer = BaselineTrainer()

        # Train baselines
        print("\n🔄 Training baselines...")
        baselines = trainer.train_from_data(df)

        # Display results
        if args.display:
            trainer.display_baselines()

        # Save baselines
        print(f"\nSaving baselines to {args.output}")
        trainer.save_baselines(args.output)

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
