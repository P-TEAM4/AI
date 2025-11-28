"""
Baseline Trainer for learning tier-based statistics

This module handles:
1. Loading raw match data (local or API)
2. Processing and calculating tier averages
3. Saving/loading learned baselines
4. Updating baselines dynamically
"""

import json
import os
from typing import Dict, List, Optional
from pathlib import Path
import pandas as pd
from datetime import datetime


class BaselineTrainer:
    """티어별 베이스라인을 학습하고 관리하는 클래스"""

    def __init__(self, data_dir: str = "data", baseline_file: str = "tier_baselines.json"):
        """
        Initialize BaselineTrainer

        Args:
            data_dir: Directory containing match data
            baseline_file: Filename for saving/loading baselines
        """
        self.data_dir = Path(data_dir)
        self.baseline_file = self.data_dir / baseline_file
        self.baselines: Dict[str, Dict[str, float]] = {}
        self.metadata: Dict[str, any] = {}

    def load_match_data(self, filepath: str) -> pd.DataFrame:
        """
        Load match data from JSON file

        Args:
            filepath: Path to match data JSON file

        Returns:
            DataFrame with match statistics
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Convert to DataFrame
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            # If data is nested by tier
            all_data = []
            for tier, matches in data.items():
                for match in matches:
                    match["tier"] = tier
                    all_data.append(match)
            df = pd.DataFrame(all_data)
        else:
            raise ValueError("Unsupported data format")

        return df

    def calculate_per_minute_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate per-minute statistics

        Args:
            df: DataFrame with raw match data

        Returns:
            DataFrame with normalized per-minute stats
        """
        # Ensure game_duration exists and is valid
        if "game_duration" not in df.columns:
            raise ValueError("game_duration column is required")

        df = df.copy()
        df["game_minutes"] = df["game_duration"] / 60

        # Calculate per-minute stats
        if "gold" in df.columns:
            df["gold_per_min"] = df["gold"] / df["game_minutes"]

        if "vision_score" in df.columns:
            df["vision_score_per_min"] = df["vision_score"] / df["game_minutes"]

        # cs_per_min should already be calculated, but verify
        if "total_cs" in df.columns and "cs_per_min" not in df.columns:
            df["cs_per_min"] = df["total_cs"] / df["game_minutes"]

        return df

    def train_from_data(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Train baselines from match data

        Args:
            df: DataFrame with match statistics (must have 'tier' column)

        Returns:
            Dictionary of tier baselines
        """
        if "tier" not in df.columns:
            raise ValueError("DataFrame must have 'tier' column")

        # Calculate per-minute stats
        df = self.calculate_per_minute_stats(df)

        baselines = {}
        tiers = df["tier"].unique()

        for tier in tiers:
            tier_df = df[df["tier"] == tier]

            if len(tier_df) == 0:
                continue

            baseline = {
                "avg_kda": round(tier_df["kda"].mean(), 2) if "kda" in tier_df else 0.0,
                "avg_cs_per_min": (
                    round(tier_df["cs_per_min"].mean(), 2) if "cs_per_min" in tier_df else 0.0
                ),
                "avg_gold_per_min": (
                    round(tier_df["gold_per_min"].mean(), 2) if "gold_per_min" in tier_df else 0.0
                ),
                "avg_vision_score_per_min": (
                    round(tier_df["vision_score_per_min"].mean(), 2)
                    if "vision_score_per_min" in tier_df
                    else 0.0
                ),
                "avg_damage_share": (
                    round(tier_df["damage_share"].mean(), 3) if "damage_share" in tier_df else 0.0
                ),
                "sample_size": len(tier_df),
                "win_rate": round(tier_df["win"].mean() * 100, 2) if "win" in tier_df else 50.0,
            }

            baselines[tier.upper()] = baseline

        self.baselines = baselines
        self.metadata = {
            "trained_at": datetime.now().isoformat(),
            "total_samples": len(df),
            "tiers_trained": list(baselines.keys()),
        }

        return baselines

    def save_baselines(self, filepath: Optional[str] = None):
        """
        Save learned baselines to JSON file

        Args:
            filepath: Custom filepath (defaults to self.baseline_file)
        """
        if filepath is None:
            filepath = self.baseline_file
        else:
            filepath = Path(filepath)

        # Create directory if not exists
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "baselines": self.baselines,
            "metadata": self.metadata,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"Baselines saved to {filepath}")

    def load_baselines(self, filepath: Optional[str] = None) -> Dict[str, Dict[str, float]]:
        """
        Load baselines from JSON file

        Args:
            filepath: Custom filepath (defaults to self.baseline_file)

        Returns:
            Dictionary of tier baselines
        """
        filepath = filepath or self.baseline_file

        if not filepath.exists():
            raise FileNotFoundError(f"Baseline file not found: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.baselines = data.get("baselines", {})
        self.metadata = data.get("metadata", {})

        print(f"Baselines loaded from {filepath}")
        print(f"   Trained at: {self.metadata.get('trained_at', 'Unknown')}")
        print(f"   Total samples: {self.metadata.get('total_samples', 0)}")

        return self.baselines

    def get_baseline(self, tier: str) -> Dict[str, float]:
        """
        Get baseline for specific tier

        Args:
            tier: Tier name (e.g., "GOLD", "PLATINUM")

        Returns:
            Baseline statistics for the tier
        """
        tier = tier.upper()
        if tier not in self.baselines:
            # Return GOLD as default
            return self.baselines.get("GOLD", {})

        return self.baselines[tier]

    def display_baselines(self):
        """Display baselines in table format"""
        if not self.baselines:
            print("No baselines loaded")
            return

        df = pd.DataFrame(self.baselines).T
        print("\n" + "=" * 100)
        print("LEARNED TIER BASELINES")
        print("=" * 100)
        print(df.to_string())
        print("=" * 100 + "\n")

    def generate_settings_code(self) -> str:
        """
        Generate Python code for settings.py

        Returns:
            Python code string
        """
        if not self.baselines:
            return "# No baselines to generate"

        lines = ["BASELINES: Dict[str, Dict[str, float]] = {"]

        for tier, stats in self.baselines.items():
            lines.append(f'    "{tier}": {{')
            for key, value in stats.items():
                if key not in ["sample_size", "win_rate"]:
                    lines.append(f'        "{key}": {value},')
            lines.append("    },")

        lines.append("}")

        return "\n".join(lines)


def main():
    """Example usage"""
    trainer = BaselineTrainer()

    # Example: Load data from file
    # df = trainer.load_match_data("data/raw_match_data.json")

    # Example: Train from DataFrame
    # baselines = trainer.train_from_data(df)

    # Example: Save baselines
    # trainer.save_baselines()

    # Example: Load existing baselines
    try:
        baselines = trainer.load_baselines()
        trainer.display_baselines()
    except FileNotFoundError:
        print("No baseline file found. Please train first.")

    # Generate code for settings.py
    print("\nGenerated code for settings.py:")
    print("=" * 80)
    print(trainer.generate_settings_code())
    print("=" * 80)


if __name__ == "__main__":
    main()
