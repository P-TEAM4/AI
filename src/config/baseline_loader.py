"""
Dynamic Baseline Loader

Loads tier baselines from:
1. Trained JSON file (if available)
2. Hardcoded defaults (fallback)
3. API updates (future)
"""

from typing import Dict
from pathlib import Path
import json
from src.config.settings import TierBaseline as DefaultTierBaseline


class DynamicBaselineLoader:
    """동적으로 베이스라인을 로드하는 클래스"""

    def __init__(self, baseline_file: str = "data/tier_baselines.json"):
        """
        Initialize DynamicBaselineLoader

        Args:
            baseline_file: Path to learned baseline JSON file
        """
        self.baseline_file = Path(baseline_file)
        self.baselines: Dict[str, Dict[str, float]] = {}
        self.use_learned = False

        # Try to load learned baselines
        self._load_baselines()

    def _load_baselines(self):
        """Load baselines from JSON file or use defaults"""
        if self.baseline_file.exists():
            try:
                with open(self.baseline_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                self.baselines = data.get("baselines", {})
                metadata = data.get("metadata", {})

                self.use_learned = True
                print(f"Using learned baselines from {self.baseline_file}")
                print(f"   Trained at: {metadata.get('trained_at', 'Unknown')}")
                print(f"   Total samples: {metadata.get('total_samples', 0)}")

            except Exception as e:
                print(f"Failed to load learned baselines: {e}")
                print("   Falling back to default baselines")
                self._use_default_baselines()
        else:
            print(f"No learned baseline file found at {self.baseline_file}")
            print("   Using default baselines from settings.py")
            self._use_default_baselines()

    def _use_default_baselines(self):
        """Use default baselines from settings.py"""
        self.baselines = DefaultTierBaseline.BASELINES.copy()
        self.use_learned = False

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
            print(f"Tier {tier} not found, using GOLD as default")
            return self.baselines.get("GOLD", DefaultTierBaseline.get_baseline("GOLD"))

        return self.baselines[tier]

    def get_all_tiers(self) -> list:
        """Get list of all available tiers"""
        return list(self.baselines.keys())

    def is_using_learned_baselines(self) -> bool:
        """Check if using learned baselines"""
        return self.use_learned

    def reload_baselines(self):
        """Reload baselines from file"""
        print("Reloading baselines...")
        self._load_baselines()


# Singleton instance
_baseline_loader = None


def get_baseline_loader() -> DynamicBaselineLoader:
    """Get singleton instance of baseline loader"""
    global _baseline_loader
    if _baseline_loader is None:
        _baseline_loader = DynamicBaselineLoader()
    return _baseline_loader
