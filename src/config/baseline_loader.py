"""
Dynamic Baseline Loader

Loads tier baselines from:
1. Trained JSON file (if available)
2. Hardcoded defaults (fallback)
3. API updates (future)
"""

from typing import Dict, Optional
from pathlib import Path
import json
import re
from src.config.settings import TierBaseline as DefaultTierBaseline


class DynamicBaselineLoader:
    """동적으로 베이스라인을 로드하는 클래스"""

    def __init__(self, baseline_file: Optional[str] = None, data_dir: str = "data"):
        """
        Initialize DynamicBaselineLoader

        Args:
            baseline_file: Path to specific baseline JSON file (optional)
            data_dir: Directory containing baseline files
        """
        self.data_dir = Path(data_dir)
        self.baseline_file = Path(baseline_file) if baseline_file else None
        self.baselines: Dict[str, Dict[str, float]] = {}
        self.use_learned = False

        # Try to load learned baselines
        self._load_baselines()

    def _find_latest_baseline_file(self) -> Optional[Path]:
        """
        Find the latest tier_baselines_*.json file in data directory

        Returns:
            Path to latest baseline file or None
        """
        if not self.data_dir.exists():
            return None

        # Find all tier_baselines_*.json files
        baseline_files = list(self.data_dir.glob("tier_baselines_*.json"))

        if not baseline_files:
            return None

        # Extract version numbers and sort
        def extract_version(filepath: Path) -> tuple:
            """Extract version number from filename (e.g., tier_baselines_15.23.json -> (15, 23))"""
            match = re.search(r'tier_baselines_(\d+)\.(\d+)\.json', filepath.name)
            if match:
                return (int(match.group(1)), int(match.group(2)))
            return (0, 0)

        # Sort by version (latest first)
        baseline_files.sort(key=extract_version, reverse=True)
        return baseline_files[0]

    def _load_baselines(self):
        """Load baselines from JSON file or use defaults"""
        # If specific file provided, use it
        if self.baseline_file:
            target_file = self.baseline_file
        else:
            # Otherwise, find latest version automatically
            target_file = self._find_latest_baseline_file()

        if target_file and target_file.exists():
            try:
                with open(target_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                self.baselines = data.get("baselines", {})
                metadata = data.get("metadata", {})

                self.use_learned = True
                print(f"Using learned baselines from {target_file}")
                print(f"   Trained at: {metadata.get('trained_at', 'Unknown')}")
                print(f"   Total samples: {metadata.get('total_samples', 0)}")

            except Exception as e:
                print(f"Failed to load learned baselines: {e}")
                print("   Falling back to default baselines")
                self._use_default_baselines()
        else:
            print(f"No learned baseline file found in {self.data_dir}")
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
            raise ValueError(f"Tier {tier} not found in baselines. Available tiers: {list(self.baselines.keys())}")

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
