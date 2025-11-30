"""Rule-based model for gap calculation and performance analysis"""

from typing import Dict, List, Tuple
import numpy as np
from src.config.baseline_loader import get_baseline_loader
from src.api.models import PlayerStats, GapAnalysisResult


class RuleBasedGapAnalyzer:
    """
    Rule-based analyzer for calculating performance gaps between player stats
    and tier baseline averages.

    Uses dynamic baseline loader that can load from:
    - Trained JSON file (data/tier_baselines.json)
    - Default hardcoded values (fallback)
    """

    def __init__(self):
        self.baseline_loader = get_baseline_loader()
        print(f"RuleBasedGapAnalyzer initialized with {'learned' if self.baseline_loader.is_using_learned_baselines() else 'default'} baselines")

    def calculate_kda(self, kills: int, deaths: int, assists: int) -> float:
        """
        Calculate KDA ratio

        Args:
            kills: Number of kills
            deaths: Number of deaths
            assists: Number of assists

        Returns:
            KDA ratio (perfect score if deaths = 0)
        """
        if deaths == 0:
            return float(kills + assists)
        return round((kills + assists) / deaths, 2)

    def calculate_cs_per_min(self, total_cs: int, game_duration_seconds: int) -> float:
        """
        Calculate CS per minute

        Args:
            total_cs: Total creep score
            game_duration_seconds: Game duration in seconds

        Returns:
            CS per minute
        """
        if game_duration_seconds == 0:
            return 0.0
        minutes = game_duration_seconds / 60
        return round(total_cs / minutes, 2)

    def get_tier_baseline(self, tier: str) -> Dict[str, float]:
        """
        Get baseline statistics for a specific tier

        Args:
            tier: Player tier (e.g., GOLD, PLATINUM, DIAMOND)

        Returns:
            Dictionary containing baseline statistics
        """
        return self.baseline_loader.get_baseline(tier)

    def calculate_gaps(
        self, player_stats: Dict[str, float], tier_avg: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate raw gaps between player stats and tier average

        Args:
            player_stats: Player's statistics
            tier_avg: Tier baseline averages

        Returns:
            Dictionary of gaps (player - tier_avg)
        """
        gaps = {}
        for key in tier_avg.keys():
            stat_key = key.replace("avg_", "")
            if stat_key in player_stats:
                gaps[stat_key] = round(player_stats[stat_key] - tier_avg[key], 2)
        return gaps

    def normalize_gaps(
        self, gaps: Dict[str, float], tier_avg: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Normalize gaps to percentage differences

        Args:
            gaps: Raw gap values
            tier_avg: Tier baseline averages

        Returns:
            Normalized gaps as percentages
        """
        normalized = {}
        for key, gap_value in gaps.items():
            avg_key = f"avg_{key}"
            if avg_key in tier_avg and tier_avg[avg_key] != 0:
                normalized[key] = round((gap_value / tier_avg[avg_key]) * 100, 2)
            else:
                normalized[key] = 0.0
        return normalized

    def calculate_overall_score(self, normalized_gaps: Dict[str, float]) -> float:
        """
        Calculate overall performance score based on normalized gaps

        Args:
            normalized_gaps: Normalized gap percentages

        Returns:
            Overall score (0-100, where 50 is tier average)
        """
        # Weights for different statistics
        weights = {
            "kda": 0.25,
            "cs_per_min": 0.20,
            "gold_per_min": 0.20,
            "vision_score_per_min": 0.15,
            "damage_share": 0.20,
        }

        weighted_sum = 0.0
        total_weight = 0.0

        for key, weight in weights.items():
            if key in normalized_gaps:
                # Normalize to 0-100 scale (50 is baseline)
                contribution = 50 + (normalized_gaps[key] * 0.5)
                # Clamp between 0 and 100
                contribution = max(0, min(100, contribution))
                weighted_sum += contribution * weight
                total_weight += weight

        if total_weight == 0:
            return 50.0

        return round(weighted_sum / total_weight, 2)

    def identify_strengths_weaknesses(
        self, normalized_gaps: Dict[str, float], threshold: float = 10.0
    ) -> Tuple[List[str], List[str]]:
        """
        Identify player strengths and weaknesses based on gaps

        Args:
            normalized_gaps: Normalized gap percentages
            threshold: Percentage threshold for classification

        Returns:
            Tuple of (strengths, weaknesses) lists
        """
        strengths = []
        weaknesses = []

        stat_names = {
            "kda": "KDA",
            "cs_per_min": "Farming (CS/min)",
            "gold_per_min": "Gold Efficiency",
            "vision_score_per_min": "Vision Control",
            "damage_share": "Damage Output",
        }

        for key, gap in normalized_gaps.items():
            stat_name = stat_names.get(key, key)
            if gap >= threshold:
                strengths.append(f"{stat_name} (+{gap}%)")
            elif gap <= -threshold:
                weaknesses.append(f"{stat_name} ({gap}%)")

        return strengths, weaknesses

    def generate_recommendations(
        self, weaknesses: List[str], normalized_gaps: Dict[str, float]
    ) -> List[str]:
        """
        Generate improvement recommendations based on weaknesses

        Args:
            weaknesses: List of identified weaknesses
            normalized_gaps: Normalized gap percentages

        Returns:
            List of recommendations
        """
        recommendations = []

        if any("KDA" in w for w in weaknesses):
            recommendations.append(
                "Focus on reducing deaths and improving positioning in team fights"
            )

        if any("Farming" in w for w in weaknesses):
            recommendations.append(
                "Practice last-hitting and wave management to improve CS/min"
            )

        if any("Gold" in w for w in weaknesses):
            recommendations.append(
                "Optimize gold efficiency through better item builds and objective control"
            )

        if any("Vision" in w for w in weaknesses):
            recommendations.append(
                "Place more wards and buy control wards to improve vision score"
            )

        if any("Damage" in w for w in weaknesses):
            recommendations.append(
                "Work on champion mechanics and target selection to increase damage output"
            )

        if not recommendations:
            recommendations.append(
                "Maintain current performance level and focus on consistency"
            )

        return recommendations

    def analyze_gap(
        self, player_stats: PlayerStats, tier: str, division: str = "I"
    ) -> GapAnalysisResult:
        """
        Perform complete gap analysis for a player

        Args:
            player_stats: Player statistics
            tier: Player tier
            division: Player division (not used in current implementation)

        Returns:
            GapAnalysisResult containing complete analysis
        """
        # Get tier baseline
        tier_avg = self.get_tier_baseline(tier)

        # Calculate gold_per_min if not provided
        if player_stats.gold_per_min is None and player_stats.game_duration is not None and player_stats.game_duration > 0:
            gold_per_min = player_stats.gold / (player_stats.game_duration / 60)
        else:
            gold_per_min = player_stats.gold_per_min or 0.0

        # Calculate vision_score_per_min if not provided
        if player_stats.vision_score_per_min is None and player_stats.game_duration is not None and player_stats.game_duration > 0:
            vision_score_per_min = player_stats.vision_score / (player_stats.game_duration / 60)
        else:
            vision_score_per_min = player_stats.vision_score_per_min or 0.0

        # Prepare player stats dictionary
        player_dict = {
            "kda": player_stats.kda,
            "cs_per_min": player_stats.cs_per_min,
            "gold_per_min": gold_per_min,
            "vision_score_per_min": vision_score_per_min,
            "damage_share": player_stats.damage_share,
        }

        # Calculate gaps
        gaps = self.calculate_gaps(player_dict, tier_avg)
        normalized_gaps = self.normalize_gaps(gaps, tier_avg)

        # Calculate overall score
        overall_score = self.calculate_overall_score(normalized_gaps)

        # Identify strengths and weaknesses
        strengths, weaknesses = self.identify_strengths_weaknesses(normalized_gaps)

        # Generate recommendations
        recommendations = self.generate_recommendations(weaknesses, normalized_gaps)

        return GapAnalysisResult(
            tier=tier,
            player_stats=player_dict,
            tier_baseline=tier_avg,
            gaps=gaps,
            normalized_gaps=normalized_gaps,
            overall_score=overall_score,
            strengths=strengths,
            weaknesses=weaknesses,
            recommendations=recommendations,
        )

    def compare_with_multiple_tiers(
        self, player_stats: PlayerStats
    ) -> Dict[str, GapAnalysisResult]:
        """
        Compare player stats with multiple tier baselines

        Args:
            player_stats: Player statistics

        Returns:
            Dictionary mapping tier names to gap analysis results
        """
        results = {}
        for tier in self.baseline_loader.get_all_tiers():
            results[tier] = self.analyze_gap(player_stats, tier)
        return results

    def suggest_target_tier(self, player_stats: PlayerStats) -> str:
        """
        Suggest appropriate target tier based on player statistics

        Args:
            player_stats: Player statistics

        Returns:
            Suggested tier name
        """
        all_analyses = self.compare_with_multiple_tiers(player_stats)

        # Find tier with overall_score closest to 50 (perfect match)
        best_tier = None
        best_score_diff = float('inf')

        for tier, analysis in all_analyses.items():
            score_diff = abs(analysis.overall_score - 50)
            if score_diff < best_score_diff:
                best_score_diff = score_diff
                best_tier = tier

        return best_tier or "GOLD"
