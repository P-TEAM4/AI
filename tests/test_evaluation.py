"""Evaluation tests for Rule-based model accuracy"""

import pytest
from src.models.rule_based import RuleBasedGapAnalyzer
from src.api.models import PlayerStats


@pytest.fixture
def analyzer():
    """Create analyzer instance"""
    return RuleBasedGapAnalyzer()


class TestBaselineAccuracy:
    """Test baseline values are reasonable"""

    def test_tier_progression(self, analyzer):
        """Test that higher tiers have better stats"""
        tiers = ["BRONZE", "SILVER", "GOLD", "PLATINUM", "DIAMOND"]

        for i in range(len(tiers) - 1):
            lower_tier = analyzer.get_tier_baseline(tiers[i])
            higher_tier = analyzer.get_tier_baseline(tiers[i + 1])

            # Higher tier should have better stats
            assert higher_tier["avg_kda"] > lower_tier["avg_kda"]
            assert higher_tier["avg_cs_per_min"] > lower_tier["avg_cs_per_min"]
            assert higher_tier["avg_gold"] > lower_tier["avg_gold"]

    def test_baseline_reasonableness(self, analyzer):
        """Test baseline values are within reasonable ranges"""
        for tier in analyzer.tier_baseline.get_all_tiers():
            baseline = analyzer.get_tier_baseline(tier)

            # KDA should be positive and reasonable
            assert 0 < baseline["avg_kda"] < 10

            # CS/min should be reasonable (pros get ~10)
            assert 0 < baseline["avg_cs_per_min"] < 12

            # Gold should be reasonable
            assert 5000 < baseline["avg_gold"] < 25000

            # Vision score should be reasonable
            assert 0 < baseline["avg_vision_score"] < 100

            # Damage share should be between 0 and 1
            assert 0 < baseline["avg_damage_share"] < 1


class TestGapAnalysisAccuracy:
    """Test gap analysis produces meaningful results"""

    def test_perfect_tier_match(self, analyzer):
        """Test player matching tier baseline exactly"""
        gold_baseline = analyzer.get_tier_baseline("GOLD")

        # Create player with exact gold baseline stats
        player_stats = PlayerStats(
            kills=int(gold_baseline["avg_kda"] * 2),
            deaths=2,
            assists=int(gold_baseline["avg_kda"] * 2),
            kda=gold_baseline["avg_kda"],
            cs=int(gold_baseline["avg_cs_per_min"] * 30),
            cs_per_min=gold_baseline["avg_cs_per_min"],
            gold=gold_baseline["avg_gold"],
            vision_score=gold_baseline["avg_vision_score"],
            damage_dealt=20000,
            damage_share=gold_baseline["avg_damage_share"],
            champion_name="Test"
        )

        result = analyzer.analyze_gap(player_stats, "GOLD")

        # Overall score should be close to 50 (perfect match)
        assert 45 <= result.overall_score <= 55

        # All normalized gaps should be close to 0
        for gap in result.normalized_gaps.values():
            assert -10 <= gap <= 10

    def test_above_tier_performance(self, analyzer):
        """Test player performing above their tier"""
        player_stats = PlayerStats(
            kills=15,
            deaths=2,
            assists=12,
            kda=13.5,  # Very high KDA
            cs=250,
            cs_per_min=10.0,  # Pro-level CS
            gold=20000,  # High gold
            vision_score=50,
            damage_dealt=35000,
            damage_share=0.30,
            champion_name="Faker"
        )

        result = analyzer.analyze_gap(player_stats, "GOLD")

        # Should score above 50 (better than tier)
        assert result.overall_score > 50

        # Should have more strengths than weaknesses
        assert len(result.strengths) >= len(result.weaknesses)

        # Recommendations should acknowledge good performance
        assert len(result.recommendations) > 0

    def test_below_tier_performance(self, analyzer):
        """Test player performing below their tier"""
        player_stats = PlayerStats(
            kills=3,
            deaths=10,
            assists=4,
            kda=0.7,  # Low KDA
            cs=80,
            cs_per_min=3.0,  # Low CS
            gold=8000,  # Low gold
            vision_score=10,
            damage_dealt=8000,
            damage_share=0.12,
            champion_name="Beginner"
        )

        result = analyzer.analyze_gap(player_stats, "DIAMOND")

        # Should score below 50 (worse than tier)
        assert result.overall_score < 50

        # Should have more weaknesses than strengths
        assert len(result.weaknesses) >= len(result.strengths)

        # Should have improvement recommendations
        assert len(result.recommendations) > 0


class TestTierSuggestion:
    """Test tier suggestion accuracy"""

    def test_suggest_appropriate_tier(self, analyzer):
        """Test suggesting correct tier based on performance"""
        # Diamond-level stats
        diamond_stats = PlayerStats(
            kills=10,
            deaths=3,
            assists=8,
            kda=6.0,
            cs=225,
            cs_per_min=7.5,
            gold=16000,
            vision_score=35,
            damage_dealt=25000,
            damage_share=0.24,
            champion_name="Test"
        )

        suggested_tier = analyzer.suggest_target_tier(diamond_stats)

        # Should suggest high tier (Diamond or close)
        high_tiers = ["PLATINUM", "EMERALD", "DIAMOND", "MASTER", "GRANDMASTER"]
        assert suggested_tier in high_tiers

    def test_suggest_lower_tier(self, analyzer):
        """Test suggesting lower tier for weaker performance"""
        bronze_stats = PlayerStats(
            kills=4,
            deaths=8,
            assists=6,
            kda=1.25,
            cs=110,
            cs_per_min=4.5,
            gold=9000,
            vision_score=15,
            damage_dealt=12000,
            damage_share=0.18,
            champion_name="Test"
        )

        suggested_tier = analyzer.suggest_target_tier(bronze_stats)

        # Should suggest lower tier
        low_tiers = ["IRON", "BRONZE", "SILVER", "GOLD"]
        assert suggested_tier in low_tiers


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_zero_deaths_kda(self, analyzer):
        """Test KDA calculation with zero deaths"""
        kda = analyzer.calculate_kda(10, 0, 5)
        assert kda == 15.0  # Perfect KDA

    def test_zero_cs_per_min(self, analyzer):
        """Test CS/min with zero duration"""
        cs_per_min = analyzer.calculate_cs_per_min(100, 0)
        assert cs_per_min == 0.0

    def test_invalid_tier(self, analyzer):
        """Test handling of invalid tier"""
        baseline = analyzer.get_tier_baseline("INVALID_TIER")
        # Should return default (GOLD)
        assert baseline is not None
        assert "avg_kda" in baseline


def test_overall_score_range(analyzer):
    """Test overall score is always between 0 and 100"""
    # Test with extreme values
    extreme_high = {
        "kda": 1000.0,
        "cs_per_min": 1000.0,
        "gold": 1000.0,
        "vision_score": 1000.0,
        "damage_share": 1000.0
    }

    extreme_low = {
        "kda": -1000.0,
        "cs_per_min": -1000.0,
        "gold": -1000.0,
        "vision_score": -1000.0,
        "damage_share": -1000.0
    }

    score_high = analyzer.calculate_overall_score(extreme_high)
    score_low = analyzer.calculate_overall_score(extreme_low)

    assert 0 <= score_high <= 100
    assert 0 <= score_low <= 100
