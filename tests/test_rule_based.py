"""Tests for rule-based gap analyzer"""

import pytest
from src.models.rule_based import RuleBasedGapAnalyzer
from src.api.models import PlayerStats


@pytest.fixture
def analyzer():
    """Create analyzer instance"""
    return RuleBasedGapAnalyzer()


@pytest.fixture
def sample_player_stats():
    """Create sample player stats"""
    return PlayerStats(
        kills=10,
        deaths=3,
        assists=8,
        kda=6.0,
        cs=180,
        cs_per_min=7.5,
        gold=15000,
        vision_score=35,
        damage_dealt=25000,
        damage_share=0.24,
        champion_name="Ahri",
        position="MIDDLE",
    )


def test_calculate_kda(analyzer):
    """Test KDA calculation"""
    assert analyzer.calculate_kda(10, 5, 15) == 5.0
    assert analyzer.calculate_kda(10, 0, 5) == 15.0
    assert analyzer.calculate_kda(0, 5, 0) == 0.0


def test_calculate_cs_per_min(analyzer):
    """Test CS per minute calculation"""
    assert analyzer.calculate_cs_per_min(180, 1440) == 7.5  # 24 minutes
    assert analyzer.calculate_cs_per_min(0, 1000) == 0.0


def test_get_tier_baseline(analyzer):
    """Test getting tier baseline"""
    gold_baseline = analyzer.get_tier_baseline("GOLD")
    assert "avg_kda" in gold_baseline
    assert "avg_cs_per_min" in gold_baseline
    assert gold_baseline["avg_kda"] == 2.8


def test_analyze_gap(analyzer, sample_player_stats):
    """Test gap analysis"""
    result = analyzer.analyze_gap(sample_player_stats, "DIAMOND")

    assert result.tier == "DIAMOND"
    assert "kda" in result.gaps
    assert "kda" in result.normalized_gaps
    assert 0 <= result.overall_score <= 100
    assert isinstance(result.strengths, list)
    assert isinstance(result.weaknesses, list)
    assert isinstance(result.recommendations, list)


def test_identify_strengths_weaknesses(analyzer):
    """Test strength and weakness identification"""
    normalized_gaps = {"kda": 15.0, "cs_per_min": -12.0, "gold": 5.0}

    strengths, weaknesses = analyzer.identify_strengths_weaknesses(normalized_gaps)

    assert len(strengths) >= 1
    assert len(weaknesses) >= 1
    assert any("KDA" in s for s in strengths)
    assert any("Farming" in w for w in weaknesses)


def test_suggest_target_tier(analyzer, sample_player_stats):
    """Test tier suggestion"""
    suggested_tier = analyzer.suggest_target_tier(sample_player_stats)

    assert suggested_tier in analyzer.tier_baseline.get_all_tiers()


def test_calculate_overall_score(analyzer):
    """Test overall score calculation"""
    normalized_gaps = {
        "kda": 10.0,
        "cs_per_min": 5.0,
        "gold": -5.0,
        "vision_score": 0.0,
        "damage_share": 8.0,
    }

    score = analyzer.calculate_overall_score(normalized_gaps)

    assert 0 <= score <= 100
    assert isinstance(score, float)
