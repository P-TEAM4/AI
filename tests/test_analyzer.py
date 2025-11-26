"""Tests for match analyzer service"""

import pytest
from unittest.mock import Mock, patch
from src.services.analyzer import MatchAnalyzer
from src.api.models import PlayerStats


@pytest.fixture
def analyzer():
    """Create analyzer instance"""
    return MatchAnalyzer()


def test_create_player_stats_from_match(analyzer):
    """Test creating PlayerStats from match data"""
    match_stats = {
        "kills": 10,
        "deaths": 3,
        "assists": 8,
        "total_cs": 180,
        "gold": 15000,
        "vision_score": 35,
        "damage_dealt": 25000,
        "champion_name": "Ahri",
        "position": "MIDDLE",
        "game_duration": 1800,  # 30 minutes
        "win": True
    }

    player_stats = analyzer.create_player_stats_from_match(match_stats)

    assert isinstance(player_stats, PlayerStats)
    assert player_stats.kills == 10
    assert player_stats.deaths == 3
    assert player_stats.assists == 8
    assert player_stats.kda == 6.0  # (10 + 8) / 3
    assert player_stats.cs == 180
    assert player_stats.cs_per_min == 6.0  # 180 / 30
    assert player_stats.champion_name == "Ahri"


def test_calculate_average_stats(analyzer):
    """Test calculating average stats from multiple games"""
    performances = [
        {
            "kills": 10,
            "deaths": 3,
            "assists": 8,
            "total_cs": 180,
            "gold": 15000,
            "vision_score": 35,
            "damage_dealt": 25000,
            "game_duration": 1800,
            "win": True
        },
        {
            "kills": 8,
            "deaths": 5,
            "assists": 12,
            "total_cs": 160,
            "gold": 13000,
            "vision_score": 30,
            "damage_dealt": 22000,
            "game_duration": 1800,
            "win": False
        }
    ]

    avg_stats = analyzer._calculate_average_stats(performances)

    assert avg_stats["avg_kills"] == 9.0  # (10 + 8) / 2
    assert avg_stats["avg_deaths"] == 4.0  # (3 + 5) / 2
    assert avg_stats["avg_assists"] == 10.0  # (8 + 12) / 2
    assert avg_stats["avg_cs"] == 170.0  # (180 + 160) / 2


def test_extract_champion_pool(analyzer):
    """Test extracting champion pool from performances"""
    performances = [
        {"champion_name": "Ahri", "win": True},
        {"champion_name": "Ahri", "win": True},
        {"champion_name": "Ahri", "win": False},
        {"champion_name": "Lux", "win": True},
        {"champion_name": "Syndra", "win": False},
    ]

    champion_pool = analyzer._extract_champion_pool(performances)

    # Should be sorted by games played
    assert len(champion_pool) <= 5
    assert champion_pool[0]["champion"] == "Ahri"
    assert champion_pool[0]["games"] == 3
    assert champion_pool[0]["wins"] == 2
    assert champion_pool[0]["win_rate"] == pytest.approx(66.67, rel=0.1)


def test_analyze_performance_trend(analyzer):
    """Test performance trend analysis"""
    # 10 games: first 5 wins, last 5 losses
    performances = [
        {"win": False},  # Recent
        {"win": False},
        {"win": False},
        {"win": False},
        {"win": False},
        {"win": True},   # Older
        {"win": True},
        {"win": True},
        {"win": True},
        {"win": True},
    ]

    trend = analyzer._analyze_performance_trend(performances)

    assert trend["trend"] == "declining"
    assert trend["recent_win_rate"] == 0.0
    assert trend["older_win_rate"] == 100.0
    assert trend["total_games"] == 10


def test_extract_key_moments(analyzer):
    """Test extracting key moments from match"""
    match_data = {
        "info": {
            "participants": [
                {
                    "puuid": "test_puuid",
                    "pentaKills": 1,
                    "quadraKills": 0,
                    "tripleKills": 2
                }
            ]
        }
    }

    key_moments = analyzer._extract_key_moments(match_data, "test_puuid")

    assert len(key_moments) == 2  # 1 penta + 2 triple
    assert any(m["type"] == "pentakill" for m in key_moments)
    assert any(m["type"] == "triple_kill" for m in key_moments)


def test_empty_performances(analyzer):
    """Test handling empty performances"""
    avg_stats = analyzer._calculate_average_stats([])

    assert avg_stats == {}


def test_single_game_trend(analyzer):
    """Test trend analysis with single game"""
    performances = [{"win": True}]

    trend = analyzer._analyze_performance_trend(performances)

    assert trend["trend"] == "insufficient_data"
