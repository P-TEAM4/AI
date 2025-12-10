"""Tests for API endpoints"""

import pytest
from fastapi.testclient import TestClient
from src.api.routes import app


@pytest.fixture
def client():
    """Create test client"""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def mock_match_data():
    """Sample match data for mocking"""
    return {
        "metadata": {"matchId": "KR_1234567890"},
        "info": {
            "gameDuration": 1800,
            "participants": [
                {
                    "puuid": "test-puuid",
                    "summonerName": "TestPlayer",
                    "championName": "Ahri",
                    "teamPosition": "MIDDLE",
                    "teamId": 100,
                    "win": True,
                    "kills": 10,
                    "deaths": 2,
                    "assists": 8,
                    "totalDamageDealtToChampions": 24000,
                    "totalMinionsKilled": 180,
                    "neutralMinionsKilled": 10,
                    "visionScore": 45,
                    "goldEarned": 13500,
                    "participantId": 1,
                    "challenges": {
                        "goldPerMinute": 450.0,
                        "damagePerMinute": 800.0,
                        "killParticipation": 0.75,
                        "soloKills": 2,
                        "controlWardsPlaced": 3,
                        "wardTakedowns": 5,
                        "wardTakedownsBefore20M": 3,
                        "skillshotsDodged": 15,
                        "skillshotsHit": 25,
                    },
                    "longestTimeSpentLiving": 600,
                }
            ]
        }
    }


@pytest.fixture
def mock_player_stats():
    """Sample player stats for mocking"""
    return {
        "summonerName": "TestPlayer",
        "championName": "Ahri",
        "teamPosition": "MIDDLE",
        "win": True,
        "kills": 10,
        "deaths": 2,
        "assists": 8,
        "kda": 9.0,
        "totalDamageDealtToChampions": 24000,
        "totalMinionsKilled": 180,
        "neutralMinionsKilled": 10,
        "cs_per_min": 6.33,
        "visionScore": 45,
        "goldEarned": 13500,
    }


def test_health_check(client):
    """Test health check endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "timestamp" in data


def test_health(client):
    """Test health endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_get_available_tiers(client):
    """Test get available tiers endpoint"""
    response = client.get("/api/v1/tiers")
    assert response.status_code == 200
    data = response.json()
    assert "tiers" in data
    assert "baselines" in data
    assert "GOLD" in data["tiers"]
    assert "DIAMOND" in data["tiers"]


def test_analyze_gap(client, mock_match_data, mock_player_stats):
    """Test gap analysis endpoint with mocked Riot API"""
    from unittest.mock import patch, MagicMock
    from src.api.models import PlayerStats
    from src.api import routes

    # Create PlayerStats object
    player_stats = PlayerStats(
        kills=10,
        deaths=2,
        assists=8,
        kda=9.0,
        cs=190,
        cs_per_min=6.33,
        gold=13500,
        vision_score=45,
        damage_dealt=24000,
        damage_share=0.25,
        champion_name="Ahri",
        position="MIDDLE",
        game_duration=1800,
    )

    # Mock the riot_client of the match_analyzer instance
    mock_riot_client = MagicMock()
    mock_riot_client.get_match_details.return_value = mock_match_data
    mock_riot_client.extract_player_stats_from_match.return_value = mock_player_stats

    routes.match_analyzer.riot_client = mock_riot_client

    with patch.object(routes.match_analyzer, 'create_player_stats_from_match', return_value=player_stats):
        request_data = {
            "match_id": "KR_1234567890",
            "puuid": "test-puuid",
            "tier": "DIAMOND",
        }

        response = client.post("/api/v1/analyze/gap", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert "overall_score" in data
        assert "strengths" in data
        assert "weaknesses" in data
        assert "tier" in data
        assert data["tier"] == "DIAMOND"


def test_suggest_tier(client, mock_match_data, mock_player_stats):
    """Test tier suggestion endpoint with mocked Riot API"""
    from unittest.mock import patch, MagicMock
    from src.api.models import PlayerStats
    from src.api import routes

    # Create PlayerStats object with good performance
    player_stats = PlayerStats(
        kills=10,
        deaths=2,
        assists=8,
        kda=9.0,
        cs=190,
        cs_per_min=6.33,
        gold=13500,
        vision_score=45,
        damage_dealt=24000,
        damage_share=0.25,
        champion_name="Ahri",
        position="MIDDLE",
        game_duration=1800,
    )

    # Mock the riot_client of the match_analyzer instance
    mock_riot_client = MagicMock()
    mock_riot_client.get_match_details.return_value = mock_match_data
    mock_riot_client.extract_player_stats_from_match.return_value = mock_player_stats

    routes.match_analyzer.riot_client = mock_riot_client

    with patch.object(routes.match_analyzer, 'create_player_stats_from_match', return_value=player_stats):
        request_data = {
            "match_id": "KR_1234567890",
            "puuid": "test-puuid",
            "tier": "GOLD",
        }

        response = client.post("/api/v1/suggest-tier", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert "suggested_tier" in data
        assert "current_tier" in data
        assert "comparisons" in data
        assert data["current_tier"] == "GOLD"
        assert isinstance(data["comparisons"], dict)
