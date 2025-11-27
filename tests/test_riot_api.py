"""Tests for Riot API client"""

import pytest
from unittest.mock import Mock, patch
from src.services.riot_api import RiotAPIClient


@pytest.fixture
def riot_client():
    """Create Riot API client instance"""
    return RiotAPIClient(api_key="test_api_key")


def test_client_initialization(riot_client):
    """Test Riot API client initialization"""
    assert riot_client.api_key == "test_api_key"
    assert "X-Riot-Token" in riot_client.headers
    assert riot_client.headers["X-Riot-Token"] == "test_api_key"


@patch('src.services.riot_api.requests.get')
def test_get_account_by_riot_id(mock_get, riot_client):
    """Test get account by Riot ID"""
    # Mock response
    mock_response = Mock()
    mock_response.json.return_value = {
        "puuid": "test_puuid_123",
        "gameName": "TestPlayer",
        "tagLine": "KR1"
    }
    mock_response.raise_for_status = Mock()
    mock_get.return_value = mock_response

    # Call function
    result = riot_client.get_account_by_riot_id("TestPlayer", "KR1")

    # Assertions
    assert result["puuid"] == "test_puuid_123"
    assert result["gameName"] == "TestPlayer"
    mock_get.assert_called_once()


@patch('src.services.riot_api.requests.get')
def test_get_match_details(mock_get, riot_client):
    """Test get match details"""
    # Mock response
    mock_response = Mock()
    mock_response.json.return_value = {
        "info": {
            "gameDuration": 1800,
            "participants": []
        }
    }
    mock_response.raise_for_status = Mock()
    mock_get.return_value = mock_response

    # Call function
    result = riot_client.get_match_details("KR_123456")

    # Assertions
    assert "info" in result
    assert result["info"]["gameDuration"] == 1800
    mock_get.assert_called_once()


def test_extract_player_stats_from_match(riot_client):
    """Test extracting player stats from match data"""
    # Mock match data
    match_data = {
        "info": {
            "gameDuration": 1800,
            "participants": [
                {
                    "puuid": "test_puuid",
                    "championName": "Ahri",
                    "kills": 10,
                    "deaths": 3,
                    "assists": 8,
                    "totalMinionsKilled": 150,
                    "neutralMinionsKilled": 30,
                    "goldEarned": 15000,
                    "visionScore": 35,
                    "totalDamageDealtToChampions": 25000,
                    "teamPosition": "MIDDLE",
                    "win": True,
                    "teamId": 100
                },
                {
                    "puuid": "other_puuid",
                    "totalDamageDealtToChampions": 20000,
                    "teamId": 100
                }
            ]
        }
    }

    # Call function
    result = riot_client.extract_player_stats_from_match(match_data, "test_puuid")

    # Assertions
    assert result is not None
    assert result["champion_name"] == "Ahri"
    assert result["kills"] == 10
    assert result["deaths"] == 3
    assert result["assists"] == 8
    assert result["total_cs"] == 180  # 150 + 30
    assert result["gold"] == 15000
    assert result["vision_score"] == 35
    assert result["win"] is True


def test_extract_player_stats_player_not_found(riot_client):
    """Test extracting stats when player not in match"""
    match_data = {
        "info": {
            "participants": [
                {"puuid": "other_puuid"}
            ]
        }
    }

    result = riot_client.extract_player_stats_from_match(match_data, "nonexistent_puuid")

    assert result is None


@patch('src.services.riot_api.requests.get')
def test_api_error_handling(mock_get, riot_client):
    """Test API error handling"""
    # Mock error response
    mock_get.side_effect = Exception("API Error")

    # Should raise exception
    with pytest.raises(Exception):
        riot_client.get_account_by_riot_id("TestPlayer", "KR1")
