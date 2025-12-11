"""Tests for impact score functionality"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from src.api.impact_score import (
    extract_features_from_match,
    riot_get
)


@pytest.fixture
def sample_match_data():
    """Sample match data for testing"""
    return {
        "metadata": {
            "matchId": "KR_1234567890",
            "participants": ["puuid1", "puuid2", "puuid3", "puuid4", "puuid5",
                           "puuid6", "puuid7", "puuid8", "puuid9", "puuid10"]
        },
        "info": {
            "gameDuration": 1800,  # 30 minutes
            "participants": [
                {
                    "puuid": "puuid1",
                    "participantId": 1,
                    "teamId": 100,
                    "win": True,
                    "championName": "Ahri",
                    "teamPosition": "MIDDLE",
                    "kills": 10,
                    "deaths": 3,
                    "assists": 8,
                    "visionScore": 35,
                    "longestTimeSpentLiving": 600,
                    "challenges": {
                        "goldPerMinute": 450.0,
                        "damagePerMinute": 800.0,
                        "killParticipation": 0.75,
                        "soloKills": 2,
                        "controlWardsPlaced": 3,
                        "wardTakedowns": 5,
                        "wardTakedownsBefore20M": 3,
                        "skillshotsDodged": 15,
                        "skillshotsHit": 25
                    }
                },
                {
                    "puuid": "puuid2",
                    "participantId": 2,
                    "teamId": 100,
                    "win": True,
                    "championName": "Lee Sin",
                    "teamPosition": "JUNGLE",
                    "kills": 5,
                    "deaths": 4,
                    "assists": 12,
                    "visionScore": 40,
                    "longestTimeSpentLiving": 500,
                    "challenges": {
                        "goldPerMinute": 400.0,
                        "damagePerMinute": 700.0,
                        "killParticipation": 0.70,
                        "soloKills": 1,
                        "controlWardsPlaced": 5,
                        "wardTakedowns": 8,
                        "wardTakedownsBefore20M": 5,
                        "skillshotsDodged": 20,
                        "skillshotsHit": 30
                    }
                }
            ]
        }
    }


@pytest.fixture
def sample_timeline_data():
    """Sample timeline data for testing"""
    return {
        "info": {
            "frames": [
                {
                    "timestamp": i * 60000,  # Every minute
                    "participantFrames": {
                        "1": {
                            "minionsKilled": i * 8,
                            "jungleMinionsKilled": i * 1,
                            "totalGold": 500 + i * 300,
                            "xp": 200 + i * 150
                        },
                        "2": {
                            "minionsKilled": i * 2,
                            "jungleMinionsKilled": i * 6,
                            "totalGold": 500 + i * 280,
                            "xp": 200 + i * 140
                        }
                    }
                }
                for i in range(31)  # 0-30 minutes
            ]
        }
    }


class TestFeatureExtraction:
    """Test feature extraction from match data"""

    @patch('src.api.impact_score.add_derived_features')
    def test_extract_features_basic(self, mock_add_derived, sample_match_data, sample_timeline_data):
        """Test basic feature extraction"""
        # Mock add_derived_features to return the input DataFrame
        mock_add_derived.side_effect = lambda df: df

        df = extract_features_from_match(sample_match_data, sample_timeline_data, "puuid1")

        assert len(df) == 1
        assert df.iloc[0]["puuid"] == "puuid1"
        assert df.iloc[0]["kills"] == 10
        assert df.iloc[0]["deaths"] == 3
        assert df.iloc[0]["assists"] == 8

    @patch('src.api.impact_score.add_derived_features')
    def test_extract_features_kda_calculation(self, mock_add_derived, sample_match_data, sample_timeline_data):
        """Test KDA is calculated correctly"""
        mock_add_derived.side_effect = lambda df: df

        df = extract_features_from_match(sample_match_data, sample_timeline_data, "puuid1")

        kda = df.iloc[0]["kda"]
        expected_kda = (10 + 8) / 3  # (kills + assists) / deaths
        assert abs(kda - expected_kda) < 0.01

    @patch('src.api.impact_score.add_derived_features')
    def test_extract_features_vision_per_minute(self, mock_add_derived, sample_match_data, sample_timeline_data):
        """Test vision score per minute calculation"""
        mock_add_derived.side_effect = lambda df: df

        df = extract_features_from_match(sample_match_data, sample_timeline_data, "puuid1")

        vision_per_min = df.iloc[0]["visionScorePerMinute"]
        expected = 35 / 30  # vision_score / game_minutes
        assert abs(vision_per_min - expected) < 0.01

    @patch('src.api.impact_score.add_derived_features')
    def test_extract_features_timeline_data(self, mock_add_derived, sample_match_data, sample_timeline_data):
        """Test timeline-based features are extracted"""
        mock_add_derived.side_effect = lambda df: df

        df = extract_features_from_match(sample_match_data, sample_timeline_data, "puuid1")

        # Check 15-minute stats
        assert "cs_15" in df.columns
        assert "gold_15" in df.columns
        assert "xp_15" in df.columns

        # Check 10-minute stats
        assert "laneMinionsFirst10Minutes" in df.columns
        assert "jungleCsBefore10Minutes" in df.columns
        assert "early_cs_total" in df.columns

        # Values should be greater than 0 for a real game
        assert df.iloc[0]["cs_15"] > 0
        assert df.iloc[0]["gold_15"] > 0

    @patch('src.api.impact_score.add_derived_features')
    def test_extract_features_player_not_found(self, mock_add_derived, sample_match_data, sample_timeline_data):
        """Test error when player not found in match"""
        from fastapi import HTTPException

        mock_add_derived.side_effect = lambda df: df

        with pytest.raises(HTTPException) as exc_info:
            extract_features_from_match(sample_match_data, sample_timeline_data, "nonexistent_puuid")

        assert exc_info.value.status_code == 404
        assert "not found" in str(exc_info.value.detail).lower()

    @patch('src.api.impact_score.add_derived_features')
    def test_extract_features_challenge_data(self, mock_add_derived, sample_match_data, sample_timeline_data):
        """Test that challenge data is extracted correctly"""
        mock_add_derived.side_effect = lambda df: df

        df = extract_features_from_match(sample_match_data, sample_timeline_data, "puuid1")

        # Check challenge stats
        assert df.iloc[0]["goldPerMinute"] == 450.0
        assert df.iloc[0]["damagePerMinute"] == 800.0
        assert df.iloc[0]["killParticipation"] == 0.75
        assert df.iloc[0]["soloKills"] == 2

    @patch('src.api.impact_score.add_derived_features')
    def test_extract_features_missing_challenge_data(self, mock_add_derived, sample_match_data, sample_timeline_data):
        """Test handling of missing challenge data"""
        # Remove challenges from match data
        sample_match_data["info"]["participants"][0]["challenges"] = {}

        mock_add_derived.side_effect = lambda df: df

        df = extract_features_from_match(sample_match_data, sample_timeline_data, "puuid1")

        # Should default to 0
        assert df.iloc[0]["goldPerMinute"] == 0
        assert df.iloc[0]["soloKills"] == 0

    @patch('src.api.impact_score.add_derived_features')
    def test_extract_features_zero_deaths_kda(self, mock_add_derived, sample_match_data, sample_timeline_data):
        """Test KDA calculation with zero deaths"""
        # Set deaths to 0
        sample_match_data["info"]["participants"][0]["deaths"] = 0

        mock_add_derived.side_effect = lambda df: df

        df = extract_features_from_match(sample_match_data, sample_timeline_data, "puuid1")

        kda = df.iloc[0]["kda"]
        # KDA with 0 deaths should be kills + assists
        expected_kda = 10 + 8  # kills + assists
        assert kda == expected_kda


class TestRiotAPIHelpers:
    """Test Riot API helper functions"""

    @patch('src.api.impact_score.session')
    @patch('src.api.impact_score.RIOT_API_KEY', 'test_api_key')
    def test_riot_get_success(self, mock_session):
        """Test successful Riot API GET request"""
        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {"test": "data"}
        mock_response.status_code = 200
        mock_session.get.return_value = mock_response

        result = riot_get("https://test.api.com")

        assert result == {"test": "data"}
        mock_session.get.assert_called_once()

    @patch('src.api.impact_score.session')
    @patch('src.api.impact_score.RIOT_API_KEY', None)
    def test_riot_get_no_api_key(self, mock_session):
        """Test Riot API request without API key"""
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            riot_get("https://test.api.com")

        assert exc_info.value.status_code == 500
        assert "RIOT_API_KEY" in str(exc_info.value.detail)

    @patch('src.api.impact_score.session')
    @patch('src.api.impact_score.RIOT_API_KEY', 'test_api_key')
    def test_riot_get_rate_limit(self, mock_session):
        """Test Riot API rate limit handling"""
        # First request returns 429 (rate limit)
        mock_response_429 = Mock()
        mock_response_429.ok = False
        mock_response_429.status_code = 429
        mock_response_429.headers = {"Retry-After": "1"}

        # Second request succeeds
        mock_response_200 = Mock()
        mock_response_200.ok = True
        mock_response_200.json.return_value = {"test": "data"}
        mock_response_200.status_code = 200

        mock_session.get.side_effect = [mock_response_429, mock_response_200]

        result = riot_get("https://test.api.com")

        assert result == {"test": "data"}
        assert mock_session.get.call_count == 2

    @patch('src.api.impact_score.session')
    @patch('src.api.impact_score.RIOT_API_KEY', 'test_api_key')
    def test_riot_get_error(self, mock_session):
        """Test Riot API error handling"""
        from fastapi import HTTPException

        mock_response = Mock()
        mock_response.ok = False
        mock_response.status_code = 404
        mock_response.text = "Not found"
        mock_session.get.return_value = mock_response

        with pytest.raises(HTTPException) as exc_info:
            riot_get("https://test.api.com")

        assert exc_info.value.status_code == 404
        assert "Riot API error" in str(exc_info.value.detail)


class TestImpactScoreIntegration:
    """Integration tests for impact score (requires mocking)"""

    @pytest.mark.skip(reason="Requires full model and Riot API mocking - integration test")
    def test_get_impact_by_riot_id_success(self):
        """Test getting impact score by Riot ID (integration test)"""
        # This would require mocking the entire flow:
        # 1. Riot ID -> PUUID
        # 2. Get match IDs
        # 3. Get match data
        # 4. Get timeline data
        # 5. Extract features
        # 6. Run model prediction
        pass

    @pytest.mark.skip(reason="Requires full model and Riot API mocking - integration test")
    def test_get_impact_by_riot_id_no_matches(self):
        """Test error when player has no ranked matches (integration test)"""
        pass


class TestEdgeCases:
    """Test edge cases and error handling"""

    @patch('src.api.impact_score.add_derived_features')
    def test_short_game_duration(self, mock_add_derived, sample_match_data, sample_timeline_data):
        """Test with very short game duration"""
        # Set game duration to 10 minutes
        sample_match_data["info"]["gameDuration"] = 600
        sample_timeline_data["info"]["frames"] = sample_timeline_data["info"]["frames"][:11]

        mock_add_derived.side_effect = lambda df: df

        df = extract_features_from_match(sample_match_data, sample_timeline_data, "puuid1")

        # Should still extract features without errors
        assert len(df) == 1
        assert df.iloc[0]["puuid"] == "puuid1"

    @patch('src.api.impact_score.add_derived_features')
    def test_missing_timeline_frames(self, mock_add_derived, sample_match_data):
        """Test with minimal timeline data"""
        minimal_timeline = {
            "info": {
                "frames": [
                    {
                        "timestamp": 0,
                        "participantFrames": {
                            "1": {
                                "minionsKilled": 0,
                                "jungleMinionsKilled": 0,
                                "totalGold": 500,
                                "xp": 200
                            }
                        }
                    }
                ]
            }
        }

        mock_add_derived.side_effect = lambda df: df

        df = extract_features_from_match(sample_match_data, minimal_timeline, "puuid1")

        # Should handle missing frames gracefully
        assert len(df) == 1
        assert df.iloc[0]["cs_15"] >= 0
        assert df.iloc[0]["gold_15"] >= 0
