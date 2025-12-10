"""Tests for highlight extraction functionality"""

import pytest

# Try importing highlight_extractor, skip all tests if dependencies are missing
try:
    from src.api.highlight_extractor import (
        extract_highlights_from_timeline,
        calculate_kill_importance,
        calculate_death_importance,
        calculate_objective_importance,
        get_top_highlights
    )
    HIGHLIGHT_EXTRACTOR_AVAILABLE = True
except ImportError as e:
    HIGHLIGHT_EXTRACTOR_AVAILABLE = False
    pytestmark = pytest.mark.skip(f"highlight_extractor dependencies not available: {e}")


@pytest.fixture
def sample_timeline_data():
    """Sample timeline data for testing"""
    return {
        'metadata': {
            'participants': ['puuid1', 'puuid2', 'puuid3', 'puuid4', 'puuid5']
        },
        'info': {
            'frames': [
                {
                    'timestamp': 180000,  # 3분
                    'events': [
                        {
                            'type': 'CHAMPION_KILL',
                            'timestamp': 180000,
                            'killerId': 1,
                            'victimId': 6,
                            'assistingParticipantIds': [2, 3],
                            'shutdownBounty': 450,
                            'bounty': 300
                        }
                    ]
                },
                {
                    'timestamp': 300000,  # 5분
                    'events': [
                        {
                            'type': 'CHAMPION_KILL',
                            'timestamp': 300000,
                            'killerId': 6,
                            'victimId': 1,
                            'assistingParticipantIds': [],
                            'bounty': 350
                        }
                    ]
                },
                {
                    'timestamp': 1200000,  # 20분
                    'events': [
                        {
                            'type': 'ELITE_MONSTER_KILL',
                            'timestamp': 1200000,
                            'killerId': 1,
                            'monsterType': 'BARON_NASHOR',
                            'assistingParticipantIds': [2, 3, 4, 5]
                        }
                    ]
                }
            ]
        }
    }


class TestHighlightExtraction:
    """Test highlight extraction from timeline"""

    def test_extract_highlights_with_kills(self, sample_timeline_data):
        """Test extracting kill highlights"""
        highlights = extract_highlights_from_timeline(sample_timeline_data, 'puuid1')

        # Should have at least 1 kill highlight
        kill_highlights = [h for h in highlights if h['type'] == 'kill']
        assert len(kill_highlights) >= 1

        # Check kill highlight structure
        kill_highlight = kill_highlights[0]
        assert 'timestamp' in kill_highlight
        assert 'type' in kill_highlight
        assert 'category' in kill_highlight
        assert 'importance' in kill_highlight
        assert 'description' in kill_highlight
        assert kill_highlight['category'] == 'highlight'

    def test_extract_highlights_with_deaths(self, sample_timeline_data):
        """Test extracting death highlights"""
        highlights = extract_highlights_from_timeline(sample_timeline_data, 'puuid1')

        # Should have at least 1 death highlight
        death_highlights = [h for h in highlights if h['type'] == 'death']
        assert len(death_highlights) >= 1

        # Check death highlight structure
        death_highlight = death_highlights[0]
        assert death_highlight['category'] == 'mistake'
        assert death_highlight['importance'] > 0

    def test_extract_highlights_with_objectives(self, sample_timeline_data):
        """Test extracting objective highlights"""
        highlights = extract_highlights_from_timeline(sample_timeline_data, 'puuid1')

        # Should have objective highlights
        obj_highlights = [h for h in highlights if h['type'] == 'objective']
        assert len(obj_highlights) >= 1

        # Baron should have high importance
        baron_highlights = [h for h in obj_highlights if 'BARON_NASHOR' in h['description']]
        if baron_highlights:
            assert baron_highlights[0]['importance'] == 10.0

    def test_extract_highlights_player_not_found(self):
        """Test with player not in timeline"""
        timeline_data = {
            'metadata': {'participants': ['puuid1', 'puuid2']},
            'info': {'frames': []}
        }

        highlights = extract_highlights_from_timeline(timeline_data, 'nonexistent_puuid')
        assert highlights == []

    def test_extract_highlights_empty_timeline(self):
        """Test with empty timeline"""
        timeline_data = {
            'metadata': {'participants': ['puuid1']},
            'info': {'frames': []}
        }

        highlights = extract_highlights_from_timeline(timeline_data, 'puuid1')
        assert highlights == []

    def test_highlights_sorted_by_importance(self, sample_timeline_data):
        """Test that highlights are sorted by importance"""
        highlights = extract_highlights_from_timeline(sample_timeline_data, 'puuid1')

        if len(highlights) > 1:
            # Check that importance is in descending order
            for i in range(len(highlights) - 1):
                assert highlights[i]['importance'] >= highlights[i + 1]['importance']


class TestImportanceCalculation:
    """Test importance calculation functions"""

    def test_calculate_kill_importance_basic(self):
        """Test basic kill importance"""
        event = {
            'assistingParticipantIds': [],
            'shutdownBounty': 0
        }
        importance = calculate_kill_importance(event, 10.0)
        assert 0 < importance <= 10.0
        assert importance == 5.0  # Base score

    def test_calculate_kill_importance_early_game(self):
        """Test early game kill has higher importance"""
        event = {
            'assistingParticipantIds': [],
            'shutdownBounty': 0
        }
        early_importance = calculate_kill_importance(event, 3.0)
        late_importance = calculate_kill_importance(event, 20.0)

        assert early_importance > late_importance

    def test_calculate_kill_importance_shutdown(self):
        """Test shutdown kill has higher importance"""
        event_no_shutdown = {
            'assistingParticipantIds': [],
            'shutdownBounty': 0
        }
        event_shutdown = {
            'assistingParticipantIds': [],
            'shutdownBounty': 500
        }

        importance_no_shutdown = calculate_kill_importance(event_no_shutdown, 10.0)
        importance_shutdown = calculate_kill_importance(event_shutdown, 10.0)

        assert importance_shutdown > importance_no_shutdown

    def test_calculate_kill_importance_teamfight(self):
        """Test teamfight kill has higher importance"""
        event_solo = {
            'assistingParticipantIds': [],
            'shutdownBounty': 0
        }
        event_teamfight = {
            'assistingParticipantIds': [2, 3, 4, 5],
            'shutdownBounty': 0
        }

        importance_solo = calculate_kill_importance(event_solo, 10.0)
        importance_teamfight = calculate_kill_importance(event_teamfight, 10.0)

        assert importance_teamfight > importance_solo

    def test_calculate_kill_importance_max_cap(self):
        """Test importance is capped at 10.0"""
        event = {
            'assistingParticipantIds': [2, 3, 4, 5],
            'shutdownBounty': 1000
        }
        importance = calculate_kill_importance(event, 2.0)  # Early game + shutdown + teamfight

        assert importance <= 10.0

    def test_calculate_death_importance_basic(self):
        """Test basic death importance"""
        event = {'bounty': 300}
        importance = calculate_death_importance(event, 10.0)
        assert 0 < importance <= 10.0

    def test_calculate_death_importance_early_game(self):
        """Test early game death is more important"""
        event = {'bounty': 300}
        early_importance = calculate_death_importance(event, 3.0)
        late_importance = calculate_death_importance(event, 20.0)

        assert early_importance > late_importance

    def test_calculate_death_importance_high_bounty(self):
        """Test high bounty death is more important"""
        event_low = {'bounty': 300}
        event_high = {'bounty': 500}

        importance_low = calculate_death_importance(event_low, 10.0)
        importance_high = calculate_death_importance(event_high, 10.0)

        assert importance_high > importance_low

    def test_calculate_objective_importance_baron(self):
        """Test Baron has highest importance"""
        event_baron = {'monsterType': 'BARON_NASHOR'}
        importance = calculate_objective_importance(event_baron, 20.0)
        assert importance == 10.0

    def test_calculate_objective_importance_dragon(self):
        """Test Dragon importance"""
        event_dragon = {'monsterType': 'DRAGON'}
        importance = calculate_objective_importance(event_dragon, 10.0)
        assert importance == 7.0

    def test_calculate_objective_importance_turret(self):
        """Test turret importance varies by type"""
        event_nexus = {'buildingType': 'NEXUS_TURRET'}
        event_outer = {'buildingType': 'OUTER_TURRET'}

        importance_nexus = calculate_objective_importance(event_nexus, 30.0)
        importance_outer = calculate_objective_importance(event_outer, 10.0)

        assert importance_nexus > importance_outer

    def test_calculate_objective_importance_unknown(self):
        """Test unknown objective has default importance"""
        event = {'monsterType': 'UNKNOWN_TYPE'}
        importance = calculate_objective_importance(event, 10.0)
        assert importance == 5.0  # Default value


class TestTopHighlights:
    """Test getting top N highlights"""

    def test_get_top_highlights_all(self):
        """Test getting top highlights without category filter"""
        highlights = [
            {'type': 'kill', 'category': 'highlight', 'importance': 8.0},
            {'type': 'death', 'category': 'mistake', 'importance': 6.0},
            {'type': 'kill', 'category': 'highlight', 'importance': 7.0},
            {'type': 'objective', 'category': 'highlight', 'importance': 10.0},
        ]

        top_3 = get_top_highlights(highlights, top_n=3)
        assert len(top_3) == 3
        assert top_3[0]['importance'] == 10.0
        assert top_3[1]['importance'] == 8.0
        assert top_3[2]['importance'] == 7.0

    def test_get_top_highlights_category_filter(self):
        """Test getting top highlights with category filter"""
        highlights = [
            {'type': 'kill', 'category': 'highlight', 'importance': 8.0},
            {'type': 'death', 'category': 'mistake', 'importance': 6.0},
            {'type': 'kill', 'category': 'highlight', 'importance': 7.0},
            {'type': 'death', 'category': 'mistake', 'importance': 5.0},
        ]

        top_mistakes = get_top_highlights(highlights, top_n=2, category='mistake')
        assert len(top_mistakes) == 2
        assert all(h['category'] == 'mistake' for h in top_mistakes)
        assert top_mistakes[0]['importance'] == 6.0

    def test_get_top_highlights_empty_list(self):
        """Test with empty highlights list"""
        highlights = []
        top = get_top_highlights(highlights, top_n=5)
        assert top == []

    def test_get_top_highlights_more_than_available(self):
        """Test requesting more highlights than available"""
        highlights = [
            {'type': 'kill', 'category': 'highlight', 'importance': 8.0},
            {'type': 'kill', 'category': 'highlight', 'importance': 7.0},
        ]

        top_5 = get_top_highlights(highlights, top_n=5)
        assert len(top_5) == 2  # Only 2 available

    def test_get_top_highlights_category_not_found(self):
        """Test filtering by category that doesn't exist"""
        highlights = [
            {'type': 'kill', 'category': 'highlight', 'importance': 8.0},
            {'type': 'kill', 'category': 'highlight', 'importance': 7.0},
        ]

        top_mistakes = get_top_highlights(highlights, top_n=3, category='mistake')
        assert top_mistakes == []
