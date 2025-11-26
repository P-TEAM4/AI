"""Tests for API endpoints"""

import pytest
from fastapi.testclient import TestClient
from src.api.routes import app


@pytest.fixture
def client():
    """Create test client"""
    with TestClient(app) as test_client:
        yield test_client


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


def test_analyze_gap(client):
    """Test gap analysis endpoint"""
    request_data = {
        "player_stats": {
            "kills": 10,
            "deaths": 3,
            "assists": 8,
            "kda": 6.0,
            "cs": 180,
            "cs_per_min": 7.5,
            "gold": 15000,
            "vision_score": 35,
            "damage_dealt": 25000,
            "damage_share": 0.24,
            "champion_name": "Ahri",
        },
        "tier": "DIAMOND",
        "division": "II",
    }

    response = client.post("/api/v1/analyze/gap", json=request_data)
    assert response.status_code == 200
    data = response.json()
    assert "tier" in data
    assert "gaps" in data
    assert "overall_score" in data
    assert "strengths" in data
    assert "weaknesses" in data
    assert "recommendations" in data


def test_suggest_tier(client):
    """Test tier suggestion endpoint"""
    request_data = {
        "player_stats": {
            "kills": 8,
            "deaths": 5,
            "assists": 10,
            "kda": 3.6,
            "cs": 150,
            "cs_per_min": 6.5,
            "gold": 13000,
            "vision_score": 28,
            "damage_dealt": 20000,
            "damage_share": 0.22,
            "champion_name": "Lux",
        },
        "tier": "GOLD",
    }

    response = client.post("/api/v1/suggest-tier", json=request_data)
    assert response.status_code == 200
    data = response.json()
    assert "suggested_tier" in data
    assert "current_tier" in data
    assert "comparisons" in data
