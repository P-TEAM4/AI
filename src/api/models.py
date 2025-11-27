"""Pydantic models for API request/response"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class PlayerStats(BaseModel):
    """Player statistics model"""

    kills: int = Field(ge=0, description="Number of kills")
    deaths: int = Field(ge=0, description="Number of deaths")
    assists: int = Field(ge=0, description="Number of assists")
    kda: float = Field(ge=0, description="KDA ratio")
    cs: int = Field(ge=0, description="Total creep score")
    cs_per_min: float = Field(ge=0, description="CS per minute")
    gold: int = Field(ge=0, description="Total gold earned")
    gold_per_min: Optional[float] = Field(default=None, ge=0, description="Gold per minute (auto-calculated if not provided)")
    vision_score: int = Field(ge=0, description="Vision score")
    vision_score_per_min: Optional[float] = Field(default=None, ge=0, description="Vision score per minute (auto-calculated if not provided)")
    damage_dealt: int = Field(ge=0, description="Total damage dealt")
    damage_share: float = Field(ge=0, le=1, description="Percentage of team's total damage")
    champion_name: str = Field(description="Champion name")
    position: Optional[str] = Field(default=None, description="Lane position")
    game_duration: Optional[int] = Field(default=None, ge=0, description="Game duration in seconds")


class TierInfo(BaseModel):
    """Player tier information"""

    tier: str = Field(description="Tier (IRON, BRONZE, SILVER, etc.)")
    division: str = Field(description="Division (I, II, III, IV)")
    lp: int = Field(ge=0, description="League Points")
    wins: int = Field(ge=0, description="Number of wins")
    losses: int = Field(ge=0, description="Number of losses")


class MatchRequest(BaseModel):
    """Match analysis request model"""

    match_id: str = Field(description="Match ID")
    summoner_name: str = Field(description="Summoner name")
    tag_line: str = Field(description="Tag line (e.g., KR1)")


class ProfileRequest(BaseModel):
    """Player profile analysis request model"""

    summoner_name: str = Field(description="Summoner name")
    tag_line: str = Field(description="Tag line")
    recent_games: int = Field(default=20, ge=1, le=100, description="Number of recent games to analyze")


class GapAnalysisRequest(BaseModel):
    """Gap analysis request model"""

    player_stats: PlayerStats
    tier: str = Field(description="Player tier")
    division: Optional[str] = Field(default="I", description="Player division")


class GapAnalysisResult(BaseModel):
    """Gap analysis result model"""

    tier: str
    player_avg: Dict[str, float]
    tier_avg: Dict[str, float]
    gaps: Dict[str, float]
    normalized_gaps: Dict[str, float]
    overall_score: float
    strengths: List[str]
    weaknesses: List[str]
    recommendations: List[str]


class MatchAnalysisResult(BaseModel):
    """Match analysis result model"""

    match_id: str
    summoner_name: str
    win: bool
    game_duration: int
    player_stats: PlayerStats
    gap_analysis: GapAnalysisResult
    key_moments: List[Dict[str, Any]]
    impact_score: float


class ProfileAnalysisResult(BaseModel):
    """Player profile analysis result model"""

    summoner_name: str
    tier_info: TierInfo
    games_analyzed: int
    avg_stats: Dict[str, float]
    gap_analysis: GapAnalysisResult
    champion_pool: List[Dict[str, Any]]
    performance_trend: Dict[str, Any]


class HealthCheckResponse(BaseModel):
    """Health check response model"""

    status: str
    version: str
    timestamp: datetime
