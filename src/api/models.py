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

    match_id: str = Field(description="Match ID (e.g., KR_1234567890)")
    puuid: str = Field(description="Player PUUID")
    tier: str = Field(description="Player tier (e.g., GOLD, PLATINUM)")


class GapAnalysisResult(BaseModel):
    """Gap analysis result model"""

    tier: str
    player_stats: Dict[str, float]
    tier_baseline: Dict[str, float]
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


# Analysis API Models
class AnalysisGenerateRequest(BaseModel):
    """AI 분석 생성 요청"""

    match_id: str = Field(description="Match ID (e.g., KR_1234567890)")
    puuid: str = Field(description="Player PUUID")
    game_name: str = Field(description="Riot ID 이름")
    tag_line: str = Field(description="Riot ID 태그")
    tier: Optional[str] = Field(default="GOLD", description="Player tier")


class ImpactFeature(BaseModel):
    """Impact Score 주요 지표"""

    name: str
    display_name: str = Field(alias="displayName")
    direction: str
    shap: float
    value: float

    class Config:
        populate_by_name = True


class AnalysisResult(BaseModel):
    """AI 분석 결과"""

    match_id: str
    puuid: str
    game_name: str
    tag_line: str
    champion: str
    role: str
    win: bool

    # Impact Score
    impact_score: float
    baseline_proba: float = Field(alias="baselineProba")
    predicted_proba: float = Field(alias="predictedProba")
    summary: str
    top_features: List[ImpactFeature]

    # Gap Analysis
    gap_analysis: GapAnalysisResult

    # Match Info
    game_duration: int
    player_stats: PlayerStats

    class Config:
        populate_by_name = True


# Highlight API Models
class HighlightGenerateRequest(BaseModel):
    """하이라이트 생성 요청"""

    match_id: str = Field(description="Match ID")
    game_name: str = Field(description="Riot ID 이름")
    tag_line: str = Field(description="Riot ID 태그")
    top_highlights: int = Field(default=5, description="잘한 장면 개수")
    top_mistakes: int = Field(default=3, description="못한 장면 개수")


class ClipInfo(BaseModel):
    """클립 정보"""

    clip_path: str
    timestamp: float
    type: str
    base_importance: float
    impact_score: float
    combined_importance: float
    description: str
    impact_description: str
    details: Dict[str, Any]


class PlayerInfo(BaseModel):
    """플레이어 정보"""

    champion_name: str = Field(alias="championName")
    team_position: str = Field(alias="teamPosition")
    win: bool
    kills: int
    deaths: int
    assists: int

    class Config:
        populate_by_name = True


class MatchSummary(BaseModel):
    """매치 요약"""

    player: Dict[str, str]
    impact_analysis: Dict[str, float]
    key_moments: Dict[str, Optional[Dict[str, Any]]]
    highlight_count: int
    mistake_count: int


class HighlightResult(BaseModel):
    """하이라이트 생성 결과"""

    match_id: str
    player: Dict[str, str]
    match_info: Optional[PlayerInfo]
    match_summary: Optional[MatchSummary]
    highlights: List[ClipInfo]
    mistakes: List[ClipInfo]
    video_path: str
    total_clips: int


class TimelineHighlightRequest(BaseModel):
    """타임라인 데이터로 하이라이트 추출 요청 (테스트용)"""

    timeline_data: Dict[str, Any] = Field(description="Riot API timeline JSON data")
    target_puuid: str = Field(description="Target player PUUID")
    top_n: int = Field(default=5, ge=1, le=20, description="Number of highlights to return")


class HighlightInfo(BaseModel):
    """하이라이트 정보"""

    timestamp: float
    type: str
    category: str
    importance: float
    description: str
    details: Dict[str, Any]


class TimelineHighlightResponse(BaseModel):
    """타임라인 하이라이트 추출 결과"""

    highlights: List[HighlightInfo] = Field(description="잘한 부분 (highlight)")
    mistakes: List[HighlightInfo] = Field(description="못한 부분 (mistake)")
    all_events: List[HighlightInfo] = Field(description="모든 이벤트")
    total_count: int
