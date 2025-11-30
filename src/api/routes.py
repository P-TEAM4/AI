"""FastAPI routes for the LoL Highlight & Analysis API"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from src.api.models import (
    HealthCheckResponse,
    MatchRequest,
    ProfileRequest,
    GapAnalysisRequest,
    MatchAnalysisResult,
    ProfileAnalysisResult,
    GapAnalysisResult,
)
from src.services.analyzer import MatchAnalyzer
from src.models.rule_based import RuleBasedGapAnalyzer
from src import __version__

# Initialize FastAPI app
app = FastAPI(
    title="LoL Highlight & Analysis API",
    description="AI-powered League of Legends match analysis and highlight generation",
    version=__version__,
)

# Add CORS middleware for Spring Boot integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
match_analyzer = MatchAnalyzer()
gap_analyzer = RuleBasedGapAnalyzer()


@app.get("/", response_model=HealthCheckResponse)
async def health_check():
    """
    Health check endpoint

    Returns service status and version information
    """
    return HealthCheckResponse(
        status="healthy", version=__version__, timestamp=datetime.now()
    )


@app.get("/health", response_model=HealthCheckResponse)
async def health():
    """Alternative health check endpoint"""
    return HealthCheckResponse(
        status="healthy", version=__version__, timestamp=datetime.now()
    )


@app.post("/api/v1/analyze/match", response_model=MatchAnalysisResult)
async def analyze_match(request: MatchRequest):
    """
    Analyze a specific match for a player

    Args:
        request: Match analysis request containing match_id, summoner_name, tag_line

    Returns:
        Complete match analysis including player stats, gap analysis, and key moments

    Raises:
        HTTPException: If match not found or analysis fails
    """
    try:
        result = match_analyzer.analyze_match(
            match_id=request.match_id,
            summoner_name=request.summoner_name,
            tag_line=request.tag_line,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/api/v1/analyze/profile", response_model=ProfileAnalysisResult)
async def analyze_profile(request: ProfileRequest):
    """
    Analyze player profile based on recent games

    Args:
        request: Profile analysis request containing summoner_name, tag_line, recent_games

    Returns:
        Complete profile analysis including average stats, gap analysis, champion pool, and trends

    Raises:
        HTTPException: If player not found or analysis fails
    """
    try:
        result = match_analyzer.analyze_profile(
            summoner_name=request.summoner_name,
            tag_line=request.tag_line,
            recent_games=request.recent_games,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/api/v1/analyze/gap", response_model=GapAnalysisResult)
async def analyze_gap(request: GapAnalysisRequest):
    """
    Perform gap analysis on player statistics from a specific match

    Args:
        request: Gap analysis request containing match_id and puuid

    Returns:
        Gap analysis result with strengths, weaknesses, and recommendations

    Raises:
        HTTPException: If match not found or analysis fails
    """
    try:
        # Get match details
        match_data = match_analyzer.riot_client.get_match_details(request.match_id)
        player_stats_raw = match_analyzer.riot_client.extract_player_stats_from_match(
            match_data, request.puuid
        )

        if not player_stats_raw:
            raise ValueError(f"Player not found in match {request.match_id}")

        # Create PlayerStats object
        player_stats = match_analyzer.create_player_stats_from_match(player_stats_raw)

        # Perform gap analysis
        result = gap_analyzer.analyze_gap(
            player_stats=player_stats,
            tier=request.tier,
            division="I",
        )
        return result

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gap analysis failed: {str(e)}")


@app.get("/api/v1/tiers")
async def get_available_tiers():
    """
    Get list of available tiers for baseline comparison

    Returns:
        List of tier names and their baseline statistics
    """
    tiers = gap_analyzer.tier_baseline.get_all_tiers()
    tier_data = {}

    for tier in tiers:
        tier_data[tier] = gap_analyzer.tier_baseline.get_baseline(tier)

    return {"tiers": tiers, "baselines": tier_data}


@app.post("/api/v1/suggest-tier")
async def suggest_tier(request: GapAnalysisRequest):
    """
    Suggest appropriate target tier based on player statistics from a match

    Args:
        request: Gap analysis request containing match_id and puuid

    Returns:
        Suggested tier and comparison with multiple tiers

    Raises:
        HTTPException: If analysis fails
    """
    try:
        # Get match details
        match_data = match_analyzer.riot_client.get_match_details(request.match_id)
        player_stats_raw = match_analyzer.riot_client.extract_player_stats_from_match(
            match_data, request.puuid
        )

        if not player_stats_raw:
            raise ValueError(f"Player not found in match {request.match_id}")

        # Create PlayerStats object
        player_stats = match_analyzer.create_player_stats_from_match(player_stats_raw)

        # Suggest tier
        suggested_tier = gap_analyzer.suggest_target_tier(player_stats)
        all_comparisons = gap_analyzer.compare_with_multiple_tiers(player_stats)

        return {
            "suggested_tier": suggested_tier,
            "current_tier": request.tier,
            "comparisons": {tier: analysis.dict() for tier, analysis in all_comparisons.items()},
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tier suggestion failed: {str(e)}")


# For running with uvicorn
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
