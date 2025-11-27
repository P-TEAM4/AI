"""Helper utility functions"""

import logging
from typing import Any, Dict
from datetime import datetime


def setup_logging(log_file: str = "logs/app.log", level: str = "INFO"):
    """
    Setup logging configuration

    Args:
        log_file: Path to log file
        level: Logging level
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )


def format_duration(seconds: int) -> str:
    """
    Format game duration from seconds to MM:SS

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted duration string
    """
    minutes = seconds // 60
    secs = seconds % 60
    return f"{minutes:02d}:{secs:02d}"


def calculate_win_rate(wins: int, losses: int) -> float:
    """
    Calculate win rate percentage

    Args:
        wins: Number of wins
        losses: Number of losses

    Returns:
        Win rate percentage
    """
    total = wins + losses
    if total == 0:
        return 0.0
    return round((wins / total) * 100, 2)


def validate_tier(tier: str) -> bool:
    """
    Validate if tier is valid

    Args:
        tier: Tier name

    Returns:
        True if valid, False otherwise
    """
    valid_tiers = [
        "IRON",
        "BRONZE",
        "SILVER",
        "GOLD",
        "PLATINUM",
        "EMERALD",
        "DIAMOND",
        "MASTER",
        "GRANDMASTER",
        "CHALLENGER",
    ]
    return tier.upper() in valid_tiers


def sanitize_summoner_name(name: str) -> str:
    """
    Sanitize summoner name for API requests

    Args:
        name: Summoner name

    Returns:
        Sanitized name
    """
    return name.strip()


def timestamp_to_datetime(timestamp: int) -> datetime:
    """
    Convert Unix timestamp to datetime

    Args:
        timestamp: Unix timestamp in milliseconds

    Returns:
        Datetime object
    """
    return datetime.fromtimestamp(timestamp / 1000)
