"""Helper utility functions"""

import logging
from typing import Any, Dict
from datetime import datetime, timezone, timedelta


# KST timezone (UTC+9)
KST = timezone(timedelta(hours=9))


class KSTFormatter(logging.Formatter):
    """Custom formatter to use KST timezone"""

    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, tz=KST)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.strftime("%Y-%m-%d %H:%M:%S")


def setup_logging(log_file: str = "logs/app.log", level: str = "INFO"):
    """
    Setup logging configuration with KST timezone

    Args:
        log_file: Path to log file
        level: Logging level
    """
    formatter = KSTFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        handlers=[file_handler, stream_handler],
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
