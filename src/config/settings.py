"""Application settings and configuration"""

import os
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Application settings"""

    # API Configuration
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))

    # Riot API Configuration
    RIOT_API_KEY: str = os.getenv("RIOT_API_KEY", "")
    RIOT_API_BASE_URL: str = "https://asia.api.riotgames.com"
    RIOT_API_REGION_URL: str = "https://kr.api.riotgames.com"

    # Database Configuration (TBD)
    DB_HOST: str = os.getenv("DB_HOST", "localhost")
    DB_PORT: int = int(os.getenv("DB_PORT", "5432"))
    DB_NAME: str = os.getenv("DB_NAME", "lol_highlights")
    DB_USER: str = os.getenv("DB_USER", "")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD", "")

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", "logs/app.log")

    # Model Configuration
    MODEL_PATH: str = "models/"
    DATA_PATH: str = "data/"


class TierBaseline:
    """Tier-based baseline statistics for gap calculation

    Note:
    - avg_gold_per_min: normalized for game duration
    - avg_vision_score_per_min: normalized for game duration (based on 30min average)
    """

    BASELINES: Dict[str, Dict[str, float]] = {
        "IRON": {
            "avg_kda": 2.0,
            "avg_cs_per_min": 4.5,
            "avg_gold_per_min": 333,  # ~10000 / 30min
            "avg_vision_score_per_min": 0.5,  # ~15 / 30min
            "avg_damage_share": 0.18,
        },
        "BRONZE": {
            "avg_kda": 2.3,
            "avg_cs_per_min": 5.0,
            "avg_gold_per_min": 367,  # ~11000 / 30min
            "avg_vision_score_per_min": 0.6,  # ~18 / 30min
            "avg_damage_share": 0.19,
        },
        "SILVER": {
            "avg_kda": 2.5,
            "avg_cs_per_min": 5.5,
            "avg_gold_per_min": 400,  # ~12000 / 30min
            "avg_vision_score_per_min": 0.73,  # ~22 / 30min
            "avg_damage_share": 0.20,
        },
        "GOLD": {
            "avg_kda": 2.8,
            "avg_cs_per_min": 6.0,
            "avg_gold_per_min": 433,  # ~13000 / 30min
            "avg_vision_score_per_min": 0.83,  # ~25 / 30min
            "avg_damage_share": 0.21,
        },
        "PLATINUM": {
            "avg_kda": 3.0,
            "avg_cs_per_min": 6.5,
            "avg_gold_per_min": 467,  # ~14000 / 30min
            "avg_vision_score_per_min": 0.93,  # ~28 / 30min
            "avg_damage_share": 0.22,
        },
        "EMERALD": {
            "avg_kda": 3.2,
            "avg_cs_per_min": 7.0,
            "avg_gold_per_min": 500,  # ~15000 / 30min
            "avg_vision_score_per_min": 1.07,  # ~32 / 30min
            "avg_damage_share": 0.23,
        },
        "DIAMOND": {
            "avg_kda": 3.5,
            "avg_cs_per_min": 7.5,
            "avg_gold_per_min": 533,  # ~16000 / 30min
            "avg_vision_score_per_min": 1.17,  # ~35 / 30min
            "avg_damage_share": 0.24,
        },
        "MASTER": {
            "avg_kda": 4.0,
            "avg_cs_per_min": 8.0,
            "avg_gold_per_min": 567,  # ~17000 / 30min
            "avg_vision_score_per_min": 1.33,  # ~40 / 30min
            "avg_damage_share": 0.25,
        },
        "GRANDMASTER": {
            "avg_kda": 4.5,
            "avg_cs_per_min": 8.5,
            "avg_gold_per_min": 600,  # ~18000 / 30min
            "avg_vision_score_per_min": 1.5,  # ~45 / 30min
            "avg_damage_share": 0.26,
        },
        "CHALLENGER": {
            "avg_kda": 5.0,
            "avg_cs_per_min": 9.0,
            "avg_gold_per_min": 633,  # ~19000 / 30min
            "avg_vision_score_per_min": 1.67,  # ~50 / 30min
            "avg_damage_share": 0.27,
        },
    }

    @classmethod
    def get_baseline(cls, tier: str) -> Dict[str, float]:
        """Get baseline statistics for a specific tier"""
        return cls.BASELINES.get(tier.upper(), cls.BASELINES["GOLD"])

    @classmethod
    def get_all_tiers(cls) -> list:
        """Get list of all tiers"""
        return list(cls.BASELINES.keys())


settings = Settings()
tier_baseline = TierBaseline()
