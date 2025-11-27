"""Match and player analysis service"""

from typing import Dict, List, Any
from src.models.rule_based import RuleBasedGapAnalyzer
from src.services.riot_api import RiotAPIClient
from src.api.models import (
    PlayerStats,
    TierInfo,
    GapAnalysisResult,
    MatchAnalysisResult,
    ProfileAnalysisResult,
)


class MatchAnalyzer:
    """Service for analyzing matches and player performance"""

    def __init__(self, api_key: str = None):
        self.gap_analyzer = RuleBasedGapAnalyzer()
        self.riot_client = RiotAPIClient(api_key=api_key) if api_key else RiotAPIClient()

    def create_player_stats_from_match(
        self, match_stats: Dict[str, Any]
    ) -> PlayerStats:
        """
        Convert raw match statistics to PlayerStats model

        Args:
            match_stats: Raw match statistics from Riot API

        Returns:
            PlayerStats object
        """
        kills = match_stats.get("kills", 0)
        deaths = match_stats.get("deaths", 0)
        assists = match_stats.get("assists", 0)
        total_cs = match_stats.get("total_cs", 0)
        game_duration = match_stats.get("game_duration", 1)

        kda = self.gap_analyzer.calculate_kda(kills, deaths, assists)
        cs_per_min = self.gap_analyzer.calculate_cs_per_min(total_cs, game_duration)

        # For damage_share, we'd need team total damage - defaulting to 0.20 for now
        # This should be calculated from match data in production
        damage_share = match_stats.get("damage_share", 0.20)

        return PlayerStats(
            kills=kills,
            deaths=deaths,
            assists=assists,
            kda=kda,
            cs=total_cs,
            cs_per_min=cs_per_min,
            gold=match_stats.get("gold", 0),
            vision_score=match_stats.get("vision_score", 0),
            damage_dealt=match_stats.get("damage_dealt", 0),
            damage_share=damage_share,
            champion_name=match_stats.get("champion_name", "Unknown"),
            position=match_stats.get("position"),
        )

    def analyze_match(
        self, match_id: str, summoner_name: str, tag_line: str
    ) -> MatchAnalysisResult:
        """
        Analyze a specific match for a player

        Args:
            match_id: Match ID
            summoner_name: Summoner name
            tag_line: Tag line

        Returns:
            Complete match analysis
        """
        # Get player account
        account = self.riot_client.get_account_by_riot_id(summoner_name, tag_line)
        puuid = account.get("puuid")

        # Get match details
        match_data = self.riot_client.get_match_details(match_id)
        player_stats_raw = self.riot_client.extract_player_stats_from_match(
            match_data, puuid
        )

        if not player_stats_raw:
            raise ValueError(f"Player not found in match {match_id}")

        # Get player rank
        summoner = self.riot_client.get_summoner_by_puuid(puuid)
        rank_info = self.riot_client.get_rank_info(summoner["id"])

        # Find Solo/Duo rank
        tier = "GOLD"  # Default
        for rank in rank_info:
            if rank.get("queueType") == "RANKED_SOLO_5x5":
                tier = rank.get("tier", "GOLD")
                break

        # Create player stats
        player_stats = self.create_player_stats_from_match(player_stats_raw)

        # Perform gap analysis
        gap_analysis = self.gap_analyzer.analyze_gap(player_stats, tier)

        # Extract key moments (simplified - would need timeline analysis)
        key_moments = self._extract_key_moments(match_data, puuid)

        # Calculate impact score (based on gap analysis)
        impact_score = gap_analysis.overall_score

        return MatchAnalysisResult(
            match_id=match_id,
            summoner_name=summoner_name,
            win=player_stats_raw.get("win", False),
            game_duration=player_stats_raw.get("game_duration", 0),
            player_stats=player_stats,
            gap_analysis=gap_analysis,
            key_moments=key_moments,
            impact_score=impact_score,
        )

    def analyze_profile(
        self, summoner_name: str, tag_line: str, recent_games: int = 20
    ) -> ProfileAnalysisResult:
        """
        Analyze player profile based on recent games

        Args:
            summoner_name: Summoner name
            tag_line: Tag line
            recent_games: Number of recent games to analyze

        Returns:
            Complete profile analysis
        """
        # Get player account
        account = self.riot_client.get_account_by_riot_id(summoner_name, tag_line)
        puuid = account.get("puuid")

        # Get summoner info
        summoner = self.riot_client.get_summoner_by_puuid(puuid)

        # Get rank info
        rank_info = self.riot_client.get_rank_info(summoner["id"])
        tier_info = self._extract_tier_info(rank_info)

        # Get recent performances
        performances = self.riot_client.get_player_recent_performance(
            summoner_name, tag_line, recent_games
        )

        if not performances:
            raise ValueError("No recent games found")

        # Calculate average stats
        avg_stats = self._calculate_average_stats(performances)

        # Create average player stats for gap analysis
        avg_player_stats = PlayerStats(
            kills=int(avg_stats["avg_kills"]),
            deaths=int(avg_stats["avg_deaths"]),
            assists=int(avg_stats["avg_assists"]),
            kda=avg_stats["avg_kda"],
            cs=int(avg_stats["avg_cs"]),
            cs_per_min=avg_stats["avg_cs_per_min"],
            gold=int(avg_stats["avg_gold"]),
            vision_score=int(avg_stats["avg_vision_score"]),
            damage_dealt=int(avg_stats["avg_damage"]),
            damage_share=0.20,  # Would need team data
            champion_name="Average",
        )

        # Perform gap analysis
        gap_analysis = self.gap_analyzer.analyze_gap(
            avg_player_stats, tier_info.tier
        )

        # Extract champion pool
        champion_pool = self._extract_champion_pool(performances)

        # Analyze performance trend
        performance_trend = self._analyze_performance_trend(performances)

        return ProfileAnalysisResult(
            summoner_name=summoner_name,
            tier_info=tier_info,
            games_analyzed=len(performances),
            avg_stats=avg_stats,
            gap_analysis=gap_analysis,
            champion_pool=champion_pool,
            performance_trend=performance_trend,
        )

    def _extract_tier_info(self, rank_info: List[Dict[str, Any]]) -> TierInfo:
        """Extract tier information from rank data"""
        for rank in rank_info:
            if rank.get("queueType") == "RANKED_SOLO_5x5":
                return TierInfo(
                    tier=rank.get("tier", "UNRANKED"),
                    division=rank.get("rank", "I"),
                    lp=rank.get("leaguePoints", 0),
                    wins=rank.get("wins", 0),
                    losses=rank.get("losses", 0),
                )

        # Default if no ranked info
        return TierInfo(tier="UNRANKED", division="I", lp=0, wins=0, losses=0)

    def _calculate_average_stats(
        self, performances: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate average statistics from multiple games"""
        if not performances:
            return {}

        total_stats = {
            "kills": 0,
            "deaths": 0,
            "assists": 0,
            "cs": 0,
            "gold": 0,
            "vision_score": 0,
            "damage": 0,
            "duration": 0,
        }

        for perf in performances:
            total_stats["kills"] += perf.get("kills", 0)
            total_stats["deaths"] += perf.get("deaths", 0)
            total_stats["assists"] += perf.get("assists", 0)
            total_stats["cs"] += perf.get("total_cs", 0)
            total_stats["gold"] += perf.get("gold", 0)
            total_stats["vision_score"] += perf.get("vision_score", 0)
            total_stats["damage"] += perf.get("damage_dealt", 0)
            total_stats["duration"] += perf.get("game_duration", 0)

        num_games = len(performances)

        avg_kills = total_stats["kills"] / num_games
        avg_deaths = total_stats["deaths"] / num_games
        avg_assists = total_stats["assists"] / num_games

        return {
            "avg_kills": round(avg_kills, 2),
            "avg_deaths": round(avg_deaths, 2),
            "avg_assists": round(avg_assists, 2),
            "avg_kda": self.gap_analyzer.calculate_kda(
                total_stats["kills"], total_stats["deaths"], total_stats["assists"]
            ),
            "avg_cs": round(total_stats["cs"] / num_games, 2),
            "avg_cs_per_min": round(
                (total_stats["cs"] / (total_stats["duration"] / 60)), 2
            )
            if total_stats["duration"] > 0
            else 0,
            "avg_gold": round(total_stats["gold"] / num_games, 2),
            "avg_vision_score": round(total_stats["vision_score"] / num_games, 2),
            "avg_damage": round(total_stats["damage"] / num_games, 2),
        }

    def _extract_champion_pool(
        self, performances: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract most played champions from performances"""
        champion_stats = {}

        for perf in performances:
            champ = perf.get("champion_name", "Unknown")
            if champ not in champion_stats:
                champion_stats[champ] = {"games": 0, "wins": 0}

            champion_stats[champ]["games"] += 1
            if perf.get("win", False):
                champion_stats[champ]["wins"] += 1

        champion_pool = []
        for champ, stats in champion_stats.items():
            win_rate = (stats["wins"] / stats["games"] * 100) if stats["games"] > 0 else 0
            champion_pool.append(
                {
                    "champion": champ,
                    "games": stats["games"],
                    "wins": stats["wins"],
                    "win_rate": round(win_rate, 2),
                }
            )

        # Sort by number of games played
        champion_pool.sort(key=lambda x: x["games"], reverse=True)
        return champion_pool[:5]  # Top 5 champions

    def _analyze_performance_trend(
        self, performances: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze performance trend over recent games"""
        if len(performances) < 2:
            return {"trend": "insufficient_data"}

        # Calculate win rate for first half vs second half
        mid_point = len(performances) // 2
        recent_half = performances[:mid_point]
        older_half = performances[mid_point:]

        recent_wins = sum(1 for p in recent_half if p.get("win", False))
        older_wins = sum(1 for p in older_half if p.get("win", False))

        recent_wr = (recent_wins / len(recent_half) * 100) if recent_half else 0
        older_wr = (older_wins / len(older_half) * 100) if older_half else 0

        trend = "stable"
        if recent_wr > older_wr + 10:
            trend = "improving"
        elif recent_wr < older_wr - 10:
            trend = "declining"

        return {
            "trend": trend,
            "recent_win_rate": round(recent_wr, 2),
            "older_win_rate": round(older_wr, 2),
            "total_games": len(performances),
            "total_wins": recent_wins + older_wins,
        }

    def _extract_key_moments(
        self, match_data: Dict[str, Any], puuid: str
    ) -> List[Dict[str, Any]]:
        """Extract key moments from match (simplified version)"""
        # This is a simplified version
        # In production, you'd analyze timeline data for kills, objectives, etc.
        key_moments = []

        participants = match_data.get("info", {}).get("participants", [])
        for participant in participants:
            if participant.get("puuid") == puuid:
                # Add pentakill if exists
                if participant.get("pentaKills", 0) > 0:
                    key_moments.append(
                        {"type": "pentakill", "count": participant["pentaKills"]}
                    )

                # Add quadrakill
                if participant.get("quadraKills", 0) > 0:
                    key_moments.append(
                        {"type": "quadrakill", "count": participant["quadraKills"]}
                    )

                # Add triple kill
                if participant.get("tripleKills", 0) > 0:
                    key_moments.append(
                        {"type": "triple_kill", "count": participant["tripleKills"]}
                    )

                break

        return key_moments
