"""Riot Games API client for fetching match and player data"""

import requests
from typing import Dict, List, Optional, Any
from src.config.settings import settings


class RiotAPIClient:
    """Client for interacting with Riot Games API"""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Riot API client

        Args:
            api_key: Riot Games API key (defaults to settings)
        """
        self.api_key = api_key or settings.RIOT_API_KEY
        self.base_url = settings.RIOT_API_BASE_URL
        self.region_url = settings.RIOT_API_REGION_URL
        self.headers = {"X-Riot-Token": self.api_key}

    def _make_request(self, url: str) -> Dict[str, Any]:
        """
        Make HTTP request to Riot API

        Args:
            url: Full URL to request

        Returns:
            JSON response as dictionary

        Raises:
            requests.HTTPError: If request fails
        """
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def get_account_by_riot_id(
        self, game_name: str, tag_line: str, region: str = "asia"
    ) -> Dict[str, Any]:
        """
        Get account information by Riot ID (game name + tag line)

        Args:
            game_name: Summoner name
            tag_line: Tag line (e.g., KR1)
            region: Region (default: asia for KR)

        Returns:
            Account information including PUUID
        """
        url = f"https://{region}.api.riotgames.com/riot/account/v1/accounts/by-riot-id/{game_name}/{tag_line}"
        return self._make_request(url)

    def get_summoner_by_puuid(self, puuid: str, region: str = "kr") -> Dict[str, Any]:
        """
        Get summoner information by PUUID

        Args:
            puuid: Player UUID
            region: Platform region (default: kr)

        Returns:
            Summoner information
        """
        url = f"https://{region}.api.riotgames.com/lol/summoner/v4/summoners/by-puuid/{puuid}"
        return self._make_request(url)

    def get_rank_info(self, summoner_id: str, region: str = "kr") -> List[Dict[str, Any]]:
        """
        Get ranked information for a summoner

        Args:
            summoner_id: Summoner ID
            region: Platform region (default: kr)

        Returns:
            List of rank entries (Solo/Duo, Flex, etc.)
        """
        url = f"https://{region}.api.riotgames.com/lol/league/v4/entries/by-summoner/{summoner_id}"
        return self._make_request(url)

    def get_match_ids(
        self,
        puuid: str,
        start: int = 0,
        count: int = 20,
        queue: Optional[int] = None,
        region: str = "asia",
    ) -> List[str]:
        """
        Get list of match IDs for a player

        Args:
            puuid: Player UUID
            start: Start index
            count: Number of matches to retrieve
            queue: Queue type filter (420 for Ranked Solo/Duo)
            region: Region (default: asia)

        Returns:
            List of match IDs
        """
        url = f"https://{region}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids"
        params = {"start": start, "count": count}
        if queue:
            params["queue"] = queue

        response = requests.get(url, headers=self.headers, params=params)
        response.raise_for_status()
        return response.json()

    def get_match_details(self, match_id: str, region: str = "asia") -> Dict[str, Any]:
        """
        Get detailed match information

        Args:
            match_id: Match ID
            region: Region (default: asia)

        Returns:
            Match details including all player statistics
        """
        url = f"https://{region}.api.riotgames.com/lol/match/v5/matches/{match_id}"
        return self._make_request(url)

    def get_match_timeline(self, match_id: str, region: str = "asia") -> Dict[str, Any]:
        """
        Get match timeline with events

        Args:
            match_id: Match ID
            region: Region (default: asia)

        Returns:
            Timeline data with kills, objectives, gold, etc.
        """
        url = f"https://{region}.api.riotgames.com/lol/match/v5/matches/{match_id}/timeline"
        return self._make_request(url)

    def extract_player_stats_from_match(
        self, match_data: Dict[str, Any], puuid: str
    ) -> Optional[Dict[str, Any]]:
        """
        Extract specific player's statistics from match data

        Args:
            match_data: Complete match data
            puuid: Player UUID to extract stats for

        Returns:
            Player statistics or None if not found
        """
        participants = match_data.get("info", {}).get("participants", [])

        # Find player and their team
        player_participant = None
        team_id = None

        for participant in participants:
            if participant.get("puuid") == puuid:
                player_participant = participant
                team_id = participant.get("teamId")
                break

        if not player_participant:
            return None

        # Calculate team damage total for damage share
        team_damage_total = sum(
            p.get("totalDamageDealtToChampions", 0)
            for p in participants
            if p.get("teamId") == team_id
        )

        player_damage = player_participant.get("totalDamageDealtToChampions", 0)
        damage_share = player_damage / team_damage_total if team_damage_total > 0 else 0

        return {
            "champion_name": player_participant.get("championName"),
            "kills": player_participant.get("kills", 0),
            "deaths": player_participant.get("deaths", 0),
            "assists": player_participant.get("assists", 0),
            "total_cs": player_participant.get("totalMinionsKilled", 0)
            + player_participant.get("neutralMinionsKilled", 0),
            "gold": player_participant.get("goldEarned", 0),
            "vision_score": player_participant.get("visionScore", 0),
            "damage_dealt": player_damage,
            "damage_share": damage_share,
            "position": player_participant.get("teamPosition"),
            "win": player_participant.get("win", False),
            "game_duration": match_data.get("info", {}).get("gameDuration", 0),
        }

    def get_player_recent_performance(
        self, game_name: str, tag_line: str, num_games: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Get recent match performance for a player

        Args:
            game_name: Summoner name
            tag_line: Tag line
            num_games: Number of recent games to fetch

        Returns:
            List of match statistics
        """
        # Get PUUID
        account = self.get_account_by_riot_id(game_name, tag_line)
        puuid = account.get("puuid")

        if not puuid:
            return []

        # Get recent match IDs (Ranked Solo/Duo only - queue 420)
        match_ids = self.get_match_ids(puuid, count=num_games, queue=420)

        # Fetch match details
        performances = []
        for match_id in match_ids:
            try:
                match_data = self.get_match_details(match_id)
                player_stats = self.extract_player_stats_from_match(match_data, puuid)
                if player_stats:
                    player_stats["match_id"] = match_id
                    performances.append(player_stats)
            except Exception as e:
                print(f"Error fetching match {match_id}: {e}")
                continue

        return performances
