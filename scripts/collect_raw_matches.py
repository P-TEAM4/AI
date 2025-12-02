"""
티어별 Raw 매치 데이터 수집 스크립트

- 티어/디비전별로 소환사를 가져오고
- 각 소환자의 최근 경기 ID를 모은 뒤
- 매치 전체 JSON을 tier별 폴더에 저장

저장 구조 예시:
data/raw/
  ├── IRON/
  │    ├── KR_1234.json
  │    ├── KR_5678.json
  ├── BRONZE/
  ├── SILVER/
  ...
"""

import os
import time
import json
from typing import Dict, List

import requests
from dotenv import load_dotenv

load_dotenv()

RIOT_API_KEY = os.getenv("RIOT_API_KEY")
if not RIOT_API_KEY:
    raise ValueError("RIOT_API_KEY not found in environment variables")

HEADERS = {"X-Riot-Token": RIOT_API_KEY}


class RawMatchCollector:
    """티어별 Raw 매치 데이터 수집기"""

    def __init__(self, rate_limit_delay: float = 1.2, save_root: str = "data/raw"):
        self.rate_limit_delay = rate_limit_delay
        self.save_root = save_root

    # ---------------- Riot API helpers ---------------- #

    def _get(self, url: str, params: Dict | None = None) -> Dict | List:
        """간단한 GET 래퍼 + rate limit용 sleep"""
        time.sleep(self.rate_limit_delay)
        res = requests.get(url, headers=HEADERS, params=params)
        res.raise_for_status()
        return res.json()

    def get_ranked_players_by_tier(
        self, tier: str, division: str, page: int = 1
    ) -> List[Dict]:
        """
        특정 티어/디비전의 랭크 플레이어 리스트 가져오기

        tier: IRON, BRONZE, SILVER, GOLD, PLATINUM, EMERALD, DIAMOND, MASTER, GRANDMASTER, CHALLENGER ...
        division: I, II, III, IV (MASTER 이상에서는 사용되지 않음)
        """
        try:
            # MASTER / GRANDMASTER / CHALLENGER 는 단일 리그 엔드포인트 사용 (entries 필드에 플레이어 리스트가 들어있음)
            if tier == "CHALLENGER":
                url = "https://kr.api.riotgames.com/lol/league/v4/challengerleagues/by-queue/RANKED_SOLO_5x5"
                params = {"api_key": RIOT_API_KEY}
                data = self._get(url, params=params)
                entries = data.get("entries", [])
                return entries

            elif tier == "GRANDMASTER":
                url = "https://kr.api.riotgames.com/lol/league/v4/grandmasterleagues/by-queue/RANKED_SOLO_5x5"
                params = {"api_key": RIOT_API_KEY}
                data = self._get(url, params=params)
                entries = data.get("entries", [])
                return entries

            elif tier == "MASTER":
                url = "https://kr.api.riotgames.com/lol/league/v4/masterleagues/by-queue/RANKED_SOLO_5x5"
                params = {"api_key": RIOT_API_KEY}
                data = self._get(url, params=params)
                entries = data.get("entries", [])
                return entries

            # 그 외 티어는 /entries 엔드포인트 사용 (리스트 반환)
            else:
                url = f"https://kr.api.riotgames.com/lol/league/v4/entries/RANKED_SOLO_5x5/{tier}/{division}"
                params = {"page": page, "api_key": RIOT_API_KEY}
                data = self._get(url, params=params)
                return data

        except Exception as e:
            print(f"[ERROR] get_ranked_players_by_tier {tier} {division} page {page}: {e}")
            return []

    def get_match_ids_by_puuid(self, puuid: str, count: int = 20) -> List[str]:
        """플레이어의 최근 경기 ID 리스트 가져오기 (솔랭만 queue=420)"""
        url = (
            f"https://asia.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids"
        )
        params = {"start": 0, "count": count, "queue": 420}
        try:
            data = self._get(url, params=params)
            return data
        except Exception as e:
            print(f"[ERROR] get_match_ids_by_puuid {puuid}: {e}")
            return []

    def get_match_details(self, match_id: str) -> Dict:
        """경기 전체 JSON 가져오기"""
        url = f"https://asia.api.riotgames.com/lol/match/v5/matches/{match_id}"
        try:
            data = self._get(url)
            return data
        except Exception as e:
            print(f"[ERROR] get_match_details {match_id}: {e}")
            return {}

    # ---------------- 저장 관련 ---------------- #

    def save_match_json(self, match_data: Dict, tier: str):
        """
        매치 JSON을 data/raw/<tier>/<matchId>.json 로 저장
        이미 존재하면 스킵
        """
        if not match_data:
            return

        metadata = match_data.get("metadata", {})
        match_id = metadata.get("matchId")
        if not match_id:
            return

        tier_dir = os.path.join(self.save_root, tier)
        os.makedirs(tier_dir, exist_ok=True)

        path = os.path.join(tier_dir, f"{match_id}.json")
        if os.path.exists(path):
            # 중복 저장 방지
            print(f"  [SKIP] {match_id} already exists")
            return

        with open(path, "w", encoding="utf-8") as f:
            json.dump(match_data, f, ensure_ascii=False, indent=2)

        print(f"  [SAVED] {path}")

    # ---------------- 메인 로직 ---------------- #

    def collect_for_tier(
        self,
        tier: str,
        division: str = "I",
        target_matches: int = 300,
        max_players: int = 80,
        matches_per_player: int = 10,
    ):
        """
        한 티어에 대해 Raw 매치를 일정 개수 이상 수집.

        target_matches: 최소한 모으고 싶은 매치 수
        max_players: 최대 스캔할 소환사 수
        matches_per_player: 소환자당 가져올 최근 경기 수
        """
        print("\n" + "=" * 60)
        print(f"Collect raw matches for {tier} {division}")
        print("=" * 60)

        # 1) 티어에서 소환사 목록 모으기 (여러 page)
        players: List[Dict] = []
        page = 1
        while len(players) < max_players:
            page_players = self.get_ranked_players_by_tier(tier, division, page)
            if not page_players:
                break
            players.extend(page_players)
            print(f"  [INFO] fetched {len(players)} players so far...")
            page += 1

        players = players[:max_players]
        print(f"  [INFO] total players to scan: {len(players)}")

        collected_matches = 0

        # 2 각 소환사에 대해 puuid -> matchIds -> match detail -> 저장
        for idx, player in enumerate(players, start=1):
            if collected_matches >= target_matches:
                break

            summoner_name = player.get("summonerName", player.get("leagueId", "Unknown"))
            puuid = player.get("puuid")

            print(f"\n[PLAYER {idx}/{len(players)}] {summoner_name}")

            if not puuid:
                continue

            match_ids = self.get_match_ids_by_puuid(puuid, count=matches_per_player)
            print(f"  [INFO] got {len(match_ids)} matchIds")

            for mid in match_ids:
                if collected_matches >= target_matches:
                    break

                match_data = self.get_match_details(mid)
                if not match_data:
                    continue

                self.save_match_json(match_data, tier)
                collected_matches += 1

        print(f"\n[Done] {tier} {division}: collected {collected_matches} matches\n")


def main():
    collector = RawMatchCollector(rate_limit_delay=1.2, save_root="data/raw")

    # 필요한 티어만 골라서 추가하면 됨
    tiers_to_collect = [
        ("MASTER", ""),
        ("GRANDMASTER", ""),
        ("CHALLENGER", ""),
    ]

    for tier, division in tiers_to_collect:
        collector.collect_for_tier(
            tier=tier,
            division=division,
            target_matches=300,      # 티어당 최소 매치 수
            max_players=80,          # 최대 소환사 수
            matches_per_player=10,   # 소환사당 최근 경기 수
        )


if __name__ == "__main__":
    main()