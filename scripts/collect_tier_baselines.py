"""
티어별 Baseline 데이터 수집 스크립트

Rule-based 모델에서 사용할 티어별 평균 통계를 수집합니다.
"""

import asyncio
import requests
import time
from typing import Dict, List
from collections import defaultdict
import pandas as pd
import json
from dotenv import load_dotenv
import os

load_dotenv()

RIOT_API_KEY = os.getenv("RIOT_API_KEY")
if not RIOT_API_KEY:
    raise ValueError("RIOT_API_KEY not found in environment variables")

HEADERS = {"X-Riot-Token": RIOT_API_KEY}


class TierBaselineCollector:
    """티어별 Baseline 데이터 수집기"""

    def __init__(self):
        self.tier_data = defaultdict(list)
        self.rate_limit_delay = 1.2  # 초당 1회 미만으로 호출

    def get_ranked_players_by_tier(
        self, tier: str, division: str, page: int = 1
    ) -> List[Dict]:
        """
        특정 티어/디비전의 랭크 플레이어 리스트 가져오기

        Args:
            tier: IRON, BRONZE, SILVER, GOLD, PLATINUM, EMERALD, DIAMOND
            division: I, II, III, IV
            page: 페이지 번호 (1부터 시작)

        Returns:
            플레이어 정보 리스트
        """
        url = f"https://kr.api.riotgames.com/lol/league/v4/entries/RANKED_SOLO_5x5/{tier}/{division}"

        params = {"page": page}

        try:
            response = requests.get(url, headers=HEADERS, params=params)
            response.raise_for_status()
            time.sleep(self.rate_limit_delay)
            return response.json()
        except Exception as e:
            print(f"Error fetching {tier} {division} page {page}: {e}")
            return []

    def get_player_recent_matches(self, puuid: str, count: int = 20) -> List[str]:
        """플레이어의 최근 경기 ID 가져오기"""
        url = f"https://asia.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids"

        params = {"start": 0, "count": count, "queue": 420}  # Ranked Solo/Duo

        try:
            response = requests.get(url, headers=HEADERS, params=params)
            response.raise_for_status()
            time.sleep(self.rate_limit_delay)
            return response.json()
        except Exception as e:
            print(f"Error fetching matches for {puuid}: {e}")
            return []

    def get_match_details(self, match_id: str) -> Dict:
        """경기 상세 정보 가져오기"""
        url = f"https://asia.api.riotgames.com/lol/match/v5/matches/{match_id}"

        try:
            response = requests.get(url, headers=HEADERS)
            response.raise_for_status()
            time.sleep(self.rate_limit_delay)
            return response.json()
        except Exception as e:
            print(f"Error fetching match {match_id}: {e}")
            return {}

    def extract_player_stats(
        self, match_data: Dict, puuid: str
    ) -> Dict:
        """경기 데이터에서 플레이어 통계 추출"""
        if not match_data or "info" not in match_data:
            return {}

        participants = match_data["info"].get("participants", [])
        game_duration = match_data["info"].get("gameDuration", 1)

        for participant in participants:
            if participant.get("puuid") == puuid:
                kills = participant.get("kills", 0)
                deaths = participant.get("deaths", 0)
                assists = participant.get("assists", 0)

                # KDA 계산
                kda = (kills + assists) / max(deaths, 1)

                # CS/min 계산
                total_cs = participant.get("totalMinionsKilled", 0) + participant.get(
                    "neutralMinionsKilled", 0
                )
                cs_per_min = total_cs / (game_duration / 60) if game_duration > 0 else 0

                # Damage Share 계산 (팀 전체 데미지 대비)
                team_id = participant.get("teamId")
                total_team_damage = sum(
                    p.get("totalDamageDealtToChampions", 0)
                    for p in participants
                    if p.get("teamId") == team_id
                )
                damage_dealt = participant.get("totalDamageDealtToChampions", 0)
                damage_share = (
                    damage_dealt / total_team_damage if total_team_damage > 0 else 0
                )

                return {
                    "kda": round(kda, 2),
                    "cs_per_min": round(cs_per_min, 2),
                    "gold": participant.get("goldEarned", 0),
                    "vision_score": participant.get("visionScore", 0),
                    "damage_share": round(damage_share, 3),
                    "win": participant.get("win", False),
                }

        return {}

    def collect_tier_data(
        self, tier: str, division: str = "I", num_players: int = 50
    ):
        """
        특정 티어의 데이터 수집

        Args:
            tier: 티어 이름
            division: 디비전
            num_players: 수집할 플레이어 수
        """
        print(f"\n{'='*60}")
        print(f"Collecting data for {tier} {division}")
        print(f"{'='*60}\n")

        # 1. 해당 티어의 플레이어 리스트 가져오기
        players = []
        page = 1

        while len(players) < num_players:
            page_players = self.get_ranked_players_by_tier(tier, division, page)
            if not page_players:
                break

            players.extend(page_players)
            page += 1
            print(f"Fetched {len(players)} players so far...")

        players = players[:num_players]
        print(f"Total players to analyze: {len(players)}\n")

        # 2. 각 플레이어의 최근 경기 분석
        tier_stats = []

        for idx, player in enumerate(players, 1):
            summoner_id = player.get("summonerId")
            summoner_name = player.get("summonerName", "Unknown")

            print(f"[{idx}/{len(players)}] Analyzing {summoner_name}...")

            # PUUID 가져오기
            try:
                summoner_url = f"https://kr.api.riotgames.com/lol/summoner/v4/summoners/{summoner_id}"
                response = requests.get(summoner_url, headers=HEADERS)
                response.raise_for_status()
                time.sleep(self.rate_limit_delay)

                puuid = response.json().get("puuid")

                if not puuid:
                    continue

                # 최근 경기 ID 가져오기
                match_ids = self.get_player_recent_matches(puuid, count=10)

                # 각 경기 분석
                for match_id in match_ids[:5]:  # 최근 5경기만
                    match_data = self.get_match_details(match_id)
                    stats = self.extract_player_stats(match_data, puuid)

                    if stats:
                        stats["tier"] = tier
                        stats["division"] = division
                        tier_stats.append(stats)

            except Exception as e:
                print(f"Error analyzing {summoner_name}: {e}")
                continue

        self.tier_data[tier].extend(tier_stats)
        print(f"\nCollected {len(tier_stats)} match stats for {tier}\n")

    def calculate_tier_averages(self) -> Dict[str, Dict[str, float]]:
        """수집된 데이터로 티어별 평균 계산"""
        tier_averages = {}

        for tier, stats_list in self.tier_data.items():
            if not stats_list:
                continue

            df = pd.DataFrame(stats_list)

            averages = {
                "avg_kda": round(df["kda"].mean(), 2),
                "avg_cs_per_min": round(df["cs_per_min"].mean(), 2),
                "avg_gold": int(df["gold"].mean()),
                "avg_vision_score": int(df["vision_score"].mean()),
                "avg_damage_share": round(df["damage_share"].mean(), 3),
                "sample_size": len(df),
                "win_rate": round(df["win"].mean() * 100, 2),
            }

            tier_averages[tier] = averages

        return tier_averages

    def save_results(self, tier_averages: Dict, filename: str = "data/tier_baselines.json"):
        """결과를 JSON 파일로 저장"""
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(tier_averages, f, indent=2, ensure_ascii=False)

        print(f"\nResults saved to {filename}")

    def display_results(self, tier_averages: Dict):
        """결과를 테이블 형태로 출력"""
        print("\n" + "=" * 100)
        print("TIER BASELINE STATISTICS")
        print("=" * 100)

        df = pd.DataFrame(tier_averages).T
        print(df.to_string())
        print("=" * 100 + "\n")


def main():
    """메인 실행 함수"""
    collector = TierBaselineCollector()

    # 수집할 티어 목록 (Master 이상은 제외 - 인원이 적음)
    tiers_to_collect = [
        ("IRON", "I", 30),
        ("BRONZE", "I", 30),
        ("SILVER", "I", 40),
        ("GOLD", "I", 50),
        ("PLATINUM", "I", 50),
        ("EMERALD", "I", 40),
        ("DIAMOND", "I", 40),
    ]

    # Master 이상은 수동으로 설정하거나 적은 샘플로 수집
    manual_high_tiers = {
        "MASTER": {
            "avg_kda": 4.0,
            "avg_cs_per_min": 8.0,
            "avg_gold": 17000,
            "avg_vision_score": 40,
            "avg_damage_share": 0.25,
        },
        "GRANDMASTER": {
            "avg_kda": 4.5,
            "avg_cs_per_min": 8.5,
            "avg_gold": 18000,
            "avg_vision_score": 45,
            "avg_damage_share": 0.26,
        },
        "CHALLENGER": {
            "avg_kda": 5.0,
            "avg_cs_per_min": 9.0,
            "avg_gold": 19000,
            "avg_vision_score": 50,
            "avg_damage_share": 0.27,
        },
    }

    # 데이터 수집
    for tier, division, num_players in tiers_to_collect:
        try:
            collector.collect_tier_data(tier, division, num_players)
        except Exception as e:
            print(f"Error collecting {tier} {division}: {e}")
            continue

    # 평균 계산
    tier_averages = collector.calculate_tier_averages()

    # 고티어 수동 데이터 추가
    tier_averages.update(manual_high_tiers)

    # 결과 출력 및 저장
    collector.display_results(tier_averages)
    collector.save_results(tier_averages)

    # Python 코드 생성
    print("\nGenerated Python code for settings.py:")
    print("\n" + "=" * 80)
    print("BASELINES: Dict[str, Dict[str, float]] = {")
    for tier, stats in tier_averages.items():
        print(f'    "{tier}": {{')
        for key, value in stats.items():
            if key not in ["sample_size", "win_rate"]:
                print(f'        "{key}": {value},')
        print("    },")
    print("}")
    print("=" * 80)


if __name__ == "__main__":
    main()
