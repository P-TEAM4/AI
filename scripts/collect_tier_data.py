"""
티어별 매치 데이터 수집 스크립트

Riot API를 통해 티어별로  매치 데이터를 수집합니다.
Rate Limit을 준수하며 안전하게 데이터를 수집하고 저장합니다.

사용법:
    python scripts/collect_tier_data.py
"""

import sys
from pathlib import Path
import time
import json
from typing import Dict, List, Optional
from datetime import datetime
import random
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services.riot_api import RiotAPIClient
from dotenv import load_dotenv
import os

# .env 파일을 프로젝트 루트에서 로드
project_root = Path(__file__).parent.parent
load_dotenv(project_root / ".env")


class TierDataCollector:
    """티어별 매치 데이터 수집기"""

    def __init__(self, api_key: str, region: str = "kr"):
        """
        초기화

        Args:
            api_key: Riot API 키
            region: 지역 코드 (기본값: kr)
        """
        self.api_key = api_key
        self.region = region
        self.data_dir = Path("data/tier_collections")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # 진행 상황 저장 디렉토리
        self.progress_dir = Path("data/collection_progress")
        self.progress_dir.mkdir(parents=True, exist_ok=True)

        # Rate Limit 설정 (개발용 API Key 기준 - 안정적 설정)
        self.requests_per_second = 10  # 초당 10개 (20개 중 50%)
        self.requests_per_2min = 90    # 2분당 90개 (100개 중 90%)
        self.request_interval = 0.6    # 0.6초 간격 (분당 100개 = 2분당 200개이지만 90개 제한)

        # 2분 슬라이딩 윈도우를 위한 타임스탬프 큐
        self.request_timestamps = []

        # 429 에러 발생 시 재시도 설정
        self.max_retries = 3
        self.retry_delay = 5  # 5초 대기 후 재시도

        # 401 에러 플래그 (API 키 만료)
        self.api_key_expired = False

        # Session 생성
        import requests
        self.session = requests.Session()
        self.session.headers.update({"X-Riot-Token": self.api_key})

        # 현재 게임 버전 조회
        self.game_version = self.get_current_game_version()

    def get_current_game_version(self) -> str:
        """
        현재 게임 버전 조회

        Returns:
            게임 버전 (예: "15.23")
        """
        try:
            # Riot API에서 현재 버전 정보 가져오기
            url = "https://ddragon.leagueoflegends.com/api/versions.json"
            response = self.session.get(url)
            response.raise_for_status()
            versions = response.json()

            # 최신 버전 (예: "15.23.1" -> "15.23")
            latest_version = versions[0]
            major_minor = ".".join(latest_version.split(".")[:2])

            print(f"현재 게임 버전: {major_minor} (Full: {latest_version})")
            return major_minor

        except Exception as e:
            print(f"게임 버전 조회 실패: {e}")
            print(f"기본 버전 사용: unknown")
            return "unknown"

    def wait_for_rate_limit(self):
        """Rate Limit 대기 (슬라이딩 윈도우 방식)"""
        current_time = time.time()

        # 2분(120초) 이전의 타임스탬프 제거
        self.request_timestamps = [
            ts for ts in self.request_timestamps if current_time - ts < 120
        ]

        # 2분 윈도우 내 요청이 제한에 도달했는지 확인
        if len(self.request_timestamps) >= self.requests_per_2min:
            # 윈도우가 충분히 비워질 때까지 대기
            # 가장 오래된 요청부터 순차적으로 만료되도록 대기
            oldest_request = self.request_timestamps[0]
            wait_time = 120 - (current_time - oldest_request) + 0.5  # 0.5초 버퍼

            if wait_time > 0:
                print(f"   Rate limit 대기 중... {wait_time:.1f}초 (2분 윈도우: {len(self.request_timestamps)}/{self.requests_per_2min})")
                time.sleep(wait_time)

                # 대기 후 타임스탬프 정리
                current_time = time.time()
                self.request_timestamps = [
                    ts for ts in self.request_timestamps if current_time - ts < 120
                ]

        # 초당 요청 간격 대기
        time.sleep(self.request_interval)

        # 현재 요청 타임스탬프 기록
        self.request_timestamps.append(time.time())

    def get_summoners_by_tier(
        self, tier: str, division: str, page: int = 1
    ) -> List[Dict]:
        """
        특정 티어/디비전의 소환사 목록 조회

        Args:
            tier: 티어 (IRON, BRONZE, SILVER, GOLD, PLATINUM, EMERALD, DIAMOND, MASTER, GRANDMASTER, CHALLENGER)
            division: 디비전 (I, II, III, IV) - MASTER 이상은 무시됨
            page: 페이지 번호 (1부터 시작)

        Returns:
            소환사 목록
        """
        self.wait_for_rate_limit()

        tier_upper = tier.upper()
        queue = "RANKED_SOLO_5x5"

        try:
            # MASTER 이상은 디비전이 없음
            if tier_upper in ["MASTER", "GRANDMASTER", "CHALLENGER"]:
                url = f"https://{self.region}.api.riotgames.com/lol/league/v4/{tier_upper.lower()}leagues/by-queue/{queue}"
            else:
                url = f"https://{self.region}.api.riotgames.com/lol/league/v4/entries/{queue}/{tier_upper}/{division}"
                url += f"?page={page}"

            response = self.session.get(url)
            response.raise_for_status()

            data = response.json()

            # MASTER 이상은 entries 필드 안에 있음
            if tier_upper in ["MASTER", "GRANDMASTER", "CHALLENGER"]:
                return data.get("entries", [])
            else:
                return data

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                print(f"\n 401 에러: API 키가 만료되었거나 유효하지 않습니다.")
                print(f"   Riot Developer Portal에서 새 API 키를 발급받으세요.")
                self.api_key_expired = True
                return []
            else:
                print(f"   Error fetching summoners: {e}")
                return []
        except Exception as e:
            print(f"   Error fetching summoners: {e}")
            return []

    def get_match_ids_by_puuid(
        self, puuid: str, count: int = 20, start: int = 0
    ) -> List[str]:
        """
        PUUID로 매치 ID 목록 조회 (429 에러 재시도 포함)

        Args:
            puuid: 플레이어 PUUID
            count: 가져올 매치 수 (최대 100)
            start: 시작 인덱스

        Returns:
            매치 ID 목록
        """
        for attempt in range(self.max_retries):
            self.wait_for_rate_limit()

            try:
                url = f"https://asia.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids"
                params = {"start": start, "count": min(count, 100), "queue": 420}  # 솔로랭크만

                response = self.session.get(url, params=params)
                response.raise_for_status()
                match_ids = response.json()
                return match_ids

            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    retry_after = int(e.response.headers.get('Retry-After', 120))
                    print(f"   429 에러: {retry_after}초 대기 후 재시도...")
                    time.sleep(retry_after)
                    continue
                elif e.response.status_code == 401:
                    print(f"\n 401 에러: API 키가 만료되었거나 유효하지 않습니다.")
                    self.api_key_expired = True
                    return []
                else:
                    print(f"   Error fetching match IDs: {e}")
                    return []
            except Exception as e:
                print(f"   Error fetching match IDs: {e}")
                return []

        return []

    def get_match_detail(self, match_id: str) -> Optional[Dict]:
        """
        매치 상세 정보 조회 (429 에러 재시도 포함)

        Args:
            match_id: 매치 ID

        Returns:
            매치 상세 정보
        """
        for attempt in range(self.max_retries):
            self.wait_for_rate_limit()

            try:
                url = f"https://asia.api.riotgames.com/lol/match/v5/matches/{match_id}"
                response = self.session.get(url)
                response.raise_for_status()
                match_data = response.json()
                return match_data

            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    # Rate limit 초과 - 대기 후 재시도
                    retry_after = int(e.response.headers.get('Retry-After', 120))
                    print(f"   429 에러: {retry_after}초 대기 후 재시도... (시도 {attempt + 1}/{self.max_retries})")
                    time.sleep(retry_after)
                    continue
                elif e.response.status_code == 401:
                    print(f"\n[ERROR] 401 에러: API 키가 만료되었거나 유효하지 않습니다.")
                    self.api_key_expired = True
                    return None
                else:
                    print(f"   Error fetching match detail: {e}")
                    return None
            except Exception as e:
                print(f"   Error fetching match detail: {e}")
                return None

        print(f"   매치 {match_id} 조회 실패: 최대 재시도 횟수 초과")
        return None

    def extract_player_data(self, match_data: Dict) -> List[Dict]:
        """
        매치 데이터에서 플레이어별 데이터 추출

        Args:
            match_data: 매치 상세 정보

        Returns:
            플레이어 데이터 리스트
        """
        if not match_data or "info" not in match_data:
            return []

        info = match_data["info"]
        game_duration = info.get("gameDuration", 0)
        game_minutes = game_duration / 60.0

        if game_minutes < 10 or game_minutes > 60:
            return []  # 비정상 게임 제외

        players_data = []

        for participant in info.get("participants", []):
            # KDA 계산
            kills = participant.get("kills", 0)
            deaths = participant.get("deaths", 0)
            assists = participant.get("assists", 0)
            kda = (kills + assists) / deaths if deaths > 0 else (kills + assists)

            # CS 계산
            total_minions_killed = participant.get("totalMinionsKilled", 0)
            neutral_minions_killed = participant.get("neutralMinionsKilled", 0)
            total_cs = total_minions_killed + neutral_minions_killed
            cs_per_min = total_cs / game_minutes if game_minutes > 0 else 0

            # Gold
            gold_earned = participant.get("goldEarned", 0)
            gold_per_min = gold_earned / game_minutes if game_minutes > 0 else 0

            # Vision
            vision_score = participant.get("visionScore", 0)
            vision_score_per_min = (
                vision_score / game_minutes if game_minutes > 0 else 0
            )

            # Damage (팀별 계산 필요)
            damage = participant.get("totalDamageDealtToChampions", 0)

            player_data = {
                "match_id": match_data["metadata"]["matchId"],
                "puuid": participant.get("puuid"),
                "summoner_name": participant.get("summonerName"),
                "champion": participant.get("championName"),
                "tier": participant.get("tier", "UNKNOWN"),  # API에서 제공 안됨
                "kills": kills,
                "deaths": deaths,
                "assists": assists,
                "kda": round(kda, 2),
                "total_cs": total_cs,
                "cs_per_min": round(cs_per_min, 2),
                "gold_earned": gold_earned,
                "gold_per_min": round(gold_per_min, 2),
                "vision_score": vision_score,
                "vision_score_per_min": round(vision_score_per_min, 2),
                "total_damage_dealt_to_champions": damage,
                "damage_share": 0.0,  # 나중에 계산
                "win": participant.get("win", False),
                "game_duration": game_duration,
                "game_minutes": round(game_minutes, 2),
                "team_id": participant.get("teamId"),
                "position": participant.get("teamPosition"),
            }

            players_data.append(player_data)

        # 팀별 Damage Share 계산
        for team_id in [100, 200]:
            team_players = [p for p in players_data if p["team_id"] == team_id]
            team_damage = sum(p["total_damage_dealt_to_champions"] for p in team_players)

            if team_damage > 0:
                for player in team_players:
                    player["damage_share"] = round(
                        player["total_damage_dealt_to_champions"] / team_damage, 3
                    )

        return players_data

    def save_progress(self, tier: str, collected_match_ids: set, all_players_data: List[Dict]):
        """
        진행 상황 저장

        Args:
            tier: 티어
            collected_match_ids: 수집된 매치 ID 목록
            all_players_data: 수집된 플레이어 데이터
        """
        progress_file = self.progress_dir / f"{tier.lower()}_v{self.game_version}_progress.json"

        progress_data = {
            "tier": tier.upper(),
            "game_version": self.game_version,
            "last_updated": datetime.now().isoformat(),
            "collected_match_ids": list(collected_match_ids),
            "total_matches": len(collected_match_ids),
            "total_players": len(all_players_data),
            "player_data": all_players_data
        }

        with open(progress_file, "w", encoding="utf-8") as f:
            json.dump(progress_data, f, indent=2, ensure_ascii=False)

    def load_progress(self, tier: str) -> tuple:
        """
        진행 상황 로드

        Args:
            tier: 티어

        Returns:
            (collected_match_ids, all_players_data) 튜플
        """
        progress_file = self.progress_dir / f"{tier.lower()}_v{self.game_version}_progress.json"

        if not progress_file.exists():
            return set(), []

        try:
            with open(progress_file, "r", encoding="utf-8") as f:
                progress_data = json.load(f)

            collected_match_ids = set(progress_data.get("collected_match_ids", []))
            all_players_data = progress_data.get("player_data", [])

            print(f"   이전 진행 상황 로드: {len(collected_match_ids)}개 매치, {len(all_players_data)}개 플레이어 데이터")

            return collected_match_ids, all_players_data
        except Exception as e:
            print(f"   진행 상황 로드 실패: {e}")
            return set(), []

    def delete_progress(self, tier: str):
        """
        진행 상황 파일 삭제

        Args:
            tier: 티어
        """
        progress_file = self.progress_dir / f"{tier.lower()}_v{self.game_version}_progress.json"
        if progress_file.exists():
            progress_file.unlink()

    def collect_tier_data(
        self, tier: str, target_matches: int = 3000, matches_per_summoner: int = 10, resume: bool = True
    ) -> List[Dict]:
        """
        특정 티어의 매치 데이터 수집

        Args:
            tier: 티어 (IRON, BRONZE, SILVER, GOLD, PLATINUM, EMERALD, DIAMOND, MASTER, GRANDMASTER, CHALLENGER)
            target_matches: 목표 매치 수 (기본값: 3000개)
            matches_per_summoner: 소환사당 수집할 매치 수 (기본값: 10)
            resume: 이전 진행 상황에서 이어서 수집할지 여부 (기본값: True)

        Returns:
            수집된 플레이어 데이터 리스트
        """
        print(f"\n{'='*80}")
        print(f"{tier} 티어 데이터 수집 시작")
        print(f"목표: {target_matches:,}개 매치 (소환사당 {matches_per_summoner}경기)")
        print(f"{'='*80}")

        tier_upper = tier.upper()

        # 이전 진행 상황 로드
        if resume:
            collected_match_ids, all_players_data = self.load_progress(tier_upper)
            if collected_match_ids:
                print(f"   이어서 수집 시작: 현재 {len(collected_match_ids)}/{target_matches} 매치")
        else:
            collected_match_ids = set()
            all_players_data = []

        # 디비전 목록 (MASTER 이상은 디비전 없음)
        if tier_upper in ["MASTER", "GRANDMASTER", "CHALLENGER"]:
            divisions = [None]
        else:
            divisions = ["I", "II", "III", "IV"]

        start_time = time.time()
        summoner_count = 0  # 수집한 소환사 수 (통계용)

        for division in divisions:
            if len(collected_match_ids) >= target_matches:  # 매치 수 기준
                break

            division_str = division if division else ""
            print(f"\n[{tier_upper} {division_str}] 소환사 목록 조회 중...")

            # 소환사 목록 조회 (필요한 만큼만)
            summoners = []
            remaining_matches = target_matches - len(collected_match_ids)
            needed_summoners = (remaining_matches // matches_per_summoner) * 2  # 여유있게 2배

            # 한 페이지당 약 205명이므로 필요한 페이지 수 계산
            pages_needed = (needed_summoners // 205) + 1
            max_pages = min(pages_needed, 5)  # 최대 5페이지

            for page in range(1, max_pages + 1):
                if len(collected_match_ids) >= target_matches:
                    break

                # 401 에러 체크
                if self.api_key_expired:
                    print(f"\n[SAVE] 진행 상황 저장 중...")
                    self.save_progress(tier_upper, collected_match_ids, all_players_data)
                    print(f"   저장 완료: {len(collected_match_ids)}개 매치")
                    print(f"\n[INFO] API 키를 갱신한 후 다시 실행하면 이어서 수집됩니다.")
                    return all_players_data

                page_summoners = self.get_summoners_by_tier(tier_upper, division or "I", page)
                if not page_summoners:
                    break
                summoners.extend(page_summoners)
                print(f"   페이지 {page}: {len(page_summoners)}명 조회")

                if tier_upper in ["MASTER", "GRANDMASTER", "CHALLENGER"]:
                    break  # MASTER 이상은 페이지 없음

            print(f"   총 {len(summoners)}명 소환사 발견")

            # 소환사별로 매치 수집
            random.shuffle(summoners)  # 랜덤하게 섞어서 다양성 확보

            for idx, summoner in enumerate(summoners):
                if len(collected_match_ids) >= target_matches:  # 매치 수 기준
                    break

                # 401 에러 체크
                if self.api_key_expired:
                    print(f"\n[SAVE] 진행 상황 저장 중...")
                    self.save_progress(tier_upper, collected_match_ids, all_players_data)
                    print(f"   저장 완료: {len(collected_match_ids)}개 매치")
                    print(f"\n[INFO] API 키를 갱신한 후 다시 실행하면 이어서 수집됩니다.")
                    return all_players_data

                summoner_id = summoner.get("summonerId")
                puuid = summoner.get("puuid")

                if not puuid:
                    continue

                # 매치 ID 조회 (소환사당 지정된 개수만큼만)
                match_ids = self.get_match_ids_by_puuid(puuid, count=matches_per_summoner)

                if not match_ids:
                    continue

                summoner_match_count = 0  # 이 소환사에게서 수집한 매치 수
                new_matches_this_summoner = 0  # 이 소환사에게서 새로 수집한 매치 수

                for match_id in match_ids:
                    if match_id in collected_match_ids:
                        continue

                    # 매치 상세 정보 조회
                    match_data = self.get_match_detail(match_id)
                    if not match_data:
                        continue

                    # 플레이어 데이터 추출
                    players_data = self.extract_player_data(match_data)

                    # 티어 정보 업데이트
                    for player_data in players_data:
                        player_data["tier"] = tier_upper

                    all_players_data.extend(players_data)
                    collected_match_ids.add(match_id)
                    summoner_match_count += 1
                    new_matches_this_summoner += 1

                # 소환사 처리 완료 표시 (매 10명마다 요약)
                if (idx + 1) % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = len(collected_match_ids) / elapsed if elapsed > 0 else 0
                    print(f"   [진행] 소환사: {idx + 1}/{len(summoners)} | 매치: {len(collected_match_ids)}/{target_matches} | 속도: {rate:.1f} 매치/초")

                    # 매치 10개마다 진행 상황 저장
                    if len(collected_match_ids) % 10 == 0:
                        self.save_progress(tier_upper, collected_match_ids, all_players_data)

                # 이 소환사로부터 매치를 수집했으면 카운트 증가 (통계용)
                if summoner_match_count > 0:
                    summoner_count += 1

                    # 진행 상황 출력 (100매치마다)
                    if len(collected_match_ids) % 100 == 0:
                        elapsed = time.time() - start_time
                        rate = len(collected_match_ids) / elapsed if elapsed > 0 else 0
                        remaining = target_matches - len(collected_match_ids)
                        eta = remaining / rate if rate > 0 else 0

                        print(
                            f"   진행: {len(collected_match_ids):,}/{target_matches:,} 매치 "
                            f"({len(collected_match_ids)/target_matches*100:.1f}%) | "
                            f"유저: {summoner_count:,}명 | "
                            f"속도: {rate:.2f} 매치/초 | "
                            f"ETA: {eta/60:.0f}분"
                        )

        elapsed = time.time() - start_time
        print(f"\n{tier_upper} 티어 수집 완료!")
        print(f"   총 소환사 수: {summoner_count:,}명")
        print(f"   총 매치 수: {len(collected_match_ids):,}")
        print(f"   총 플레이어 데이터: {len(all_players_data):,}")
        print(f"   평균 매치/소환사: {len(collected_match_ids)/summoner_count:.1f}경기" if summoner_count > 0 else "")
        print(f"   소요 시간: {elapsed/60:.1f}분")

        # 최종 진행 상황 저장
        self.save_progress(tier_upper, collected_match_ids, all_players_data)

        return all_players_data

    def is_tier_already_collected(self, tier: str) -> bool:
        """
        해당 티어의 현재 버전 데이터가 이미 수집되었는지 확인

        Args:
            tier: 티어

        Returns:
            이미 수집되었으면 True, 아니면 False
        """
        base_filename = f"{tier.lower()}_tier_v{self.game_version}.json"
        filepath = self.data_dir / base_filename
        return filepath.exists()

    def get_next_version_filename(self, tier: str) -> Path:
        """
        버전별 파일명 생성 (중복 시 -1, -2 등 추가)

        Args:
            tier: 티어

        Returns:
            파일 경로
        """
        base_filename = f"{tier.lower()}_tier_v{self.game_version}.json"
        filepath = self.data_dir / base_filename

        # 파일이 존재하지 않으면 그대로 반환
        if not filepath.exists():
            return filepath

        # 파일이 존재하면 -1, -2, -3... 형태로 번호 증가
        counter = 1
        while True:
            filename = f"{tier.lower()}_tier_v{self.game_version}-{counter}.json"
            filepath = self.data_dir / filename
            if not filepath.exists():
                return filepath
            counter += 1

    def save_tier_data(self, tier: str, data: List[Dict]):
        """
        티어별 데이터 저장 (CSV 형식으로 저장)

        Args:
            tier: 티어
            data: 플레이어 데이터 리스트
        """
        import pandas as pd

        # CSV 파일명 생성
        base_filename = f"{tier.lower()}_tier_v{self.game_version}.csv"
        filepath = self.data_dir / base_filename

        # 파일이 존재하면 -1, -2 추가
        if filepath.exists():
            counter = 1
            while True:
                filename = f"{tier.lower()}_tier_v{self.game_version}-{counter}.csv"
                filepath = self.data_dir / filename
                if not filepath.exists():
                    break
                counter += 1

        # DataFrame으로 변환 후 CSV 저장
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False, encoding="utf-8")

        print(f"\n데이터 저장 완료: {filepath}")
        print(f"   파일 크기: {filepath.stat().st_size / 1024 / 1024:.2f} MB")
        print(f"   총 플레이어: {len(data):,}")
        print(f"   총 매치: {len(set(p['match_id'] for p in data)):,}")

        # 최종 파일 저장 완료 시 진행 상황 파일 삭제
        self.delete_progress(tier)

    def collect_all_tiers(self, matches_per_tier: int = 3000, matches_per_summoner: int = 10):
        """
        모든 티어의 데이터 수집

        Args:
            matches_per_tier: 티어당 매치 수 (기본값: 3000개)
            matches_per_summoner: 소환사당 매치 수 (기본값: 10)
        """
        # 티어별 목표 매치 수 (인구 고려)
        tier_targets = {
            "IRON": matches_per_tier,
            "BRONZE": matches_per_tier,
            "SILVER": matches_per_tier,
            "GOLD": matches_per_tier,
            "PLATINUM": matches_per_tier,
            "EMERALD": matches_per_tier,
            "DIAMOND": matches_per_tier,
            "MASTER": min(matches_per_tier, 2000),      # 고티어는 인구가 적음
            "GRANDMASTER": min(matches_per_tier, 1000), # 더 적음
            "CHALLENGER": min(matches_per_tier, 500),   # 가장 적음
        }

        print("="*80)
        print("티어별 매치 데이터 수집 시작")
        print("="*80)
        print(f"기본 목표: {matches_per_tier:,}개 매치/티어")
        print(f"고티어 조정: Master {tier_targets['MASTER']:,}, GM {tier_targets['GRANDMASTER']:,}, Challenger {tier_targets['CHALLENGER']:,}")
        total_matches = sum(tier_targets.values())
        print(f"총 목표: {total_matches:,}개 매치")
        print(f"예상 소요 시간: Rate limit 고려 시 수 시간 ~ 1일")
        print("="*80)

        overall_start = time.time()

        for tier, target_matches in tier_targets.items():
            try:
                # 이미 수집된 티어인지 확인
                if self.is_tier_already_collected(tier):
                    print(f"\n[SKIP] {tier} 티어는 이미 v{self.game_version} 버전 데이터가 수집되어 있습니다. 건너뜁니다.")
                    continue

                data = self.collect_tier_data(tier, target_matches, matches_per_summoner)

                # 401 에러로 중단되었는지 확인
                if self.api_key_expired:
                    print(f"\n\n[WARN] API 키 만료로 수집 중단됨")
                    print(f"   현재까지 수집된 데이터는 진행 상황 파일에 저장되었습니다.")
                    print(f"   .env 파일에서 RIOT_API_KEY를 갱신한 후 다시 실행하세요.")
                    break

                self.save_tier_data(tier, data)

            except KeyboardInterrupt:
                print(f"\n\n사용자에 의해 중단됨")
                print(f"현재까지 수집된 데이터는 저장되었습니다.")
                break

            except Exception as e:
                print(f"\n{tier} 티어 수집 중 오류 발생: {e}")
                print(f"다음 티어로 계속 진행합니다...")
                continue

        overall_elapsed = time.time() - overall_start
        print(f"\n{'='*80}")
        print(f"전체 수집 완료!")
        print(f"총 소요 시간: {overall_elapsed/3600:.1f}시간")
        print(f"수집된 파일: {self.data_dir}")
        print(f"{'='*80}")


def main():
    """메인 함수"""
    api_key = os.getenv("RIOT_API_KEY")

    if not api_key:
        print("Error: RIOT_API_KEY가 .env 파일에 설정되지 않았습니다.")
        return

    collector = TierDataCollector(api_key, region="kr")

    # 티어별 데이터 수집
    # - matches_per_tier: 티어당 수집할 매치 수 (예: 3000개)
    # - matches_per_summoner: 소환사 1명당 수집할 경기 수 (예: 10)
    # - 매치 ID로 중복 자동 제거
    # - 고티어는 자동으로 인구 고려하여 조정됨 (Master: 2000개, GM: 1000개, Challenger: 500개)
    collector.collect_all_tiers(matches_per_tier=3000, matches_per_summoner=10)


if __name__ == "__main__":
    main()
