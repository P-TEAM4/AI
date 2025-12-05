"""
매치 타임라인 수집 스크립트

collect_tier_data.py로 수집된 매치들의 타임라인 전문을 가져옵니다.
ML 학습용 데이터로 사용됩니다.

저장 구조:
data/timelines/
  ├── v15.23/
  │    ├── KR_1234.json
  │    └── KR_5678.json
  └── v15.24/
       └── KR_9999.json

사용법:
    # 특정 버전의 타임라인 수집
    python scripts/timeline_fetcher.py --version v15.23

    # 모든 버전의 타임라인 수집
    python scripts/timeline_fetcher.py --all
"""

import os
import json
import time
from typing import Dict, List, Optional, Set
from pathlib import Path
from glob import glob

import requests
from dotenv import load_dotenv

load_dotenv()

RIOT_API_KEY = os.getenv("RIOT_API_KEY")
if not RIOT_API_KEY:
    raise ValueError("RIOT_API_KEY not found in environment variables")

HEADERS = {"X-Riot-Token": RIOT_API_KEY}


class TimelineFetcher:
    """타임라인 전문 수집기"""

    def __init__(
        self,
        cache_dir: str = "data/timelines",
        rate_limit_delay: float = 1.2,
    ):
        """
        초기화

        Args:
            cache_dir: 타임라인 저장 디렉토리
            rate_limit_delay: API 호출 간격 (초)
        """
        self.cache_dir = Path(cache_dir)
        self.rate_limit_delay = rate_limit_delay
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_timeline_path(self, match_id: str, version: str) -> Path:
        """타임라인 파일 경로 생성"""
        version_dir = self.cache_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)
        return version_dir / f"{match_id}.json"

    def _fetch_timeline_from_api(self, match_id: str) -> Optional[Dict]:
        """Riot API에서 타임라인 조회"""
        url = f"https://asia.api.riotgames.com/lol/match/v5/matches/{match_id}/timeline"

        try:
            time.sleep(self.rate_limit_delay)
            response = requests.get(url, headers=HEADERS)
            response.raise_for_status()
            return response.json()

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                print(f"  [ERROR] 매치를 찾을 수 없음: {match_id}")
            elif e.response.status_code == 401:
                print(f"  [ERROR] API 키가 유효하지 않습니다")
                raise  # 401은 중단
            elif e.response.status_code == 429:
                retry_after = int(e.response.headers.get("Retry-After", 120))
                print(f"  [WARN] Rate limit 초과, {retry_after}초 대기...")
                time.sleep(retry_after)
                # 재시도
                return self._fetch_timeline_from_api(match_id)
            else:
                print(f"  [ERROR] API 호출 실패 ({match_id}): {e}")
            return None

        except Exception as e:
            print(f"  [ERROR] 타임라인 조회 실패 ({match_id}): {e}")
            return None

    def get_timeline(self, match_id: str, version: str) -> Optional[Dict]:
        """
        타임라인 조회 및 저장

        Args:
            match_id: 매치 ID
            version: 게임 버전 (예: "v15.23")

        Returns:
            타임라인 JSON 또는 None
        """
        # 이미 존재하면 스킵
        timeline_path = self._get_timeline_path(match_id, version)
        if timeline_path.exists():
            print(f"  [SKIP] {match_id} (already exists)")
            return None

        # API 호출
        print(f"  [FETCH] {match_id}")
        timeline = self._fetch_timeline_from_api(match_id)

        if not timeline:
            return None

        # 저장
        try:
            with open(timeline_path, "w", encoding="utf-8") as f:
                json.dump(timeline, f, ensure_ascii=False, indent=2)
            print(f"  [SAVED] {timeline_path}")
            return timeline
        except Exception as e:
            print(f"  [ERROR] 저장 실패 ({match_id}): {e}")
            return None

    def collect_from_tier_data(self, tier_data_file: Path) -> int:
        """
        collect_tier_data.py 결과물에서 매치 ID 추출하여 타임라인 수집

        Args:
            tier_data_file: 티어 데이터 파일 경로
                           (예: data/tier_collections/master_tier_v15.23.json)

        Returns:
            수집된 타임라인 개수
        """
        print(f"\n{'='*80}")
        print(f"타임라인 수집: {tier_data_file.name}")
        print(f"{'='*80}")

        # 1) 파일 로드
        try:
            with open(tier_data_file, "r", encoding="utf-8") as f:
                tier_data = json.load(f)
        except Exception as e:
            print(f"[ERROR] 파일 로드 실패: {e}")
            return 0

        # 2) 메타데이터에서 버전 추출
        metadata = tier_data.get("metadata", {})
        version = f"v{metadata.get('game_version', 'unknown')}"
        total_matches = metadata.get("total_matches", 0)

        print(f"게임 버전: {version}")
        print(f"총 매치 수: {total_matches:,}")

        # 3) 매치 ID 추출 (중복 제거)
        match_ids: Set[str] = set()
        for player_data in tier_data.get("data", []):
            match_id = player_data.get("match_id")
            if match_id:
                match_ids.add(match_id)

        print(f"고유 매치 수: {len(match_ids):,}")

        # 4) 이미 존재하는 타임라인 개수 확인
        version_dir = self.cache_dir / version
        existing_count = len(list(version_dir.glob("*.json"))) if version_dir.exists() else 0
        print(f"기존 타임라인: {existing_count:,}")

        # 5) 타임라인 수집
        collected = 0
        failed = 0
        skipped = 0

        for idx, match_id in enumerate(sorted(match_ids), start=1):
            print(f"\n[{idx}/{len(match_ids)}]", end=" ")

            result = self.get_timeline(match_id, version)

            if result:
                collected += 1
            elif result is None and self._get_timeline_path(match_id, version).exists():
                skipped += 1
            else:
                failed += 1

            # 진행 상황 출력 (10개마다)
            if idx % 10 == 0:
                print(f"\n  진행: {idx}/{len(match_ids)} | 수집: {collected} | 스킵: {skipped} | 실패: {failed}")

        print(f"\n{'='*80}")
        print(f"타임라인 수집 완료!")
        print(f"  총 매치: {len(match_ids):,}")
        print(f"  새로 수집: {collected:,}")
        print(f"  이미 존재: {skipped:,}")
        print(f"  실패: {failed:,}")
        print(f"  저장 위치: {self.cache_dir / version}")
        print(f"{'='*80}\n")

        return collected

    def collect_all_versions(self, tier_data_dir: str = "data/tier_collections"):
        """
        모든 버전의 타임라인 수집

        Args:
            tier_data_dir: 티어 데이터 디렉토리
        """
        tier_data_files = sorted(glob(f"{tier_data_dir}/*_tier_v*.json"))

        if not tier_data_files:
            print(f"[WARN] {tier_data_dir}에 티어 데이터 파일이 없습니다.")
            return

        print(f"\n발견된 티어 데이터 파일: {len(tier_data_files)}개")
        for f in tier_data_files:
            print(f"  - {Path(f).name}")

        total_collected = 0

        for tier_file in tier_data_files:
            try:
                collected = self.collect_from_tier_data(Path(tier_file))
                total_collected += collected
            except KeyboardInterrupt:
                print(f"\n\n사용자에 의해 중단됨")
                print(f"현재까지 수집된 타임라인: {total_collected:,}개")
                break
            except Exception as e:
                print(f"\n[ERROR] {tier_file} 처리 중 오류: {e}")
                continue

        print(f"\n{'='*80}")
        print(f"전체 수집 완료!")
        print(f"총 수집된 타임라인: {total_collected:,}개")
        print(f"저장 위치: {self.cache_dir}")
        print(f"{'='*80}\n")


def main():
    """메인 함수"""
    import argparse

    parser = argparse.ArgumentParser(description="매치 타임라인 수집")
    parser.add_argument(
        "--version",
        type=str,
        help="특정 버전의 타임라인 수집 (예: v15.23)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="모든 버전의 타임라인 수집",
    )
    parser.add_argument(
        "--file",
        type=str,
        help="특정 티어 데이터 파일의 타임라인 수집",
    )

    args = parser.parse_args()

    fetcher = TimelineFetcher(cache_dir="data/timelines", rate_limit_delay=1.2)

    if args.file:
        # 특정 파일
        fetcher.collect_from_tier_data(Path(args.file))

    elif args.version:
        # 특정 버전
        tier_files = glob(f"data/tier_collections/*_v{args.version.lstrip('v')}.json")
        if not tier_files:
            print(f"[ERROR] {args.version} 버전의 티어 데이터를 찾을 수 없습니다.")
            return

        for tier_file in tier_files:
            fetcher.collect_from_tier_data(Path(tier_file))

    elif args.all:
        # 모든 버전
        fetcher.collect_all_versions()

    else:
        # 기본: 모든 버전
        print("옵션을 지정하지 않아 모든 버전의 타임라인을 수집합니다.")
        print("특정 버전만 수집하려면 --version v15.23 옵션을 사용하세요.\n")
        fetcher.collect_all_versions()


if __name__ == "__main__":
    main()
