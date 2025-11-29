"""
타임라인 데이터 샘플 가져오기

Usage:
    python scripts/fetch_timeline_sample.py
"""

import os
import sys
from pathlib import Path
import requests
import json
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

# .env 파일 로드
project_root = Path(__file__).parent.parent
load_dotenv(project_root / ".env")


def get_timeline_sample():
    """타임라인 데이터 샘플 가져오기"""

    api_key = os.getenv("RIOT_API_KEY")
    if not api_key:
        print("Error: RIOT_API_KEY not found in .env")
        return

    headers = {'X-Riot-Token': api_key}

    # Step 1: Gold 티어 소환사 목록 조회
    print("[1] Fetching Gold tier summoners...")
    summoners_url = "https://kr.api.riotgames.com/lol/league/v4/entries/RANKED_SOLO_5x5/GOLD/I?page=1"

    response = requests.get(summoners_url, headers=headers)
    time.sleep(1.2)

    if response.status_code != 200:
        print(f"Failed to get summoners: {response.status_code}")
        print(response.text)
        return

    summoners = response.json()
    print(f"Found {len(summoners)} summoners")

    # Step 2: 첫 번째 소환사의 PUUID로 매치 ID 조회
    for summoner in summoners[:5]:  # 최대 5명 시도
        puuid = summoner.get('puuid')
        if not puuid:
            continue

        print(f"\n[2] Fetching matches for PUUID: {puuid[:20]}...")
        matches_url = f"https://asia.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids"
        params = {"start": 0, "count": 3, "queue": 420}

        response = requests.get(matches_url, params=params, headers=headers)
        time.sleep(1.2)

        if response.status_code != 200:
            print(f"Failed to get matches: {response.status_code}")
            continue

        match_ids = response.json()
        print(f"Found {len(match_ids)} matches")

        # Step 3: 각 매치의 타임라인 조회
        for match_id in match_ids:
            print(f"\n[3] Fetching timeline for: {match_id}")
            timeline_url = f"https://asia.api.riotgames.com/lol/match/v5/matches/{match_id}/timeline"

            response = requests.get(timeline_url, headers=headers)
            time.sleep(1.2)

            if response.status_code == 200:
                print("[SUCCESS] Got timeline data!\n")
                timeline = response.json()

                # 전체 구조 출력
                print("=" * 80)
                print("TIMELINE DATA STRUCTURE")
                print("=" * 80)
                print(f"Match ID: {match_id}")
                print(f"Frame Interval: {timeline['info']['frameInterval']}ms")
                print(f"Total Frames: {len(timeline['info']['frames'])}")
                print(f"Participants: {len(timeline['metadata']['participants'])}")

                # 첫 번째 프레임 상세
                print("\n" + "=" * 80)
                print("FRAME 1 (1분 시점)")
                print("=" * 80)
                frame1 = timeline['info']['frames'][1] if len(timeline['info']['frames']) > 1 else timeline['info']['frames'][0]

                # 한 명의 플레이어 데이터 예시
                participant_id = list(frame1['participantFrames'].keys())[0]
                player_frame = frame1['participantFrames'][participant_id]

                print(f"\nPlayer {participant_id} stats:")
                print(json.dumps(player_frame, indent=2, ensure_ascii=False))

                # 이벤트 예시
                print("\n" + "=" * 80)
                print("EVENTS IN FRAME 1")
                print("=" * 80)
                if frame1.get('events'):
                    for event in frame1['events'][:5]:  # 첫 5개 이벤트만
                        print(json.dumps(event, indent=2, ensure_ascii=False))
                        print("-" * 40)

                # 전체 데이터를 파일로 저장
                output_dir = Path("data/timeline_samples")
                output_dir.mkdir(parents=True, exist_ok=True)
                output_file = output_dir / f"{match_id}_timeline.json"

                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(timeline, f, indent=2, ensure_ascii=False)

                print("\n" + "=" * 80)
                print(f"Full timeline saved to: {output_file}")
                print("=" * 80)

                return  # 성공하면 종료

            else:
                print(f"Failed: {response.status_code} - {response.text[:100]}")

    print("\nFailed to fetch timeline from any match")


if __name__ == "__main__":
    get_timeline_sample()
