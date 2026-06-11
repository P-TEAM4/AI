# AI/src/api/highlight_extractor.py

import ffmpeg
import os
from typing import List, Dict
import numpy as np


def extract_highlights_from_timeline(timeline_data: dict, target_puuid: str) -> List[Dict]:
    """
    타임라인에서 주요 이벤트를 추출하여 하이라이트 리스트 생성

    Args:
        timeline_data: Riot API 타임라인 데이터
        target_puuid: 대상 플레이어 PUUID

    Returns:
        하이라이트 이벤트 리스트
    """
    highlights = []
    frames = timeline_data['info']['frames']

    # 참가자 ID 찾기
    participant_id = None
    for p in timeline_data['metadata']['participants']:
        if p == target_puuid:
            participant_id = timeline_data['metadata']['participants'].index(p) + 1
            break

    if not participant_id:
        return []

    # 이벤트 순회하며 하이라이트 추출
    for frame_idx, frame in enumerate(frames):
        for event in frame.get('events', []):
            event_type = event.get('type')
            event_seconds = event['timestamp'] / 1000  # ms to seconds (이벤트 실제 시간)
            event_minutes = event_seconds / 60

            # 킬 이벤트
            if event_type == 'CHAMPION_KILL':
                killer_id = event.get('killerId')
                victim_id = event.get('victimId')
                assistants = event.get('assistingParticipantIds', [])

                # 플레이어가 킬을 했을 때
                if killer_id == participant_id:
                    importance = calculate_kill_importance(event, event_minutes)
                    highlights.append({
                        'timestamp': event_seconds,
                        'type': 'kill',
                        'category': 'highlight',
                        'importance': importance,
                        'description': f"킬 ({event_minutes:.1f}분)",
                        'details': {
                            'victim_id': victim_id,
                            'assistants': assistants,
                            'bounty': event.get('bounty', 0),
                            'shutdown_bounty': event.get('shutdownBounty', 0)
                        }
                    })

                # 플레이어가 죽었을 때
                if victim_id == participant_id:
                    importance = calculate_death_importance(event, event_minutes)
                    highlights.append({
                        'timestamp': event_seconds,
                        'type': 'death',
                        'category': 'mistake',
                        'importance': importance,
                        'description': f"데스 ({event_minutes:.1f}분)",
                        'details': {
                            'killer_id': killer_id,
                            'assistants': assistants
                        }
                    })

            # 오브젝트 획득
            elif event_type in ['ELITE_MONSTER_KILL', 'BUILDING_KILL']:
                killer_id = event.get('killerId')
                if killer_id == participant_id or participant_id in event.get('assistingParticipantIds', []):
                    obj_type = event.get('monsterType') or event.get('buildingType')
                    importance = calculate_objective_importance(event, event_minutes)

                    highlights.append({
                        'timestamp': event_seconds,
                        'type': 'objective',
                        'category': 'highlight',
                        'importance': importance,
                        'description': f"{obj_type} 획득 ({event_minutes:.1f}분)",
                        'details': {
                            'object_type': obj_type
                        }
                    })

    # 중요도 순으로 정렬
    highlights.sort(key=lambda x: x['importance'], reverse=True)

    return highlights


def calculate_kill_importance(event: dict, timestamp_minutes: float) -> float:
    """킬의 중요도 계산"""
    importance = 5.0  # 기본 점수

    # 초반 킬은 더 중요
    if timestamp_minutes < 5:
        importance += 3.0

    # 셧다운 골드가 있으면 중요
    shutdown_bounty = event.get('shutdownBounty', 0)
    if shutdown_bounty > 0:
        importance += min(shutdown_bounty / 100, 5.0)

    # 어시스트 수가 많으면 팀파이트
    assistants = len(event.get('assistingParticipantIds', []))
    if assistants >= 3:
        importance += 2.0

    return min(importance, 10.0)


def calculate_death_importance(event: dict, timestamp_minutes: float) -> float:
    """데스의 중요도 계산 (높을수록 나쁜 데스)"""
    importance = 5.0  # 기본 점수

    # 초반 데스는 더 치명적
    if timestamp_minutes < 5:
        importance += 3.0

    # 바운티 골드를 줬으면 중요
    bounty = event.get('bounty', 0)
    if bounty > 300:
        importance += 2.0

    return min(importance, 10.0)


def calculate_objective_importance(event: dict, timestamp_minutes: float) -> float:
    """오브젝트의 중요도 계산"""
    obj_type = event.get('monsterType') or event.get('buildingType')

    importance_map = {
        'BARON_NASHOR': 10.0,
        'DRAGON': 7.0,
        'RIFTHERALD': 6.0,
        'TOWER_PLATE': 4.0,
        'OUTER_TURRET': 5.0,
        'INNER_TURRET': 7.0,
        'BASE_TURRET': 9.0,
        'NEXUS_TURRET': 10.0
    }

    return importance_map.get(obj_type, 5.0)


def create_clip(
    video_path: str,
    highlight: Dict,
    output_dir: str = "clips",
    before_seconds: int = None,
    after_seconds: int = None,
    game_start_offset: float = 0.0,
) -> str:
    """
    영상에서 클립 추출

    이벤트 종류별 비대칭 윈도우:
      데스: 전 20초 / 후 5초 — 결정적 실수는 죽기 10~20초 전에 발생
      킬:  전 10초 / 후 15초 — 킬 후 전환(웨이브·오브젝트)이 핵심 코칭 포인트

    Args:
        video_path: 원본 영상 경로
        highlight: 하이라이트 정보
        output_dir: 클립 저장 디렉토리
        before_seconds: 이벤트 전 초 (None이면 이벤트 종류로 자동 결정)
        after_seconds: 이벤트 후 초 (None이면 이벤트 종류로 자동 결정)
        game_start_offset: 녹화 시작 시점의 게임 내 시간(초, 보통 음수)

    Returns:
        생성된 클립 파일 경로
    """
    os.makedirs(output_dir, exist_ok=True)

    event_type = highlight.get('type', 'kill')
    if before_seconds is None:
        before_seconds = 20 if event_type == 'death' else 10
    if after_seconds is None:
        after_seconds = 10 if event_type == 'death' else 15

    timestamp = highlight['timestamp']
    video_timestamp = timestamp - game_start_offset
    start_time = max(0, video_timestamp - before_seconds)
    duration = before_seconds + after_seconds

    # 출력 파일명 생성
    clip_filename = f"{highlight['type']}_{int(timestamp)}_{highlight['category']}.mp4"
    output_path = os.path.join(output_dir, clip_filename)

    try:
        # FFmpeg로 클립 추출
        (
            ffmpeg
            .input(video_path, ss=start_time, t=duration)
            .output(
                output_path,
                vcodec='libx264',
                acodec='aac',
                **{'b:v': '2M', 'b:a': '128k'}
            )
            .overwrite_output()
            .run(quiet=True)
        )

        print(f"[INFO] Created clip: {output_path}")
        # Gemini 프롬프트용: 클립 내 이벤트 발생 시점(초) = 영상 내 위치 - 시작점
        highlight['clip_event_sec'] = int(video_timestamp - start_time)
        return output_path

    except ffmpeg.Error as e:
        print(f"[ERROR] Failed to create clip: {e}")
        return None


def get_top_highlights(
    highlights: List[Dict],
    top_n: int = 5,
    category: str = None
) -> List[Dict]:
    """
    상위 N개 하이라이트 반환

    Args:
        highlights: 하이라이트 리스트
        top_n: 반환할 개수
        category: 'highlight' 또는 'mistake', None이면 전체

    Returns:
        상위 하이라이트 리스트
    """
    if category:
        filtered = [h for h in highlights if h['category'] == category]
    else:
        filtered = highlights

    # 중요도 순으로 정렬 (내림차순)
    sorted_highlights = sorted(filtered, key=lambda x: x['importance'], reverse=True)

    return sorted_highlights[:top_n]


# 사용 예시
if __name__ == "__main__":
    # 예시: 타임라인에서 하이라이트 추출
    timeline_data = {
        'metadata': {'participants': ['puuid1', 'puuid2']},
        'info': {
            'frames': [
                {
                    'timestamp': 180000,  # 3분
                    'events': [
                        {
                            'type': 'CHAMPION_KILL',
                            'timestamp': 180000,
                            'killerId': 1,
                            'victimId': 6,
                            'assistingParticipantIds': [2, 3],
                            'shutdownBounty': 450
                        }
                    ]
                }
            ]
        }
    }

    highlights = extract_highlights_from_timeline(timeline_data, 'puuid1')

    # 잘한 부분 Top 3
    top_highlights = get_top_highlights(highlights, top_n=3, category='highlight')
    print("잘한 부분:")
    for h in top_highlights:
        print(f"  - {h['description']}: {h['importance']:.1f}점")

    # 못한 부분 Top 3
    top_mistakes = get_top_highlights(highlights, top_n=3, category='mistake')
    print("\n개선할 부분:")
    for h in top_mistakes:
        print(f"  - {h['description']}: {h['importance']:.1f}점")
