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


def merge_nearby_highlights(highlights: List[Dict], merge_window: float = 10.0) -> List[Dict]:
    """
    인접한 이벤트를 단일 클립으로 병합 (더블킬/트리플킬/한타 대응)

    같은 카테고리(highlight/mistake)의 이벤트가 merge_window 초 이내에
    연달아 발생하면 단일 확장 클립으로 묶는다.

    Args:
        highlights: 하이라이트 리스트
        merge_window: 병합 기준 시간 간격 (초)

    Returns:
        병합된 하이라이트 리스트
    """
    if not highlights:
        return []

    sorted_h = sorted(highlights, key=lambda x: x['timestamp'])
    merged = []
    i = 0

    while i < len(sorted_h):
        group = [sorted_h[i]]
        j = i + 1

        while j < len(sorted_h):
            nxt = sorted_h[j]
            # 같은 카테고리이고 이전 이벤트와 merge_window 이내
            if (nxt['category'] == group[-1]['category'] and
                    nxt['timestamp'] - group[-1]['timestamp'] <= merge_window):
                group.append(nxt)
                j += 1
            else:
                break

        if len(group) == 1:
            merged.append(group[0])
        else:
            # 가장 중요도가 높은 이벤트를 기반으로 병합 이벤트 생성
            best = max(group, key=lambda x: x.get('combined_importance', x.get('importance', 0)))
            kill_count = sum(1 for g in group if g.get('type') == 'kill')
            if kill_count >= 3:
                merged_type = 'teamfight'
            elif kill_count == 2:
                merged_type = 'double_kill'
            else:
                merged_type = best['type']

            merged_event = best.copy()
            merged_event['type'] = merged_type
            merged_event['timestamp'] = group[0]['timestamp']       # 첫 이벤트 기준
            merged_event['timestamp_end'] = group[-1]['timestamp']  # 마지막 이벤트
            merged_event['kill_count'] = kill_count
            merged_event['importance'] = max(g.get('combined_importance', g.get('importance', 0)) for g in group)
            merged_event['description'] = (
                f"{merged_type} ({group[0]['timestamp']/60:.1f}분, {len(group)}이벤트)"
            )
            merged.append(merged_event)

        i = j

    return merged


def merge_overlapping_events(highlights: List[Dict], overlap_threshold: float = 0.5) -> List[Dict]:
    """
    highlight(킬)와 mistake(데스) 클립 구간이 겹치면 gold_delta 기준으로 하나로 합친다.

    킬 직후 죽거나, 팀파이트 안에서 킬+데스가 같은 영상에 담길 때
    두 클립이 중복 생성되는 문제를 방지한다.

    병합 규칙:
      - 클립 구간 오버랩 비율 >= overlap_threshold (기본 50%)
      - net_gold_delta >= 0  →  highlight (킬이 이득)
      - net_gold_delta <  0  →  mistake   (교환했지만 손해)
      - type = 'trade'로 설정해 Gemini 프롬프트가 복합 상황임을 인지

    Args:
        highlights: merge_nearby_highlights 이후의 전체 이벤트 리스트 (highlight + mistake 혼재)
        overlap_threshold: 클립 구간 오버랩 비율 기준 (0~1)

    Returns:
        병합된 이벤트 리스트
    """
    def _clip_window(h: Dict):
        ts     = h['timestamp']
        ts_end = h.get('timestamp_end', ts)
        t      = h.get('type', 'kill')
        before = 20 if t == 'death' else 10
        after  = 10 if t == 'death' else 15
        return (ts - before, ts_end + after)

    def _overlap_ratio(a, b) -> float:
        start = max(a[0], b[0])
        end   = min(a[1], b[1])
        if end <= start:
            return 0.0
        overlap = end - start
        shorter = min(a[1] - a[0], b[1] - b[0])
        return overlap / shorter if shorter > 0 else 0.0

    hl = [h for h in highlights if h.get('category') == 'highlight']
    ms = [h for h in highlights if h.get('category') == 'mistake']

    used_hl: set = set()
    used_ms: set = set()
    combined: List[Dict] = []

    for i, h in enumerate(hl):
        for j, m in enumerate(ms):
            if i in used_hl or j in used_ms:
                continue
            if _overlap_ratio(_clip_window(h), _clip_window(m)) < overlap_threshold:
                continue

            # 교전 구간: 두 이벤트의 타임스탬프 전체를 커버
            all_ts  = [h['timestamp'], h.get('timestamp_end', h['timestamp']),
                       m['timestamp'], m.get('timestamp_end', m['timestamp'])]
            ts_min  = min(all_ts)
            ts_max  = max(all_ts)

            h_gold  = h.get('gold_delta', 0)
            m_gold  = m.get('gold_delta', 0)
            net     = h_gold - abs(m_gold)   # 양수 = 킬이 더 이득

            base    = h if net >= 0 else m
            ev      = base.copy()
            ev['timestamp']     = ts_min
            ev['timestamp_end'] = ts_max if ts_max > ts_min else ts_min
            ev['type']          = 'trade'
            ev['category']      = 'highlight' if net >= 0 else 'mistake'
            ev['has_kill']      = True
            ev['has_death']     = True
            ev['net_gold_delta'] = net
            ev['importance']    = max(
                h.get('combined_importance', h.get('importance', 0)),
                m.get('combined_importance', m.get('importance', 0))
            )
            ev['description']   = (
                f"킬+데스 교환 ({ts_min/60:.1f}분, net {'+' if net>=0 else ''}{net:.0f}G)"
            )

            combined.append(ev)
            used_hl.add(i)
            used_ms.add(j)

    result  = [h for i, h in enumerate(hl) if i not in used_hl]
    result += [m for j, m in enumerate(ms) if j not in used_ms]
    result += combined
    return result


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
      데스: 전 20초 / 후 10초 — 결정적 실수는 죽기 10~20초 전에 발생
      킬:  전 10초 / 후 15초 — 킬 후 전환(웨이브·오브젝트)이 핵심 코칭 포인트
      병합 이벤트(더블킬/한타): 첫 이벤트 전 10초 ~ 마지막 이벤트 후 15초

    Args:
        video_path: 원본 영상 경로
        highlight: 하이라이트 정보 (timestamp_end 있으면 병합 이벤트로 처리)
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
    timestamp_end = highlight.get('timestamp_end')  # 병합 이벤트인 경우

    if timestamp_end and timestamp_end > timestamp:
        # 병합 이벤트: 첫 이벤트 전 before_seconds ~ 마지막 이벤트 후 after_seconds
        video_start = timestamp - game_start_offset
        video_end = timestamp_end - game_start_offset
        start_time = max(0, video_start - before_seconds)
        duration = (video_end - video_start) + before_seconds + after_seconds
        clip_filename = f"{event_type}_{int(timestamp)}_to_{int(timestamp_end)}_{highlight['category']}.mp4"
    else:
        video_timestamp = timestamp - game_start_offset
        start_time = max(0, video_timestamp - before_seconds)
        duration = before_seconds + after_seconds
        clip_filename = f"{event_type}_{int(timestamp)}_{highlight['category']}.mp4"

    # clip_event_sec: 클립 내 첫 이벤트 발생 시점(Gemini 프롬프트용)
    highlight['clip_event_sec'] = int((timestamp - game_start_offset) - start_time)

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

        print(f"[INFO] Created clip: {output_path} (duration={duration:.1f}s)")
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
