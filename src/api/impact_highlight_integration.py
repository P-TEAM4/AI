# AI/src/api/impact_highlight_integration.py

import numpy as np
import pandas as pd
from typing import Dict, List
import sys
import os

# player_rating 모듈 import
models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
if models_dir not in sys.path:
    sys.path.insert(0, models_dir)

from player_rating import add_derived_features


def extract_stats_at_frame(frame: dict, participant_id: int, match_data: dict) -> dict:
    """
    특정 타임라인 프레임에서 플레이어 스탯 추출

    Args:
        frame: 타임라인 프레임
        participant_id: 플레이어 ID
        match_data: 전체 매치 데이터

    Returns:
        해당 시점의 플레이어 스탯
    """
    participant_frame = frame['participantFrames'].get(str(participant_id), {})

    # 기본 스탯 추출
    timestamp_minutes = frame['timestamp'] / 60000

    stats = {
        'timestamp': timestamp_minutes,
        'totalGold': participant_frame.get('totalGold', 0),
        'level': participant_frame.get('level', 1),
        'xp': participant_frame.get('xp', 0),
        'currentGold': participant_frame.get('currentGold', 0),
        'minionsKilled': participant_frame.get('minionsKilled', 0),
        'jungleMinionsKilled': participant_frame.get('jungleMinionsKilled', 0),
    }

    return stats


def calculate_winrate_at_timestamp(
    timeline_data: dict,
    match_data: dict,
    participant_id: int,
    timestamp_minutes: float,
    model,
    feature_cols: List[str]
) -> float:
    """
    특정 시점의 승률 예측

    Args:
        timeline_data: 타임라인 데이터
        match_data: 매치 데이터
        participant_id: 플레이어 ID
        timestamp_minutes: 분 단위 시간
        model: 학습된 모델
        feature_cols: feature 컬럼 리스트

    Returns:
        예측 승률 (0~1)
    """
    # 해당 시점의 프레임 찾기
    frames = timeline_data['info']['frames']
    target_frame_idx = min(int(timestamp_minutes), len(frames) - 1)

    if target_frame_idx < 0:
        return 0.5  # 기본값

    frame = frames[target_frame_idx]

    # 플레이어 정보 가져오기
    participant_info = None
    for p in match_data['info']['participants']:
        if p['participantId'] == participant_id:
            participant_info = p
            break

    if not participant_info:
        return 0.5

    # 해당 시점의 스탯으로 feature 구성
    participant_frame = frame['participantFrames'].get(str(participant_id), {})

    # 모델이 사용하는 feature 추출
    feature_dict = {
        # 타임라인 기반 feature
        'cs_15': participant_frame.get('minionsKilled', 0) + participant_frame.get('jungleMinionsKilled', 0),
        'gold_15': participant_frame.get('totalGold', 0),
        'xp_15': participant_frame.get('xp', 0),
        'early_cs_total': participant_frame.get('minionsKilled', 0) + participant_frame.get('jungleMinionsKilled', 0),
        'laneMinionsFirst10Minutes': participant_frame.get('minionsKilled', 0),
        'jungleCsBefore10Minutes': participant_frame.get('jungleMinionsKilled', 0),

        # 게임 전체 스탯 (최종값 사용 - 근사치)
        'goldPerMinute': participant_info.get('challenges', {}).get('goldPerMinute', 0),
        'damagePerMinute': participant_info.get('challenges', {}).get('damagePerMinute', 0),
        'visionScorePerMinute': participant_info.get('challenges', {}).get('visionScorePerMinute', 0),
        'kda': (participant_info['kills'] + participant_info['assists']) / max(1, participant_info['deaths']),
        'kda_norm': 0.5,  # 팀 내 정규화는 실시간으로 불가능
        'killParticipation': participant_info.get('challenges', {}).get('killParticipation', 0),
        'soloKills': participant_info.get('challenges', {}).get('soloKills', 0),
        'vision_norm': 0.5,
        'controlWardsPlaced': participant_info.get('challenges', {}).get('controlWardsPlaced', 0),
        'wardTakedowns': participant_info.get('challenges', {}).get('wardTakedowns', 0),
        'wardTakedownsBefore20M': participant_info.get('challenges', {}).get('wardTakedownsBefore20M', 0),
        'skillshotsDodged': participant_info.get('challenges', {}).get('skillshotsDodged', 0),
        'skillshotsHit': participant_info.get('challenges', {}).get('skillshotsHit', 0),
        'longestTimeSpentLiving': participant_info.get('longestTimeSpentLiving', 0),
        'death_rate': participant_info['deaths']
    }

    # DataFrame 생성 및 예측
    X = pd.DataFrame([feature_dict])
    X = X[feature_cols].fillna(0)

    winrate = model.predict_proba(X.to_numpy())[0, 1]

    return winrate


def calculate_event_impact_score(
    event: dict,
    timeline_data: dict,
    match_data: dict,
    participant_id: int,
    model,
    feature_cols: List[str]
) -> float:
    """
    이벤트가 게임에 미친 영향 계산

    이벤트 전후의 승률 변화를 계산하여 Impact Score 산출

    Returns:
        Impact Score 변화량 (-100 ~ +100)
    """
    timestamp_minutes = event['timestamp'] / 60000

    # 이벤트 직전 승률
    if timestamp_minutes > 1:
        winrate_before = calculate_winrate_at_timestamp(
            timeline_data, match_data, participant_id,
            timestamp_minutes - 1, model, feature_cols
        )
    else:
        winrate_before = 0.5

    # 이벤트 직후 승률
    winrate_after = calculate_winrate_at_timestamp(
        timeline_data, match_data, participant_id,
        timestamp_minutes, model, feature_cols
    )

    # 승률 변화를 Impact Score로 변환 (0~100 스케일)
    impact_delta = (winrate_after - winrate_before) * 100

    return impact_delta


def enrich_highlights_with_impact(
    highlights: List[Dict],
    timeline_data: dict,
    match_data: dict,
    participant_id: int,
    model,
    feature_cols: List[str]
) -> List[Dict]:
    """
    하이라이트에 Impact Score 정보 추가

    Args:
        highlights: 추출된 하이라이트 리스트
        timeline_data: 타임라인 데이터
        match_data: 매치 데이터
        participant_id: 플레이어 ID
        model: Impact Score 모델
        feature_cols: feature 컬럼 리스트

    Returns:
        Impact Score 정보가 추가된 하이라이트 리스트
    """
    enriched = []

    for highlight in highlights:
        # 기존 중요도 점수
        base_importance = highlight['importance']

        # 이벤트의 Impact Score 계산
        impact_score = calculate_event_impact_score(
            highlight,
            timeline_data,
            match_data,
            participant_id,
            model,
            feature_cols
        )

        # 통합 중요도 점수 (기존 + Impact Score)
        combined_importance = base_importance + abs(impact_score) * 0.5

        # 설명 생성
        if highlight['category'] == 'highlight':
            if impact_score > 5:
                impact_description = f"이 플레이로 승률이 {impact_score:.1f}% 증가했습니다!"
            elif impact_score > 0:
                impact_description = f"긍정적인 플레이 (승률 +{impact_score:.1f}%)"
            else:
                impact_description = "플레이 성공했지만 큰 영향은 없었습니다"
        else:  # mistake
            if impact_score < -5:
                impact_description = f"이 실수로 승률이 {abs(impact_score):.1f}% 감소했습니다"
            elif impact_score < 0:
                impact_description = f"부정적인 플레이 (승률 {impact_score:.1f}%)"
            else:
                impact_description = "실수했지만 회복 가능한 수준입니다"

        enriched.append({
            **highlight,
            'impact_score': impact_score,
            'combined_importance': combined_importance,
            'impact_description': impact_description
        })

    # 통합 중요도로 재정렬
    enriched.sort(key=lambda x: x['combined_importance'], reverse=True)

    return enriched


def generate_match_summary(
    highlights: List[Dict],
    match_data: dict,
    participant_id: int
) -> dict:
    """
    매치 전체 요약 생성
    """
    # 플레이어 정보
    participant = None
    for p in match_data['info']['participants']:
        if p['participantId'] == participant_id:
            participant = p
            break

    # Impact Score 합계
    total_positive_impact = sum(h['impact_score'] for h in highlights if h['impact_score'] > 0)
    total_negative_impact = sum(h['impact_score'] for h in highlights if h['impact_score'] < 0)
    net_impact = total_positive_impact + total_negative_impact

    # 주요 하이라이트
    top_play = max(highlights, key=lambda x: x['impact_score']) if highlights else None
    worst_play = min(highlights, key=lambda x: x['impact_score']) if highlights else None

    summary = {
        'player': {
            'champion': participant['championName'],
            'role': participant['teamPosition'],
            'kda': f"{participant['kills']}/{participant['deaths']}/{participant['assists']}"
        },
        'impact_analysis': {
            'total_positive': round(total_positive_impact, 1),
            'total_negative': round(total_negative_impact, 1),
            'net_impact': round(net_impact, 1)
        },
        'key_moments': {
            'best_play': {
                'description': top_play['description'],
                'impact': round(top_play['impact_score'], 1),
                'timestamp': top_play['timestamp']
            } if top_play else None,
            'worst_play': {
                'description': worst_play['description'],
                'impact': round(worst_play['impact_score'], 1),
                'timestamp': worst_play['timestamp']
            } if worst_play else None
        },
        'highlight_count': len([h for h in highlights if h['category'] == 'highlight']),
        'mistake_count': len([h for h in highlights if h['category'] == 'mistake'])
    }

    return summary


# 사용 예시
if __name__ == "__main__":
    print("Impact Score + Highlight Integration Module")
    print("이 모듈은 하이라이트 클립에 Impact Score 분석을 추가합니다.")
