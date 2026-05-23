"""
백엔드 통합 API 테스트
"""

import requests
import json
from typing import Dict, Any

# API URL
API_URL = "http://localhost:8001"

# 샘플 데이터
SAMPLE_MATCH_DATA = {
    "match_id": "KR_7053936104",
    "game_duration": 1845,
    "game_creation": 1735719600000,
    "queue_id": 420
}

SAMPLE_PLAYER_STATS = {
    "puuid": "test-puuid-12345",
    "summonerName": "TestPlayer",
    "gameName": "TestPlayer",
    "tagLine": "KR1",
    "championName": "Jinx",
    "championId": 222,
    "teamId": 100,
    "teamPosition": "BOTTOM",
    "participantId": 1,
    "win": True,
    "kills": 12,
    "deaths": 3,
    "assists": 8,
    "goldEarned": 15420,
    "goldSpent": 14800,
    "totalMinionsKilled": 245,
    "neutralMinionsKilled": 18,
    "totalDamageDealtToChampions": 28540,
    "magicDamageDealtToChampions": 2340,
    "physicalDamageDealtToChampions": 25200,
    "trueDamageDealtToChampions": 1000,
    "visionScore": 42,
    "wardsPlaced": 15,
    "wardsKilled": 8,
    "visionWardsBoughtInGame": 5,
    "longestTimeSpentLiving": 650,
    "challenges": {
        "killParticipation": 0.74,
        "soloKills": 2,
        "controlWardsPlaced": 5,
        "wardTakedownsBefore20M": 3,
        "skillshotsDodged": 45,
        "skillshotsHit": 89,
        "goldPerMinute": 502.3,
        "damagePerMinute": 930.1
    }
}

SAMPLE_TIMELINE_DATA = {
    "frames": {
        "10": [
            {
                "timestamp": 600000,
                "participantId": 1,
                "level": 7,
                "currentGold": 1250,
                "totalGold": 4320,
                "xp": 5840,
                "minionsKilled": 82,
                "jungleMinionsKilled": 0
            }
        ],
        "15": [
            {
                "timestamp": 900000,
                "participantId": 1,
                "level": 10,
                "currentGold": 2340,
                "totalGold": 7850,
                "xp": 9240,
                "minionsKilled": 134,
                "jungleMinionsKilled": 4
            }
        ]
    }
}


def test_health_check():
    """Health check 테스트"""
    print("=" * 60)
    print("Health Check 테스트")
    print("=" * 60)

    response = requests.get(f"{API_URL}/health")

    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True

    print("✅ Health check 성공!\n")


def test_single_match_analysis():
    """단일 매치 분석 테스트"""
    print("=" * 60)
    print("단일 매치 분석 테스트")
    print("=" * 60)

    request_data = {
        "match_data": SAMPLE_MATCH_DATA,
        "player_stats": SAMPLE_PLAYER_STATS,
        "timeline_data": SAMPLE_TIMELINE_DATA,
        "tier": "GOLD"
    }

    response = requests.post(
        f"{API_URL}/api/v1/backend/analyze/match",
        json=request_data
    )

    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"\n매치 ID: {data['match_id']}")
        print(f"챔피언: {data['champion']} ({data['role']})")
        print(f"승패: {'승리' if data['win'] else '패배'}")
        print(f"\n=== Impact Score ===")
        impact = data['impactResult']
        print(f"Impact Score: {impact['impact_score']:.1f}")
        print(f"예측 승률: {impact['predicted_proba']:.1%}")
        print(f"요약: {impact['summary']}")

        print(f"\n=== 긍정적 요인 ===")
        for feat in impact['topPositiveFeatures']:
            print(f"  - {feat['displayName']}: {feat['value']:.2f} (SHAP: {feat['shap']:.3f})")

        print(f"\n=== 부정적 요인 ===")
        for feat in impact['topNegativeFeatures']:
            print(f"  - {feat['displayName']}: {feat['value']:.2f} (SHAP: {feat['shap']:.3f})")

        print(f"\n=== Raw Stats ===")
        raw = impact['rawStats']
        print(f"  KDA: {raw['kda']:.2f} ({raw['kills']}/{raw['deaths']}/{raw['assists']})")
        print(f"  분당 골드: {raw['goldPerMinute']:.1f}")
        print(f"  분당 데미지: {raw['damagePerMinute']:.1f}")
        print(f"  15분 CS: {raw['cs_15']:.0f}")

        print("\n✅ 단일 매치 분석 성공!\n")
    else:
        print(f"❌ 에러: {response.text}\n")


def test_single_match_without_timeline():
    """타임라인 없이 매치 분석 테스트"""
    print("=" * 60)
    print("타임라인 없이 매치 분석 테스트")
    print("=" * 60)

    request_data = {
        "match_data": SAMPLE_MATCH_DATA,
        "player_stats": SAMPLE_PLAYER_STATS,
        # timeline_data는 생략
        "tier": "GOLD"
    }

    response = requests.post(
        f"{API_URL}/api/v1/backend/analyze/match",
        json=request_data
    )

    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        impact = data['impactResult']
        print(f"Impact Score: {impact['impact_score']:.1f}")
        print(f"예측 승률: {impact['predicted_proba']:.1%}")

        raw = impact['rawStats']
        print(f"15분 CS: {raw['cs_15']:.0f} (타임라인 없어서 0)")
        print(f"15분 골드: {raw['gold_15']:.0f} (타임라인 없어서 0)")

        print("\n✅ 타임라인 없이도 분석 성공!\n")
    else:
        print(f"❌ 에러: {response.text}\n")


def test_batch_analysis():
    """일괄 분석 테스트"""
    print("=" * 60)
    print("일괄 분석 테스트 (3개 매치)")
    print("=" * 60)

    # 3개의 매치 데이터 생성 (약간씩 다르게)
    matches = []
    for i in range(3):
        match_data = SAMPLE_MATCH_DATA.copy()
        match_data["match_id"] = f"KR_700000000{i}"

        player_stats = SAMPLE_PLAYER_STATS.copy()
        player_stats["kills"] = 10 + i * 2
        player_stats["deaths"] = 2 + i
        player_stats["win"] = (i % 2 == 0)  # 승-패-승

        matches.append({
            "match_data": match_data,
            "player_stats": player_stats,
            "timeline_data": SAMPLE_TIMELINE_DATA,
            "tier": "GOLD"
        })

    request_data = {
        "matches": matches,
        "aggregate": True
    }

    response = requests.post(
        f"{API_URL}/api/v1/backend/analyze/batch",
        json=request_data
    )

    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"\n총 매치: {data['totalMatches']}")
        print(f"성공: {data['successful']}")
        print(f"실패: {data['failed']}")

        if data.get('averageImpactScore') is not None:
            print(f"\n=== 평균 지표 ===")
            print(f"평균 Impact Score: {data['averageImpactScore']:.1f}")
            print(f"승률: {data['winRate']:.1%}")

            avg = data['avgStats']
            print(f"평균 KDA: {avg['kda']:.2f}")
            print(f"평균 분당 골드: {avg['goldPerMinute']:.1f}")
            print(f"평균 분당 데미지: {avg['damagePerMinute']:.1f}")

        print(f"\n=== 개별 매치 결과 ===")
        for result in data['results']:
            print(f"{result['match_id']}: Impact {result['impactResult']['impact_score']:.1f}, "
                  f"{'승' if result['win'] else '패'}")

        print("\n✅ 일괄 분석 성공!\n")
    else:
        print(f"❌ 에러: {response.text}\n")


def test_extreme_stats():
    """극단적인 통계 테스트"""
    print("=" * 60)
    print("극단적인 통계 테스트 (30킬 캐리)")
    print("=" * 60)

    # 극단적으로 잘한 경기
    extreme_stats = SAMPLE_PLAYER_STATS.copy()
    extreme_stats["kills"] = 30
    extreme_stats["deaths"] = 1
    extreme_stats["assists"] = 15
    extreme_stats["goldEarned"] = 25000
    extreme_stats["totalDamageDealtToChampions"] = 55000
    extreme_stats["visionScore"] = 80

    request_data = {
        "match_data": SAMPLE_MATCH_DATA,
        "player_stats": extreme_stats,
        "timeline_data": SAMPLE_TIMELINE_DATA,
        "tier": "GOLD"
    }

    response = requests.post(
        f"{API_URL}/api/v1/backend/analyze/match",
        json=request_data
    )

    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        impact = data['impactResult']
        print(f"Impact Score: {impact['impact_score']:.1f}")
        print(f"예측 승률: {impact['predicted_proba']:.1%}")
        print(f"요약: {impact['summary']}")

        raw = impact['rawStats']
        print(f"KDA: {raw['kda']:.2f}")

        print("\n✅ 극단적 통계도 정상 분석!\n")
    else:
        print(f"❌ 에러: {response.text}\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("백엔드 통합 API 테스트 시작")
    print("=" * 60 + "\n")

    try:
        # 1. Health Check
        test_health_check()

        # 2. 단일 매치 분석
        test_single_match_analysis()

        # 3. 타임라인 없이 분석
        test_single_match_without_timeline()

        # 4. 일괄 분석
        test_batch_analysis()

        # 5. 극단적 통계
        test_extreme_stats()

        print("=" * 60)
        print("✅ 모든 테스트 완료!")
        print("=" * 60)

    except requests.exceptions.ConnectionError:
        print("\n❌ API 서버에 연결할 수 없습니다.")
        print("다음 명령으로 서버를 먼저 실행하세요:")
        print("  cd AI && python -m src.api.backend_integration")
    except Exception as e:
        print(f"\n❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
