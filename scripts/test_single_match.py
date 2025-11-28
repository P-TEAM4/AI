"""단일 매치 데이터 구조 확인"""

import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("RIOT_API_KEY")
headers = {"X-Riot-Token": api_key}

print("=" * 80)
print("단일 매치 데이터 구조 확인")
print("=" * 80)

# 1단계: GOLD 티어 소환사 목록에서 PUUID 가져오기
print("\n1. GOLD 티어 소환사 목록 조회...")
url = "https://kr.api.riotgames.com/lol/league/v4/entries/RANKED_SOLO_5x5/GOLD/I?page=1"
response = requests.get(url, headers=headers)
summoners = response.json()
print(f"   조회된 소환사: {len(summoners)}명")

# 첫 번째 소환사의 puuid 가져오기
first_summoner = summoners[0]
puuid = first_summoner.get("puuid")

if not puuid:
    # puuid가 없으면 summonerId로 조회
    summoner_id = first_summoner.get("summonerId")
    print(f"\n2. summonerId로 소환사 정보 조회...")
    url = f"https://kr.api.riotgames.com/lol/summoner/v4/summoners/{summoner_id}"
    response = requests.get(url, headers=headers)
    summoner_data = response.json()
    puuid = summoner_data.get("puuid")

print(f"   PUUID: {puuid[:30]}...")

# 2단계: 매치 ID 목록 가져오기
print(f"\n3. 최근 매치 ID 조회...")
url = f"https://asia.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids"
params = {"start": 0, "count": 1, "queue": 420}  # 솔로랭크 1개
response = requests.get(url, headers=headers, params=params)
match_ids = response.json()
print(f"   매치 ID: {match_ids}")

if not match_ids:
    print("   매치 없음")
    exit()

match_id = match_ids[0]

# 3단계: 매치 상세 정보 가져오기
print(f"\n4. 매치 상세 정보 조회...")
url = f"https://asia.api.riotgames.com/lol/match/v5/matches/{match_id}"
response = requests.get(url, headers=headers)
match_data = response.json()

# 4단계: 매치 데이터 구조 분석
print(f"\n" + "=" * 80)
print(f"매치 데이터 구조")
print(f"=" * 80)

print(f"\n[Metadata]")
metadata = match_data.get("metadata", {})
for key, value in metadata.items():
    if isinstance(value, list):
        print(f"  {key}: [{len(value)}개 항목]")
    else:
        print(f"  {key}: {value}")

print(f"\n[Info - 게임 정보]")
info = match_data.get("info", {})
game_keys = ["gameCreation", "gameDuration", "gameMode", "gameType", "gameVersion", "queueId", "mapId"]
for key in game_keys:
    print(f"  {key}: {info.get(key)}")

print(f"\n[Info - 참가자 수]")
participants = info.get("participants", [])
print(f"  participants: {len(participants)}명")

print(f"\n[Participant 0 - 첫 번째 플레이어 데이터]")
if participants:
    player = participants[0]
    print(f"  기본 정보:")
    print(f"    puuid: {player.get('puuid', '')[:30]}...")
    print(f"    summonerName: {player.get('summonerName')}")
    print(f"    championName: {player.get('championName')}")
    print(f"    teamId: {player.get('teamId')}")
    print(f"    teamPosition: {player.get('teamPosition')}")
    print(f"    win: {player.get('win')}")

    print(f"\n  KDA:")
    print(f"    kills: {player.get('kills')}")
    print(f"    deaths: {player.get('deaths')}")
    print(f"    assists: {player.get('assists')}")

    print(f"\n  CS:")
    print(f"    totalMinionsKilled: {player.get('totalMinionsKilled')}")
    print(f"    neutralMinionsKilled: {player.get('neutralMinionsKilled')}")

    print(f"\n  Gold & Damage:")
    print(f"    goldEarned: {player.get('goldEarned')}")
    print(f"    totalDamageDealtToChampions: {player.get('totalDamageDealtToChampions')}")

    print(f"\n  Vision:")
    print(f"    visionScore: {player.get('visionScore')}")
    print(f"    wardsPlaced: {player.get('wardsPlaced')}")
    print(f"    wardsKilled: {player.get('wardsKilled')}")

    print(f"\n  기타:")
    print(f"    champLevel: {player.get('champLevel')}")
    print(f"    champExperience: {player.get('champExperience')}")
    print(f"    turretKills: {player.get('turretKills')}")
    print(f"    inhibitorKills: {player.get('inhibitorKills')}")

    print(f"\n  모든 필드 수: {len(player)}개")
    print(f"\n  전체 필드 목록:")
    for i, key in enumerate(sorted(player.keys())):
        if i % 5 == 0:
            print(f"\n    ", end="")
        print(f"{key:30s} ", end="")

# 5단계: JSON 파일로 저장
output_file = "data/sample_match.json"
os.makedirs("data", exist_ok=True)
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(match_data, f, indent=2, ensure_ascii=False)

print(f"\n\n" + "=" * 80)
print(f"전체 매치 데이터가 {output_file}에 저장되었습니다.")
print(f"파일 크기: {os.path.getsize(output_file) / 1024:.1f} KB")
print(f"=" * 80)
