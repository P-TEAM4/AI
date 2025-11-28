"""API 빠른 테스트"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("RIOT_API_KEY")
print(f"API Key: {api_key[:20]}...")

# 1단계: GOLD 티어 소환사 목록 조회
url = "https://kr.api.riotgames.com/lol/league/v4/entries/RANKED_SOLO_5x5/GOLD/I?page=1"
headers = {"X-Riot-Token": api_key}

print(f"\nAPI 호출: {url}")
response = requests.get(url, headers=headers)
print(f"Status: {response.status_code}")

if response.status_code == 200:
    data = response.json()
    print(f"조회된 소환사 수: {len(data)}")

    if data:
        first = data[0]
        print(f"\n첫 번째 소환사:")
        print(f"  summonerName: {first.get('summonerName')}")
        print(f"  tier: {first.get('tier')}")
        print(f"  rank: {first.get('rank')}")
        print(f"  leaguePoints: {first.get('leaguePoints')}")
        print(f"  wins: {first.get('wins')}")
        print(f"  losses: {first.get('losses')}")
        print(f"  puuid: {first.get('puuid', 'N/A')[:20]}...")
else:
    print(f"Error: {response.text}")
