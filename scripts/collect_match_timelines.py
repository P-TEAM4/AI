import os
import json
import requests
from glob import glob
from dotenv import load_dotenv
import time

load_dotenv()
API_KEY = os.getenv("RIOT_API_KEY")
HEADERS = {"X-Riot-Token": API_KEY}

def get_timeline(match_id):
    url = f"https://asia.api.riotgames.com/lol/match/v5/matches/{match_id}/timeline"
    params = {"api_key" : API_KEY}
    res = requests.get(url, headers=HEADERS, params=params)
    res.raise_for_status()
    return res.json()

def save_timeline(match_id, tier, data):
    save_dir = f"data/raw/{tier}/timeline"
    os.makedirs(save_dir, exist_ok=True)
    path = f"{save_dir}/{match_id}.json"

    if os.path.exists(path):
        print(f"[SKIP] timeline exists: {match_id}")
        return

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[SAVED] {path}")

# 메인 로직
def fetch_missing_timelines():
    tiers = ["EMERALD", "DIAMOND", "MASTER", "GRANDMASTER", "CHALLENGER"]

    for tier in tiers:
        match_files = glob(f"data/raw/{tier}/*.json")

        for fpath in match_files:
            with open(fpath, "r", encoding="utf-8") as f:
                match = json.load(f)
            
            match_id = match["metadata"]["matchId"]
            timeline_path = f"data/raw/{tier}/timeline/{match_id}.json"

            if os.path.exists(timeline_path):
                continue

            print(f"[FETCH] {tier} {match_id}")

            try:
                time.sleep(1.2)
                timeline = get_timeline(match_id)
                save_timeline(match_id, tier, timeline)
            except Exception as e:
                print(f"[ERROR] timeline failed {match_id}: {e}")

if __name__ == "__main__":
    fetch_missing_timelines()