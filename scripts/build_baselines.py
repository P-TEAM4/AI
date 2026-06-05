"""
Tier Baseline Builder

티어별 실제 KR 랭크 경기 데이터를 수집해 평균 스탯을 계산하고
data/tier_baselines_X.Y.json 에 저장합니다.

사용법:
    cd AI/
    python scripts/build_baselines.py

기능:
    - 티어당 10,000개 고유 경기 수집 (매치 ID 중복 없음)
    - 각 경기에서 전체 10명 플레이어 스탯 추출
    - 체크포인트 자동 저장 (100경기마다)
    - 중단 후 재실행 시 이어서 수집
    - 완료 후 data/tier_baselines_<version>.json 저장
"""

import sys
import json
import time
import random
import logging
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import requests
from dotenv import load_dotenv
import os

sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv(Path(__file__).parent.parent / ".env")

# ─── 설정 ─────────────────────────────────────────────────────────────────────

RIOT_API_KEY = os.getenv("RIOT_API_KEY", "")
KR_BASE      = "https://kr.api.riotgames.com"
ASIA_BASE    = "https://asia.api.riotgames.com"

TARGET_MATCHES_PER_TIER = 10_000
MATCHES_PER_SUMMONER    = 5     # 소환사당 최근 경기 수 (너무 많으면 한 소환사에 편향)
MIN_GAME_MINUTES        = 15    # 리메이크 제외

TIER_TARGETS = {
    "IRON":         TARGET_MATCHES_PER_TIER,
    "BRONZE":       TARGET_MATCHES_PER_TIER,
    "SILVER":       TARGET_MATCHES_PER_TIER,
    "GOLD":         TARGET_MATCHES_PER_TIER,
    "PLATINUM":     TARGET_MATCHES_PER_TIER,
    "EMERALD":      TARGET_MATCHES_PER_TIER,
    "DIAMOND":      TARGET_MATCHES_PER_TIER,
    "MASTER":       2_000,
    "GRANDMASTER":    500,
    "CHALLENGER":     300,
}
DIVISIONS = ["I", "II", "III", "IV"]

CHECKPOINT_DIR = Path("data/baselines_checkpoints")
OUTPUT_DIR     = Path("data")

# Rate limit (개발 키: 20req/sec, 100req/2min — 여유있게 10req/sec)
REQUEST_INTERVAL = 1.3    # 1.3초 간격 → ~46req/min (dev key: 100req/2min 제한 안전 준수)
MAX_RETRIES      = 3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("data/build_baselines.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ─── HTTP 클라이언트 ───────────────────────────────────────────────────────────

class RiotClient:
    def __init__(self, api_key: str):
        self.session = requests.Session()
        self.session.headers["X-Riot-Token"] = api_key
        self._last_request = 0.0

    def _get(self, url: str, params: dict = None) -> Optional[dict]:
        for attempt in range(MAX_RETRIES):
            # Rate limit
            elapsed = time.monotonic() - self._last_request
            if elapsed < REQUEST_INTERVAL:
                time.sleep(REQUEST_INTERVAL - elapsed)
            self._last_request = time.monotonic()

            try:
                resp = self.session.get(url, params=params, timeout=10)

                if resp.status_code == 200:
                    return resp.json()
                elif resp.status_code == 429:
                    retry_after = int(resp.headers.get("Retry-After", 120))
                    log.warning(f"429 Rate limited — {retry_after}s 대기...")
                    time.sleep(retry_after)
                elif resp.status_code == 404:
                    return None
                elif resp.status_code == 401:
                    log.error("401 API 키 만료 또는 무효. .env 파일에서 RIOT_API_KEY를 갱신하세요.")
                    raise SystemExit(1)
                else:
                    log.warning(f"HTTP {resp.status_code} for {url} (시도 {attempt+1}/{MAX_RETRIES})")
                    time.sleep(2 ** attempt)

            except requests.exceptions.Timeout:
                log.warning(f"Timeout: {url} (시도 {attempt+1}/{MAX_RETRIES})")
                time.sleep(2 ** attempt)
            except requests.exceptions.ConnectionError:
                log.warning(f"Connection error: {url} (시도 {attempt+1}/{MAX_RETRIES})")
                time.sleep(5)

        return None

    def get_league_entries(self, tier: str, division: str, page: int) -> List[dict]:
        url = f"{KR_BASE}/lol/league/v4/entries/RANKED_SOLO_5x5/{tier}/{division}"
        data = self._get(url, params={"page": page})
        return data if data else []

    def get_apex_league(self, tier: str) -> List[dict]:
        """MASTER / GRANDMASTER / CHALLENGER"""
        url = f"{KR_BASE}/lol/league/v4/{tier.lower()}leagues/by-queue/RANKED_SOLO_5x5"
        data = self._get(url)
        return data.get("entries", []) if data else []

    def get_summoner(self, summoner_id: str) -> Optional[dict]:
        url = f"{KR_BASE}/lol/summoner/v4/summoners/{summoner_id}"
        return self._get(url)

    def get_match_ids(self, puuid: str, count: int = 5) -> List[str]:
        url = f"{ASIA_BASE}/lol/match/v5/matches/by-puuid/{puuid}/ids"
        data = self._get(url, params={"queue": 420, "count": count})
        return data if data else []

    def get_match(self, match_id: str) -> Optional[dict]:
        url = f"{ASIA_BASE}/lol/match/v5/matches/{match_id}"
        return self._get(url)

    def get_game_version(self) -> str:
        try:
            resp = self.session.get("https://ddragon.leagueoflegends.com/api/versions.json", timeout=5)
            versions = resp.json()
            return ".".join(versions[0].split(".")[:2])  # "16.11.1" → "16.11"
        except Exception:
            return "unknown"


# ─── 스탯 추출 ─────────────────────────────────────────────────────────────────

def extract_match_stats(match_data: dict) -> List[dict]:
    """경기에서 전체 10명 스탯 추출 (리메이크 제외)"""
    info = match_data.get("info", {})
    duration = info.get("gameDuration", 0)
    minutes = duration / 60.0

    if minutes < MIN_GAME_MINUTES:
        return []

    participants = info.get("participants", [])
    result = []

    for team_id in [100, 200]:
        team = [p for p in participants if p.get("teamId") == team_id]
        team_dmg = sum(p.get("totalDamageDealtToChampions", 0) for p in team)

        for p in team:
            kills   = p.get("kills", 0)
            deaths  = p.get("deaths", 0)
            assists = p.get("assists", 0)
            cs      = p.get("totalMinionsKilled", 0) + p.get("neutralMinionsKilled", 0)
            gold    = p.get("goldEarned", 0)
            vision  = p.get("visionScore", 0)
            dmg     = p.get("totalDamageDealtToChampions", 0)

            result.append({
                "kda":                  round((kills + assists) / max(1, deaths), 3),
                "cs_per_min":           round(cs / minutes, 3),
                "gold_per_min":         round(gold / minutes, 3),
                "vision_score_per_min": round(vision / minutes, 3),
                "damage_share":         round(dmg / team_dmg, 4) if team_dmg > 0 else 0.0,
            })

    return result


# ─── 체크포인트 ────────────────────────────────────────────────────────────────

def load_checkpoint(tier: str) -> dict:
    path = CHECKPOINT_DIR / f"{tier.lower()}.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {"seen_matches": [], "seen_summoners": [], "stats": []}

def save_checkpoint(tier: str, seen_matches: Set[str], seen_summoners: Set[str], stats: List[dict]):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = CHECKPOINT_DIR / f"{tier.lower()}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "seen_matches":   list(seen_matches),
            "seen_summoners": list(seen_summoners),
            "stats":          stats,
        }, f, ensure_ascii=False)


# ─── 티어별 수집 ───────────────────────────────────────────────────────────────

def collect_tier(client: RiotClient, tier: str, target: int) -> List[dict]:
    log.info(f"{'='*60}")
    log.info(f"{tier} 수집 시작 — 목표 {target:,}경기")

    ckpt = load_checkpoint(tier)
    seen_matches:   Set[str] = set(ckpt["seen_matches"])
    seen_summoners: Set[str] = set(ckpt["seen_summoners"])
    stats:          List[dict] = ckpt["stats"]

    log.info(f"{tier}: 재개 — {len(seen_matches)}경기 완료, {len(stats)}개 스탯")

    def flush():
        save_checkpoint(tier, seen_matches, seen_summoners, stats)

    # 소환사 목록 생성기
    def iter_summoners():
        if tier in ("MASTER", "GRANDMASTER", "CHALLENGER"):
            entries = client.get_apex_league(tier)
            random.shuffle(entries)
            for e in entries:
                yield e.get("summonerId"), e.get("puuid")
        else:
            for division in DIVISIONS:
                page = 1
                while True:
                    entries = client.get_league_entries(tier, division, page)
                    if not entries:
                        break
                    random.shuffle(entries)
                    for e in entries:
                        yield e.get("summonerId"), e.get("puuid")
                    if len(entries) < 205:
                        break
                    page += 1

    start = time.time()
    summoner_count = 0

    for summoner_id, puuid_from_entry in iter_summoners():
        if len(seen_matches) >= target:
            break
        if not summoner_id or summoner_id in seen_summoners:
            continue

        seen_summoners.add(summoner_id)

        # puuid 조회 (league entry에 없으면 summoner API 호출)
        puuid = puuid_from_entry
        if not puuid:
            summoner = client.get_summoner(summoner_id)
            if not summoner:
                continue
            puuid = summoner.get("puuid")
        if not puuid:
            continue

        match_ids = client.get_match_ids(puuid, count=MATCHES_PER_SUMMONER)
        new_matches = 0

        for match_id in match_ids:
            if match_id in seen_matches:
                continue
            if len(seen_matches) >= target:
                break

            match_data = client.get_match(match_id)
            if not match_data:
                continue

            extracted = extract_match_stats(match_data)
            if extracted:
                stats.extend(extracted)
                seen_matches.add(match_id)
                new_matches += 1

        summoner_count += 1

        # 진행 상황 출력 및 체크포인트 (100경기마다)
        if len(seen_matches) % 100 < new_matches or summoner_count % 50 == 0:
            elapsed = time.time() - start
            rate = len(seen_matches) / elapsed if elapsed > 0 else 0
            remaining = max(0, target - len(seen_matches))
            eta = remaining / rate if rate > 0 else 0
            log.info(
                f"{tier}: {len(seen_matches):,}/{target:,} 경기 | "
                f"소환사 {summoner_count:,}명 | "
                f"속도 {rate:.1f}경기/s | "
                f"ETA {eta/60:.0f}분"
            )
            flush()

    flush()
    log.info(f"{tier} 완료 — {len(seen_matches):,}경기, {len(stats):,}개 스탯")
    return stats


# ─── 평균 계산 ─────────────────────────────────────────────────────────────────

def calculate_baseline(stats: List[dict]) -> dict:
    if not stats:
        return {}
    keys = ["kda", "cs_per_min", "gold_per_min", "vision_score_per_min", "damage_share"]
    totals = defaultdict(float)
    for s in stats:
        for k in keys:
            totals[k] += s.get(k, 0.0)
    n = len(stats)
    return {f"avg_{k}": round(totals[k] / n, 4) for k in keys}


# ─── 메인 ──────────────────────────────────────────────────────────────────────

def main():
    if not RIOT_API_KEY:
        log.error("RIOT_API_KEY가 .env 파일에 없습니다.")
        sys.exit(1)

    OUTPUT_DIR.mkdir(exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    client = RiotClient(RIOT_API_KEY)

    version = client.get_game_version()
    log.info(f"게임 버전: {version}")

    baselines = {}
    total_samples = 0

    for tier, target in TIER_TARGETS.items():
        try:
            stats = collect_tier(client, tier, target)
            if stats:
                baselines[tier] = calculate_baseline(stats)
                total_samples += len(stats)
                log.info(f"{tier} 베이스라인: {baselines[tier]}")
            else:
                log.warning(f"{tier}: 수집된 스탯 없음, 건너뜀")
        except KeyboardInterrupt:
            log.warning("사용자에 의해 중단됨 — 지금까지 수집된 데이터 저장 중...")
            break
        except Exception as e:
            log.error(f"{tier} 수집 오류: {e}")
            continue

    if not baselines:
        log.error("수집된 베이스라인 없음")
        sys.exit(1)

    # 파일명: tier_baselines_16.11.json (baseline_loader.py 형식 맞춤)
    output_path = OUTPUT_DIR / f"tier_baselines_{version}.json"
    output = {
        "metadata": {
            "trained_at":    datetime.now().isoformat(),
            "total_samples": total_samples,
            "matches_target_per_tier": TARGET_MATCHES_PER_TIER,
            "queue":         "RANKED_SOLO_5x5",
            "region":        "KR",
            "game_version":  version,
        },
        "baselines": baselines,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    log.info(f"\n저장 완료: {output_path}")
    log.info(f"총 샘플 수: {total_samples:,}")

    print("\n=== 최종 티어 베이스라인 ===")
    for tier, avg in baselines.items():
        print(f"\n{tier}:")
        for k, v in avg.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
