import os
import glob
import json
import pandas as pd
from typing import Dict, Any, Optional
from tqdm import tqdm

# ------------------------------------------------
# 설정
# ------------------------------------------------
RAW_ROOT = "data/raw"
OUTPUT_CSV = "data/processed/matches_advanced.csv"

# ------------------------------------------------
# 1) 파일 경로 생성기
# ------------------------------------------------
def iter_match_files(root: str):
    tier_dirs = glob.glob(os.path.join(root, "*"))
    for tier_dir in tier_dirs:
        if not os.path.isdir(tier_dir):
            continue
        tier = os.path.basename(tier_dir)
        pattern = os.path.join(tier_dir, "*.json")
        for fpath in glob.glob(pattern):
            yield fpath, tier

# ------------------------------------------------
# 2) 타임라인 데이터 로드
# ------------------------------------------------
def load_timeline_for_match(match_id: str, tier: str, root: str) -> Optional[Dict[str, Any]]:
    timeline_path = os.path.join(root, tier, "timeline", f"{match_id}.json")
    if not os.path.exists(timeline_path):
        return None
    try:
        with open(timeline_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

# ------------------------------------------------
# 3) 핵심: 특정 시간대의 종합 스탯 추출 (골드, XP, CS, 딜량)
# ------------------------------------------------
def extract_stats_at_minute(timeline_info: Dict[str, Any], minute: int) -> Dict[str, float]:
    """
    특정 시간(분)까지의 팀별 누적 스탯 차이(Team100 - Team200)를 계산합니다.
    """
    frames = timeline_info.get("frames", [])
    if not frames:
        return {}

    target_ms = minute * 60 * 1000
    candidate = None
    
    # 해당 시간과 가장 가까운(이전) 프레임 찾기
    for fr in frames:
        ts = fr.get("timestamp")
        if ts is None: continue
        if ts <= target_ms:
            candidate = fr
        else:
            break
            
    if candidate is None:
        return {}

    pf = candidate.get("participantFrames", {})
    
    stats = {
        "gold_100": 0, "gold_200": 0,
        "xp_100": 0, "xp_200": 0,
        "cs_100": 0, "cs_200": 0,
        "dmg_100": 0, "dmg_200": 0
    }
    
    for pid, pdata in pf.items():
        try:
            pid_int = int(pid)
            
            # 데이터 추출
            total_gold = pdata.get("totalGold", 0)
            xp = pdata.get("xp", 0)
            cs = pdata.get("minionsKilled", 0) + pdata.get("jungleMinionsKilled", 0)
            dmg = pdata.get("damageStats", {}).get("totalDamageDoneToChampions", 0)

            # 랭크 게임 하드코딩 (1~5: 블루, 6~10: 레드)
            if 1 <= pid_int <= 5:
                stats["gold_100"] += total_gold
                stats["xp_100"] += xp
                stats["cs_100"] += cs
                stats["dmg_100"] += dmg
            elif 6 <= pid_int <= 10:
                stats["gold_200"] += total_gold
                stats["xp_200"] += xp
                stats["cs_200"] += cs
                stats["dmg_200"] += dmg
                
        except ValueError:
            continue

    # 차이(Diff) 계산 (블루팀 기준)
    return {
        f"gold_diff_{minute}": stats["gold_100"] - stats["gold_200"],
        f"xp_diff_{minute}": stats["xp_100"] - stats["xp_200"],
        f"cs_diff_{minute}": stats["cs_100"] - stats["cs_200"],
        f"dmg_diff_{minute}": stats["dmg_100"] - stats["dmg_200"]
    }

# ------------------------------------------------
# 4) 이벤트 기반 Feature 추출 (오브젝트, 시야, 포탑방패, 무력행사)
# ------------------------------------------------
def extract_event_features(timeline_info: Dict[str, Any]) -> Dict[str, Any]:
    frames = timeline_info.get("frames", [])
    
    # 초기화
    first_tower_time = None
    first_dragon_time = None
    first_baron_time = None
    
    # 아타칸 / 무력행사 관련
    atakhan_events = []
    feat_state: Dict[tuple, int] = {} # (teamId, featType) -> value
    
    # 시야 / 포탑 방패
    wards_placed_100 = 0
    wards_placed_200 = 0
    plates_100 = 0
    plates_200 = 0
    
    # 15분 기준 (시야, 방패 등은 초반 스노우볼링 지표이므로 15분까지만 집계)
    LIMIT_MS = 15 * 60 * 1000 

    for fr in frames:
        for ev in fr.get("events", []):
            ts = ev.get("timestamp")
            etype = ev.get("type")
            
            # 1. 첫 오브젝트 시간
            if etype == "BUILDING_KILL" and ev.get("buildingType") == "TOWER_BUILDING":
                if first_tower_time is None: first_tower_time = ts
            
            elif etype == "ELITE_MONSTER_KILL":
                mtype = ev.get("monsterType")
                if mtype == "DRAGON" and first_dragon_time is None:
                    first_dragon_time = ts
                elif mtype == "BARON_NASHOR" and first_baron_time is None:
                    first_baron_time = ts
                elif mtype == "ATAKHAN":
                    atakhan_events.append(ev)
            
            # 2. 무력행사 (Force of Will) 상태 추적
            elif etype == "FEAT_UPDATE":
                if ev.get("teamId") and ev.get("featType") is not None:
                    feat_state[(ev["teamId"], ev["featType"])] = ev["featValue"]

            # 3. 15분 제한 데이터 (시야, 포탑 방패)
            if ts <= LIMIT_MS:
                if etype == "WARD_PLACED":
                    creator = ev.get("creatorId", 0)
                    if 1 <= creator <= 5: wards_placed_100 += 1
                    elif 6 <= creator <= 10: wards_placed_200 += 1
                
                elif etype == "TURRET_PLATE_DESTROYED":
                    # killerId가 없거나 0인 경우(미니언 처치 등)는 teamId로 판단
                    team_id = 0
                    if "killerId" in ev and 1 <= ev["killerId"] <= 5:
                        team_id = 100
                    elif "killerId" in ev and 6 <= ev["killerId"] <= 10:
                        team_id = 200
                    # killerId가 명확하지 않으면 laneType 등으로 추정해야 하지만 여기선 생략
                    
                    if team_id == 100: plates_100 += 1
                    elif team_id == 200: plates_200 += 1

    # 아타칸 킬 수
    atakhan_kills_100 = sum(1 for e in atakhan_events if e.get("killerTeamId") == 100)
    atakhan_kills_200 = sum(1 for e in atakhan_events if e.get("killerTeamId") == 200)
    
    atakan_diff = atakhan_kills_100 - atakhan_kills_200

    # 무력행사 버프 획득 여부 (2개 이상 조건 달성 시)
    def get_force_buff_team(state):
        for team in [100, 200]:
            cnt = 0
            for ft in [0, 1, 2]:
                v = state.get((team, ft))
                if v is not None and 0 < v < 1000: # 성공 코드 (보통 1~3)
                    cnt += 1
            if cnt >= 2: return team
        return 0

    force_buff_team = get_force_buff_team(feat_state)

    def to_sec(t): return t / 1000.0 if t else None

    return {
        "first_tower_time": to_sec(first_tower_time),
        "first_dragon_time": to_sec(first_dragon_time),
        "first_baron_time": to_sec(first_baron_time),
        "atakan_diff": atakan_diff,
        "force_buff_team": force_buff_team,
        # 15분 데이터
        "ward_placed_diff_15": wards_placed_100 - wards_placed_200,
        "plates_diff_15": plates_100 - plates_200
    }

# ------------------------------------------------
# 5) 메인 로직: 매치별 데이터 병합
# ------------------------------------------------
def process_match(match_json: Dict[str, Any], tier: str, root: str) -> Optional[Dict[str, Any]]:
    info = match_json.get("info", {})
    meta = match_json.get("metadata", {})
    match_id = meta.get("matchId")
    
    if not info or not match_id: return None

    try:
        team100 = next(t for t in info["teams"] if t["teamId"] == 100)
        team200 = next(t for t in info["teams"] if t["teamId"] == 200)
        winner = 1 if team100.get("win") else 0
    except Exception:
        return None

    duration = info["gameDuration"]
    if "gameEndTimestamp" not in info: duration /= 1000.0
    
    row = {
        "matchId": match_id,
        "tier": tier,
        "winner": winner,
        "duration": duration
    }

    # 오브젝트 횟수 차이 (team100 - team200)
    try:
        obj100 = team100.get("objectives", {})
        obj200 = team200.get("objectives", {})

        row["dragon_diff"] = obj100.get("dragon", {}).get("kills", 0) - obj200.get("dragon", {}).get("kills", 0)
        row["baron_diff"] = obj100.get("baron", {}).get("kills", 0) - obj200.get("baron", {}).get("kills", 0)
        row["herald_diff"] = obj100.get("riftHerald", {}).get("kills", 0) - obj200.get("riftHerald", {}).get("kills", 0)
    except Exception:
        # objectives 정보가 예상과 다를 경우를 대비해 조용히 0으로 채운다
        row.setdefault("dragon_diff", 0)
        row.setdefault("baron_diff", 0)
        row.setdefault("herald_diff", 0)

    # 타임라인 데이터 로드 및 Feature 추가
    timeline = load_timeline_for_match(match_id, tier, root)
    if timeline:
        # 15분 시점의 종합 스탯 (XP, CS, DMG, GOLD)
        stats_15 = extract_stats_at_minute(timeline.get("info", {}), 15)
        row.update(stats_15)
        
        # 이벤트 기반 스탯 (오브젝트, 시야, 무력행사)
        event_feats = extract_event_features(timeline.get("info", {}))
        row.update(event_feats)
    
    return row

def build_dataset(raw_root: str, out_path: str):
    all_files = list(iter_match_files(raw_root))
    print(f"총 {len(all_files)}개 파일 처리 시작...")
    
    rows = []
    for fpath, tier in tqdm(all_files, desc="Processing"):
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                match_json = json.load(f)
            
            row = process_match(match_json, tier, raw_root)
            if row and row["duration"] >= 180: # 3분 이상 게임만
                rows.append(row)
        except Exception:
            continue

    # DataFrame 변환 및 저장
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df = pd.DataFrame(rows)
    
    # 결측치 처리 (타임라인 없는 경우 등 0으로 채우기)
    df.fillna(0, inplace=True)
    
    df.to_csv(out_path, index=False)
    print(f"저장 완료: {out_path} (총 {len(df)}행)")

if __name__ == "__main__":
    build_dataset(RAW_ROOT, OUTPUT_CSV)