import os
import glob
import json
import pandas as pd
from typing import Dict, Any, Optional, List
from tqdm import tqdm

# ------------------------------------------------
# 설정
# ------------------------------------------------
RAW_ROOT = "data/raw"
OUTPUT_CSV = "data/processed/player_stats.csv"

# ------------------------------------------------
# 1) 파일 경로 생성기 (기존 preprocess.py와 동일)
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
# 2) 타임라인 로드 (기존 함수와 동일)
#    matchId + tier 기준으로 timeline 파일을 찾는다.
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
# 3) 타임라인에서 특정 플레이어의 15분 스탯 추출
#    totalGold, xp, cs 정도만 유저 단위로 가져온다.
# ------------------------------------------------
def extract_player_15min_stats(timeline_info: Dict[str, Any], participant_id: int) -> Dict[str, float]:
    frames = timeline_info.get("frames", [])
    if not frames:
        return {}

    target_ms = 15 * 60 * 1000
    candidate = None
    for fr in frames:
        ts = fr.get("timestamp")
        if ts is None:
            continue
        if ts <= target_ms:
            candidate = fr
        else:
            break

    if candidate is None:
        return {}

    pf = candidate.get("participantFrames", {})
    pframe = pf.get(str(participant_id))
    if pframe is None:
        return {}

    total_gold = pframe.get("totalGold", 0)
    xp = pframe.get("xp", 0)
    cs = pframe.get("minionsKilled", 0) + pframe.get("jungleMinionsKilled", 0)

    return {
        "gold_15": total_gold,
        "xp_15": xp,
        "cs_15": cs
    }

# ------------------------------------------------
# 3.5) 타임라인에서 objective 참여도 추출
# ------------------------------------------------
def extract_objective_participation(timeline_info: Dict[str, Any], participants: List[Dict[str, Any]]) -> Dict[int, Dict[str, float]]:
    pid_to_team = {p.get("participantId"): p.get("teamId") for p in participants}

    keys = ["dragon", "baron", "atakan", "herald", "tower", "plate"]
    player_counts = {pid: {k: 0 for k in keys} for pid in pid_to_team}
    team_counts = {100: {k: 0 for k in keys}, 200: {k: 0 for k in keys}}

    for frame in timeline_info.get("frames", []):
        for event in frame.get("events", []):
            etype = event.get("type")
            if etype == "ELITE_MONSTER_KILL":
                monster_type = event.get("monsterType")
                mapping = {
                    "DRAGON": "dragon",
                    "BARON_NASHOR": "baron",
                    "ATAKHAN": "atakan",
                    "RIFTHERALD": "herald"
                }
                key = mapping.get(monster_type)
                if key is None:
                    continue
                team_id = event.get("killerTeamId")
                if team_id not in (100, 200):
                    continue
                team_counts[team_id][key] += 1

                killer_id = event.get("killerId")
                if killer_id in player_counts:
                    player_counts[killer_id][key] += 1
                assisting_ids = event.get("assistingParticipantIds", [])
                for aid in assisting_ids:
                    if aid in player_counts:
                        player_counts[aid][key] += 1

            elif etype == "BUILDING_KILL":
                building_type = event.get("buildingType")
                if building_type == "TOWER_BUILDING":
                    key = "tower"

                    killer_id = event.get("killerId")
                    killer_team = pid_to_team.get(killer_id)
                    if killer_team not in (100, 200):
                        continue
                    team_counts[killer_team][key] += 1

                    if killer_id in player_counts:
                        player_counts[killer_id][key] += 1
                    assisting_ids = event.get("assistingParticipantIds", [])
                    for aid in assisting_ids:
                        if aid in player_counts:
                            player_counts[aid][key] += 1

            elif etype == "TURRET_PLATE_DESTROYED":
                key = "plate"

                killer_id = event.get("killerId")
                killer_team = pid_to_team.get(killer_id)
                if killer_team not in (100, 200):
                    continue
                team_counts[killer_team][key] += 1

                if killer_id in player_counts:
                    player_counts[killer_id][key] += 1
                assisting_ids = event.get("assistingParticipantIds", [])
                for aid in assisting_ids:
                    if aid in player_counts:
                        player_counts[aid][key] += 1

    result = {}
    for pid, counts in player_counts.items():
        team_id = pid_to_team.get(pid)
        feats = {}
        for key in keys:
            team_total = team_counts[team_id][key] if team_id in team_counts else 0
            ratio = counts[key] / team_total if team_total > 0 else 0.0
            feats[f"{key}_participation"] = ratio
        result[pid] = feats

    return result

# ------------------------------------------------
# 4) 한 경기에서 "유저 단위 row"들 생성
# ------------------------------------------------
def process_player_rows(match_json: Dict[str, Any],
                        tier: str,
                        timeline: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    info = match_json.get("info", {})
    meta = match_json.get("metadata", {})
    match_id = meta.get("matchId")
    if not info or not match_id:
        return []

    participants = info.get("participants", [])
    if len(participants) != 10:
        # 랭크게임이 아닌 경우 등은 스킵
        return []

    # 팀별 합계 데미지/골드 계산 → share 계산용
    team_totals = {
        100: {"dmg": 0, "gold": 0},
        200: {"dmg": 0, "gold": 0},
    }
    for p in participants:
        t = p.get("teamId")
        dmg = p.get("totalDamageDealtToChampions", 0)
        gold = p.get("goldEarned", 0)
        if t in team_totals:
            team_totals[t]["dmg"] += dmg
            team_totals[t]["gold"] += gold

    # 타임라인 info
    tl_info = timeline.get("info", {}) if timeline else None

    # ------------------------------------------------
    # objective 참여도 미리 계산
    # ------------------------------------------------
    if tl_info is not None:
        obj_participation = extract_objective_participation(tl_info, participants)
    else:
        obj_participation = {}

    rows = []
    for p in participants:
        team_id = p.get("teamId")
        if team_id not in (100, 200):
            continue

        win = 1 if p.get("win") else 0
        dmg = p.get("totalDamageDealtToChampions", 0)
        gold = p.get("goldEarned", 0)

        # 기본 스탯
        row = {
            "matchId": match_id,
            "tier": tier,
            "teamId": team_id,
            "win": win,
            "puuid": p.get("puuid"),
            "champion": p.get("championName"),
            "role": p.get("teamPosition"),  # TOP/JUNGLE/MID/BOT/UTILITY
            "kills": p.get("kills", 0),
            "deaths": p.get("deaths", 0),
            "assists": p.get("assists", 0),
            "kda": (p.get("kills", 0) + p.get("assists", 0)) / max(1, p.get("deaths", 0)),
            "totalDamage": dmg,
            "visionScore": p.get("visionScore", 0),
        }

        # 팀 대비 비율
        team_d = team_totals[team_id]["dmg"]
        team_g = team_totals[team_id]["gold"]
        row["dmg_share"] = dmg / team_d if team_d > 0 else 0.0
        row["gold_share"] = gold / team_g if team_g > 0 else 0.0

        # 15분 스탯 (타임라인이 있는 경우만)
        if tl_info is not None:
            pid = p.get("participantId")
            if pid is not None:
                p15 = extract_player_15min_stats(tl_info, pid)
                row.update(p15)

        # objective 참여도 업데이트
        pid = p.get("participantId")
        if obj_participation and pid in obj_participation:
            row.update(obj_participation[pid])

        rows.append(row)

    return rows

# ------------------------------------------------
# 5) 전체 데이터셋 빌드
# ------------------------------------------------
def build_player_dataset(raw_root: str, out_path: str):
    all_files = list(iter_match_files(raw_root))
    print(f"총 {len(all_files)}개 매치 파일에서 플레이어 스탯 생성...")

    all_rows = []
    for fpath, tier in tqdm(all_files, desc="Processing(player)"):
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                match_json = json.load(f)

            meta = match_json.get("metadata", {})
            match_id = meta.get("matchId")
            if not match_id:
                continue

            # 타임라인 로드
            timeline = load_timeline_for_match(match_id, tier, raw_root)

            rows = process_player_rows(match_json, tier, timeline)
            all_rows.extend(rows)
        except Exception:
            # 문제가 있는 매치는 스킵
            continue

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df = pd.DataFrame(all_rows)

    # 결측치 기본 0으로 처리 (puuid 등 문자열 컬럼은 건드리지 않음)
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    df[num_cols] = df[num_cols].fillna(0)

    df.to_csv(out_path, index=False)
    print(f"저장 완료: {out_path} (총 {len(df)}행)")

if __name__ == "__main__":
    build_player_dataset(RAW_ROOT, OUTPUT_CSV)