import os
import asyncio
import base64
import logging
import requests
from typing import Optional

log = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

# 레벨별 데스 타이머 (초) — LoL 위키 기준 근사값
DEATH_TIMERS = {
    1: 10, 2: 10, 3: 12, 4: 12, 5: 14, 6: 16,
    7: 20, 8: 24, 9: 28, 10: 32, 11: 36,
    12: 40, 13: 44, 14: 48, 15: 54, 16: 60,
    17: 66, 18: 72,
}
SNOWBALL_EXTRA_SEC = 15  # 리스폰 후 추가 측정 시간


# ─── 타임라인 파싱 ────────────────────────────────────────────────────────────

def _build_snapshots(frames: list) -> list[dict]:
    """프레임별 누적 골드·킬스코어 스냅샷을 만든다."""
    blue_kills = red_kills = 0
    snapshots = []
    for frame in frames:
        ts_ms = frame.get("timestamp", 0)
        pf = frame.get("participantFrames", {})

        blue_gold = sum(pf.get(str(i), {}).get("totalGold", 0) for i in range(1, 6))
        red_gold  = sum(pf.get(str(i), {}).get("totalGold", 0) for i in range(6, 11))

        for event in frame.get("events", []):
            if event.get("type") == "CHAMPION_KILL":
                killer = event.get("killerId", 0)
                if 1 <= killer <= 5:
                    blue_kills += 1
                elif 6 <= killer <= 10:
                    red_kills += 1

        snapshots.append({
            "ts_ms": ts_ms,
            "blue_gold": blue_gold,
            "red_gold": red_gold,
            "blue_kills": blue_kills,
            "red_kills": red_kills,
            "pf": pf,
        })
    return snapshots


def _snap_at_or_before(snapshots: list, target_ms: int) -> Optional[dict]:
    result = None
    for s in snapshots:
        if s["ts_ms"] <= target_ms:
            result = s
        else:
            break
    return result


def _snap_at_or_after(snapshots: list, target_ms: int) -> Optional[dict]:
    for s in snapshots:
        if s["ts_ms"] >= target_ms:
            return s
    return snapshots[-1] if snapshots else None


def extract_game_moments(timeline_data: dict, participant_id: int, top_n: int = 5) -> list[dict]:
    """
    플레이어의 킬/데스 이벤트별로 스노우볼 정보를 반환한다.

    - 킬: 직전 프레임 vs 이벤트+15초 프레임
    - 데스: 직전 프레임 vs (데스타이머 + 15초) 프레임
      → 리스폰 후 15초까지 적이 얼마나 이득 봤는지 측정
    """
    frames = timeline_data.get("info", {}).get("frames", [])
    if not frames:
        return []

    player_team = "blue" if participant_id <= 5 else "red"
    enemy_team  = "red"  if player_team == "blue" else "blue"

    snapshots = _build_snapshots(frames)

    # 플레이어의 킬/데스 이벤트 수집
    raw_events = []
    for frame in frames:
        for event in frame.get("events", []):
            if event.get("type") != "CHAMPION_KILL":
                continue
            killer = event.get("killerId", 0)
            victim = event.get("victimId", 0)
            ts_ms  = event.get("timestamp", 0)
            if killer == participant_id:
                raw_events.append({"kind": "kill", "ts_ms": ts_ms})
            elif victim == participant_id:
                raw_events.append({"kind": "death", "ts_ms": ts_ms})

    raw_events = raw_events[:top_n]

    moments = []
    for ev in raw_events:
        ts_ms  = ev["ts_ms"]
        minute = ts_ms // 60000
        second = (ts_ms % 60000) // 1000

        pre = _snap_at_or_before(snapshots, max(0, ts_ms - 1000))
        if not pre:
            continue

        if ev["kind"] == "death":
            # 데스 직전 프레임에서 플레이어 레벨 조회
            level = pre["pf"].get(str(participant_id), {}).get("level", 6)
            death_timer_sec = DEATH_TIMERS.get(level, 30)
            window_ms = (death_timer_sec + SNOWBALL_EXTRA_SEC) * 1000
            label = f"리스폰({death_timer_sec}초)+15초 후"
        else:
            window_ms = 15_000
            label = "15초 후"

        post = _snap_at_or_after(snapshots, ts_ms + window_ms)
        if not post:
            continue

        pre_diff  = pre[f"{player_team}_gold"]  - pre[f"{enemy_team}_gold"]
        post_diff = post[f"{player_team}_gold"] - post[f"{enemy_team}_gold"]
        gold_delta = post_diff - pre_diff

        moments.append({
            "kind":       ev["kind"],
            "time":       f"{minute}분{second:02d}초",
            "label":      label,
            "pre_score":  f"{pre['blue_kills']}:{pre['red_kills']}",
            "post_score": f"{post['blue_kills']}:{post['red_kills']}",
            "pre_gold_diff":  pre_diff,
            "post_gold_diff": post_diff,
            "gold_delta": gold_delta,
        })

    return moments


# ─── 프롬프트 빌드 ────────────────────────────────────────────────────────────

def _fmt_gold(g: int) -> str:
    return f"+{g:,}G" if g >= 0 else f"{g:,}G"


def build_prompt(
    champion: str,
    role: str,
    win: bool,
    tier: str,
    kda: str,
    cs_per_min: float,
    damage_share: float,
    vision_score: float,
    gold_per_min: float,
    game_duration_min: float,
    strengths: list[str],
    weaknesses: list[str],
    impact_score: float,
    moments: list[dict],
) -> str:
    cs_per_min = cs_per_min or 0.0
    damage_share = damage_share or 0.0
    vision_score = vision_score or 0.0
    gold_per_min = gold_per_min or 0.0
    impact_score = impact_score or 0.0

    result = "승리" if win else "패배"

    moments_text = ""
    if moments:
        lines = []
        for m in moments:
            kind = "킬" if m["kind"] == "kill" else "데스"
            lines.append(
                f"  - {m['time']} [{kind}] "
                f"직전: 스코어 {m['pre_score']}, 골드차 {_fmt_gold(m['pre_gold_diff'])} → "
                f"{m['label']}: 스코어 {m['post_score']}, 골드차 {_fmt_gold(m['post_gold_diff'])} "
                f"(스노우볼 {_fmt_gold(m['gold_delta'])})"
            )
        moments_text = "\n[주요 순간 및 스노우볼]\n" + "\n".join(lines)

    return f"""당신은 리그 오브 레전드 코치입니다. 아래 경기 데이터를 보고 플레이어에게 구체적인 피드백을 한국어로 제공해주세요.

[경기 정보]
- 챔피언: {champion} / 포지션: {role}
- 결과: {result} / 현재 티어: {tier}
- 게임 시간: {game_duration_min:.0f}분
- KDA: {kda}
- 분당 CS: {cs_per_min:.1f}
- 딜 기여도: {damage_share:.1f}%
- 분당 시야점수: {vision_score:.2f}
- 분당 골드: {gold_per_min:.1f}
- 임팩트 스코어 (승리기여도): {impact_score:.1f}점 (0~100)
{moments_text}

위 데이터를 바탕으로 아래 형식으로 정확히 답해주세요.
추상적인 표현("CS가 부족하다") 대신 구체적인 행동("라인전에서 CS를 놓치는 대신 하드 갱킹 타이밍을 잡아야 했습니다")으로 작성하세요.
스노우볼 데이터가 있다면 해당 시간대 이벤트를 직접 언급하세요. 각 항목은 1~2문장으로 작성하세요.

강점1: (이 게임에서 잘한 구체적인 플레이)
강점2: (이 게임에서 잘한 구체적인 플레이)
강점3: (이 게임에서 잘한 구체적인 플레이)
약점1: (이 게임에서 잘못된 구체적인 결정 또는 행동)
약점2: (이 게임에서 잘못된 구체적인 결정 또는 행동)
약점3: (이 게임에서 잘못된 구체적인 결정 또는 행동)
개선1: (다음 게임에서 바로 실천할 수 있는 행동 변화)
개선2: (다음 게임에서 바로 실천할 수 있는 행동 변화)
개선3: (다음 게임에서 바로 실천할 수 있는 행동 변화)"""


# ─── Gemini 호출 ──────────────────────────────────────────────────────────────

def parse_response(text: str) -> dict:
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    strengths, weaknesses, improvements = [], [], []
    for line in lines:
        if line.startswith("강점"):
            val = line.split(":", 1)[-1].strip()
            if val:
                strengths.append(val)
        elif line.startswith("약점"):
            val = line.split(":", 1)[-1].strip()
            if val:
                weaknesses.append(val)
        elif line.startswith("개선"):
            val = line.split(":", 1)[-1].strip()
            if val:
                improvements.append(val)
    return {"strengths": strengths, "weaknesses": weaknesses, "improvements": improvements}


def get_coaching(
    champion: str,
    role: str,
    win: bool,
    tier: str,
    kda: str,
    cs_per_min: float,
    damage_share: float,
    vision_score: float,
    gold_per_min: float,
    game_duration_min: float,
    strengths: list[str],
    weaknesses: list[str],
    impact_score: float,
    timeline_data: Optional[dict] = None,
    participant_id: Optional[int] = None,
) -> Optional[dict]:
    if not GEMINI_API_KEY:
        log.warning("GEMINI_API_KEY not set — skipping LLM coaching")
        return None

    moments = []
    if timeline_data and participant_id:
        try:
            moments = extract_game_moments(timeline_data, participant_id)
        except Exception as e:
            log.warning("Failed to extract game moments: %s", e)

    prompt = build_prompt(
        champion, role, win, tier, kda,
        cs_per_min, damage_share, vision_score, gold_per_min,
        game_duration_min, strengths, weaknesses, impact_score, moments,
    )

    try:
        resp = requests.post(
            f"{GEMINI_URL}?key={GEMINI_API_KEY}",
            json={"contents": [{"parts": [{"text": prompt}]}]},
            timeout=15,
        )
        if resp.status_code != 200:
            log.warning("Gemini API error %d: %s", resp.status_code, resp.text[:200])
            return None

        data = resp.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        result = parse_response(text)

        if not result["strengths"] and not result["weaknesses"]:
            log.warning("Gemini response could not be parsed")
            return None

        return result

    except Exception as e:
        log.warning("Gemini coaching failed: %s", e)
        return None


# ─── 클립 영상 분석 ───────────────────────────────────────────────────────────

def analyze_clip(
    clip_path: str,
    event_kind: str,       # "kill" | "death"
    time_str: str,         # 예: "22분45초"
    pre_score: str,        # 예: "9:11"
    pre_gold_diff: int,    # 직전 골드차 (아군 - 적군)
    snowball_gold: int,    # 스노우볼 골드 변화
    snowball_label: str,   # 예: "리스폰(36초)+15초 후"
    champion: str,
    role: str,
) -> Optional[str]:
    """
    15초 클립 영상 + 게임 상황을 Gemini Vision에 전달해 플레이 피드백을 반환합니다.
    실패 시 None 반환 (fallback 없음 — 텍스트 코칭보다 없는 게 나음).
    """
    if not GEMINI_API_KEY:
        return None

    if not os.path.exists(clip_path):
        log.warning("Clip file not found: %s", clip_path)
        return None

    try:
        with open(clip_path, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        log.warning("Failed to read clip: %s", e)
        return None

    kind_map = {"kill": "킬", "death": "데스", "objective": "오브젝트 획득"}
    kind_str = kind_map.get(event_kind, event_kind)
    gold_diff_str = _fmt_gold(pre_gold_diff)
    snowball_str  = _fmt_gold(snowball_gold)

    prompt = f"""당신은 리그 오브 레전드 코치입니다. 이 15초 클립을 보고 플레이에 대한 구체적인 피드백을 2~3문장으로 제공해주세요.

[상황 정보]
- 챔피언: {champion} ({role})
- 이벤트: {time_str} [{kind_str}]
- 당시 킬 스코어: {pre_score}
- 당시 골드차: {gold_diff_str} (양수=아군 우세, 음수=적군 우세)
- {snowball_label}: {snowball_str} 스노우볼 발생

영상을 보고 실제 플레이에서 어떤 결정이 잘못됐는지 또는 잘됐는지 구체적으로 분석해주세요.
예시 피드백 스타일: "시야 없이 강 쪽으로 진입한 것이 원인입니다", "상대 스킬을 맞은 상태로 교전을 시작했습니다", "킬 이후 오브젝트를 챙기지 않고 귀환했습니다"
추상적인 표현("포지셔닝이 아쉽다") 대신 영상에서 보이는 행동을 직접 지적해주세요."""

    try:
        resp = requests.post(
            f"{GEMINI_URL}?key={GEMINI_API_KEY}",
            json={
                "contents": [{
                    "parts": [
                        {
                            "inlineData": {
                                "mimeType": "video/mp4",
                                "data": video_b64,
                            }
                        },
                        {"text": prompt},
                    ]
                }]
            },
            timeout=300,
        )

        if resp.status_code != 200:
            log.warning("Gemini clip analysis error %d: %s", resp.status_code, resp.text[:200])
            return None

        data = resp.json()
        coaching = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        log.info("Clip coaching generated for %s %s", event_kind, time_str)
        return coaching

    except Exception as e:
        log.warning("Gemini clip analysis failed: %s", e)
        return None


async def analyze_clip_async(
    clip_path: str,
    event_kind: str,
    time_str: str,
    pre_score: str,
    pre_gold_diff: int,
    snowball_gold: int,
    snowball_label: str,
    champion: str,
    role: str,
) -> Optional[str]:
    """analyze_clip의 비동기 래퍼 — asyncio.to_thread로 스레드풀에서 실행"""
    return await asyncio.to_thread(
        analyze_clip,
        clip_path, event_kind, time_str, pre_score,
        pre_gold_diff, snowball_gold, snowball_label,
        champion, role,
    )
