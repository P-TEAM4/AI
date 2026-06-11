import os
import asyncio
import logging
import time
import requests
from typing import Optional

log = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"

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


SUMMONER_SPELL_NAMES = {
    1: "정화", 3: "탈진", 4: "점멸", 6: "유령질주",
    7: "회복", 11: "강타", 12: "순간이동", 13: "투명",
    14: "점화", 21: "방어막", 32: "표식",
}


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
    summoner_spells: Optional[tuple[int, int]] = None,
    ally_champions: Optional[list[str]] = None,
    enemy_champions: Optional[list[str]] = None,
) -> str:
    cs_per_min = cs_per_min or 0.0
    damage_share = damage_share or 0.0
    vision_score = vision_score or 0.0
    gold_per_min = gold_per_min or 0.0
    impact_score = impact_score or 0.0

    result = "승리" if win else "패배"

    spell1, spell2 = summoner_spells or (0, 0)
    spell1_name = SUMMONER_SPELL_NAMES.get(spell1, f"주문{spell1}")
    spell2_name = SUMMONER_SPELL_NAMES.get(spell2, f"주문{spell2}")
    spells_text = f"{spell1_name} / {spell2_name}"

    allies_text = ", ".join(ally_champions) if ally_champions else "정보 없음"
    enemies_text = ", ".join(enemy_champions) if enemy_champions else "정보 없음"

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

    return f"""당신은 챌린저 출신 리그 오브 레전드 전문 코치입니다. 첨부된 경기 영상과 아래 데이터를 함께 분석하여 플레이어에게 한국어로 코칭 피드백을 제공하세요.

[경기 정보]
- 플레이어 챔피언: {champion} ({role}) / 소환사 주문: {spells_text}
- 아군 팀: {allies_text}
- 적군 팀: {enemies_text}
- 결과: {result} / 현재 티어: {tier}
- 게임 시간: {game_duration_min:.0f}분
- KDA: {kda}
- 분당 CS: {cs_per_min:.1f}
- 딜 기여도: {damage_share:.1f}%
- 분당 시야점수: {vision_score:.2f}
- 분당 골드: {gold_per_min:.1f}
- 임팩트 스코어 (승리기여도): {impact_score:.1f}점 (0~100)
{moments_text}

[분석 기준]
- {tier} 티어 평균과 비교해 어떤 지표가 부족하고 어떤 지표가 우수한지 기준점을 제시하세요. (예: "{tier} {role} 기준 분당 CS 평균은 약 X인데...")
- 약점은 "결과"가 아니라 "원인이 된 의사결정"을 지적하세요. ("죽었다"가 아니라 "와드 없이 적 정글 위치 미확인 상태에서 라인을 깊게 밀었다")
- 개선 사항은 다음 게임에서 측정 가능하고 즉시 실천 가능한 단일 행동으로 제시하세요. (예: "리콜 직후 복귀 시 미니맵을 3초간 확인하고 적 정글 동선을 예측한 뒤 라인 복귀")
- 가장 승패에 큰 영향을 준 순간 1개를 골라 우선적으로 다루세요.

[제약]
- 위 데이터에 명시된 챔피언과 소환사 주문만 언급하세요. 데이터에 없는 챔피언, 스킬, 아이템은 절대 추측하거나 언급하지 마세요.
- 추상적인 표현("CS가 부족하다") 대신 구체적인 행동과 수치 기반 서술을 사용하세요.
- 각 항목은 2~3문장으로 작성하고, 가능한 한 수치를 포함하세요.

아래 형식으로 정확히 답하세요.

총평: (이 게임의 승패를 가른 핵심 요인 1가지를 2~3문장으로 요약)
강점1: (잘한 구체적인 플레이 + 수치 근거)
강점2: (잘한 구체적인 플레이 + 수치 근거)
강점3: (데이터 지표 기반 강점 + 티어 평균 대비 평가)
약점1: (가장 치명적이었던 의사결정 + 그 직전 상황에서 했어야 할 판단)
약점2: (반복적으로 나타난 나쁜 습관 + 수치 근거)
약점3: (데이터 지표 기반 약점 + 티어 평균 대비 평가)
개선1: (약점1을 고치기 위한, 다음 게임에서 즉시 실천 가능한 단일 행동)
개선2: (약점2를 고치기 위한, 다음 게임에서 즉시 실천 가능한 단일 행동)
개선3: (약점3을 고치기 위한, 측정 가능한 수치 목표 포함 행동)"""


# ─── 경기 분석 전용 프롬프트 (텍스트 데이터 기반, 영상 없음) ─────────────────────

def build_match_analysis_prompt(
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
    summoner_spells: Optional[tuple[int, int]] = None,
    ally_champions: Optional[list[str]] = None,
    enemy_champions: Optional[list[str]] = None,
) -> str:
    cs_per_min = cs_per_min or 0.0
    damage_share = damage_share or 0.0
    vision_score = vision_score or 0.0
    gold_per_min = gold_per_min or 0.0
    impact_score = impact_score or 0.0

    result = "승리" if win else "패배"

    spell1, spell2 = summoner_spells or (0, 0)
    spell1_name = SUMMONER_SPELL_NAMES.get(spell1, f"주문{spell1}")
    spell2_name = SUMMONER_SPELL_NAMES.get(spell2, f"주문{spell2}")
    spells_text = f"{spell1_name} / {spell2_name}"

    allies_text = ", ".join(ally_champions) if ally_champions else "정보 없음"
    enemies_text = ", ".join(enemy_champions) if enemy_champions else "정보 없음"

    if role in ("BOTTOM", "UTILITY"):
        bot_partner = ally_champions[0] if ally_champions else "불명확"
        lane_context = (
            f"바텀 2v2 라인 — 아군 파트너: {bot_partner} / 적군 전체: {enemies_text}\n"
            f"  (라인전은 원딜+서포트 2v2 구도로 평가하세요)"
        )
    else:
        lane_context = f"적군 팀: {enemies_text}"

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
        moments_text = "\n[주요 순간 타임라인]\n" + "\n".join(lines)

    return f"""당신은 챌린저 출신 리그 오브 레전드 전문 코치입니다. 아래 경기 데이터만을 기반으로 한국어로 경기 전체를 단계별로 분석하세요. 영상은 없으며 수치와 타임라인 데이터만 근거로 사용하세요.

[경기 정보]
- 플레이어 챔피언: {champion} ({role}) / 소환사 주문: {spells_text}
- 아군 팀: {allies_text}
- {lane_context}
- 결과: {result} / 현재 티어: {tier} / 게임 시간: {game_duration_min:.0f}분
- KDA: {kda} / 분당 CS: {cs_per_min:.1f} / 딜 기여도: {damage_share:.1f}%
- 분당 시야점수: {vision_score:.2f} / 분당 골드: {gold_per_min:.1f}
- 임팩트 스코어 (0~100): {impact_score:.1f}점
{moments_text}

[분석 지침]
- 각 단계(라인전·중반전·후반전)를 타임라인 데이터와 수치를 근거로 평가하세요.
- {tier} 티어 {role} 평균 수치와 비교해 우열을 명시하세요.
- 의사결정 평가는 "결과"가 아닌 "원인"을 서술하세요. ("죽었다" → "골드차 불리한 상황에서 시야 없이 라인을 과도하게 밀었다")
- 핵심패턴은 게임 내내 반복된 좋은/나쁜 습관 1가지씩을 꼽으세요.
- 개선점은 다음 게임에서 즉시 실천 가능한 단일 행동으로 제시하세요.

[제약 — 반드시 준수]
- 명시된 챔피언({champion}, 아군: {allies_text}, 적군: {enemies_text})과 소환사 주문({spells_text})만 언급하세요.
- 데이터에 없는 스킬·아이템·포지션·챔피언은 절대 추측하지 마세요.
- 각 항목은 2~3문장, 수치 반드시 포함.

아래 형식으로 정확히 답하세요 (콜론 뒤 내용만 작성, 다른 텍스트 없이):

총평: (승패를 가른 핵심 요인과 전체 흐름을 2~3문장으로)
라인전: (0~14분 라인전 교전·CS·골드 흐름 평가. {tier} 티어 기준 대비 수치 포함)
중반전: (오브젝트 참여·로밍·팀파이트 기여 평가. 주요 순간 타임라인 참고)
후반전: (포지셔닝·의사결정·게임 마무리 평가)
핵심패턴: (게임 내내 반복된 습관 — 잘한 것 1가지, 고쳐야 할 것 1가지)
개선1: (가장 시급한 즉시 실천 행동 — 수치 목표 포함)
개선2: (두 번째 개선 행동)
개선3: (세 번째 개선 행동)"""


def parse_match_response(text: str) -> dict:
    """경기 분석 전용 응답 파서 (단계별 포맷)."""
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    result = {
        "summary": "",
        "early_game": "",
        "mid_game": "",
        "late_game": "",
        "key_pattern": "",
        "improvements": [],
    }
    for line in lines:
        if line.startswith("총평"):
            result["summary"] = line.split(":", 1)[-1].strip()
        elif line.startswith("라인전"):
            result["early_game"] = line.split(":", 1)[-1].strip()
        elif line.startswith("중반전"):
            result["mid_game"] = line.split(":", 1)[-1].strip()
        elif line.startswith("후반전"):
            result["late_game"] = line.split(":", 1)[-1].strip()
        elif line.startswith("핵심패턴"):
            result["key_pattern"] = line.split(":", 1)[-1].strip()
        elif line.startswith("개선"):
            val = line.split(":", 1)[-1].strip()
            if val:
                result["improvements"].append(val)
    return result


# ─── Gemini 호출 ──────────────────────────────────────────────────────────────

def parse_response(text: str) -> dict:
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    summary = ""
    strengths, weaknesses, improvements = [], [], []
    for line in lines:
        if line.startswith("총평"):
            summary = line.split(":", 1)[-1].strip()
        elif line.startswith("강점"):
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
    return {"summary": summary, "strengths": strengths, "weaknesses": weaknesses, "improvements": improvements}


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
    summoner_spells: Optional[tuple[int, int]] = None,
    ally_champions: Optional[list[str]] = None,
    enemy_champions: Optional[list[str]] = None,
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

    prompt = build_match_analysis_prompt(
        champion, role, win, tier, kda,
        cs_per_min, damage_share, vision_score, gold_per_min,
        game_duration_min, strengths, weaknesses, impact_score, moments,
        summoner_spells=summoner_spells,
        ally_champions=ally_champions,
        enemy_champions=enemy_champions,
    )

    import time
    for attempt in range(3):
        try:
            resp = requests.post(
                f"{GEMINI_URL}?key={GEMINI_API_KEY}",
                json={"contents": [{"parts": [{"text": prompt}]}]},
                timeout=60,
            )
            if resp.status_code == 503:
                log.warning("Gemini 503 (attempt %d/3), retrying in 5s...", attempt + 1)
                time.sleep(5)
                continue
            if resp.status_code != 200:
                log.warning("Gemini API error %d: %s", resp.status_code, resp.text[:200])
                return None

            data = resp.json()
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            result = parse_match_response(text)

            if not result["summary"] and not result["early_game"]:
                log.warning("Gemini match response could not be parsed. Raw text:\n%s", text[:500])
                return None

            return result

        except Exception as e:
            log.warning("Gemini coaching failed (attempt %d/3): %s", attempt + 1, e)
            if attempt < 2:
                time.sleep(3)
    return None


# ─── 클립 영상 분석 ───────────────────────────────────────────────────────────

GEMINI_UPLOAD_URL = "https://generativelanguage.googleapis.com/upload/v1beta/files"
GEMINI_FILES_URL  = "https://generativelanguage.googleapis.com/v1beta/files"


def _upload_video_file(clip_path: str) -> Optional[str]:
    """Gemini File API로 영상 업로드 후 file_uri 반환. 실패 시 None."""
    file_size = os.path.getsize(clip_path)
    display_name = os.path.basename(clip_path)

    # 업로드 세션 시작
    init_resp = requests.post(
        f"{GEMINI_UPLOAD_URL}?key={GEMINI_API_KEY}",
        headers={
            "X-Goog-Upload-Protocol": "resumable",
            "X-Goog-Upload-Command": "start",
            "X-Goog-Upload-Header-Content-Length": str(file_size),
            "X-Goog-Upload-Header-Content-Type": "video/mp4",
            "Content-Type": "application/json",
        },
        json={"file": {"display_name": display_name}},
        timeout=30,
    )
    if init_resp.status_code != 200:
        log.warning("File API upload init failed %d: %s", init_resp.status_code, init_resp.text[:200])
        return None

    upload_url = init_resp.headers.get("X-Goog-Upload-URL")
    if not upload_url:
        log.warning("File API did not return upload URL")
        return None

    # 파일 본문 업로드
    with open(clip_path, "rb") as f:
        upload_resp = requests.post(
            upload_url,
            headers={
                "Content-Length": str(file_size),
                "X-Goog-Upload-Offset": "0",
                "X-Goog-Upload-Command": "upload, finalize",
            },
            data=f,
            timeout=120,
        )
    if upload_resp.status_code not in (200, 201):
        log.warning("File API upload failed %d: %s", upload_resp.status_code, upload_resp.text[:200])
        return None

    file_info = upload_resp.json().get("file", {})
    file_name = file_info.get("name")
    file_uri  = file_info.get("uri")

    if not file_name or not file_uri:
        log.warning("File API response missing name/uri")
        return None

    # PROCESSING 상태 대기 (최대 30초)
    for _ in range(15):
        state_resp = requests.get(
            f"{GEMINI_FILES_URL}/{file_name.split('/')[-1]}?key={GEMINI_API_KEY}",
            timeout=10,
        )
        if state_resp.status_code == 200:
            state = state_resp.json().get("state", "ACTIVE")
            if state == "ACTIVE":
                break
            if state == "FAILED":
                log.warning("File API processing failed for %s", clip_path)
                return None
        time.sleep(2)

    return file_uri


def _delete_uploaded_file(file_uri: str) -> None:
    """업로드된 파일 삭제 (48시간 자동 삭제되지만 즉시 정리)."""
    try:
        file_name = file_uri.split("/files/")[-1]
        requests.delete(
            f"{GEMINI_FILES_URL}/{file_name}?key={GEMINI_API_KEY}",
            timeout=10,
        )
    except Exception:
        pass


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
    summoner_spells: Optional[tuple[int, int]] = None,
    ally_champions: Optional[list[str]] = None,
    enemy_champions: Optional[list[str]] = None,
    event_sec_in_clip: Optional[int] = None,  # 클립 내 이벤트 발생 시점(초)
    lane_opponent: Optional[str] = None,       # 직접 상대 라이너 챔피언
) -> Optional[str]:
    """
    15초 클립 영상 + 게임 상황을 Gemini Vision에 전달해 플레이 피드백을 반환합니다.
    File API로 먼저 업로드 후 uri 참조 방식 사용 (20MB 인라인 제한 우회).
    실패 시 None 반환 (fallback 없음 — 텍스트 코칭보다 없는 게 나음).
    """
    if not GEMINI_API_KEY:
        return None

    if not os.path.exists(clip_path):
        log.warning("Clip file not found: %s", clip_path)
        return None

    kind_map = {"kill": "킬", "death": "데스", "objective": "오브젝트 획득"}
    kind_str = kind_map.get(event_kind, event_kind)
    gold_diff_str = _fmt_gold(pre_gold_diff)
    snowball_str  = _fmt_gold(snowball_gold)

    spell1, spell2 = summoner_spells or (0, 0)
    spell1_name = SUMMONER_SPELL_NAMES.get(spell1, f"주문{spell1}")
    spell2_name = SUMMONER_SPELL_NAMES.get(spell2, f"주문{spell2}")
    allies_text   = ", ".join(ally_champions)  if ally_champions  else "정보 없음"
    enemies_text  = ", ".join(enemy_champions) if enemy_champions else "정보 없음"
    if role in ("BOTTOM", "UTILITY"):
        bot_partner = next((p for p in (ally_champions or []) if p != champion), "불명확")
        opponent_text = (
            f"2v2 바텀 라인 (원딜+서포트 vs 원딜+서포트) — "
            f"아군 파트너: {bot_partner} / 적군 전체: {enemies_text}"
        )
    else:
        opponent_text = lane_opponent if lane_opponent else "영상에서 직접 확인"

    if event_kind == "kill":
        post_check = (
            "이벤트 직후: 킬로 얻은 시간을 웨이브 푸시, 오브젝트, 타워 채굴 등 "
            "실질 이득으로 연결했는지 확인하세요. 킬만 따고 아무것도 챙기지 않고 "
            "물러났다면 그것 자체가 지적 대상입니다."
        )
        snowball_hint = (
            "킬인데도 골드 이득이 작거나 음수라면, 킬 이후 후속 플레이가 "
            "이득으로 연결되지 못한 원인을 영상에서 찾아 지적하세요."
        )
    else:
        post_check = (
            "이벤트 직후: 죽어 있는 동안 적이 가져간 것(웨이브, 타워, 오브젝트, "
            "시야 장악)을 확인하고, 데스 직전 그 위치에 있어야 할 전략적 이유가 "
            "있었는지 평가하세요."
        )
        snowball_hint = (
            "데스로 인한 골드 손실이 크다면, 단순히 죽은 것이 아니라 "
            "'그 타이밍, 그 위치에서' 죽은 것의 비용(놓친 웨이브·오브젝트)을 함께 설명하세요."
        )

    event_anchor = f"클립의 약 {event_sec_in_clip}초 시점" if event_sec_in_clip is not None else "클립 중간 시점"

    prompt = f"""당신은 프로팀에서 VOD 리뷰를 진행하는 리그 오브 레전드 전문 코치입니다. 이 클립은 플레이어({champion})의 {kind_str} 장면입니다. 프로팀 피드백 세션처럼 날카롭고 구체적으로, 그러나 실행 가능하게 분석하세요.

[상황 정보]
- 플레이어 챔피언: {champion} ({role}) / 소환사 주문: {spell1_name} / {spell2_name}
- 직접 상대 라이너: {opponent_text} ← 이 챔피언이 라인 상대임. 나머지는 다른 포지션
- 아군 팀 전체: {allies_text}
- 적군 팀 전체: {enemies_text}
- 이벤트: 게임 내 {time_str} [{kind_str}] — 이 이벤트는 {event_anchor}에 발생합니다
- 당시 킬 스코어: {pre_score}
- 당시 골드차: {gold_diff_str} (양수=아군 우세, 음수=적군 우세)
- {snowball_label} 골드 변화: {snowball_str} ← 이 장면이 경기에 미친 실제 파급력

[영상에서 반드시 확인할 것]
1. 이벤트 직전: 플레이어와 상대의 체력·마나 상태, 미니맵에 보이는 적의 수(미니맵에 안 보이는 적 = 위험 신호), 주변 시야 확보 여부, 미니언 웨이브 위치, 가장 가까운 아군과의 거리.
2. 교전 중: 스킬을 어떤 순서로 사용했고 적중했는지, 점멸 등 소환사 주문 사용 타이밍이 적절했는지(생존용으로 아껴야 했는지/공격용으로 써도 됐는지), 교전 시작 시점에 누구의 사거리 안에 서 있었는지.
3. {post_check}

[분석 원칙]
- 데스는 죽는 순간이 아니라 그 몇 초 전 결정에서 시작됩니다. 결과가 사실상 확정된 "최초의 결정"이 클립의 몇 초 시점인지 찾아내 지목하세요.
- 당시 골드차({gold_diff_str})와 스코어({pre_score})를 근거로, 이 교전이 애초에 시도할 가치가 있었는지 리스크 대비 리턴을 평가하세요. 불리한 상황의 무리한 교전과 유리한 상황의 소극적 플레이는 둘 다 지적 대상입니다.
- {snowball_hint}
- 영상에서 직접 보이는 것만 근거로 사용하세요. 위에 명시된 챔피언과 소환사 주문만 언급하고, 화면에서 확인되지 않는 챔피언·스킬·아이템·쿨다운은 절대 추측하지 마세요.
- "포지셔닝이 아쉽다" 같은 추상적 표현 금지. "클립 4초경 시야 없는 강 부쉬 옆으로 붙어 이동했다"처럼 화면에 보이는 행동을 직접 서술하세요.

아래 형식으로 정확히 답하세요. 각 항목은 1~2문장이며, 장면을 언급할 때는 "클립 X초경" 형태로 시점을 지목하세요.

결정적 판단: (이 장면의 결과를 만든 핵심 결정 1가지 + 클립 내 시점)
판단 분석: (그 결정이 좋았던/나빴던 이유 — 시야·체력·스킬·인원수·골드 상황 중 어떤 정보에 근거했거나 무엇을 놓쳤는지)
대안 플레이: (같은 상황에서 정확히 어떻게 움직였어야 했는지 — 서 있을 위치, 아껴야 할 스킬, 진입/이탈 타이밍 단위로)
연습 포인트: (이 장면에서 뽑아낼 수 있는, 다음 게임에 일반화 가능한 원칙 1가지)"""

    file_uri = _upload_video_file(clip_path)
    if not file_uri:
        return None

    try:
        resp = requests.post(
            f"{GEMINI_URL}?key={GEMINI_API_KEY}",
            json={
                "contents": [{
                    "parts": [
                        {
                            "fileData": {"mimeType": "video/mp4", "fileUri": file_uri},
                            "videoMetadata": {"fps": 5},
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
    finally:
        if file_uri:
            _delete_uploaded_file(file_uri)


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
    summoner_spells: Optional[tuple[int, int]] = None,
    ally_champions: Optional[list[str]] = None,
    enemy_champions: Optional[list[str]] = None,
    event_sec_in_clip: Optional[int] = None,
    lane_opponent: Optional[str] = None,
) -> Optional[str]:
    """analyze_clip의 비동기 래퍼 — asyncio.to_thread로 스레드풀에서 실행"""
    return await asyncio.to_thread(
        analyze_clip,
        clip_path, event_kind, time_str, pre_score,
        pre_gold_diff, snowball_gold, snowball_label,
        champion, role,
        summoner_spells, ally_champions, enemy_champions,
        event_sec_in_clip, lane_opponent,
    )
