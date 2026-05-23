"""
LLM-based Match Report Generator

OpenAI GPT-4를 사용하여 경기 데이터를 분석하고
풍부한 자연어 리포트를 생성합니다.
"""

import os
from typing import Dict, List
from openai import OpenAI


def format_highlights(highlights: List[Dict]) -> str:
    """하이라이트를 텍스트로 포맷팅"""
    if not highlights:
        return "하이라이트 없음"

    lines = []
    for i, h in enumerate(highlights[:10], 1):  # 상위 10개만
        timestamp_min = int(h['timestamp'] / 60)
        timestamp_sec = int(h['timestamp'] % 60)
        time_str = f"{timestamp_min}:{timestamp_sec:02d}"

        impact = h.get('impact_score', 0)
        impact_str = f"{impact:+.1f}%" if impact != 0 else ""

        lines.append(f"{i}. [{time_str}] {h['description']} {impact_str}")

    return "\n".join(lines)


def format_team_rankings(team_scores: List[Dict]) -> str:
    """팀 기여도를 텍스트로 포맷팅"""
    if not team_scores:
        return "팀 정보 없음"

    lines = []
    for i, member in enumerate(team_scores, 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}위"
        current = " (나)" if member.get('is_current_player') else ""

        lines.append(
            f"{medal} {member['summoner_name']}{current} - "
            f"{member['champion']} ({member['role']}) - "
            f"Impact: {member['impact_score']:.1f}/100"
        )

    return "\n".join(lines)


def format_impact_features(impact_result: Dict) -> str:
    """Impact Score 주요 기여 요소를 텍스트로 포맷팅"""
    if not impact_result:
        return ""

    lines = []

    # top_positive_features와 top_negative_features 형식 처리
    if 'top_positive_features' in impact_result:
        positive = impact_result.get('top_positive_features', [])[:3]
        negative = impact_result.get('top_negative_features', [])[:3]

        if positive:
            lines.append("[긍정적 요소]")
            for feat in positive:
                name = feat.get('displayName', feat.get('name', ''))
                shap = feat.get('shap', 0)
                lines.append(f"↑ {name}: {shap:+.2f}")

        if negative:
            lines.append("\n[부정적 요소]")
            for feat in negative:
                name = feat.get('displayName', feat.get('name', ''))
                shap = feat.get('shap', 0)
                lines.append(f"↓ {name}: {shap:+.2f}")

    # 구버전 topFeatures 형식도 지원
    elif 'topFeatures' in impact_result:
        features = impact_result['topFeatures'][:5]
        for feat in features:
            direction = "↑" if feat['direction'] == 'up' else "↓"
            feature_name = feat['feature']
            contribution = feat['contribution']
            lines.append(f"{direction} {feature_name}: {contribution:+.2f}")

    return "\n".join(lines)


def format_gap_analysis(gap_result) -> str:
    """Gap Analysis 결과를 텍스트로 포맷팅"""
    if not gap_result:
        return ""

    # gap_result가 Pydantic 모델이면 dict로 변환
    if hasattr(gap_result, 'model_dump'):
        gap_data = gap_result.model_dump()
    else:
        gap_data = gap_result

    lines = []

    tier = gap_data.get("tier", "N/A")
    division = gap_data.get("division", "")
    # total_score / overall_score 둘 다 대응
    score = gap_data.get("total_score", gap_data.get("overall_score", 0))

    lines.append(f"티어: {tier} {division}")
    lines.append(f"종합 점수: {score:.1f}/100")

    # -------------------- 강점 --------------------
    strengths = gap_data.get("strengths", [])
    if strengths:
        lines.append("\n[강점]")
        for s in strengths[:3]:
            # 1) 문자열이면 그대로 출력
            if isinstance(s, str):
                lines.append(f" {s}")
            # 2) dict이면 상세 포맷
            elif isinstance(s, dict):
                name = s.get("displayName") or s.get("metric", "")
                player_val = s.get("playerValue", "")
                percentile = s.get("percentile", None)

                if percentile is not None:
                    lines.append(
                        f" {name}: {player_val} (상위 {100 - percentile:.0f}%)"
                    )
                else:
                    lines.append(f" {name}: {player_val}")
            # 3) 그 외 타입은 그냥 문자열 캐스팅
            else:
                lines.append(f" {str(s)}")

    # -------------------- 약점 --------------------
    weaknesses = gap_data.get("weaknesses", [])
    if weaknesses:
        lines.append("\n[약점]")
        for w in weaknesses[:3]:
            if isinstance(w, str):
                # "Gold Efficiency (-16.38%)" 이런 거
                lines.append(f" {w}")
            elif isinstance(w, dict):
                name = w.get("displayName") or w.get("metric", "")
                player_val = w.get("playerValue", "")
                gap_value = w.get("gap", 0)
                lines.append(
                    f" {name}: {player_val} (기준치 대비 {gap_value:+.0f})"
                )
            else:
                lines.append(f" {str(w)}")

    return "\n".join(lines)


def generate_match_report(
    match_data: Dict,
    highlights: List[Dict],
    team_impact_scores: List[Dict],
    impact_result: Dict,
    gap_result = None,
    api_key: str = None
) -> str:
    """
    OpenAI GPT-4를 사용하여 경기 분석 리포트 생성

    Args:
        match_data: 매치 기본 정보 (챔피언, 포지션, KDA 등)
        highlights: 하이라이트 목록 (킬, 오브젝트, 실수 등)
        team_impact_scores: 팀원별 Impact Score 랭킹
        impact_result: Impact Score 상세 분석 결과
        gap_result: Gap Analysis 결과 (선택사항)
        api_key: OpenAI API 키 (없으면 환경변수에서 읽음)

    Returns:
        자연어로 작성된 경기 분석 리포트
    """
    # API 키 설정
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        return "⚠️ OpenAI API 키가 설정되지 않았습니다. .env 파일에 OPENAI_API_KEY를 추가해주세요."

    # OpenAI 클라이언트 초기화
    client = OpenAI(api_key=api_key)

    # 플레이어 정보 추출
    player_info = match_data.get('player', {})
    game_info = match_data.get('game_info', {})

    champion = player_info.get('champion', '알 수 없음')
    role = player_info.get('position', '알 수 없음')
    win = player_info.get('win', False)
    result_text = "승리" if win else "패배"

    # KDA 정보 (highlights에서 추출)
    kills = 0
    deaths = 0
    assists = 0

    # 팀 기여도에서 현재 플레이어 찾기
    current_player = None
    for member in team_impact_scores:
        if member.get('is_current_player'):
            current_player = member
            kills = member.get('kills', 0)
            deaths = member.get('deaths', 0)
            assists = member.get('assists', 0)
            break

    kda_text = f"{kills}/{deaths}/{assists}"
    impact_score = impact_result.get('impact_score', 0) if impact_result else 0

    # 게임 시간
    duration = game_info.get('duration_formatted', '알 수 없음')

    # 하이라이트 분류
    highlight_plays = [h for h in highlights if h.get('category') == 'highlight']
    mistakes = [h for h in highlights if h.get('category') == 'mistake']

    # Gap Analysis 정보 (있으면 추가)
    gap_info = ""
    if gap_result:
        gap_info = f"\n# Gap Analysis (티어별 비교)\n{format_gap_analysis(gap_result)}\n"

    # 프롬프트 구성
    prompt = f"""당신은 리그오브레전드 전문 코치입니다. 다음 경기 데이터를 분석하여 선수에게 친절하고 구체적인 피드백을 제공해주세요.

# 경기 정보
- 챔피언: {champion}
- 포지션: {role}
- 전적: {kda_text} ({result_text})
- 게임 시간: {duration}
- Impact Score: {impact_score:.1f}/100

# 팀 기여도 순위
{format_team_rankings(team_impact_scores)}

# Impact Score 주요 기여 요소
{format_impact_features(impact_result)}
{gap_info}
# 주요 하이라이트 ({len(highlight_plays)}개)
{format_highlights(highlight_plays)}

# 주요 실수 ({len(mistakes)}개)
{format_highlights(mistakes)}

다음 형식으로 친절하고 상세한 리포트를 작성해주세요:

## 📊 경기 요약
(전반적인 플레이 평가 2-3문장)

## ✨ 칭찬할 점
(가장 잘한 플레이를 하이라이트 데이터를 참고하여 상세 분석. 구체적인 시간대와 상황 언급)

## 💡 개선 포인트
(실수 분석 및 구체적인 개선 방법. 데스, 격차 등 실제 데이터 기반으로 조언. Gap Analysis의 약점 지표도 참고)

## 🎯 다음 경기 조언
(실천 가능한 구체적인 팁 3가지. Gap Analysis의 약점을 개선하는 방법 포함)

리포트는 친근하고 동기부여가 되는 톤으로 작성하되, 구체적인 데이터와 시간대를 언급하여 신뢰도를 높여주세요.
Gap Analysis 데이터가 있다면 티어 기준치와 비교하여 어떤 지표를 개선해야 하는지 구체적으로 조언해주세요.
"""

    try:
        # OpenAI API 호출
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "당신은 리그오브레전드 게임을 깊이 이해하고 있는 전문 코치입니다. "
                               "선수들의 데이터를 분석하여 건설적이고 구체적인 피드백을 제공합니다."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            max_tokens=2000,
            top_p=0.9
        )

        # 응답 추출
        report = response.choices[0].message.content

        return report

    except Exception as e:
        error_message = f"⚠️ 리포트 생성 중 오류가 발생했습니다: {str(e)}"
        print(f"OpenAI API Error: {e}")
        return error_message


# 테스트용
if __name__ == "__main__":
    print("LLM Reporter Module")
    print("OpenAI GPT-4를 사용하여 경기 분석 리포트를 생성합니다.")

    # 샘플 데이터로 테스트
    sample_match_data = {
        'player': {
            'champion': 'Jax',
            'position': 'TOP',
            'win': True
        },
        'game_info': {
            'duration_formatted': '28:55'
        }
    }

    sample_highlights = [
        {
            'description': 'BARON_NASHOR 획득 (27.0분)',
            'timestamp': 1581.451,
            'category': 'highlight',
            'impact_score': 15.2
        }
    ]

    sample_team_scores = [
        {
            'summoner_name': '옷장세번째칸후드',
            'champion': 'Jax',
            'role': 'TOP',
            'impact_score': 85.3,
            'kills': 4,
            'deaths': 2,
            'assists': 8,
            'is_current_player': True
        }
    ]

    sample_impact = {
        'impactScore': 85.3,
        'topFeatures': [
            {'feature': 'goldPerMinute', 'contribution': 12.5, 'direction': 'up'},
            {'feature': 'kda', 'contribution': 8.3, 'direction': 'up'}
        ]
    }

    # API 키가 설정되어 있으면 테스트 실행
    if os.getenv("OPENAI_API_KEY"):
        report = generate_match_report(
            sample_match_data,
            sample_highlights,
            sample_team_scores,
            sample_impact
        )
        print("\n" + "="*60)
        print(report)
        print("="*60)
    else:
        print("\n⚠️ OPENAI_API_KEY 환경변수를 설정해주세요.")
