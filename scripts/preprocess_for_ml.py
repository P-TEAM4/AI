"""
ML 학습을 위한 데이터 전처리

플레이어별 데이터 → 매치별 집계 데이터로 변환
- 팀별 통계 계산 (킬, 골드, CS 등)
- 팀 간 차이 계산 (gold_diff, cs_diff 등)

입력: data/tier_collections/*.csv (플레이어 레벨)
출력: data/processed/matches_aggregated.csv (매치 레벨)

사용법:
    python scripts/preprocess_for_ml.py
"""

import pandas as pd
from pathlib import Path
from glob import glob


def load_all_tier_data(tier_dir: str = "data/tier_collections") -> pd.DataFrame:
    """
    모든 티어 CSV 데이터를 하나의 DataFrame으로 통합

    Returns:
        모든 플레이어 데이터가 합쳐진 DataFrame
    """
    tier_files = glob(f"{tier_dir}/*.csv")

    if not tier_files:
        raise FileNotFoundError(f"No CSV files found in {tier_dir}")

    print(f"Found {len(tier_files)} CSV files")

    dfs = []

    for tier_file in tier_files:
        filename = Path(tier_file).name
        print(f"Loading: {filename}")

        try:
            df = pd.read_csv(tier_file)
            dfs.append(df)
            print(f"  -> {len(df)} players")
        except Exception as e:
            print(f"  [ERROR] Failed to load {filename}: {e}")
            continue

    if not dfs:
        raise ValueError("No valid CSV files loaded")

    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal players: {len(combined_df):,}")

    return combined_df


def aggregate_to_match_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    플레이어별 데이터 → 매치별 데이터로 집계

    각 매치를 팀 단위로 집계하고 팀 간 차이를 계산
    """
    print("\nAggregating to match level...")

    matches = []

    for match_id in df["match_id"].unique():
        match_df = df[df["match_id"] == match_id]

        if len(match_df) != 10:
            # 10명이 아닌 매치는 스킵
            continue

        # 팀별로 분리
        team_100 = match_df[match_df["team_id"] == 100]
        team_200 = match_df[match_df["team_id"] == 200]

        if len(team_100) != 5 or len(team_200) != 5:
            continue

        # 승리팀 결정
        winner = 100 if team_100.iloc[0]["win"] else 200

        # 팀별 집계 통계
        match_record = {
            "matchId": match_id,
            "tier": team_100.iloc[0]["tier"],
            "duration": team_100.iloc[0]["game_duration"],
            "winner": 1 if winner == 100 else 0,  # 1 = team_100 승리

            # Team 100 통계
            "team100_kills": team_100["kills"].sum(),
            "team100_deaths": team_100["deaths"].sum(),
            "team100_assists": team_100["assists"].sum(),
            "team100_gold": team_100["gold_earned"].sum(),
            "team100_cs": team_100["total_cs"].sum(),
            "team100_damage": team_100["total_damage_dealt_to_champions"].sum(),
            "team100_vision": team_100["vision_score"].sum(),
            "team100_avg_kda": team_100["kda"].mean(),

            # Team 200 통계
            "team200_kills": team_200["kills"].sum(),
            "team200_deaths": team_200["deaths"].sum(),
            "team200_assists": team_200["assists"].sum(),
            "team200_gold": team_200["gold_earned"].sum(),
            "team200_cs": team_200["total_cs"].sum(),
            "team200_damage": team_200["total_damage_dealt_to_champions"].sum(),
            "team200_vision": team_200["vision_score"].sum(),
            "team200_avg_kda": team_200["kda"].mean(),
        }

        # 차이 계산 (Team 100 - Team 200)
        match_record["gold_diff"] = match_record["team100_gold"] - match_record["team200_gold"]
        match_record["cs_diff"] = match_record["team100_cs"] - match_record["team200_cs"]
        match_record["damage_diff"] = match_record["team100_damage"] - match_record["team200_damage"]
        match_record["vision_diff"] = match_record["team100_vision"] - match_record["team200_vision"]
        match_record["kill_diff"] = match_record["team100_kills"] - match_record["team200_kills"]

        matches.append(match_record)

    print(f"Aggregated to {len(matches):,} valid matches")

    return pd.DataFrame(matches)


def main():
    print("="*80)
    print("ML 학습용 데이터 전처리")
    print("="*80)

    # 1) 모든 티어 데이터 로드
    df_players = load_all_tier_data()

    print(f"\nPlayer data shape: {df_players.shape}")
    print(f"Columns: {list(df_players.columns)}")

    # 2) 매치 레벨로 집계
    df_matches = aggregate_to_match_level(df_players)

    print(f"\nMatch data shape: {df_matches.shape}")
    print(f"Columns: {list(df_matches.columns)}")

    # 3) 저장
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "matches_aggregated.csv"
    df_matches.to_csv(output_file, index=False)

    print(f"\n저장 완료: {output_file}")
    print(f"  파일 크기: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"  매치 수: {len(df_matches):,}")
    print(f"  승리 분포: {df_matches['winner'].value_counts().to_dict()}")

    # 4) 샘플 확인
    print("\n샘플 데이터 (처음 3개):")
    print(df_matches[['matchId', 'tier', 'winner', 'gold_diff', 'kill_diff']].head(3).to_string())

    print("\n[SUCCESS] 전처리 완료! 이제 ML 모델 학습 가능합니다:")
    print("   python src/models/ml_based.py")


if __name__ == "__main__":
    main()
