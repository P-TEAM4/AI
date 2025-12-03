"""
전체 파이프라인 테스트

1. API로 매치 데이터 수집 (소량)
2. 타임라인 데이터 수집
3. CSV 저장 확인
4. ML 모델 학습 가능 여부 확인
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.collect_tier_data import TierDataCollector
from scripts.timeline_fetcher import TimelineFetcher
import os

def main():
    print("="*80)
    print("전체 파이프라인 테스트")
    print("="*80)

    api_key = os.getenv("RIOT_API_KEY")
    if not api_key:
        print("[ERROR] RIOT_API_KEY가 .env 파일에 없습니다.")
        return

    # 1단계: 매치 데이터 수집 (소량 - 10개만)
    print("\n[1단계] 매치 데이터 수집 (테스트용 10개)")
    print("-"*80)

    collector = TierDataCollector(api_key, region="kr")

    # MASTER 티어에서 10개 매치만 수집
    data = collector.collect_tier_data(
        tier="MASTER",
        target_matches=10,        # 10개만
        matches_per_summoner=5,   # 소환사당 5개
        resume=False              # 새로 시작
    )

    if not data:
        print("[ERROR] 데이터 수집 실패 (API 키 확인 필요)")
        return

    # CSV 저장
    collector.save_tier_data("TEST_MASTER", data)

    print(f"\n[SUCCESS] 매치 데이터 수집 완료: {len(data)} 플레이어")

    # 수집된 매치 ID 추출
    import pandas as pd
    df = pd.DataFrame(data)
    match_ids = df['match_id'].unique().tolist()
    print(f"   고유 매치 수: {len(match_ids)}")

    # 2단계: 타임라인 수집 (첫 3개만)
    print(f"\n[2단계] 타임라인 데이터 수집 (첫 3개)")
    print("-"*80)

    fetcher = TimelineFetcher(cache_dir="data/timelines", rate_limit_delay=1.2)

    test_match_ids = match_ids[:3]
    results = {}

    for idx, match_id in enumerate(test_match_ids, 1):
        print(f"\n[{idx}/{len(test_match_ids)}] {match_id}")
        timeline = fetcher.get_timeline(match_id, version="v15.24")
        results[match_id] = timeline

    success_count = sum(1 for v in results.values() if v is not None)
    print(f"\n[SUCCESS] 타임라인 수집 완료: {success_count}/{len(test_match_ids)}")

    # 3단계: 저장된 파일 확인
    print(f"\n[3단계] 저장된 파일 확인")
    print("-"*80)

    csv_file = Path("data/tier_collections/test_master_tier_v15.24.csv")
    if csv_file.exists():
        print(f"[OK] CSV 파일: {csv_file}")
        print(f"   크기: {csv_file.stat().st_size / 1024:.2f} KB")

        # CSV 내용 확인
        df_check = pd.read_csv(csv_file)
        print(f"   행 수: {len(df_check)}")
        print(f"   컬럼 수: {len(df_check.columns)}")
        print(f"   컬럼: {list(df_check.columns)[:5]}...")
    else:
        print(f"[ERROR] CSV 파일이 없습니다: {csv_file}")

    timeline_dir = Path("data/timelines/v15.24")
    if timeline_dir.exists():
        timeline_files = list(timeline_dir.glob("*.json"))
        print(f"\n[OK] 타임라인 파일: {len(timeline_files)}개")
        for tf in timeline_files[:3]:
            print(f"   - {tf.name} ({tf.stat().st_size / 1024:.2f} KB)")
    else:
        print(f"\n[ERROR] 타임라인 디렉토리가 없습니다: {timeline_dir}")

    # 4단계: ML 학습 가능 여부 확인
    print(f"\n[4단계] ML 모델 학습 가능 여부 확인")
    print("-"*80)

    required_columns = ['match_id', 'tier', 'win', 'kda', 'gold_earned', 'total_cs']
    missing = [col for col in required_columns if col not in df_check.columns]

    if missing:
        print(f"[ERROR] 필수 컬럼 누락: {missing}")
    else:
        print(f"[OK] 필수 컬럼 모두 존재")
        print(f"\n샘플 데이터:")
        print(df_check[required_columns].head(3).to_string())

    print("\n" + "="*80)
    print("파이프라인 테스트 완료!")
    print("="*80)
    print("\n다음 단계:")
    print("  1. API 키를 갱신하여 대량 수집: python scripts/collect_tier_data.py")
    print("  2. 타임라인 수집: python scripts/timeline_fetcher.py --all")
    print("  3. ML 모델 학습: (타임라인 전처리 후)")


if __name__ == "__main__":
    main()
