"""
테스트용: 소량 데이터 수집하여 동작 확인
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.collect_tier_data import TierDataCollector
from dotenv import load_dotenv
import os

load_dotenv()

def main():
    api_key = os.getenv("RIOT_API_KEY")

    if not api_key:
        print("Error: RIOT_API_KEY가 .env 파일에 설정되지 않았습니다.")
        return

    collector = TierDataCollector(api_key, region="kr")

    # GOLD 티어에서 10개만 테스트
    print("GOLD 티어에서 10개 매치 수집 테스트...\n")
    data = collector.collect_tier_data("GOLD", target_count=10)

    print(f"\n수집 결과:")
    print(f"  총 플레이어 데이터: {len(data)}개")

    if data:
        print(f"\n첫 번째 데이터 샘플:")
        sample = data[0]
        for key, value in list(sample.items())[:10]:
            print(f"    {key}: {value}")

    collector.save_tier_data("GOLD_TEST", data)

if __name__ == "__main__":
    main()
