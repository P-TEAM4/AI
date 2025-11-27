"""Test script for real Riot API calls"""

import sys
import os
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Load environment variables
load_dotenv('.env-dev')

from src.services.riot_api import RiotAPIClient
from src.services.analyzer import MatchAnalyzer
from src.models.rule_based import RuleBasedGapAnalyzer
from src.utils.visualizer import GapAnalysisVisualizer


def main():
    """Main function to test real Riot API"""

    print("🎮 LoL Match Analysis with Baseline Comparison\n")
    print("=" * 60)
    print("📌 This test analyzes a single match against learned baseline data")
    print("=" * 60)

    # Get API key from environment
    api_key = os.getenv('RIOT_API_KEY')
    if not api_key or api_key == 'your_riot_api_key_here':
        print("❌ Error: RIOT_API_KEY not set in .env-dev")
        return

    print(f"✅ API Key loaded: {api_key[:20]}...")

    # Initialize clients
    riot_client = RiotAPIClient(api_key=api_key)
    analyzer = MatchAnalyzer(api_key=api_key)
    gap_analyzer = RuleBasedGapAnalyzer()
    visualizer = GapAnalysisVisualizer()

    # Test 1: Get account by Riot ID
    print("\n" + "=" * 60)
    print("Test 1: Getting Account Information")
    print("=" * 60)

    # 유명한 한국 프로게이머 예시 (Faker)
    summoner_name = input("Enter summoner name (default: Hide on bush): ").strip() or "Hide on bush"
    tag_line = input("Enter tag line (default: KR1): ").strip() or "KR1"

    try:
        print(f"\n🔍 Fetching account: {summoner_name}#{tag_line}")
        account = riot_client.get_account_by_riot_id(summoner_name, tag_line)

        print(f"✅ Account found!")
        print(f"   PUUID: {account['puuid'][:20]}...")
        print(f"   Name: {account['gameName']}#{account['tagLine']}")

        puuid = account['puuid']

        # Test 2: Get most recent match
        print("\n" + "=" * 60)
        print("Test 2: Getting Most Recent Match")
        print("=" * 60)

        print(f"🔍 Fetching most recent match...")
        match_ids = riot_client.get_match_ids(puuid, count=1)

        if not match_ids:
            print("❌ No matches found")
            return

        print(f"✅ Found most recent match:")
        print(f"   {match_ids[0]}")

        # Test 3: Analyze the match (단일 경기 분석)
        print("\n" + "=" * 60)
        print("Test 3: Analyzing Match with Learned Baseline Data")
        print("=" * 60)

        latest_match_id = match_ids[0]
        print(f"🔍 Analyzing match: {latest_match_id}")
        print(f"📊 Comparing with tier baseline statistics (learned data)")

        match_details = riot_client.get_match_details(latest_match_id)
        player_stats = riot_client.extract_player_stats_from_match(match_details, puuid)

        if not player_stats:
            print("❌ Could not extract player stats")
            return

        print(f"\n✅ Match Analysis:")
        print(f"   Champion: {player_stats['champion_name']}")
        print(f"   Position: {player_stats['position']}")
        print(f"   Result: {'Victory' if player_stats['win'] else 'Defeat'}")
        print(f"   KDA: {player_stats['kills']}/{player_stats['deaths']}/{player_stats['assists']}")
        print(f"   CS: {player_stats['total_cs']} ({player_stats['total_cs'] / (player_stats['game_duration'] / 60):.1f}/min)")
        print(f"   Gold: {player_stats['gold']:,}")
        print(f"   Vision Score: {player_stats['vision_score']}")
        print(f"   Damage: {player_stats['damage_dealt']:,}")

        # Test 4: Gap Analysis
        print("\n" + "=" * 60)
        print("Test 4: Gap Analysis")
        print("=" * 60)

        # Get player's rank (assuming for this test)
        tier = input("\nEnter player's tier (default: PLATINUM): ").strip().upper() or "PLATINUM"

        # Create PlayerStats object
        from src.api.models import PlayerStats

        game_duration_min = player_stats['game_duration'] / 60

        player_stats_obj = PlayerStats(
            kills=player_stats['kills'],
            deaths=player_stats['deaths'],
            assists=player_stats['assists'],
            kda=player_stats['kills'] + player_stats['assists'] / max(player_stats['deaths'], 1),
            cs=player_stats['total_cs'],
            cs_per_min=player_stats['total_cs'] / game_duration_min,
            gold=player_stats['gold'],
            vision_score=player_stats['vision_score'],
            damage_dealt=player_stats['damage_dealt'],
            damage_share=player_stats['damage_share'],
            champion_name=player_stats['champion_name'],
            game_duration=player_stats['game_duration']
        )

        print(f"🔍 Analyzing gap for {tier} tier...")
        gap_result = gap_analyzer.analyze_gap(player_stats_obj, tier)

        print(f"\n✅ Gap Analysis Results:")
        print(f"   Overall Score: {gap_result.overall_score:.1f}/100")
        print(f"   Strengths ({len(gap_result.strengths)}):")
        for strength in gap_result.strengths[:3]:
            print(f"      • {strength}")
        print(f"   Weaknesses ({len(gap_result.weaknesses)}):")
        for weakness in gap_result.weaknesses[:3]:
            print(f"      • {weakness}")

        # Test 5: Tier Suggestion
        print("\n" + "=" * 60)
        print("Test 5: Tier Suggestion")
        print("=" * 60)

        suggested_tier = gap_analyzer.suggest_target_tier(player_stats_obj)
        print(f"✅ Suggested tier based on performance: {suggested_tier}")

        # Test 6: Visualization
        print("\n" + "=" * 60)
        print("Test 6: Generating Visualizations")
        print("=" * 60)

        generate_viz = input("\nGenerate visualizations? (y/n, default: y): ").strip().lower() or 'y'

        if generate_viz == 'y':
            print("🎨 Generating visualizations...")

            # Performance overview
            overview_path = visualizer.plot_performance_overview(gap_result.model_dump())
            print(f"✅ Performance Overview: {overview_path}")

            # Radar chart
            gold_per_min = player_stats_obj.gold / (player_stats_obj.game_duration / 60) if player_stats_obj.game_duration else 0
            vision_score_per_min = player_stats_obj.vision_score / (player_stats_obj.game_duration / 60) if player_stats_obj.game_duration else 0
            player_dict = {
                'kda': player_stats_obj.kda,
                'cs_per_min': player_stats_obj.cs_per_min,
                'gold_per_min': gold_per_min,
                'vision_score_per_min': vision_score_per_min,
                'damage_share': player_stats_obj.damage_share,
            }
            baseline = gap_analyzer.get_tier_baseline(tier)
            radar_path = visualizer.plot_radar_chart(player_dict, baseline, tier)
            print(f"✅ Radar Chart: {radar_path}")

            # Tier comparison
            all_comparisons = gap_analyzer.compare_with_multiple_tiers(player_stats_obj)
            tier_suggestions = [
                {'tier': t, 'match_score': analysis.overall_score}
                for t, analysis in all_comparisons.items()
            ]
            tier_comp_path = visualizer.plot_tier_comparison(player_dict, tier_suggestions)
            print(f"✅ Tier Comparison: {tier_comp_path}")

            print(f"\n📁 All visualizations saved to 'visualization_results/' directory")

        print("\n" + "=" * 60)
        print("✨ All tests completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
