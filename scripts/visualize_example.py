"""Example script for testing visualization functionality"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.rule_based import RuleBasedGapAnalyzer
from src.utils.visualizer import GapAnalysisVisualizer
from src.api.models import PlayerStats


def main():
    """Main function to demonstrate visualizations"""

    print("🎨 LOL Gap Analysis Visualization Demo\n")

    # Initialize analyzer and visualizer
    analyzer = RuleBasedGapAnalyzer()
    visualizer = GapAnalysisVisualizer(output_dir="visualization_results")

    # Example 1: Above-tier performance (Diamond stats in Gold tier)
    print("=" * 60)
    print("Example 1: High Performance Player (Diamond stats in Gold)")
    print("=" * 60)

    high_performer = PlayerStats(
        kills=12,
        deaths=3,
        assists=10,
        kda=7.3,
        cs=240,
        cs_per_min=8.0,
        gold=18000,
        vision_score=45,
        damage_dealt=30000,
        damage_share=0.28,
        champion_name="Zed"
    )

    # Analyze
    result = analyzer.analyze_gap(high_performer, "GOLD")
    print(f"Overall Score: {result.overall_score:.1f}/100")
    print(f"Strengths: {len(result.strengths)}")
    print(f"Weaknesses: {len(result.weaknesses)}\n")

    # Generate visualizations
    print("Generating visualizations...")

    # 1. Comprehensive overview
    overview_path = visualizer.plot_performance_overview(result.dict())
    print(f"✅ Performance Overview: {overview_path}")

    # 2. Radar chart
    player_dict = {
        'kda': high_performer.kda,
        'cs_per_min': high_performer.cs_per_min,
        'gold': high_performer.gold,
        'vision_score': high_performer.vision_score,
        'damage_share': high_performer.damage_share,
    }
    baseline = analyzer.get_tier_baseline("GOLD")
    radar_path = visualizer.plot_radar_chart(player_dict, baseline, "GOLD")
    print(f"✅ Radar Chart: {radar_path}")

    # 3. Gap bars
    gap_path = visualizer.plot_gap_bars(result.normalized_gaps, "GOLD")
    print(f"✅ Gap Bars: {gap_path}")

    # 4. Tier comparison
    all_comparisons = analyzer.compare_with_multiple_tiers(high_performer)
    tier_suggestions = [
        {'tier': tier, 'match_score': analysis.overall_score}
        for tier, analysis in all_comparisons.items()
    ]
    tier_comp_path = visualizer.plot_tier_comparison(player_dict, tier_suggestions)
    print(f"✅ Tier Comparison: {tier_comp_path}\n")

    # Example 2: Below-tier performance (Bronze stats in Diamond tier)
    print("=" * 60)
    print("Example 2: Struggling Player (Bronze stats in Diamond)")
    print("=" * 60)

    low_performer = PlayerStats(
        kills=3,
        deaths=9,
        assists=5,
        kda=0.9,
        cs=95,
        cs_per_min=3.8,
        gold=8500,
        vision_score=12,
        damage_dealt=10000,
        damage_share=0.14,
        champion_name="Annie"
    )

    # Analyze
    result2 = analyzer.analyze_gap(low_performer, "DIAMOND")
    print(f"Overall Score: {result2.overall_score:.1f}/100")
    print(f"Strengths: {len(result2.strengths)}")
    print(f"Weaknesses: {len(result2.weaknesses)}\n")

    # Generate overview
    print("Generating visualization...")
    overview_path2 = visualizer.plot_performance_overview(result2.dict())
    print(f"✅ Performance Overview: {overview_path2}\n")

    # Example 3: Perfect tier match (Gold stats in Gold tier)
    print("=" * 60)
    print("Example 3: Perfect Match (Gold stats in Gold)")
    print("=" * 60)

    gold_baseline = analyzer.get_tier_baseline("GOLD")

    perfect_match = PlayerStats(
        kills=int(gold_baseline["avg_kda"] * 2),
        deaths=2,
        assists=int(gold_baseline["avg_kda"] * 2),
        kda=gold_baseline["avg_kda"],
        cs=int(gold_baseline["avg_cs_per_min"] * 30),
        cs_per_min=gold_baseline["avg_cs_per_min"],
        gold=gold_baseline["avg_gold"],
        vision_score=gold_baseline["avg_vision_score"],
        damage_dealt=20000,
        damage_share=gold_baseline["avg_damage_share"],
        champion_name="Lux"
    )

    # Analyze
    result3 = analyzer.analyze_gap(perfect_match, "GOLD")
    print(f"Overall Score: {result3.overall_score:.1f}/100")
    print(f"Strengths: {len(result3.strengths)}")
    print(f"Weaknesses: {len(result3.weaknesses)}\n")

    # Generate overview
    print("Generating visualization...")
    overview_path3 = visualizer.plot_performance_overview(result3.dict())
    print(f"✅ Performance Overview: {overview_path3}\n")

    print("=" * 60)
    print("✨ All visualizations generated successfully!")
    print(f"📁 Check the 'visualization_results' directory")
    print("=" * 60)


if __name__ == "__main__":
    main()
