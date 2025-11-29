"""Visualization utilities for gap analysis results"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import numpy as np
from typing import Dict, List
import os
from datetime import datetime


class GapAnalysisVisualizer:
    """Visualizer for gap analysis results"""

    def __init__(self, output_dir: str = "visualization_results"):
        """
        Initialize visualizer

        Args:
            output_dir: Directory to save visualization outputs
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10

        # Try to set Korean font (for macOS)
        try:
            plt.rcParams['font.family'] = 'AppleGothic'
        except:
            pass

        # Fix minus sign display
        plt.rcParams['axes.unicode_minus'] = False

    def plot_radar_chart(
        self,
        player_stats: Dict[str, float],
        baseline_stats: Dict[str, float],
        tier: str,
        save_path: str = None
    ) -> str:
        """
        Create radar chart comparing player stats to baseline

        Args:
            player_stats: Player's normalized stats
            baseline_stats: Tier baseline normalized stats
            tier: Tier name
            save_path: Custom save path (optional)

        Returns:
            Path to saved image
        """
        # Categories
        categories = ['KDA', 'CS/min', 'Gold', 'Vision', 'Damage']

        # Extract values (normalized to 0-100 scale)
        player_values = [
            player_stats.get('kda', 0) * 10,
            player_stats.get('cs_per_min', 0) * 10,
            player_stats.get('gold_per_min', player_stats.get('gold', 0) / 200),  # fallback to old value if needed
            player_stats.get('vision_score_per_min', player_stats.get('vision_score', 0)) * 30,  # scale to ~0-50 range
            player_stats.get('damage_share', 0) * 100
        ]

        baseline_values = [
            baseline_stats.get('avg_kda', 0) * 10,
            baseline_stats.get('avg_cs_per_min', 0) * 10,
            baseline_stats.get('avg_gold_per_min', baseline_stats.get('avg_gold', 0) / 200),  # fallback to old value if needed
            baseline_stats.get('avg_vision_score_per_min', baseline_stats.get('avg_vision_score', 0)) * 30,  # scale to ~0-50 range
            baseline_stats.get('avg_damage_share', 0) * 100
        ]

        # Number of variables
        num_vars = len(categories)

        # Compute angle for each axis
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

        # Complete the circle
        player_values += player_values[:1]
        baseline_values += baseline_values[:1]
        angles += angles[:1]

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

        # Plot data
        ax.plot(angles, player_values, 'o-', linewidth=2, label='Player', color='#FF6B6B')
        ax.fill(angles, player_values, alpha=0.25, color='#FF6B6B')

        ax.plot(angles, baseline_values, 'o-', linewidth=2, label=f'{tier} Average', color='#4ECDC4')
        ax.fill(angles, baseline_values, alpha=0.25, color='#4ECDC4')

        # Fix axis to go in the right order
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)

        # Set category labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=12)

        # Set y-axis limits
        ax.set_ylim(0, 100)

        # Add legend
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12)

        # Add title
        plt.title(f'Performance Comparison - {tier} Tier', size=16, pad=20)

        # Save
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.output_dir, f"radar_chart_{timestamp}.png")

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        return save_path

    def plot_gap_bars(
        self,
        gaps: Dict[str, float],
        tier: str,
        save_path: str = None
    ) -> str:
        """
        Create bar chart showing gaps from baseline

        Args:
            gaps: Dictionary of stat gaps (positive = above baseline)
            tier: Tier name
            save_path: Custom save path (optional)

        Returns:
            Path to saved image
        """
        # Prepare data
        stats = list(gaps.keys())
        values = list(gaps.values())

        # Map stat names to readable labels
        label_map = {
            'kda': 'KDA',
            'cs_per_min': 'CS/min',
            'gold': 'Gold',
            'vision_score': 'Vision Score',
            'damage_share': 'Damage Share'
        }
        labels = [label_map.get(s, s) for s in stats]

        # Create colors based on positive/negative
        colors = ['#4ECDC4' if v >= 0 else '#FF6B6B' for v in values]

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))

        bars = ax.barh(labels, values, color=colors, alpha=0.8)

        # Add zero line
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)

        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, values)):
            if value >= 0:
                ax.text(value + 0.5, i, f'+{value:.1f}%',
                       va='center', fontsize=11, fontweight='bold')
            else:
                ax.text(value - 0.5, i, f'{value:.1f}%',
                       va='center', ha='right', fontsize=11, fontweight='bold')

        # Styling
        ax.set_xlabel('Gap from Baseline (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'Performance Gaps - {tier} Tier', fontsize=16, fontweight='bold', pad=20)
        ax.grid(axis='x', alpha=0.3)

        # Save
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.output_dir, f"gap_bars_{timestamp}.png")

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        return save_path

    def plot_performance_overview(
        self,
        gap_result: dict,
        save_path: str = None
    ) -> str:
        """
        Create comprehensive performance overview with multiple subplots

        Args:
            gap_result: Complete gap analysis result dictionary
            save_path: Custom save path (optional)

        Returns:
            Path to saved image
        """
        fig = plt.figure(figsize=(16, 10))

        # 1. Gap bars (top left)
        ax1 = plt.subplot(2, 2, 1)
        gaps = gap_result.get('normalized_gaps', {})
        stats = list(gaps.keys())
        values = list(gaps.values())

        label_map = {
            'kda': 'KDA',
            'cs_per_min': 'CS/min',
            'gold': 'Gold',
            'vision_score': 'Vision',
            'damage_share': 'Damage'
        }
        labels = [label_map.get(s, s) for s in stats]
        colors = ['#4ECDC4' if v >= 0 else '#FF6B6B' for v in values]

        ax1.barh(labels, values, color=colors, alpha=0.8)
        ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax1.set_xlabel('Gap (%)')
        ax1.set_title('Performance Gaps', fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)

        # 2. Overall score gauge (top right)
        ax2 = plt.subplot(2, 2, 2)
        score = gap_result.get('overall_score', 50)

        # Create gauge
        theta = np.linspace(0, np.pi, 100)
        r = np.ones(100)

        # Background
        ax2.fill_between(theta, 0, r, color='#CCCCCC', alpha=0.3)

        # Score arc
        score_theta = np.linspace(0, np.pi * (score / 100), 100)
        if score >= 70:
            color = '#4ECDC4'
        elif score >= 40:
            color = '#FFD93D'
        else:
            color = '#FF6B6B'

        ax2.fill_between(score_theta, 0, r, color=color, alpha=0.7)

        # Add score text
        ax2.text(np.pi/2, 0.5, f'{score:.1f}',
                ha='center', va='center', fontsize=36, fontweight='bold')
        ax2.text(np.pi/2, 0.2, 'Overall Score',
                ha='center', va='center', fontsize=12)

        ax2.set_ylim(0, 1.2)
        ax2.axis('off')
        ax2.set_title('Performance Score', fontweight='bold', pad=20)

        # 3. Strengths and Weaknesses (bottom left)
        ax3 = plt.subplot(2, 2, 3)
        ax3.axis('off')

        strengths = gap_result.get('strengths', [])[:5]
        weaknesses = gap_result.get('weaknesses', [])[:5]

        y_pos = 0.9
        ax3.text(0.05, y_pos, 'Strengths:', fontsize=14, fontweight='bold', color='#4ECDC4')
        y_pos -= 0.1

        for i, strength in enumerate(strengths, 1):
            ax3.text(0.08, y_pos, f'{i}. {strength}', fontsize=10, wrap=True)
            y_pos -= 0.08

        y_pos -= 0.05
        ax3.text(0.05, y_pos, 'Weaknesses:', fontsize=14, fontweight='bold', color='#FF6B6B')
        y_pos -= 0.1

        for i, weakness in enumerate(weaknesses, 1):
            ax3.text(0.08, y_pos, f'{i}. {weakness}', fontsize=10, wrap=True)
            y_pos -= 0.08

        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        ax3.set_title('Analysis Summary', fontweight='bold')

        # 4. Recommendations (bottom right)
        ax4 = plt.subplot(2, 2, 4)
        ax4.axis('off')

        recommendations = gap_result.get('recommendations', [])[:6]

        y_pos = 0.9
        ax4.text(0.05, y_pos, 'Recommendations:', fontsize=14, fontweight='bold', color='#6C5CE7')
        y_pos -= 0.1

        for i, rec in enumerate(recommendations, 1):
            # Wrap text if too long
            if len(rec) > 50:
                rec = rec[:47] + '...'
            ax4.text(0.08, y_pos, f'{i}. {rec}', fontsize=10, wrap=True)
            y_pos -= 0.12

        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.set_title('Improvement Plan', fontweight='bold')

        # Main title
        tier = gap_result.get('tier', 'UNKNOWN')
        fig.suptitle(f'Performance Analysis Report - {tier} Tier',
                    fontsize=18, fontweight='bold', y=0.98)

        # Save
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.output_dir, f"performance_overview_{timestamp}.png")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        return save_path

    def plot_tier_comparison(
        self,
        player_stats: Dict[str, float],
        tier_suggestions: List[Dict],
        save_path: str = None
    ) -> str:
        """
        Compare player performance across multiple tiers

        Args:
            player_stats: Player statistics
            tier_suggestions: List of tier comparison results
            save_path: Custom save path (optional)

        Returns:
            Path to saved image
        """
        fig, ax = plt.subplots(figsize=(14, 8))

        tiers = [t['tier'] for t in tier_suggestions]
        scores = [t['match_score'] for t in tier_suggestions]

        # Create gradient colors
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(tiers)))

        bars = ax.bar(tiers, scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

        # Add value labels on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{score:.1f}%', ha='center', va='bottom',
                   fontsize=12, fontweight='bold')

        # Add baseline at 50%
        ax.axhline(y=50, color='gray', linestyle='--', linewidth=2, label='Baseline (50%)')

        # Styling
        ax.set_xlabel('Tier', fontsize=14, fontweight='bold')
        ax.set_ylabel('Match Score (%)', fontsize=14, fontweight='bold')
        ax.set_title('Performance Match by Tier', fontsize=16, fontweight='bold', pad=20)
        ax.set_ylim(0, 110)
        ax.grid(axis='y', alpha=0.3)
        ax.legend(fontsize=12)

        # Rotate x labels if many tiers
        plt.xticks(rotation=45 if len(tiers) > 5 else 0)

        # Save
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.output_dir, f"tier_comparison_{timestamp}.png")

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        return save_path
