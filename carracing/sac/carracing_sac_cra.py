#!/usr/bin/env python3
"""
Comprehensive Configuration Ranking Analysis
Ranks all evaluated SAC configurations by the most significant metrics for CarRacing
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

env_name = "carracing"
alg_name = "sac"

# Set style for publication-ready plots
plt.style.use('default')
sns.set_style("whitegrid")
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
})

def load_and_process_all_configs(json_file):
    """Load and process all configurations with comprehensive metrics"""
    
    with open(json_file, 'r') as f:
        results = json.load(f)
    
    configs = []
    
    for result in results:
        config_name = result['config_name']
        
        # Extract individual seed performances for detailed analysis
        individual_performances = [seed['test_mean_reward'] for seed in result['seed_results']]
        individual_success = [1 if seed['test_mean_reward'] >= 500 else 0 for seed in result['seed_results']]  # CarRacing success threshold
        individual_efficiency = [seed.get('sample_efficiency_500') for seed in result['seed_results']]  # CarRacing efficiency threshold
        individual_efficiency = [x for x in individual_efficiency if x is not None]
        
        config_data = {
            # Basic Info
            'config_name': config_name,
            'full_config': result['config'],
            
            # Tier 1 Metrics (Most Critical)
            'test_mean_reward': result['test_mean_reward_mean'],
            'test_std_reward': result['test_mean_reward_std'],
            'success_rate': result['success_rate_mean'],
            'individual_performances': individual_performances,
            
            # Tier 2 Metrics (Important)
            'sample_efficiency': result.get('sample_efficiency_500_mean'),
            'sample_efficiency_success_rate': result.get('sample_efficiency_500_success_rate', 0),
            'performance_variance': result['test_mean_reward_std'],  # Lower = more stable
            
            # Tier 3 Metrics (Context)
            'final_training_performance': result['final_performance_mean'],
            'max_reward': result['max_reward_mean'],
            'num_seeds': result['num_seeds'],
            
            # Derived Metrics
            'solved': result['test_mean_reward_mean'] >= 500,  # CarRacing good performance threshold
            'reliability_score': 1 / (1 + result['test_mean_reward_std']),  # Higher = more reliable
        }
        
        configs.append(config_data)
    
    return pd.DataFrame(configs)

def calculate_comprehensive_rankings(df):
    """Calculate rankings across all significant metrics"""
    
    rankings = df.copy()
    
    # Rank by each metric (1 = best)
    rankings['rank_performance'] = df['test_mean_reward'].rank(ascending=False)
    rankings['rank_success_rate'] = df['success_rate'].rank(ascending=False)
    rankings['rank_stability'] = df['performance_variance'].rank(ascending=True)  # Lower variance = better
    
    # Sample efficiency ranking (only for configs that achieved it)
    efficiency_data = df[df['sample_efficiency'].notna()].copy()
    if not efficiency_data.empty:
        efficiency_ranks = efficiency_data['sample_efficiency'].rank(ascending=True)  # Lower = better
        rankings['rank_efficiency'] = np.nan
        rankings.loc[efficiency_data.index, 'rank_efficiency'] = efficiency_ranks
    else:
        rankings['rank_efficiency'] = np.nan
    
    # Composite score (lower = better, weighted by importance)
    # Weights: Performance (40%), Success Rate (30%), Stability (20%), Efficiency (10%)
    rankings['composite_score'] = (
        0.4 * rankings['rank_performance'] +
        0.3 * rankings['rank_success_rate'] +
        0.2 * rankings['rank_stability'] +
        0.1 * rankings['rank_efficiency'].fillna(rankings['rank_performance'].max() + 1)
    )
    
    # Final ranking based on composite score
    rankings['final_rank'] = rankings['composite_score'].rank(ascending=True)
    
    # Sort by final rank for presentation
    rankings = rankings.sort_values('final_rank').reset_index(drop=True)
    
    return rankings

def create_comprehensive_ranking_visualization(df_ranked):
    """Create comprehensive visualization of all ranking metrics"""
    
    plt.figure(figsize=(20, 12))
    
    # 1. Performance Ranking (Top Left)
    plt.subplot(2, 3, 1)
    
    plot_data = df_ranked.sort_values('test_mean_reward', ascending=True)
    colors = ['#ff9999' if not solved else '#90ee90' for solved in plot_data['solved']]
    bars = plt.barh(range(len(plot_data)), plot_data['test_mean_reward'], 
                    color=colors, alpha=0.8, edgecolor='black')
    
    plt.yticks(range(len(plot_data)), plot_data['config_name'])
    plt.xlabel('Test Mean Reward')
    plt.title('(A) Performance Ranking', fontweight='bold')
    plt.axvline(x=500, color='blue', linestyle='--', alpha=0.8, linewidth=2)
    plt.grid(axis='x', alpha=0.3)
    
    # Add performance values on bars
    for bar, value in zip(bars, plot_data['test_mean_reward']):
        plt.text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2, 
                f'{value:.0f}', va='center', fontweight='bold')
    
    # 2. Success Rate Ranking (Top Middle)
    plt.subplot(2, 3, 2)
    
    plot_data = df_ranked.sort_values('success_rate', ascending=True)
    bars = plt.barh(range(len(plot_data)), plot_data['success_rate'] * 100, 
                    color='orange', alpha=0.8, edgecolor='black')
    
    plt.yticks(range(len(plot_data)), plot_data['config_name'])
    plt.xlabel('Success Rate (%)')
    plt.title('(B) Success Rate Ranking\n(% Episodes > 500)', fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    
    # Add percentage labels
    for bar, value in zip(bars, plot_data['success_rate']):
        plt.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                f'{value*100:.0f}%', va='center', fontweight='bold')
    
    # 3. Stability Ranking (Top Right)
    plt.subplot(2, 3, 3)
    
    plot_data = df_ranked.sort_values('performance_variance', ascending=False)  # Higher variance = less stable
    bars = plt.barh(range(len(plot_data)), plot_data['performance_variance'], 
                    color='purple', alpha=0.8, edgecolor='black')
    
    plt.yticks(range(len(plot_data)), plot_data['config_name'])
    plt.xlabel('Performance Variance (Lower = More Stable)')
    plt.title('(C) Stability Ranking', fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    
    # 4. Sample Efficiency (Bottom Left)
    plt.subplot(2, 3, 4)
    
    # Only show configs that achieved the efficiency threshold
    efficiency_data = df_ranked[df_ranked['sample_efficiency'].notna()]
    
    if not efficiency_data.empty:
        plot_data = efficiency_data.sort_values('sample_efficiency', ascending=False)
        bars = plt.barh(range(len(plot_data)), plot_data['sample_efficiency'], 
                        color='green', alpha=0.8, edgecolor='black')
        
        plt.yticks(range(len(plot_data)), plot_data['config_name'])
        plt.xlabel('Episodes to Reach 500+ Reward')
        plt.title('(D) Sample Efficiency\n(Lower = Better)', fontweight='bold')
        plt.grid(axis='x', alpha=0.3)
        
        # Add efficiency values
        for bar, value in zip(bars, plot_data['sample_efficiency']):
            plt.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2, 
                    f'{value:.0f}', va='center', fontweight='bold')
    else:
        plt.text(0.5, 0.5, 'No configurations\nachieved 500+ reward\nduring training', 
                ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
        plt.title('(D) Sample Efficiency\n(No Data Available)', fontweight='bold')
    
    # 5. Composite Score Ranking (Bottom Middle)
    plt.subplot(2, 3, 5)
    
    plot_data = df_ranked.sort_values('composite_score', ascending=False)
    bars = plt.barh(range(len(plot_data)), plot_data['composite_score'], 
                    color='red', alpha=0.8, edgecolor='black')
    
    plt.yticks(range(len(plot_data)), plot_data['config_name'])
    plt.xlabel('Composite Score (Lower = Better)')
    plt.title('(E) Overall Ranking\n(Balanced Evaluation)', fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    
    # Add rank labels
    for bar, rank in zip(bars, plot_data['final_rank']):
        plt.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2, 
                f'#{int(rank)}', va='center', fontweight='bold')
    
    # 6. Performance Distribution (Bottom Right)
    plt.subplot(2, 3, 6)
    
    # Box plot of individual seed performances
    box_data = []
    box_labels = []
    box_colors = []
    
    for _, row in df_ranked.iterrows():
        box_data.append(row['individual_performances'])
        box_labels.append(row['config_name'])
        box_colors.append("#95e1d3")
    
    bp = plt.boxplot(box_data, labels=box_labels, patch_artist=True)
    
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor('black')
    
    plt.xticks(rotation=45)
    plt.ylabel('Individual Seed Performance')
    plt.title('(F) Performance Distribution', fontweight='bold')
    plt.axhline(y=500, color='blue', linestyle='--', alpha=0.8, linewidth=2)
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Create output directory
    os.makedirs(f'./carracing/sac/cra/', exist_ok=True)
    plt.savefig(f'./carracing/sac/cra/{env_name}_{alg_name}_cra.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_ranking_summary_table(df_ranked):
    """Create a comprehensive ranking summary table"""
    
    print("="*100)
    print("COMPREHENSIVE CONFIGURATION RANKING ANALYSIS - CARRACING SAC")
    print("="*100)
    print()
    
    print("FINAL RANKINGS (Lower rank number = Better performance):")
    print("="*70)
    
    print(f"{'Rank':<4} {'Configuration':<20} {'Performance':<15} {'Success':<10} {'Stability':<12} {'Efficiency':<12} {'Overall':<8}")
    print("-" * 90)
    
    for _, row in df_ranked.iterrows():
        rank = int(row['final_rank'])
        config = row['config_name'][:18]
        performance = f"{row['test_mean_reward']:.1f} ± {row['test_std_reward']:.1f}"
        success = f"{row['success_rate']:.0%}"
        stability = f"#{int(row['rank_stability'])}"
        
        if pd.notna(row['rank_efficiency']):
            efficiency = f"#{int(row['rank_efficiency'])}"
        else:
            efficiency = "N/A"
        
        overall = f"{row['composite_score']:.2f}"
        
        print(f"{rank:<4} {config:<20} {performance:<15} {success:<10} {stability:<12} {efficiency:<12} {overall:<8}")
    
    print("\nLegend:")
    print("- Performance: Mean ± Std test reward across seeds")
    print("- Success: Percentage of test episodes with reward > 500")
    print("- Stability: Rank based on performance variance (lower variance = better)")
    print("- Efficiency: Rank based on episodes needed to reach 500+ reward")
    print("- Overall: Composite score (lower = better)")

def identify_best_configuration(df_ranked):
    """Identify and analyze the best configuration"""
    
    best_config = df_ranked.iloc[0]
    
    print("\n" + "="*80)
    print("🏆 BEST CONFIGURATION ANALYSIS")
    print("="*80)
    
    config = best_config['full_config']
    
    print(f"Best Configuration: {best_config['config_name']}")
    print(f"Final Rank: #{int(best_config['final_rank'])}")
    print(f"Composite Score: {best_config['composite_score']:.3f}")
    print()
    
    print("Configuration Details:")
    print("-" * 25)
    for key, value in config.items():
        if key != 'total_timesteps':  # Skip timesteps as it's for tuning only
            print(f"  {key}: {value}")
    
    print()
    print("Performance Metrics:")
    print("-" * 20)
    print(f"  Mean Test Reward: {best_config['test_mean_reward']:.2f} ± {best_config['test_std_reward']:.2f}")
    print(f"  Success Rate: {best_config['success_rate']:.1%}")
    print(f"  Max Reward Achieved: {best_config['max_reward']:.1f}")
    print(f"  Performance Variance: {best_config['performance_variance']:.2f}")
    
    if pd.notna(best_config['sample_efficiency']):
        print(f"  Sample Efficiency: {best_config['sample_efficiency']:.0f} episodes to 500+")
    else:
        print(f"  Sample Efficiency: Not achieved during training")
    
    solved_status = "✅ GOOD PERFORMANCE" if best_config['solved'] else "❌ NEEDS IMPROVEMENT"
    print(f"  Status: {solved_status}")

def statistical_significance_of_winner(df_ranked):
    """Test statistical significance of the best configuration"""
    
    print("\n" + "="*80)
    print("📊 STATISTICAL SIGNIFICANCE OF BEST CONFIGURATION")
    print("="*80)
    
    best_config = df_ranked.iloc[0]
    best_performances = best_config['individual_performances']
    
    print(f"Best Configuration: {best_config['config_name']}")
    print(f"Individual Seed Performances: {best_performances}")
    print()
    
    # Compare with each other configuration
    print("Statistical Comparisons with Other Configurations:")
    print("-" * 45)
    
    for _, other_config in df_ranked.iloc[1:].iterrows():
        other_performances = other_config['individual_performances']
        
        # Mann-Whitney U test
        if len(best_performances) >= 2 and len(other_performances) >= 2:
            stat, p_value = stats.mannwhitneyu(best_performances, other_performances, alternative='two-sided')
            
            # Effect size
            pooled_std = np.sqrt((np.var(best_performances, ddof=1) + np.var(other_performances, ddof=1)) / 2)
            effect_size = abs(np.mean(best_performances) - np.mean(other_performances)) / pooled_std if pooled_std > 0 else 0
            
            significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
            
            print(f"vs {other_config['config_name'][:15]:<15}: p={p_value:.4f} {significance}, d={effect_size:.3f}")
    
    print()
    print("Legend: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")

def generate_recommendations(df_ranked):
    """Generate optimization recommendations based on results"""
    
    print("\n" + "="*80)
    print("🎯 CARRACING SAC OPTIMIZATION RECOMMENDATIONS FOR THESIS")
    print("="*80)
    print()
    
    best_config = df_ranked.iloc[0]
    
    print("1. OPTIMAL HYPERPARAMETER VALUES:")
    print("-" * 40)
    config = best_config['full_config']
    
    if 'learning_rate' in config:
        print(f"   ✅ Learning Rate: {config['learning_rate']}")
    if 'buffer_size' in config:
        print(f"   ✅ Buffer Size: {config['buffer_size']}")
    if 'batch_size' in config:
        print(f"   ✅ Batch Size: {config['batch_size']}")
    if 'tau' in config:
        print(f"   ✅ Soft Update Rate (tau): {config['tau']}")
    
    print()
    
    print("2. KEY INSIGHTS FOR CARRACING SAC OPTIMIZATION:")
    print("-" * 50)
    print("   • SAC's sample efficiency critical for expensive simulations")
    print(f"   • Proper configuration achieves {best_config['test_mean_reward']:.1f} average reward")
    print(f"   • Success rate can reach {best_config['success_rate']:.0%} with optimal settings")
    print("   • Automatic entropy tuning handles exploration-exploitation balance")
    print()
    
    print("3. CRITICAL SAC SUCCESS FACTORS FOR CARRACING:")
    print("-" * 47)
    print("   • Buffer size affects off-policy learning stability")
    print("   • Soft updates prevent catastrophic Q-value changes")
    print("   • Automatic entropy tuning adapts to continuous action space")
    print("   • Network architecture balances actor-critic learning")
    print()
    
    print("4. IMPLICATIONS FOR CONTINUOUS CONTROL ENVIRONMENTS:")
    print("-" * 58)
    print("   • SAC's off-policy nature maximizes sample efficiency")
    print("   • Entropy regularization prevents premature convergence")
    print("   • Superior performance in high-dimensional continuous control")

def main():
    """Run comprehensive configuration ranking analysis"""
    
    json_file = f'./carracing/sac/json_data/{env_name}_{alg_name}_hp.json'
    
    try:
        # Load and process all configurations
        df = load_and_process_all_configs(json_file)
        
        print("📊 COMPREHENSIVE CARRACING SAC CONFIGURATION RANKING ANALYSIS")
        print("="*69)
        print(f"Analyzing {len(df)} configurations across multiple metrics...")
        print()
        
        # Calculate comprehensive rankings
        df_ranked = calculate_comprehensive_rankings(df)
        
        # Create visualizations
        print("🎨 Generating comprehensive ranking visualizations...")
        create_comprehensive_ranking_visualization(df_ranked)
        
        # Display ranking table
        create_ranking_summary_table(df_ranked)
        
        # Identify best configuration
        identify_best_configuration(df_ranked)
        
        # Statistical validation
        statistical_significance_of_winner(df_ranked)
        
        # Generate recommendations
        generate_recommendations(df_ranked)
        
        print(f"\n✅ CarRacing SAC Analysis Complete!")
        print(f"📁 Visualizations saved to ./carracing/sac/cra/")
        
    except FileNotFoundError:
        print(f"❌ Error: Could not find {json_file}")
        print("Please ensure you have run the hyperparameter tuning first.")
        print("The file should be generated by carracing_sac_hyperparameter.py")

if __name__ == "__main__":
    main()