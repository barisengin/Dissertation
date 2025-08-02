"""
Comprehensive Configuration Ranking Analysis
Ranks all evaluated REINFORCE configurations by the most significant metrics
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

env_name = "cartpole"
alg_name = "reinforce"

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
        individual_success = [1 if seed['test_mean_reward'] >= 475 else 0 for seed in result['seed_results']]
        individual_efficiency = [seed.get('sample_efficiency_400') for seed in result['seed_results']]
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
            'sample_efficiency': result.get('sample_efficiency_400_mean'),
            'sample_efficiency_success_rate': result.get('sample_efficiency_400_success_rate', 0),
            'performance_variance': result['test_mean_reward_std'],  # Lower = more stable
            
            # Tier 3 Metrics (Context)
            'final_training_performance': result['final_performance_mean'],
            'max_reward': result['max_reward_mean'],
            'num_seeds': result['num_seeds'],
            
            # Derived Metrics
            'solved': result['test_mean_reward_mean'] >= 475,
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
        rankings.loc[rankings['sample_efficiency'].notna(), 'rank_efficiency'] = efficiency_ranks
    else:
        rankings['rank_efficiency'] = np.nan
    
    # For REINFORCE, variance is particularly important due to high policy gradient variance
    # Adjust weights to emphasize stability more
    weights = {
        'performance': 0.35,
        'success_rate': 0.25,
        'stability': 0.30,  # Higher weight for stability in REINFORCE
        'efficiency': 0.1
    }
    
    # For configs without efficiency data, redistribute weight to other metrics
    rankings['composite_score'] = 0
    
    for idx, row in rankings.iterrows():
        score = 0
        total_weight = 0
        
        # Performance (always available)
        score += weights['performance'] * row['rank_performance']
        total_weight += weights['performance']
        
        # Success rate (always available)
        score += weights['success_rate'] * row['rank_success_rate']
        total_weight += weights['success_rate']
        
        # Stability (always available) - particularly important for REINFORCE
        score += weights['stability'] * row['rank_stability']
        total_weight += weights['stability']
        
        # Efficiency (only if available)
        if not np.isnan(row['rank_efficiency']):
            score += weights['efficiency'] * row['rank_efficiency']
            total_weight += weights['efficiency']
        
        # Normalize by actual total weight
        rankings.loc[idx, 'composite_score'] = score / total_weight if total_weight > 0 else np.inf
    
    # Final ranking based on composite score (lower = better)
    rankings['final_rank'] = rankings['composite_score'].rank(ascending=True)
    
    return rankings.sort_values('final_rank')

def create_comprehensive_ranking_visualization(df_ranked):
    """Create comprehensive visualization of all rankings"""
    
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Overall Performance Ranking (Top Left)
    plt.subplot(2, 3, 1)
    
    # Sort by performance for this plot
    plot_data = df_ranked.sort_values('test_mean_reward', ascending=True)
    
    colors = "#4ecdc4"
    bars = plt.barh(range(len(plot_data)), plot_data['test_mean_reward'], 
                    color=colors, alpha=0.8, edgecolor='black')
    
    plt.yticks(range(len(plot_data)), plot_data['config_name'])
    plt.xlabel('Test Mean Reward')
    plt.title('(A) Performance Ranking', fontweight='bold')
    plt.axvline(x=475, color='blue', linestyle='--', alpha=0.8, linewidth=2, label='Solved Threshold')
    
    # Add value labels
    for i, (bar, val, std) in enumerate(zip(bars, plot_data['test_mean_reward'], plot_data['test_std_reward'])):
        plt.text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2, 
                f'{val:.1f}±{std:.1f}', va='center', fontweight='bold')
    
    plt.legend()
    plt.grid(axis='x', alpha=0.3)
    
    # 2. Success Rate Comparison (Top Middle)
    plt.subplot(2, 3, 2)
    
    plot_data = df_ranked.sort_values('success_rate', ascending=True)
    bars = plt.barh(range(len(plot_data)), plot_data['success_rate'] * 100, 
                    color='orange', alpha=0.8, edgecolor='black')
    
    plt.yticks(range(len(plot_data)), plot_data['config_name'])
    plt.xlabel('Success Rate (%)')
    plt.title('(B) Success Rate Ranking\n(≥475 Reward)', fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, plot_data['success_rate']):
        plt.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                f'{val:.0%}', va='center', fontweight='bold')
    
    # 3. Stability Ranking (Top Right) - Critical for REINFORCE
    plt.subplot(2, 3, 3)
    
    plot_data = df_ranked.sort_values('performance_variance', ascending=False)  # Higher variance = less stable
    bars = plt.barh(range(len(plot_data)), plot_data['performance_variance'], 
                    color='purple', alpha=0.8, edgecolor='black')
    
    plt.yticks(range(len(plot_data)), plot_data['config_name'])
    plt.xlabel('Performance Standard Deviation')
    plt.title('(C) Stability Ranking\n(Critical for REINFORCE)', fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, plot_data['performance_variance']):
        plt.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2, 
                f'{val:.1f}', va='center', fontweight='bold')
    
    # 4. Sample Efficiency (Bottom Left)
    plt.subplot(2, 3, 4)
    
    efficient_configs = df_ranked[df_ranked['sample_efficiency'].notna()].copy()
    if not efficient_configs.empty:
        plot_data = efficient_configs.sort_values('sample_efficiency', ascending=False)
        bars = plt.barh(range(len(plot_data)), plot_data['sample_efficiency'], 
                        color='green', alpha=0.8, edgecolor='black')
        
        plt.yticks(range(len(plot_data)), plot_data['config_name'])
        plt.xlabel('Episodes to Reach 400+ Reward')
        plt.title('(D) Sample Efficiency Ranking\n(Lower = More Efficient)', fontweight='bold')
        plt.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, plot_data['sample_efficiency']):
            plt.text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2, 
                    f'{val:.0f}', va='center', fontweight='bold')
    else:
        plt.text(0.5, 0.5, 'No configurations\nachieved 400+ reward\nconsistently', 
                ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
        plt.title('(D) Sample Efficiency Ranking', fontweight='bold')
    
    # 5. Composite Score Ranking (Bottom Middle)
    plt.subplot(2, 3, 5)
    
    plot_data = df_ranked.sort_values('composite_score', ascending=False)
    bars = plt.barh(range(len(plot_data)), plot_data['composite_score'], 
                    color='red', alpha=0.8, edgecolor='black')
    
    plt.yticks(range(len(plot_data)), plot_data['config_name'])
    plt.xlabel('Composite Score (Lower = Better)')
    plt.title('(E) Overall Ranking\n(Stability-Weighted)', fontweight='bold')
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
        box_colors.append("#4ecdc4")
    
    bp = plt.boxplot(box_data, labels=box_labels, patch_artist=True)
    
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor('black')
    
    plt.xticks(rotation=45)
    plt.ylabel('Individual Seed Performance')
    plt.title('(F) Performance Distribution\n(Variance Analysis)', fontweight='bold')
    plt.axhline(y=475, color='blue', linestyle='--', alpha=0.8, linewidth=2)
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'./cartpole/reinforce/cra/{env_name}_{alg_name}_cra.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_ranking_summary_table(df_ranked):
    """Create a comprehensive ranking summary table"""
    
    print("="*100)
    print("COMPREHENSIVE CONFIGURATION RANKING ANALYSIS - REINFORCE")
    print("="*100)
    print()
    
    print("FINAL RANKINGS (Lower rank number = Better performance):")
    print("="*70)
    
    print(f"{'Rank':<4} {'Configuration':<20} {'Performance':<15} {'Success':<10} {'Stability':<12} {'Efficiency':<12} {'Overall':<8}")
    print("-" * 90)
    
    for _, row in df_ranked.iterrows():
        rank = int(row['final_rank'])
        config = row['config_name']
        performance = f"{row['test_mean_reward']:.1f}±{row['test_std_reward']:.1f}"
        success = f"{row['success_rate']:.0%}"
        stability = f"σ={row['performance_variance']:.1f}"
        efficiency = f"{row['sample_efficiency']:.0f}" if pd.notna(row['sample_efficiency']) else "N/A"
        solved = "✅" if row['solved'] else "❌"
        
        print(f"{rank:<4} {config:<20} {performance:<15} {success:<10} {stability:<12} {efficiency:<12} {solved:<8}")
    
    print()

def identify_best_configuration(df_ranked):
    """Identify and analyze the best configuration"""
    
    best_config = df_ranked.iloc[0]  # First row is best ranked
    
    print("🏆 BEST REINFORCE CONFIGURATION IDENTIFIED")
    print("="*50)
    print()
    
    print(f"🥇 WINNER: {best_config['config_name']}")
    print(f"Final Rank: #{int(best_config['final_rank'])}")
    print()
    
    print("📊 PERFORMANCE METRICS:")
    print(f"   • Test Performance: {best_config['test_mean_reward']:.1f} ± {best_config['test_std_reward']:.1f}")
    print(f"   • Success Rate: {best_config['success_rate']:.1%}")
    print(f"   • Stability (σ): {best_config['performance_variance']:.1f}")
    print(f"   • Sample Efficiency: {best_config['sample_efficiency']:.0f} episodes" if pd.notna(best_config['sample_efficiency']) else "   • Sample Efficiency: Not achieved")
    print(f"   • Solved CartPole: {'✅ YES' if best_config['solved'] else '❌ NO'}")
    print()
    
    print("⚙️ REINFORCE CONFIGURATION DETAILS:")
    config = best_config['full_config']
    reinforce_params = ['learning_rate', 'baseline', 'entropy_coeff', 'gamma']
    for key, value in config.items():
        if key in reinforce_params or key in ['architecture', 'batch_size']:
            print(f"   • {key.replace('_', ' ').title()}: {value}")
    print()
    
    print("🎯 WHY THIS REINFORCE CONFIGURATION WINS:")
    
    # Compare to worst configuration
    worst_config = df_ranked.iloc[-1]
    
    perf_improvement = ((best_config['test_mean_reward'] - worst_config['test_mean_reward']) / worst_config['test_mean_reward'] * 100)
    stability_improvement = ((worst_config['performance_variance'] - best_config['performance_variance']) / worst_config['performance_variance'] * 100)
    
    print(f"   • {perf_improvement:.0f}% better performance than worst config")
    print(f"   • {stability_improvement:.0f}% more stable than worst config")
    print(f"   • {'Solves' if best_config['solved'] else 'Approaches solution for'} CartPole environment")
    print(f"   • Manages policy gradient variance effectively")
    
    if pd.notna(best_config['sample_efficiency']):
        print(f"   • Learns efficiently in {best_config['sample_efficiency']:.0f} episodes")
    
    print()
    
    return best_config

def statistical_significance_of_winner(df_ranked):
    """Test statistical significance of the winning configuration"""
    
    print("📈 STATISTICAL VALIDATION OF REINFORCE WINNER")
    print("="*50)
    print()
    
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
    print("🎯 REINFORCE OPTIMIZATION RECOMMENDATIONS FOR THESIS")
    print("="*80)
    print()
    
    best_config = df_ranked.iloc[0]
    
    print("1. OPTIMAL REINFORCE HYPERPARAMETER VALUES:")
    print("-" * 42)
    config = best_config['full_config']
    
    reinforce_params = ['learning_rate', 'baseline', 'entropy_coeff', 'gamma']
    for param in reinforce_params:
        if param in config:
            print(f"   ✅ {param.replace('_', ' ').title()}: {config[param]}")
    
    print()
    
    print("2. KEY INSIGHTS FOR REINFORCE OPTIMIZATION:")
    print("-" * 42)
    print("   • Policy gradient methods require careful variance management")
    print(f"   • Proper configuration achieves {best_config['test_mean_reward']:.1f} average reward")
    print(f"   • Success rate can reach {best_config['success_rate']:.0%} with optimal settings")
    print("   • Baseline implementation critical for variance reduction")
    print()
    
    print("3. CRITICAL REINFORCE SUCCESS FACTORS:")
    print("-" * 37)
    print("   • Learning rate directly affects convergence stability")
    print("   • Baseline reduces variance without introducing bias")
    print("   • Entropy coefficient balances exploration vs exploitation")
    print("   • Multiple seeds essential due to inherent variance")
    print()
    
    print("4. IMPLICATIONS FOR COMPLEX ENVIRONMENTS:")
    print("-" * 45)
    print("   • Variance management becomes more critical in complex tasks")
    print("   • Baseline implementation techniques scale to larger problems")
    print("   • Foundation for understanding advanced policy gradient methods")

def main():
    """Run comprehensive configuration ranking analysis"""
    
    json_file = f'./cartpole/reinforce/json_data/{env_name}_{alg_name}_hp.json'
    
    try:
        # Load and process all configurations
        df = load_and_process_all_configs(json_file)
        
        print("📊 COMPREHENSIVE REINFORCE CONFIGURATION RANKING ANALYSIS")
        print("="*62)
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
        best_config = identify_best_configuration(df_ranked)
        
        # Statistical validation
        statistical_significance_of_winner(df_ranked)
        
        # Generate recommendations
        generate_recommendations(df_ranked)
        
        print(f"\n✅ REINFORCE Analysis Complete!")
        print(f"📁 Visualizations saved")
        
    except FileNotFoundError:
        print(f"❌ Error: Could not find {json_file}")
        print("Update filename to match your JSON file")

if __name__ == "__main__":
    main()