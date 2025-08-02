import gym
import numpy as np
import torch
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
import json
from datetime import datetime
from scipy import stats
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

class PendulumWrapper(gym.Wrapper):
    """Pendulum-specific wrapper for preprocessing"""
    
    def __init__(self, env):
        super(PendulumWrapper, self).__init__(env)
        
    def reset(self, **kwargs):
        """Reset environment and handle new gym API"""
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple):
            obs, info = result
            return obs, info
        else:
            return result, {}
        
    def step(self, action):
        """Step environment and handle action format"""
        # Ensure action is properly formatted for Pendulum
        if isinstance(action, np.ndarray):
            action = action.astype(np.float32)
        
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, info

class TrainingCallback(BaseCallback):
    """Custom callback to track training progress"""
    
    def __init__(self, verbose=0):
        super(TrainingCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_count = 0
        self.current_episode_reward = 0
        self.current_episode_length = 0
        
    def _on_step(self) -> bool:
        """Called at each step"""
        self.current_episode_length += 1
        
        # Get reward from info if available
        if len(self.locals.get('rewards', [])) > 0:
            self.current_episode_reward += self.locals['rewards'][0]
        
        # Check if episode ended
        if len(self.locals.get('dones', [])) > 0 and self.locals['dones'][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            self.episode_count += 1
            
            # Reset for next episode
            self.current_episode_reward = 0
            self.current_episode_length = 0
        
        return True

def set_seeds(seed):
    """Set seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def calculate_metrics(episode_rewards):
    """Calculate key metrics for Pendulum evaluation"""
    if len(episode_rewards) == 0:
        return {}
    
    # Get last 100 episodes (or all if less than 100)
    recent_rewards = episode_rewards[-100:] if len(episode_rewards) >= 100 else episode_rewards
    
    # Pendulum rewards are negative (cost function), closer to 0 is better
    metrics = {
        'mean_episode_return': np.mean(recent_rewards),
        'success_rate': np.mean(np.array(recent_rewards) >= -200),  # % episodes reaching -200+ (good performance)
        'performance_variance': np.std(recent_rewards),
        'final_performance': np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards),
        'total_episodes': len(episode_rewards),
        'max_reward': np.max(episode_rewards),  # Best (highest/least negative) reward
        'min_reward': np.min(episode_rewards)   # Worst (lowest/most negative) reward
    }
    
    # Sample efficiency: episodes to reach -300+ average reward (intermediate milestone)
    if len(episode_rewards) >= 10:
        for i in range(10, len(episode_rewards)):
            avg_reward = np.mean(episode_rewards[i-10:i])
            if avg_reward >= -300:
                metrics['sample_efficiency_300'] = i
                break
        else:
            metrics['sample_efficiency_300'] = None
    else:
        metrics['sample_efficiency_300'] = None
    
    # Sample efficiency: episodes to reach -200+ average reward (good performance)
    if len(episode_rewards) >= 10:
        for i in range(10, len(episode_rewards)):
            avg_reward = np.mean(episode_rewards[i-10:i])
            if avg_reward >= -200:
                metrics['sample_efficiency_200'] = i
                break
        else:
            metrics['sample_efficiency_200'] = None
    else:
        metrics['sample_efficiency_200'] = None
    
    return metrics

def train_and_evaluate_single_seed(config, config_name, seed):
    """Train A3C (A2C) with given configuration and seed, return metrics"""
    print(f"  Seed {seed}: Training {config_name}...")
    
    try:
        # Set seed for reproducibility
        set_seeds(seed)

        # Create Pendulum environment with specified number of parallel environments
        def make_env():
            env = gym.make('Pendulum-v1')
            env = PendulumWrapper(env)
            return env
            
        env = DummyVecEnv([make_env for _ in range(config['n_envs'])])
        
        # Create A2C model with specified configuration
        model = A2C(
            "MlpPolicy",
            env,
            learning_rate=config['learning_rate'],
            n_steps=config['n_steps'],
            gamma=config['gamma'],
            gae_lambda=config['gae_lambda'],
            ent_coef=config['ent_coef'],
            vf_coef=config['vf_coef'],
            max_grad_norm=config['max_grad_norm'],
            policy_kwargs=dict(
                net_arch=config['net_arch'],
                activation_fn=torch.nn.ReLU,
            ),
            verbose=0,
            device="auto",
            seed=seed,
        )
        
        # Create callback to track training
        callback = TrainingCallback()
        
        # Train the model
        model.learn(
            total_timesteps=config['total_timesteps'],
            callback=callback,
            progress_bar=False
        )
        
        # Calculate training metrics
        episode_rewards = callback.episode_rewards
        metrics = calculate_metrics(episode_rewards)
        
        # Evaluate the trained model
        eval_env = gym.make("Pendulum-v1")
        mean_reward, std_reward = evaluate_policy(
            model, eval_env, n_eval_episodes=20, deterministic=True
        )
        
        metrics['test_mean_reward'] = mean_reward
        metrics['test_std_reward'] = std_reward
        
        # Get individual test episode rewards
        test_rewards = []
        for _ in range(20):
            obs, _ = eval_env.reset()
            episode_reward = 0
            done = False
            truncated = False
            
            while not (done or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, _ = eval_env.step(action)
                episode_reward += reward
            
            test_rewards.append(episode_reward)
        
        metrics['test_rewards'] = test_rewards
        
        eval_env.close()
        env.close()
        
        return metrics
        
    except Exception as e:
        print(f"    ERROR in seed {seed}: {str(e)}")
        # Return default metrics to prevent crash
        return {
            'mean_episode_return': -1500.0,  # Poor performance for Pendulum
            'success_rate': 0.0,
            'performance_variance': 0.0,
            'final_performance': -1500.0,
            'total_episodes': 0,
            'max_reward': -1500.0,
            'min_reward': -1500.0,
            'sample_efficiency_300': None,
            'sample_efficiency_200': None,
            'test_mean_reward': -1500.0,
            'test_std_reward': 0.0,
            'test_rewards': [-1500.0] * 20
        }

def run_config_multiple_seeds(config, config_name, num_seeds=2):
    """Run a configuration multiple times with different seeds"""
    print(f"\n{'='*50}")
    print(f"Testing configuration: {config_name}")
    print(f"Config: {config}")
    print(f"Running {num_seeds} seeds...")
    print(f"{'='*50}")

    all_metrics = []
    seeds = [42 + i for i in range(num_seeds)]  # Use consistent seeds
    
    for seed in seeds:
        try:
            metrics = train_and_evaluate_single_seed(config, config_name, seed)
            metrics['seed'] = seed
            metrics['config'] = config
            metrics['config_name'] = config_name
            all_metrics.append(metrics)
        except Exception as e:
            print(f"    Error with seed {seed}: {e}")
            continue
    
    if not all_metrics:
        print(f"    All seeds failed for {config_name}")
        return None
    
    # Aggregate metrics across seeds
    aggregated = aggregate_seed_results(all_metrics, config, config_name)

    print(f"Results for {config_name} (n={len(all_metrics)}):")
    print(f"  Test Performance: {aggregated['test_mean_reward_mean']:.2f} ± {aggregated['test_mean_reward_std']:.2f}")
    print(f"  Success Rate: {aggregated['success_rate_mean']:.2%} ± {aggregated['success_rate_std']:.2%}")
    
    # Handle None case for sample efficiency
    if aggregated.get('sample_efficiency_200_mean') is not None:
        print(f"  Sample Efficiency (-200+): {aggregated['sample_efficiency_200_mean']:.1f} episodes")
    else:
        print(f"  Sample Efficiency (-200+): Not achieved")
        
    if aggregated.get('sample_efficiency_300_mean') is not None:
        print(f"  Sample Efficiency (-300+): {aggregated['sample_efficiency_300_mean']:.1f} episodes")
    else:
        print(f"  Sample Efficiency (-300+): Not achieved")
    
    return aggregated

def aggregate_seed_results(all_metrics, config, config_name):
    """Aggregate results across multiple seeds"""
    aggregated = {
        'config': config,
        'config_name': config_name,
        'num_seeds': len(all_metrics),
        'seed_results': all_metrics
    }
    
    # Primary metrics for statistical testing
    metrics_to_aggregate = [
        'test_mean_reward', 'final_performance', 'mean_episode_return',
        'success_rate', 'performance_variance', 'max_reward', 'min_reward'
    ]

    for metric in metrics_to_aggregate:
        values = [m[metric] for m in all_metrics if metric in m and m[metric] is not None]
        if values:
            aggregated[f'{metric}_mean'] = np.mean(values)
            aggregated[f'{metric}_std'] = np.std(values) if len(values) > 1 else 0.0
            aggregated[f'{metric}_values'] = values
        else:
            aggregated[f'{metric}_mean'] = -1500.0  # Default poor performance for Pendulum
            aggregated[f'{metric}_std'] = 0.0
            aggregated[f'{metric}_values'] = []
    
    # Sample efficiency for -300+ reward (intermediate milestone)
    sample_eff_300_values = [m['sample_efficiency_300'] for m in all_metrics 
                            if m['sample_efficiency_300'] is not None]
    if sample_eff_300_values:
        aggregated['sample_efficiency_300_mean'] = np.mean(sample_eff_300_values)
        aggregated['sample_efficiency_300_std'] = np.std(sample_eff_300_values) if len(sample_eff_300_values) > 1 else 0.0
        aggregated['sample_efficiency_300_success_rate'] = len(sample_eff_300_values) / len(all_metrics)
    else:
        aggregated['sample_efficiency_300_mean'] = None
        aggregated['sample_efficiency_300_std'] = None
        aggregated['sample_efficiency_300_success_rate'] = 0
    
    # Sample efficiency for -200+ reward (good performance)
    sample_eff_200_values = [m['sample_efficiency_200'] for m in all_metrics 
                            if m['sample_efficiency_200'] is not None]
    if sample_eff_200_values:
        aggregated['sample_efficiency_200_mean'] = np.mean(sample_eff_200_values)
        aggregated['sample_efficiency_200_std'] = np.std(sample_eff_200_values) if len(sample_eff_200_values) > 1 else 0.0
        aggregated['sample_efficiency_200_success_rate'] = len(sample_eff_200_values) / len(all_metrics)
    else:
        aggregated['sample_efficiency_200_mean'] = None
        aggregated['sample_efficiency_200_std'] = None
        aggregated['sample_efficiency_200_success_rate'] = 0
    
    return aggregated

def cohens_d(group1, group2):
    """Calculate Cohen's d effect size"""
    try:
        n1, n2 = len(group1), len(group2)
        if n1 <= 1 or n2 <= 1:
            return 0.0
            
        pooled_std = np.sqrt(((n1 - 1) * np.var(group1, ddof=1) + 
                             (n2 - 1) * np.var(group2, ddof=1)) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0.0
        
        return (np.mean(group1) - np.mean(group2)) / pooled_std
    except Exception as e:
        print(f"    Warning: Cohen's d calculation failed: {e}")
        return 0.0

def statistical_comparison(results, metric='test_mean_reward_values', alpha=0.05):
    """Perform statistical comparisons between all configuration pairs"""
    print(f"\n{'='*60}")
    print(f"STATISTICAL SIGNIFICANCE TESTING")
    print(f"Metric: {metric}")
    print(f"Alpha level: {alpha}")
    print(f"{'='*60}")
    
    # Filter results that have the metric
    valid_results = [r for r in results if r is not None and 
                    metric in r and r[metric] and len(r[metric]) > 1]
    
    if len(valid_results) < 2:
        print("Not enough valid results for statistical testing")
        print(f"Valid results found: {len(valid_results)}")
        return []
    
    comparisons = []
    
    # Pairwise comparisons
    for i, j in combinations(range(len(valid_results)), 2):
        try:
            result1, result2 = valid_results[i], valid_results[j]
            name1, name2 = result1['config_name'], result2['config_name']
            values1, values2 = result1[metric], result2[metric]
            
            # Additional validation
            if len(values1) < 2 or len(values2) < 2:
                print(f"  Skipping {name1} vs {name2}: insufficient data")
                continue
            
            # Choose appropriate test (Mann-Whitney U for robustness)
            try:
                stat, p_value = stats.mannwhitneyu(values1, values2, alternative='two-sided')
                test_name = "Mann-Whitney U"
            except Exception as e:
                print(f"    Warning: Statistical test failed for {name1} vs {name2}: {e}")
                continue
            
            # Effect size
            effect_size = abs(cohens_d(values1, values2))
            
            # Interpretation
            if p_value < alpha:
                significance = "Significant"
            else:
                significance = "Not significant"
            
            # Effect size interpretation
            if effect_size < 0.2:
                effect_desc = "Negligible"
            elif effect_size < 0.5:
                effect_desc = "Small"
            elif effect_size < 0.8:
                effect_desc = "Medium"
            else:
                effect_desc = "Large"
            
            comparison = {
                'config1': name1,
                'config2': name2,
                'mean1': np.mean(values1),
                'mean2': np.mean(values2),
                'test': test_name,
                'statistic': stat,
                'p_value': p_value,
                'significant': p_value < alpha,
                'effect_size': effect_size,
                'effect_desc': effect_desc,
                'significance': significance
            }
            comparisons.append(comparison)
            
        except Exception as e:
            print(f"    Error comparing {valid_results[i]['config_name']} vs {valid_results[j]['config_name']}: {e}")
            continue
    
    if not comparisons:
        print("No valid comparisons could be performed")
        return []
    
    # Bonferroni correction for multiple comparisons
    num_comparisons = len(comparisons)
    bonferroni_alpha = alpha / num_comparisons if num_comparisons > 0 else alpha
    
    print(f"Number of comparisons: {num_comparisons}")
    print(f"Bonferroni corrected alpha: {bonferroni_alpha:.4f}")
    print()
    
    # Sort by p-value
    comparisons.sort(key=lambda x: x['p_value'])
    
    # Print results
    significant_comparisons = []
    for comp in comparisons:
        bonferroni_significant = comp['p_value'] < bonferroni_alpha
        
        print(f"{comp['config1']} vs {comp['config2']}:")
        print(f"  Means: {comp['mean1']:.2f} vs {comp['mean2']:.2f}")
        print(f"  {comp['test']}: p = {comp['p_value']:.4f}")
        print(f"  Effect size (Cohen's d): {comp['effect_size']:.3f} ({comp['effect_desc']})")
        print(f"  Uncorrected: {comp['significance']}")
        print(f"  Bonferroni corrected: {'Significant' if bonferroni_significant else 'Not significant'}")
        print()
        
        if bonferroni_significant:
            significant_comparisons.append(comp)
    
    if significant_comparisons:
        print(f"Significant differences after Bonferroni correction ({len(significant_comparisons)}):")
        for comp in significant_comparisons:
            better_config = comp['config1'] if comp['mean1'] > comp['mean2'] else comp['config2']
            print(f"  {better_config} significantly outperforms the other (p = {comp['p_value']:.4f})")
    else:
        print("No significant differences after Bonferroni correction.")
    
    return comparisons

def run_hyperparameter_tuning_with_stats(num_seeds=2):
    """Run systematic hyperparameter tuning with statistical testing for Pendulum A3C"""
    
    # Base configuration for A3C (A2C)
    base_config = {
        'learning_rate': 7e-4,
        'n_steps': 5,
        'gamma': 0.99,
        'gae_lambda': 1.0,
        'ent_coef': 0.0,
        'vf_coef': 0.25,
        'max_grad_norm': 0.5,
        'net_arch': [64, 64],
        'n_envs': 8,  # Number of parallel environments
        'total_timesteps': 200000  # Reduced for hyperparameter tuning
    }
    
    results = []
    
    # 1. Test different learning rates
    print("PHASE 1: Testing Learning Rates for Pendulum A3C")
    learning_rates = [3e-4, 7e-4, 1e-3]
    for lr in learning_rates:
        config = base_config.copy()
        config['learning_rate'] = lr
        result = run_config_multiple_seeds(config, f"lr_{lr}", num_seeds)
        if result:
            results.append(result)
    
    # Statistical comparison for learning rates
    if len(results) >= 2:
        print("\n" + "="*50)
        print("STATISTICAL ANALYSIS: Learning Rates")
        statistical_comparison(results, 'test_mean_reward_values')
    
    # 2. Test different number of parallel environments
    print("\nPHASE 2: Testing Number of Parallel Environments")
    
    # Find best learning rate
    if results:
        try:
            best_lr_result = max(results, key=lambda x: x['test_mean_reward_mean'])
            best_lr = best_lr_result['config']['learning_rate']
            print(f"Using best learning rate: {best_lr}")
        except Exception as e:
            print(f"Error finding best learning rate: {e}")
            print("Using default learning rate")
            best_lr = base_config['learning_rate']
    else:
        print("No valid learning rate results, using default")
        best_lr = base_config['learning_rate']
    
    n_envs_options = [4, 8, 16]  # Different numbers of parallel environments
    
    env_results = []
    for n_envs in n_envs_options:
        config = base_config.copy()
        config['learning_rate'] = best_lr
        config['n_envs'] = n_envs
        result = run_config_multiple_seeds(config, f"envs_{n_envs}", num_seeds)
        if result:
            env_results.append(result)
            results.append(result)

    # Statistical comparison for parallel environments
    if len(env_results) >= 2:
        print("\n" + "="*50)
        print("STATISTICAL ANALYSIS: Number of Parallel Environments")
        statistical_comparison(env_results, 'test_mean_reward_values')
    
    # 3. Test different n_steps (rollout lengths)
    print("\nPHASE 3: Testing N-Steps (Rollout Length)")
    
    # Find best configuration so far
    if results:
        try:
            best_result = max(results, key=lambda x: x['test_mean_reward_mean'])
            best_config_so_far = best_result['config'].copy()
            print(f"Using best configuration from previous phases")
        except Exception as e:
            print(f"Error finding best configuration: {e}")
            print("Using base configuration with best learning rate")
            best_config_so_far = base_config.copy()
            best_config_so_far['learning_rate'] = best_lr
    else:
        print("No valid results, using base configuration")
        best_config_so_far = base_config.copy()

    n_steps_options = [5, 10, 20]  # Different rollout lengths
    
    steps_results = []
    for n_steps in n_steps_options:
        config = best_config_so_far.copy()
        config['n_steps'] = n_steps
        result = run_config_multiple_seeds(config, f"steps_{n_steps}", num_seeds)
        if result:
            steps_results.append(result)
            results.append(result)

    # Statistical comparison for n_steps
    if len(steps_results) >= 2:
        print("\n" + "="*50)
        print("STATISTICAL ANALYSIS: N-Steps (Rollout Length)")
        statistical_comparison(steps_results, 'test_mean_reward_values')
    
    # 4. Test different entropy coefficients
    print("\nPHASE 4: Testing Entropy Coefficients")
    
    # Find best configuration so far
    if results:
        try:
            best_result = max(results, key=lambda x: x['test_mean_reward_mean'])
            best_config_so_far = best_result['config'].copy()
        except Exception as e:
            best_config_so_far = base_config.copy()
    else:
        best_config_so_far = base_config.copy()

    ent_coefs = [0.0, 0.01, 0.1]  # Different entropy coefficients
    
    ent_results = []
    for ent_coef in ent_coefs:
        config = best_config_so_far.copy()
        config['ent_coef'] = ent_coef
        result = run_config_multiple_seeds(config, f"ent_{ent_coef}", num_seeds)
        if result:
            ent_results.append(result)
            results.append(result)

    # Statistical comparison for entropy coefficients
    if len(ent_results) >= 2:
        print("\n" + "="*50)
        print("STATISTICAL ANALYSIS: Entropy Coefficients")
        statistical_comparison(ent_results, 'test_mean_reward_values')
    
    # 5. Test different network architectures
    print("\nPHASE 5: Testing Network Architectures")
    
    # Find best configuration so far
    if results:
        try:
            best_result = max(results, key=lambda x: x['test_mean_reward_mean'])
            best_config_so_far = best_result['config'].copy()
        except Exception as e:
            best_config_so_far = base_config.copy()
    else:
        best_config_so_far = base_config.copy()

    architectures = [
        ([32, 32], "32_32"),
        ([64, 64], "64_64"),
        ([128, 64], "128_64"),
        ([64, 64, 32], "64_64_32")
    ]
    
    arch_results = []
    for arch, arch_name in architectures:
        config = best_config_so_far.copy()
        config['net_arch'] = arch
        result = run_config_multiple_seeds(config, f"arch_{arch_name}", num_seeds)
        if result:
            arch_results.append(result)
            results.append(result)

    # Statistical comparison for architectures
    if len(arch_results) >= 2:
        print("\n" + "="*50)
        print("STATISTICAL ANALYSIS: Network Architectures")
        statistical_comparison(arch_results, 'test_mean_reward_values')
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        # Prepare results for JSON serialization
        results_serializable = []
        for result in results:
            if result is None:
                continue
            result_copy = {}
            for key, value in result.items():
                try:
                    if isinstance(value, np.ndarray):
                        result_copy[key] = value.tolist()
                    elif isinstance(value, (np.integer, np.floating)):
                        result_copy[key] = value.item()
                    elif key == 'seed_results':
                        # Handle nested seed results
                        seed_results_clean = []
                        for seed_result in value:
                            seed_clean = {}
                            for k, v in seed_result.items():
                                try:
                                    if isinstance(v, np.ndarray):
                                        seed_clean[k] = v.tolist()
                                    elif isinstance(v, (np.integer, np.floating)):
                                        seed_clean[k] = v.item()
                                    else:
                                        seed_clean[k] = v
                                except Exception as e:
                                    print(f"    Warning: Error serializing {k}: {e}")
                                    seed_clean[k] = str(v)
                            seed_results_clean.append(seed_clean)
                        result_copy[key] = seed_results_clean
                    else:
                        result_copy[key] = value
                except Exception as e:
                    print(f"    Warning: Error serializing {key}: {e}")
                    result_copy[key] = str(value)
            results_serializable.append(result_copy)
        
        with open(f'./pendulum/a3c/json_data/hyperparameter_results_stats_{timestamp}.json', 'w') as f:
            json.dump(results_serializable, f, indent=2)
        print(f"\nResults saved successfully to: hyperparameter_results_stats_{timestamp}.json")
        
    except Exception as e:
        print(f"Warning: Could not save results to JSON: {e}")
        print("Continuing with analysis...")
    
    # Final statistical analysis of all configurations
    if len(results) >= 2:
        print("\n" + "="*60)
        print("FINAL STATISTICAL ANALYSIS: All Configurations")
        statistical_comparison(results, 'test_mean_reward_values')
    
    # Print summary
    print(f"\n{'='*60}")
    print("PENDULUM A3C HYPERPARAMETER TUNING SUMMARY")
    print(f"{'='*60}")
    
    # Sort by mean test performance (higher is better, less negative)
    valid_results = [r for r in results if r is not None and r.get('test_mean_reward_mean') is not None]
    
    if valid_results:
        try:
            valid_results.sort(key=lambda x: x['test_mean_reward_mean'], reverse=True)
            
            print(f"\nConfigurations ranked by mean test performance (n={num_seeds} seeds each):")
            print("Note: Pendulum rewards are negative (cost function), higher values are better")
            for i, result in enumerate(valid_results):
                mean_perf = result['test_mean_reward_mean']
                std_perf = result.get('test_mean_reward_std', 0.0)
                success_rate = result.get('success_rate_mean', 0.0)
                performance_level = "🎯 Excellent" if mean_perf >= -200 else "✅ Good" if mean_perf >= -500 else "❌ Poor"
                print(f"{i+1}. {result['config_name']}: {mean_perf:.2f} ± {std_perf:.2f} "
                      f"(Success: {success_rate:.1%}) {performance_level}")
        except Exception as e:
            print(f"Error creating summary: {e}")
            print("Raw results available in the results list")
    else:
        print("No valid results to summarize")
    
    print(f"\nPendulum A3C hyperparameter tuning completed at {datetime.now()}")
    print("Note: Pendulum uses negative rewards (cost function). Values closer to 0 are better.")
    print("Good performance: > -200, Excellent performance: > -150")
    print("A3C uses parallel environments and actor-critic learning for continuous control")
    
    return valid_results

if __name__ == "__main__":
    # Run with statistical testing
    print("Starting Pendulum A3C hyperparameter tuning with statistical significance testing...")
    print("Note: A3C uses parallel environments and actor-critic learning for continuous control.")
    print("Expected training time: 2-3 hours depending on hardware")
    
    try:
        results = run_hyperparameter_tuning_with_stats(num_seeds=2)
        print(f"✅ Successfully completed Pendulum A3C hyperparameter tuning!")
        print(f"Total valid configurations tested: {len(results)}")
    except Exception as e:
        print(f"❌ Fatal error in hyperparameter tuning: {e}")
        print("Check the error logs above for details")