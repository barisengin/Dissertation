import gym
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
import json
import os
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

class SACTrainingCallback(BaseCallback):
    """Custom callback to track SAC training progress and metrics"""
    
    def __init__(self, verbose=0):
        super(SACTrainingCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_count = 0
        self.current_episode_reward = 0
        self.current_episode_length = 0
        
        # SAC-specific metrics
        self.actor_losses = []
        self.critic_losses = []
        self.entropy_losses = []
        self.entropy_coefficients = []
        
    def _on_step(self) -> bool:
        """Called at each step"""
        self.current_episode_length += 1
        
        # Get reward from info if available
        if len(self.locals.get('rewards', [])) > 0:
            self.current_episode_reward += self.locals['rewards'][0]
        
        # Try to capture SAC-specific metrics
        try:
            if hasattr(self.model, 'logger') and self.model.logger is not None:
                if hasattr(self.model.logger, 'name_to_value'):
                    actor_loss = self.model.logger.name_to_value.get('train/actor_loss', None)
                    critic_loss = self.model.logger.name_to_value.get('train/critic_loss', None)
                    entropy_loss = self.model.logger.name_to_value.get('train/ent_coef_loss', None)
                    entropy_coef = self.model.logger.name_to_value.get('train/ent_coef', None)
                    
                    if actor_loss is not None:
                        self.actor_losses.append(actor_loss)
                    if critic_loss is not None:
                        self.critic_losses.append(critic_loss)
                    if entropy_loss is not None:
                        self.entropy_losses.append(entropy_loss)
                    if entropy_coef is not None:
                        self.entropy_coefficients.append(entropy_coef)
        except Exception:
            pass  # Silently handle any logging errors
        
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

def calculate_sac_metrics(episode_rewards, callback=None):
    """Calculate key metrics for SAC Pendulum evaluation"""
    if len(episode_rewards) == 0:
        return {}
    
    # Get last 100 episodes (or all if less than 100)
    recent_rewards = episode_rewards[-100:] if len(episode_rewards) >= 100 else episode_rewards
    
    # Pendulum rewards are negative (cost function), closer to 0 is better
    metrics = {
        'mean_episode_return': np.mean(recent_rewards),
        'success_rate': np.mean(np.array(recent_rewards) >= -200),  # % episodes reaching -200+ (good performance)
        'excellent_rate': np.mean(np.array(recent_rewards) >= -150),  # % episodes reaching -150+ (excellent performance)
        'performance_variance': np.std(recent_rewards),
        'final_performance': np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards),
        'total_episodes': len(episode_rewards),
        'max_reward': np.max(episode_rewards),  # Best (highest/least negative) reward
        'min_reward': np.min(episode_rewards),   # Worst (lowest/most negative) reward
        'median_reward': np.median(episode_rewards),
        'q75_reward': np.percentile(episode_rewards, 75),  # 75th percentile
        'q25_reward': np.percentile(episode_rewards, 25),  # 25th percentile
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
    
    # Sample efficiency: episodes to reach -150+ average reward (excellent performance)
    if len(episode_rewards) >= 10:
        for i in range(10, len(episode_rewards)):
            avg_reward = np.mean(episode_rewards[i-10:i])
            if avg_reward >= -150:
                metrics['sample_efficiency_150'] = i
                break
        else:
            metrics['sample_efficiency_150'] = None
    else:
        metrics['sample_efficiency_150'] = None
    
    # SAC-specific metrics if callback provided
    if callback:
        if callback.actor_losses:
            metrics['mean_actor_loss'] = np.mean(callback.actor_losses)
            metrics['final_actor_loss'] = callback.actor_losses[-1] if callback.actor_losses else None
        
        if callback.critic_losses:
            metrics['mean_critic_loss'] = np.mean(callback.critic_losses)
            metrics['final_critic_loss'] = callback.critic_losses[-1] if callback.critic_losses else None
        
        if callback.entropy_coefficients:
            metrics['mean_entropy_coef'] = np.mean(callback.entropy_coefficients)
            metrics['final_entropy_coef'] = callback.entropy_coefficients[-1] if callback.entropy_coefficients else None
            metrics['entropy_coef_std'] = np.std(callback.entropy_coefficients)
    
    return metrics

def train_and_evaluate_sac_single_seed(config, config_name, seed):
    """Train SAC with given configuration and seed, return metrics"""
    print(f"  Seed {seed}: Training {config_name}...")
    
    try:
        # Set seed for reproducibility
        set_seeds(seed)

        # Create Pendulum environment (SAC works well with single environment due to replay buffer)
        def make_env():
            env = gym.make('Pendulum-v1')
            env = PendulumWrapper(env)
            return env
            
        env = DummyVecEnv([make_env])
        
        # Create SAC model with specified configuration
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=config['learning_rate'],
            buffer_size=config['buffer_size'],
            learning_starts=config['learning_starts'],
            batch_size=config['batch_size'],
            tau=config['tau'],
            gamma=config['gamma'],
            train_freq=config['train_freq'],
            gradient_steps=config['gradient_steps'],
            ent_coef=config['ent_coef'],
            target_update_interval=config['target_update_interval'],
            target_entropy=config['target_entropy'],
            use_sde=config.get('use_sde', False),
            sde_sample_freq=config.get('sde_sample_freq', -1),
            use_sde_at_warmup=config.get('use_sde_at_warmup', False),
            policy_kwargs=dict(
                net_arch=config['net_arch'],
                activation_fn=torch.nn.ReLU,
                n_critics=config.get('n_critics', 2),
                share_features_extractor=False,
            ),
            verbose=0,
            device="auto",
            seed=seed,
        )
        
        # Create callback to track training
        callback = SACTrainingCallback()
        
        # Train the model
        model.learn(
            total_timesteps=config['total_timesteps'],
            callback=callback,
            progress_bar=False
        )
        
        # Calculate training metrics
        episode_rewards = callback.episode_rewards
        metrics = calculate_sac_metrics(episode_rewards, callback)
        
        # Evaluate the trained model
        eval_env = gym.make("Pendulum-v1")
        mean_reward, std_reward = evaluate_policy(
            model, eval_env, n_eval_episodes=20, deterministic=True
        )
        
        metrics['test_mean_reward'] = mean_reward
        metrics['test_std_reward'] = std_reward
        
        # Get individual test episode rewards for statistical analysis
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
            'excellent_rate': 0.0,
            'performance_variance': 0.0,
            'final_performance': -1500.0,
            'total_episodes': 0,
            'max_reward': -1500.0,
            'min_reward': -1500.0,
            'median_reward': -1500.0,
            'q75_reward': -1500.0,
            'q25_reward': -1500.0,
            'sample_efficiency_300': None,
            'sample_efficiency_200': None,
            'sample_efficiency_150': None,
            'test_mean_reward': -1500.0,
            'test_std_reward': 0.0,
            'test_rewards': [-1500.0] * 20
        }

def run_sac_config_multiple_seeds(config, config_name, num_seeds=2):
    """Run a SAC configuration multiple times with different seeds"""
    print(f"\n{'='*50}")
    print(f"Testing SAC configuration: {config_name}")
    print(f"Config: {config}")
    print(f"Running {num_seeds} seeds...")
    print(f"{'='*50}")

    all_metrics = []
    seeds = [42 + i for i in range(num_seeds)]  # Use consistent seeds
    
    for seed in seeds:
        try:
            metrics = train_and_evaluate_sac_single_seed(config, config_name, seed)
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
    aggregated = aggregate_sac_seed_results(all_metrics, config, config_name)

    print(f"Results for {config_name} (n={len(all_metrics)}):")
    print(f"  Test Performance: {aggregated['test_mean_reward_mean']:.2f} ± {aggregated['test_mean_reward_std']:.2f}")
    print(f"  Success Rate: {aggregated['success_rate_mean']:.2%} ± {aggregated['success_rate_std']:.2%}")
    print(f"  Excellent Rate: {aggregated['excellent_rate_mean']:.2%} ± {aggregated['excellent_rate_std']:.2%}")
    
    # Handle None case for sample efficiency
    if aggregated.get('sample_efficiency_200_mean') is not None:
        print(f"  Sample Efficiency (-200+): {aggregated['sample_efficiency_200_mean']:.1f} episodes")
    else:
        print(f"  Sample Efficiency (-200+): Not achieved")
        
    if aggregated.get('sample_efficiency_150_mean') is not None:
        print(f"  Sample Efficiency (-150+): {aggregated['sample_efficiency_150_mean']:.1f} episodes")
    else:
        print(f"  Sample Efficiency (-150+): Not achieved")
    
    return aggregated

def aggregate_sac_seed_results(all_metrics, config, config_name):
    """Aggregate SAC results across multiple seeds"""
    aggregated = {
        'config': config,
        'config_name': config_name,
        'num_seeds': len(all_metrics),
        'seed_results': all_metrics
    }
    
    # Primary metrics for statistical testing
    metrics_to_aggregate = [
        'test_mean_reward', 'final_performance', 'mean_episode_return',
        'success_rate', 'excellent_rate', 'performance_variance', 
        'max_reward', 'min_reward', 'median_reward', 'q75_reward', 'q25_reward',
        'mean_actor_loss', 'mean_critic_loss', 'mean_entropy_coef'
    ]

    for metric in metrics_to_aggregate:
        values = [m[metric] for m in all_metrics if metric in m and m[metric] is not None]
        if values:
            aggregated[f'{metric}_mean'] = np.mean(values)
            aggregated[f'{metric}_std'] = np.std(values) if len(values) > 1 else 0.0
            aggregated[f'{metric}_values'] = values
        else:
            # Set appropriate defaults
            if 'rate' in metric:
                aggregated[f'{metric}_mean'] = 0.0
            elif 'loss' in metric:
                aggregated[f'{metric}_mean'] = None
            else:
                aggregated[f'{metric}_mean'] = -1500.0  # Default poor performance for Pendulum
            aggregated[f'{metric}_std'] = 0.0
            aggregated[f'{metric}_values'] = []
    
    # Sample efficiency metrics
    for threshold in [300, 200, 150]:
        sample_eff_values = [m[f'sample_efficiency_{threshold}'] for m in all_metrics 
                            if m[f'sample_efficiency_{threshold}'] is not None]
        if sample_eff_values:
            aggregated[f'sample_efficiency_{threshold}_mean'] = np.mean(sample_eff_values)
            aggregated[f'sample_efficiency_{threshold}_std'] = np.std(sample_eff_values) if len(sample_eff_values) > 1 else 0.0
            aggregated[f'sample_efficiency_{threshold}_success_rate'] = len(sample_eff_values) / len(all_metrics)
        else:
            aggregated[f'sample_efficiency_{threshold}_mean'] = None
            aggregated[f'sample_efficiency_{threshold}_std'] = None
            aggregated[f'sample_efficiency_{threshold}_success_rate'] = 0
    
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

def run_sac_hyperparameter_tuning_with_stats(num_seeds=2):
    """Run systematic SAC hyperparameter tuning with statistical testing for Pendulum"""
    
    # Create directories
    os.makedirs("./pendulum/sac/json_data/", exist_ok=True)
    
    # Base configuration for SAC optimized for Pendulum
    base_config = {
        'learning_rate': 3e-4,
        'buffer_size': 50000,
        'learning_starts': 1000,
        'batch_size': 256,
        'tau': 0.005,
        'gamma': 0.99,
        'train_freq': 1,
        'gradient_steps': 1,
        'ent_coef': 'auto',
        'target_update_interval': 1,
        'target_entropy': 'auto',
        'use_sde': False,
        'sde_sample_freq': -1,
        'use_sde_at_warmup': False,
        'net_arch': [64, 64],
        'n_critics': 2,
        'total_timesteps': 100000  # Reduced for hyperparameter tuning
    }
    
    results = []
    
    # Phase 1: Test different learning rates
    print("PHASE 1: Testing Learning Rates for Pendulum SAC")
    learning_rates = [1e-4, 3e-4, 1e-3]
    for lr in learning_rates:
        config = base_config.copy()
        config['learning_rate'] = lr
        result = run_sac_config_multiple_seeds(config, f"lr_{lr}", num_seeds)
        if result:
            results.append(result)
    
    # Statistical comparison for learning rates
    if len(results) >= 2:
        print("\n" + "="*50)
        print("STATISTICAL ANALYSIS: Learning Rates")
        statistical_comparison(results, 'test_mean_reward_values')
    
    # Phase 2: Test different buffer sizes
    print("\nPHASE 2: Testing Buffer Sizes")
    
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
    
    buffer_sizes = [25000, 50000, 100000]
    
    buffer_results = []
    for buffer_size in buffer_sizes:
        config = base_config.copy()
        config['learning_rate'] = best_lr
        config['buffer_size'] = buffer_size
        result = run_sac_config_multiple_seeds(config, f"buffer_{buffer_size}", num_seeds)
        if result:
            buffer_results.append(result)
            results.append(result)

    # Statistical comparison for buffer sizes
    if len(buffer_results) >= 2:
        print("\n" + "="*50)
        print("STATISTICAL ANALYSIS: Buffer Sizes")
        statistical_comparison(buffer_results, 'test_mean_reward_values')
    
    # Phase 3: Test different batch sizes
    print("\nPHASE 3: Testing Batch Sizes")
    
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

    batch_sizes = [128, 256, 512]
    
    batch_results = []
    for batch_size in batch_sizes:
        config = best_config_so_far.copy()
        config['batch_size'] = batch_size
        result = run_sac_config_multiple_seeds(config, f"batch_{batch_size}", num_seeds)
        if result:
            batch_results.append(result)
            results.append(result)

    # Statistical comparison for batch sizes
    if len(batch_results) >= 2:
        print("\n" + "="*50)
        print("STATISTICAL ANALYSIS: Batch Sizes")
        statistical_comparison(batch_results, 'test_mean_reward_values')
    
    # Phase 4: Test different tau values (soft update coefficient)
    print("\nPHASE 4: Testing Tau Values (Soft Update Coefficient)")
    
    # Find best configuration so far
    if results:
        try:
            best_result = max(results, key=lambda x: x['test_mean_reward_mean'])
            best_config_so_far = best_result['config'].copy()
        except Exception as e:
            best_config_so_far = base_config.copy()
    else:
        best_config_so_far = base_config.copy()

    tau_values = [0.001, 0.005, 0.01]
    
    tau_results = []
    for tau in tau_values:
        config = best_config_so_far.copy()
        config['tau'] = tau
        result = run_sac_config_multiple_seeds(config, f"tau_{tau}", num_seeds)
        if result:
            tau_results.append(result)
            results.append(result)

    # Statistical comparison for tau values
    if len(tau_results) >= 2:
        print("\n" + "="*50)
        print("STATISTICAL ANALYSIS: Tau Values")
        statistical_comparison(tau_results, 'test_mean_reward_values')
    
    # Phase 5: Test different learning starts values
    print("\nPHASE 5: Testing Learning Starts Values")
    
    # Find best configuration so far
    if results:
        try:
            best_result = max(results, key=lambda x: x['test_mean_reward_mean'])
            best_config_so_far = best_result['config'].copy()
        except Exception as e:
            best_config_so_far = base_config.copy()
    else:
        best_config_so_far = base_config.copy()

    learning_starts_values = [500, 1000, 2000]
    
    starts_results = []
    for learning_starts in learning_starts_values:
        config = best_config_so_far.copy()
        config['learning_starts'] = learning_starts
        result = run_sac_config_multiple_seeds(config, f"starts_{learning_starts}", num_seeds)
        if result:
            starts_results.append(result)
            results.append(result)

    # Statistical comparison for learning starts
    if len(starts_results) >= 2:
        print("\n" + "="*50)
        print("STATISTICAL ANALYSIS: Learning Starts Values")
        statistical_comparison(starts_results, 'test_mean_reward_values')
    
    # Phase 6: Test different network architectures
    print("\nPHASE 6: Testing Network Architectures")
    
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
        ([128, 128], "128_128"),
        ([256, 256], "256_256")
    ]
    
    arch_results = []
    for arch, arch_name in architectures:
        config = best_config_so_far.copy()
        config['net_arch'] = arch
        result = run_sac_config_multiple_seeds(config, f"arch_{arch_name}", num_seeds)
        if result:
            arch_results.append(result)
            results.append(result)

    # Statistical comparison for architectures
    if len(arch_results) >= 2:
        print("\n" + "="*50)
        print("STATISTICAL ANALYSIS: Network Architectures")
        statistical_comparison(arch_results, 'test_mean_reward_values')
    
    # Phase 7: Test manual vs auto entropy coefficient
    print("\nPHASE 7: Testing Entropy Coefficient Settings")
    
    # Find best configuration so far
    if results:
        try:
            best_result = max(results, key=lambda x: x['test_mean_reward_mean'])
            best_config_so_far = best_result['config'].copy()
        except Exception as e:
            best_config_so_far = base_config.copy()
    else:
        best_config_so_far = base_config.copy()

    entropy_configs = [
        ('auto', 'auto_ent'),
        (0.1, 'manual_0.1'),
        (0.2, 'manual_0.2')
    ]
    
    ent_results = []
    for ent_coef, ent_name in entropy_configs:
        config = best_config_so_far.copy()
        config['ent_coef'] = ent_coef
        result = run_sac_config_multiple_seeds(config, f"ent_{ent_name}", num_seeds)
        if result:
            ent_results.append(result)
            results.append(result)

    # Statistical comparison for entropy coefficients
    if len(ent_results) >= 2:
        print("\n" + "="*50)
        print("STATISTICAL ANALYSIS: Entropy Coefficient Settings")
        statistical_comparison(ent_results, 'test_mean_reward_values')
    
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
        
        filename = f'./pendulum/sac/json_data/sac_hyperparameter_results_stats_{timestamp}.json'
        with open(filename, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        print(f"\nResults saved successfully to: {filename}")
        
    except Exception as e:
        print(f"Warning: Could not save results to JSON: {e}")
        print("Continuing with analysis...")
    
    # Final statistical analysis of all configurations
    if len(results) >= 2:
        print("\n" + "="*60)
        print("FINAL STATISTICAL ANALYSIS: All SAC Configurations")
        statistical_comparison(results, 'test_mean_reward_values')
    
    # Print summary
    print(f"\n{'='*60}")
    print("PENDULUM SAC HYPERPARAMETER TUNING SUMMARY")
    print(f"{'='*60}")
    
    # Sort by mean test performance (higher is better, less negative)
    valid_results = [r for r in results if r is not None and r.get('test_mean_reward_mean') is not None]
    
    if valid_results:
        try:
            valid_results.sort(key=lambda x: x['test_mean_reward_mean'], reverse=True)
            
            print(f"\nSAC configurations ranked by mean test performance (n={num_seeds} seeds each):")
            print("Note: Pendulum rewards are negative (cost function), higher values are better")
            for i, result in enumerate(valid_results):
                mean_perf = result['test_mean_reward_mean']
                std_perf = result.get('test_mean_reward_std', 0.0)
                success_rate = result.get('success_rate_mean', 0.0)
                excellent_rate = result.get('excellent_rate_mean', 0.0)
                
                if mean_perf >= -150:
                    performance_level = "🏆 Excellent"
                elif mean_perf >= -200:
                    performance_level = "🎯 Good"
                elif mean_perf >= -500:
                    performance_level = "⚠️ Moderate"
                else:
                    performance_level = "❌ Poor"
                
                print(f"{i+1}. {result['config_name']}: {mean_perf:.2f} ± {std_perf:.2f}")
                print(f"   Success Rate (≥-200): {success_rate:.1%}, Excellent Rate (≥-150): {excellent_rate:.1%} {performance_level}")
        except Exception as e:
            print(f"Error creating summary: {e}")
            print("Raw results available in the results list")
    else:
        print("No valid results to summarize")
    
    # SAC-specific insights
    print(f"\nSAC-Specific Analysis:")
    print("✓ Off-policy learning with experience replay enables sample efficiency")
    print("✓ Maximum entropy objective balances exploration and exploitation")
    print("✓ Auto-tuned entropy coefficient adapts exploration dynamically")
    print("✓ Double critic networks reduce Q-value overestimation bias")
    print("✓ Continuous action space handling ideal for Pendulum control")
    
    print(f"\nPendulum SAC hyperparameter tuning completed at {datetime.now()}")
    print("Note: Pendulum uses negative rewards (cost function). Values closer to 0 are better.")
    print("Good performance: ≥ -200, Excellent performance: ≥ -150")
    print("SAC's maximum entropy framework excels at continuous control tasks")
    
    return valid_results

if __name__ == "__main__":
    # Run SAC hyperparameter tuning with statistical testing
    print("Starting Pendulum SAC hyperparameter tuning with statistical significance testing...")
    print("Note: SAC uses off-policy learning with maximum entropy for continuous control.")
    print("Expected training time: 3-4 hours depending on hardware")
    
    try:
        results = run_sac_hyperparameter_tuning_with_stats(num_seeds=2)
        print(f"✅ Successfully completed Pendulum SAC hyperparameter tuning!")
        print(f"Total valid configurations tested: {len(results)}")
    except Exception as e:
        print(f"❌ Fatal error in SAC hyperparameter tuning: {e}")
        print("Check the error logs above for details")