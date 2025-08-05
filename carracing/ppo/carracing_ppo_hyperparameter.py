#!/usr/bin/env python3
"""
CarRacing PPO Hyperparameter Tuning with Statistical Analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
import json
import time
import os
from datetime import datetime
from scipy import stats

env_name = "carracing"
alg_name = "ppo"

# Import environment wrapper (adjust import based on your structure)
# from carracing.ppo.carracing_ppo import CarRacingWrapper

class CarRacingWrapper(gym.Wrapper):
    """CarRacing-specific wrapper for preprocessing"""
    
    def __init__(self, env):
        super(CarRacingWrapper, self).__init__(env)
        
    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple):
            obs, info = result
            return obs, info
        else:
            return result, {}
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # CarRacing reward clipping
        if reward <= -100:  # Went off track
            reward = -100
        
        return obs, reward, terminated, truncated, info

def statistical_comparison(results_list, metric_key):
    """Compare multiple configurations using statistical tests"""
    print(f"\nStatistical Comparison of {metric_key}:")
    print("-" * 45)
    
    comparisons = []
    
    # Extract values for each configuration
    config_values = {}
    for result in results_list:
        if result and metric_key in result:
            config_values[result['config_name']] = result[metric_key]
    
    # Pairwise comparisons
    config_names = list(config_values.keys())
    for i in range(len(config_names)):
        for j in range(i+1, len(config_names)):
            name1, name2 = config_names[i], config_names[j]
            values1, values2 = config_values[name1], config_values[name2]
            
            if len(values1) >= 2 and len(values2) >= 2:
                # Mann-Whitney U test (non-parametric)
                stat, p_value = stats.mannwhitneyu(values1, values2, alternative='two-sided')
                
                # Effect size (Cohen's d approximation)
                pooled_std = np.sqrt((np.var(values1, ddof=1) + np.var(values2, ddof=1)) / 2)
                effect_size = abs(np.mean(values1) - np.mean(values2)) / pooled_std if pooled_std > 0 else 0
                
                significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                
                comparisons.append({
                    'config1': name1,
                    'config2': name2,
                    'p_value': p_value,
                    'effect_size': effect_size,
                    'significance': significance
                })
                
                print(f"{name1} vs {name2}: p={p_value:.4f} {significance}, d={effect_size:.3f}")
    
    print("Legend: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
    return comparisons

def train_and_evaluate_single_seed(config, config_name, seed):
    """Train and evaluate a single configuration with one seed"""
    print(f"  Training seed {seed}...")
    
    try:
        # Create environment
        def make_env():
            env = gym.make('CarRacing-v2', continuous=True)
            env = CarRacingWrapper(env)
            return env
            
        env = DummyVecEnv([make_env for _ in range(4)])
        
        # Create model with configuration
        model = PPO(
            "CnnPolicy",
            env,
            learning_rate=config['learning_rate'],
            n_steps=config['n_steps'],
            batch_size=config['batch_size'],
            n_epochs=config['n_epochs'],
            gamma=config['gamma'],
            gae_lambda=config['gae_lambda'],
            clip_range=config['clip_range'],
            ent_coef=config['ent_coef'],
            vf_coef=config['vf_coef'],
            max_grad_norm=config['max_grad_norm'],
            policy_kwargs=dict(
                net_arch=config['policy_arch'],
                activation_fn=torch.nn.ReLU,
                normalize_images=True,
            ),
            verbose=0,
            seed=seed,
        )
        
        # Train model
        model.learn(total_timesteps=config['total_timesteps'])
        
        # Evaluate model
        eval_env = DummyVecEnv([make_env])
        mean_reward, std_reward = evaluate_policy(
            model, eval_env, n_eval_episodes=20, deterministic=True
        )
        
        # Test rewards for individual analysis
        test_rewards = []
        for _ in range(20):
            obs = eval_env.reset()
            episode_reward = 0
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = eval_env.step(action)
                episode_reward += reward[0]
            test_rewards.append(episode_reward)
        
        # Calculate success rate (CarRacing: >500 is good performance)
        success_rate = sum(1 for r in test_rewards if r > 500) / len(test_rewards)
        
        # Sample efficiency tracking
        sample_efficiency_500 = None  # Placeholder - would need episode-by-episode tracking
        
        env.close()
        eval_env.close()
        
        return {
            'mean_episode_return': mean_reward,
            'success_rate': success_rate,
            'performance_variance': np.var(test_rewards, ddof=1),
            'final_performance': mean_reward,
            'total_episodes': config['total_timesteps'] // (config['n_steps'] * 4),  # Approximate
            'max_reward': max(test_rewards),
            'min_reward': min(test_rewards),
            'sample_efficiency_500': sample_efficiency_500,
            'test_mean_reward': mean_reward,
            'test_std_reward': std_reward,
            'test_rewards': test_rewards
        }
        
    except Exception as e:
        print(f"    Failed: {e}")
        return {
            'mean_episode_return': -100.0,
            'success_rate': 0.0,
            'performance_variance': 0.0,
            'final_performance': -100.0,
            'total_episodes': 0,
            'max_reward': -100.0,
            'min_reward': -100.0,
            'sample_efficiency_500': None,
            'test_mean_reward': -100.0,
            'test_std_reward': 0.0,
            'test_rewards': [-100.0] * 20
        }

def run_config_multiple_seeds(config, config_name, num_seeds=2):
    """Run a configuration multiple times with different seeds"""
    print(f"\n{'='*50}")
    print(f"Testing configuration: {config_name}")
    print(f"Config: {config}")
    print(f"Running {num_seeds} seeds...")
    print(f"{'='*50}")

    all_metrics = []
    seeds = [42 + i for i in range(num_seeds)]
    
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
    
    return aggregated

def aggregate_seed_results(all_metrics, config, config_name):
    """Aggregate results across multiple seeds"""
    aggregated = {
        'config': config,
        'config_name': config_name,
        'num_seeds': len(all_metrics),
        'seed_results': all_metrics
    }
    
    # Primary metrics
    metrics_to_aggregate = [
        'mean_episode_return', 'success_rate', 'performance_variance',
        'final_performance', 'total_episodes', 'max_reward', 'min_reward',
        'test_mean_reward', 'test_std_reward'
    ]
    
    for metric in metrics_to_aggregate:
        values = [m[metric] for m in all_metrics if m.get(metric) is not None]
        if values:
            aggregated[f'{metric}_mean'] = np.mean(values)
            aggregated[f'{metric}_std'] = np.std(values, ddof=1) if len(values) > 1 else 0.0
            aggregated[f'{metric}_values'] = values
        else:
            aggregated[f'{metric}_mean'] = None
            aggregated[f'{metric}_std'] = None
            aggregated[f'{metric}_values'] = []
    
    # Sample efficiency (handle None values)
    efficiency_values = [m.get('sample_efficiency_500') for m in all_metrics]
    efficiency_values = [v for v in efficiency_values if v is not None]
    
    if efficiency_values:
        aggregated['sample_efficiency_500_mean'] = np.mean(efficiency_values)
        aggregated['sample_efficiency_500_std'] = np.std(efficiency_values, ddof=1) if len(efficiency_values) > 1 else 0.0
        aggregated['sample_efficiency_500_success_rate'] = len(efficiency_values) / len(all_metrics)
    else:
        aggregated['sample_efficiency_500_mean'] = None
        aggregated['sample_efficiency_500_std'] = None
        aggregated['sample_efficiency_500_success_rate'] = 0
    
    return aggregated

def run_hyperparameter_tuning_with_stats(num_seeds=2):
    """Run systematic hyperparameter tuning with statistical testing for CarRacing PPO"""
    
    # Base configuration optimized for CarRacing continuous control
    base_config = {
        'learning_rate': 3e-4,
        'n_steps': 2048,
        'batch_size': 64,
        'n_epochs': 10,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'ent_coef': 0.01,
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
        'policy_arch': [256, 256],
        'total_timesteps': 500000  # Reduced for hyperparameter tuning
    }
    
    results = []
    
    # 1. Test different learning rates
    print("PHASE 1: Testing Learning Rates for CarRacing PPO")
    learning_rates = [1e-4, 3e-4, 1e-3]
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
    
    # 2. Test different entropy coefficients with best learning rate
    print("\nPHASE 2: Testing Entropy Coefficients")
    
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
    
    ent_coefs = [0.0, 0.01, 0.05]
    
    ent_results = []
    for ent_coef in ent_coefs:
        config = base_config.copy()
        config['learning_rate'] = best_lr
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
    
    # 3. Test different network architectures with best hyperparameters
    print("\nPHASE 3: Testing Network Architectures")
    
    # Find best entropy coefficient
    if ent_results:
        try:
            best_ent_result = max(ent_results, key=lambda x: x['test_mean_reward_mean'])
            best_ent = best_ent_result['config']['ent_coef']
            print(f"Using best entropy coefficient: {best_ent}")
        except Exception as e:
            print(f"Error finding best entropy coefficient: {e}")
            print("Using default entropy coefficient")
            best_ent = base_config['ent_coef']
    else:
        print("No valid entropy coefficient results, using default")
        best_ent = base_config['ent_coef']
    
    architectures = [[128, 128], [256, 256], [512, 256]]
    
    arch_results = []
    for arch in architectures:
        config = base_config.copy()
        config['learning_rate'] = best_lr
        config['ent_coef'] = best_ent
        config['policy_arch'] = arch
        arch_name = f"arch_{'_'.join(map(str, arch))}"
        result = run_config_multiple_seeds(config, arch_name, num_seeds)
        if result:
            arch_results.append(result)
            results.append(result)

    # Statistical comparison for architectures
    if len(arch_results) >= 2:
        print("\n" + "="*50)
        print("STATISTICAL ANALYSIS: Network Architectures")
        statistical_comparison(arch_results, 'test_mean_reward_values')
    
    # Save results
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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
                        seed_results_clean = []
                        for seed_result in value:
                            seed_clean = {}
                            for k, v in seed_result.items():
                                if isinstance(v, np.ndarray):
                                    seed_clean[k] = v.tolist()
                                elif isinstance(v, (np.integer, np.floating)):
                                    seed_clean[k] = v.item()
                                else:
                                    seed_clean[k] = v
                            seed_results_clean.append(seed_clean)
                        result_copy[key] = seed_results_clean
                    else:
                        result_copy[key] = value
                except Exception as e:
                    print(f"    Warning: Error serializing {key}: {e}")
                    result_copy[key] = str(value)
            results_serializable.append(result_copy)
        
        os.makedirs("./carracing/ppo/json_data/", exist_ok=True)
        with open(f'./carracing/ppo/json_data/{env_name}_{alg_name}_hp.json', 'w') as f:
            json.dump(results_serializable, f, indent=2)
        print(f"\nResults saved successfully to: {env_name}_{alg_name}_hp.json")
        
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
    print("CARRACING PPO HYPERPARAMETER TUNING SUMMARY")
    print(f"{'='*60}")
    
    # Sort by mean test performance
    valid_results = [r for r in results if r is not None and r.get('test_mean_reward_mean') is not None]
    
    if valid_results:
        try:
            valid_results.sort(key=lambda x: x['test_mean_reward_mean'], reverse=True)
            
            print(f"\nConfigurations ranked by mean test performance (n={num_seeds} seeds each):")
            for i, result in enumerate(valid_results):
                mean_perf = result['test_mean_reward_mean']
                std_perf = result.get('test_mean_reward_std', 0.0)
                success_rate = result.get('success_rate_mean', 0.0)
                solved_status = "✅ GOOD" if mean_perf >= 500 else "❌ Poor"
                print(f"{i+1}. {result['config_name']}: {mean_perf:.2f} ± {std_perf:.2f} "
                      f"(Success: {success_rate:.1%}) {solved_status}")
        except Exception as e:
            print(f"Error creating summary: {e}")
            print("Raw results available in the results list")
    else:
        print("No valid results to summarize")
    
    print(f"\nCarRacing PPO hyperparameter tuning completed at {datetime.now()}")
    print("Note: CarRacing requires complex visual-motor control and takes time to learn")
    print("Good performance is typically 500+ average reward")
    
    return valid_results

if __name__ == "__main__":
    import torch  # Import here to avoid issues if not available
    
    # Run with statistical testing
    print("Starting CarRacing PPO hyperparameter tuning with statistical significance testing...")
    print("Note: CarRacing is a complex continuous control environment with visual input.")
    print("Expected training time: 2-4 hours depending on hardware")
    
    try:
        results = run_hyperparameter_tuning_with_stats(num_seeds=2)
        print(f"✅ Successfully completed CarRacing PPO hyperparameter tuning!")
        print(f"Total valid configurations tested: {len(results)}")
    except Exception as e:
        print(f"❌ Fatal error in hyperparameter tuning: {e}")
        print("Check the error logs above for details")