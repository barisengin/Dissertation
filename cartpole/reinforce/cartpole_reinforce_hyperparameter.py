import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import json
from datetime import datetime
from scipy import stats
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

class PolicyNetwork(nn.Module):
    """Policy network for REINFORCE"""
    
    def __init__(self, state_size, action_size, hidden_sizes=[128, 128]):
        super(PolicyNetwork, self).__init__()
        self.layers = nn.ModuleList()
        
        # Input layer
        prev_size = state_size
        for hidden_size in hidden_sizes:
            self.layers.append(nn.Linear(prev_size, hidden_size))
            prev_size = hidden_size
        
        # Output layer
        self.output_layer = nn.Linear(prev_size, action_size)
        
    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        x = self.output_layer(x)
        return F.softmax(x, dim=1)

class BaselineNetwork(nn.Module):
    """Value network for baseline subtraction"""
    
    def __init__(self, state_size, hidden_sizes=[128, 128]):
        super(BaselineNetwork, self).__init__()
        self.layers = nn.ModuleList()
        
        # Input layer
        prev_size = state_size
        for hidden_size in hidden_sizes:
            self.layers.append(nn.Linear(prev_size, hidden_size))
            prev_size = hidden_size
        
        # Output layer
        self.output_layer = nn.Linear(prev_size, 1)
        
    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        x = self.output_layer(x)
        return x

class REINFORCEAgent:
    """REINFORCE algorithm implementation"""
    
    def __init__(self, state_size, action_size, lr=1e-3, gamma=0.99, use_baseline=True, hidden_sizes=[128, 128]):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.use_baseline = use_baseline
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize networks
        self.policy_net = PolicyNetwork(state_size, action_size, hidden_sizes).to(self.device)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        if use_baseline:
            self.baseline_net = BaselineNetwork(state_size, hidden_sizes).to(self.device)
            self.baseline_optimizer = optim.Adam(self.baseline_net.parameters(), lr=lr)
        
        # Storage for episode data
        self.reset_episode_data()
        
    def reset_episode_data(self):
        """Reset storage for new episode"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        
    def select_action(self, state):
        """Select action using current policy"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        probs = self.policy_net(state)
        m = Categorical(probs)
        action = m.sample()
        
        # Store for learning
        self.states.append(state)
        self.log_probs.append(m.log_prob(action))
        
        return action.item()
    
    def store_reward(self, reward):
        """Store reward for current step"""
        self.rewards.append(reward)
    
    def compute_returns(self):
        """Compute discounted returns for the episode"""
        returns = []
        R = 0
        for reward in reversed(self.rewards):
            R = reward + self.gamma * R
            returns.insert(0, R)
        
        # Convert to tensor and normalize
        returns = torch.FloatTensor(returns).to(self.device)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        return returns
    
    def update_policy(self):
        """Update policy using REINFORCE algorithm"""
        if len(self.rewards) == 0:
            return None, None
        
        # Compute returns
        returns = self.compute_returns()
        
        # Compute baseline values if using baseline
        baseline_values = None
        baseline_loss = None
        
        if self.use_baseline:
            states_tensor = torch.cat(self.states)
            baseline_values = self.baseline_net(states_tensor).squeeze()
            
            # Update baseline network
            baseline_loss = F.mse_loss(baseline_values, returns.detach())
            self.baseline_optimizer.zero_grad()
            baseline_loss.backward()
            self.baseline_optimizer.step()
            
            # Subtract baseline from returns (advantage)
            advantages = returns - baseline_values.detach()
        else:
            advantages = returns
        
        # Compute policy loss
        policy_loss = []
        for log_prob, advantage in zip(self.log_probs, advantages):
            policy_loss.append(-log_prob * advantage)
        
        policy_loss = torch.stack(policy_loss).sum()
        
        # Update policy network
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # Reset episode data
        self.reset_episode_data()
        
        return policy_loss.item(), baseline_loss.item() if baseline_loss is not None else None

def set_seeds(seed):
    """Set seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def calculate_metrics(episode_rewards):
    """Calculate key metrics for CartPole evaluation"""
    if len(episode_rewards) == 0:
        return {}
    
    # Get last 100 episodes (or all if less than 100)
    recent_rewards = episode_rewards[-100:] if len(episode_rewards) >= 100 else episode_rewards
    
    metrics = {
        'mean_episode_return': np.mean(recent_rewards),
        'success_rate': np.mean(np.array(recent_rewards) >= 195),  # % episodes reaching 195+ (CartPole solved threshold)
        'performance_variance': np.std(recent_rewards),
        'final_performance': np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards),
        'total_episodes': len(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'min_reward': np.min(episode_rewards)
    }
    
    # Sample efficiency: episodes to reach 150+ average reward (intermediate milestone)
    if len(episode_rewards) >= 10:
        for i in range(10, len(episode_rewards)):
            avg_reward = np.mean(episode_rewards[i-10:i])
            if avg_reward >= 150:
                metrics['sample_efficiency_150'] = i
                break
        else:
            metrics['sample_efficiency_150'] = None
    else:
        metrics['sample_efficiency_150'] = None
    
    # Sample efficiency: episodes to reach 195+ average reward (solved threshold)
    if len(episode_rewards) >= 10:
        for i in range(10, len(episode_rewards)):
            avg_reward = np.mean(episode_rewards[i-10:i])
            if avg_reward >= 195:
                metrics['sample_efficiency_195'] = i
                break
        else:
            metrics['sample_efficiency_195'] = None
    else:
        metrics['sample_efficiency_195'] = None
    
    return metrics

def train_and_evaluate_single_seed(config, config_name, seed):
    """Train REINFORCE with given configuration and seed, return metrics"""
    print(f"  Seed {seed}: Training {config_name}...")
    
    try:
        # Set seed for reproducibility
        set_seeds(seed)

        # Create CartPole environment
        env = gym.make("CartPole-v1")
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        
        # Create REINFORCE agent with specified configuration
        agent = REINFORCEAgent(
            state_size=state_size,
            action_size=action_size,
            lr=config['learning_rate'],
            gamma=config['gamma'],
            use_baseline=config['use_baseline'],
            hidden_sizes=config['hidden_sizes']
        )
        
        # Training parameters
        max_episodes = config['max_episodes']
        episode_rewards = []
        policy_losses = []
        baseline_losses = []
        
        # Training loop
        for episode in range(max_episodes):
            state, _ = env.reset()
            episode_reward = 0
            
            # Run episode
            while True:
                action = agent.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                agent.store_reward(reward)
                episode_reward += reward
                
                if done:
                    break
                    
                state = next_state
            
            # Update policy after episode completion
            policy_loss, baseline_loss = agent.update_policy()
            
            episode_rewards.append(episode_reward)
            if policy_loss is not None:
                policy_losses.append(policy_loss)
            if baseline_loss is not None:
                baseline_losses.append(baseline_loss)
            
            # Early stopping if solved
            if len(episode_rewards) >= 100:
                avg_reward = np.mean(episode_rewards[-100:])
                if avg_reward >= 195:
                    print(f"    Solved at episode {episode}!")
                    break
        
        # Calculate training metrics
        metrics = calculate_metrics(episode_rewards)
        
        # Evaluate the trained model
        test_rewards = []
        for _ in range(20):
            state, _ = env.reset()
            episode_reward = 0
            
            while True:
                action = agent.select_action(state)
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_reward += reward
                
                if done:
                    break
            
            test_rewards.append(episode_reward)
        
        metrics['test_mean_reward'] = np.mean(test_rewards)
        metrics['test_std_reward'] = np.std(test_rewards)
        metrics['test_rewards'] = test_rewards
        metrics['policy_losses'] = policy_losses
        metrics['baseline_losses'] = baseline_losses
        
        env.close()
        
        return metrics
        
    except Exception as e:
        print(f"    ERROR in seed {seed}: {str(e)}")
        # Return default metrics to prevent crash
        return {
            'mean_episode_return': 0.0,
            'success_rate': 0.0,
            'performance_variance': 0.0,
            'final_performance': 0.0,
            'total_episodes': 0,
            'max_reward': 0.0,
            'min_reward': 0.0,
            'sample_efficiency_150': None,
            'sample_efficiency_195': None,
            'test_mean_reward': 0.0,
            'test_std_reward': 0.0,
            'test_rewards': [0.0] * 20,
            'policy_losses': [],
            'baseline_losses': []
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
    if aggregated.get('sample_efficiency_195_mean') is not None:
        print(f"  Sample Efficiency (195+): {aggregated['sample_efficiency_195_mean']:.1f} episodes")
    else:
        print(f"  Sample Efficiency (195+): Not achieved")
        
    if aggregated.get('sample_efficiency_150_mean') is not None:
        print(f"  Sample Efficiency (150+): {aggregated['sample_efficiency_150_mean']:.1f} episodes")
    else:
        print(f"  Sample Efficiency (150+): Not achieved")
    
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
            aggregated[f'{metric}_mean'] = 0.0
            aggregated[f'{metric}_std'] = 0.0
            aggregated[f'{metric}_values'] = []
    
    # Sample efficiency for 150+ reward (intermediate milestone)
    sample_eff_150_values = [m['sample_efficiency_150'] for m in all_metrics 
                            if m['sample_efficiency_150'] is not None]
    if sample_eff_150_values:
        aggregated['sample_efficiency_150_mean'] = np.mean(sample_eff_150_values)
        aggregated['sample_efficiency_150_std'] = np.std(sample_eff_150_values) if len(sample_eff_150_values) > 1 else 0.0
        aggregated['sample_efficiency_150_success_rate'] = len(sample_eff_150_values) / len(all_metrics)
    else:
        aggregated['sample_efficiency_150_mean'] = None
        aggregated['sample_efficiency_150_std'] = None
        aggregated['sample_efficiency_150_success_rate'] = 0
    
    # Sample efficiency for 195+ reward (solved threshold)
    sample_eff_195_values = [m['sample_efficiency_195'] for m in all_metrics 
                            if m['sample_efficiency_195'] is not None]
    if sample_eff_195_values:
        aggregated['sample_efficiency_195_mean'] = np.mean(sample_eff_195_values)
        aggregated['sample_efficiency_195_std'] = np.std(sample_eff_195_values) if len(sample_eff_195_values) > 1 else 0.0
        aggregated['sample_efficiency_195_success_rate'] = len(sample_eff_195_values) / len(all_metrics)
    else:
        aggregated['sample_efficiency_195_mean'] = None
        aggregated['sample_efficiency_195_std'] = None
        aggregated['sample_efficiency_195_success_rate'] = 0
    
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
    """Run systematic hyperparameter tuning with statistical testing for CartPole REINFORCE"""
    
    # Base configuration for REINFORCE
    base_config = {
        'learning_rate': 1e-3,
        'gamma': 0.99,
        'use_baseline': True,
        'hidden_sizes': [128, 128],
        'max_episodes': 1500  # REINFORCE typically needs more episodes
    }
    
    results = []
    
    # 1. Test different learning rates
    print("PHASE 1: Testing Learning Rates for CartPole REINFORCE")
    learning_rates = [1e-4, 1e-3, 5e-3]
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
    
    # 2. Test with and without baseline
    print("\nPHASE 2: Testing Baseline Usage")
    
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
    
    baseline_options = [True, False]
    
    baseline_results = []
    for use_baseline in baseline_options:
        config = base_config.copy()
        config['learning_rate'] = best_lr
        config['use_baseline'] = use_baseline
        baseline_name = "with_baseline" if use_baseline else "no_baseline"
        result = run_config_multiple_seeds(config, f"baseline_{baseline_name}", num_seeds)
        if result:
            baseline_results.append(result)
            results.append(result)

    # Statistical comparison for baseline usage
    if len(baseline_results) >= 2:
        print("\n" + "="*50)
        print("STATISTICAL ANALYSIS: Baseline Usage")
        statistical_comparison(baseline_results, 'test_mean_reward_values')
    
    # 3. Test different discount factors
    print("\nPHASE 3: Testing Discount Factors")
    
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

    gamma_values = [0.95, 0.99, 0.995]
    
    gamma_results = []
    for gamma in gamma_values:
        config = best_config_so_far.copy()
        config['gamma'] = gamma
        result = run_config_multiple_seeds(config, f"gamma_{gamma}", num_seeds)
        if result:
            gamma_results.append(result)
            results.append(result)

    # Statistical comparison for gamma values
    if len(gamma_results) >= 2:
        print("\n" + "="*50)
        print("STATISTICAL ANALYSIS: Discount Factors")
        statistical_comparison(gamma_results, 'test_mean_reward_values')
    
    # 4. Test different network architectures
    print("\nPHASE 4: Testing Network Architectures")
    
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
        ([64, 64], "64_64"),
        ([128, 128], "128_128"),
        ([256, 128], "256_128"),
        ([128, 128, 64], "128_128_64")
    ]
    
    arch_results = []
    for arch, arch_name in architectures:
        config = best_config_so_far.copy()
        config['hidden_sizes'] = arch
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
        
        with open(f'./cartpole/reinforce/json_data/hyperparameter_results_stats_{timestamp}.json', 'w') as f:
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
    print("CARTPOLE REINFORCE HYPERPARAMETER TUNING SUMMARY")
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
                solved_status = "✅ SOLVED" if mean_perf >= 195 else "❌ Not solved"
                print(f"{i+1}. {result['config_name']}: {mean_perf:.2f} ± {std_perf:.2f} "
                      f"(Success: {success_rate:.1%}) {solved_status}")
        except Exception as e:
            print(f"Error creating summary: {e}")
            print("Raw results available in the results list")
    else:
        print("No valid results to summarize")
    
    print(f"\nCartPole REINFORCE hyperparameter tuning completed at {datetime.now()}")
    print("Note: CartPole is considered solved at 195+ average reward over 100 episodes")
    print("REINFORCE is a policy gradient method with high variance - baseline helps reduce this")
    
    return valid_results

if __name__ == "__main__":
    # Run with statistical testing
    print("Starting CartPole REINFORCE hyperparameter tuning with statistical significance testing...")
    print("Note: REINFORCE is a vanilla policy gradient method with high variance.")
    print("Expected training time: 1-2 hours depending on hardware (slower than PPO/DQN)")
    
    try:
        results = run_hyperparameter_tuning_with_stats(num_seeds=2)
        print(f"✅ Successfully completed CartPole REINFORCE hyperparameter tuning!")
        print(f"Total valid configurations tested: {len(results)}")
    except Exception as e:
        print(f"❌ Fatal error in hyperparameter tuning: {e}")
        print("Check the error logs above for details")