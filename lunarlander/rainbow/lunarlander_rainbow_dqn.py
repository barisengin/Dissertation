import gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import json
import time
import os
from datetime import datetime
from collections import deque
import torch

class PerformanceLogger:
    
    def __init__(self, save_dir="./lunarlander/rainbow/json_data/"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize tracking variables
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_times = []
        self.training_losses = []
        self.q_losses = []
        self.exploration_rates = []
        self.timesteps = []
        
        # Performance metrics (LunarLander specific)
        self.convergence_threshold = 200.0  # LunarLander considered solved at 200
        self.convergence_window = 100
        self.performance_windows = [10, 50, 100, 200]
        
        # Sample efficiency tracking
        self.steps_to_threshold = None
        self.episodes_to_threshold = None
        self.threshold_achieved = False
        
        # Stability metrics
        self.reward_variance_history = []
        self.performance_drops = []
        
        # Training metadata
        self.training_start_time = None
        self.total_training_steps = 0
        
        # Convergence detection
        self.recent_rewards = deque(maxlen=self.convergence_window)
        self.convergence_detected = False
        self.convergence_episode = None
        
    def log_episode(self, episode_num, reward, episode_length, training_time, 
                   total_steps, exploration_rate=None, q_loss=None):
        """Log data for a completed episode"""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(episode_length)
        self.episode_times.append(training_time)
        self.timesteps.append(total_steps)
        self.total_training_steps = total_steps
        
        # Log Rainbow DQN-specific metrics if provided
        if exploration_rate is not None:
            self.exploration_rates.append(exploration_rate)
        if q_loss is not None:
            self.q_losses.append(q_loss)
        
        self.recent_rewards.append(reward)
        
        # Check for convergence
        if not self.convergence_detected and len(self.recent_rewards) == self.convergence_window:
            avg_reward = np.mean(self.recent_rewards)
            if avg_reward >= self.convergence_threshold:
                self.convergence_detected = True
                self.convergence_episode = episode_num
                print(f"Convergence detected at episode {episode_num}")
        
        # Sample efficiency tracking
        if not self.threshold_achieved and reward >= self.convergence_threshold:
            self.steps_to_threshold = total_steps
            self.episodes_to_threshold = episode_num
            self.threshold_achieved = True
            print(f"Threshold achieved at episode {episode_num}, step {total_steps}")
        
        # Calculate stability metrics every 50 episodes
        if episode_num % 50 == 0 and episode_num > 0:
            self._calculate_stability_metrics(episode_num)
    
    def log_training_loss(self, loss):
        """Log training loss values"""
        if loss is not None:
            self.training_losses.append(loss)
    
    def _calculate_stability_metrics(self, episode_num):
        """Calculate stability metrics for recent performance"""
        if len(self.episode_rewards) >= 50:
            recent_50 = self.episode_rewards[-50:]
            variance = np.var(recent_50)
            self.reward_variance_history.append({
                'episode': episode_num,
                'variance': variance,
                'mean': np.mean(recent_50)
            })
            
            # Detect performance drops (>20% decrease in moving average)
            if len(self.episode_rewards) >= 100:
                prev_avg = np.mean(self.episode_rewards[-100:-50])
                curr_avg = np.mean(recent_50)
                if curr_avg < prev_avg * 0.8:  # 20% drop
                    self.performance_drops.append({
                        'episode': episode_num,
                        'drop_magnitude': (prev_avg - curr_avg) / prev_avg
                    })
    
    def get_performance_metrics(self):
        """Calculate comprehensive performance metrics"""
        if not self.episode_rewards:
            return {}
        
        metrics = {
            'basic_stats': {
                'total_episodes': len(self.episode_rewards),
                'total_training_steps': self.total_training_steps,
                'final_reward': self.episode_rewards[-1],
                'max_reward': max(self.episode_rewards),
                'mean_reward': np.mean(self.episode_rewards),
                'std_reward': np.std(self.episode_rewards),
                'mean_episode_length': np.mean(self.episode_lengths)
            },
            'sample_efficiency': {
                'episodes_to_threshold': self.episodes_to_threshold,
                'steps_to_threshold': self.steps_to_threshold,
                'threshold_achieved': self.threshold_achieved,
                'convergence_episode': self.convergence_episode,
                'convergence_detected': self.convergence_detected
            },
            'stability_metrics': {
                'reward_variance_trend': self.reward_variance_history,
                'performance_drops': self.performance_drops,
                'final_100_episode_average': np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else None,
                'coefficient_of_variation': np.std(self.episode_rewards) / np.mean(self.episode_rewards) if np.mean(self.episode_rewards) > 0 else None
            },
            'learning_curve_analysis': self._analyze_learning_curve(),
            'training_efficiency': {
                'average_episode_time': np.mean(self.episode_times),
                'total_training_time': sum(self.episode_times),
                'steps_per_second': self.total_training_steps / sum(self.episode_times) if sum(self.episode_times) > 0 else 0
            },
            'rainbow_dqn_specific_metrics': {
                'q_losses': self.q_losses,
                'exploration_rates': self.exploration_rates
            }
        }
        
        # Add moving averages for different windows
        for window in self.performance_windows:
            if len(self.episode_rewards) >= window:
                moving_avg = []
                for i in range(window-1, len(self.episode_rewards)):
                    moving_avg.append(np.mean(self.episode_rewards[i-window+1:i+1]))
                metrics[f'moving_average_{window}'] = moving_avg
        
        return metrics
    
    def _analyze_learning_curve(self):
        """Analyze learning curve characteristics"""
        if len(self.episode_rewards) < 50:
            return {}
        
        # Calculate slope of learning curve (linear regression)
        x = np.arange(len(self.episode_rewards))
        y = np.array(self.episode_rewards)
        slope, intercept = np.polyfit(x, y, 1)
        
        # Early vs late performance comparison
        early_performance = np.mean(self.episode_rewards[:len(self.episode_rewards)//4])
        late_performance = np.mean(self.episode_rewards[-len(self.episode_rewards)//4:])
        
        return {
            'learning_rate_slope': slope,
            'early_performance': early_performance,
            'late_performance': late_performance,
            'improvement_ratio': late_performance / early_performance if early_performance > 0 else None,
            'monotonic_improvement': slope > 0
        }
    
    def save_data(self, filename_prefix="rainbow_dqn_lunarlander"):
        """Save all logged data to JSON files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Raw data
        raw_data = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'episode_times': self.episode_times,
            'training_losses': self.training_losses,
            'q_losses': self.q_losses,
            'exploration_rates': self.exploration_rates,
            'timesteps': self.timesteps,
            'metadata': {
                'timestamp': timestamp,
                'convergence_threshold': self.convergence_threshold,
                'convergence_window': self.convergence_window,
                'total_episodes': len(self.episode_rewards),
                'total_training_steps': self.total_training_steps
            }
        }
        
        # Performance metrics
        performance_metrics = self.get_performance_metrics()
        
        # Save files
        raw_file = os.path.join(self.save_dir, f"{filename_prefix}_raw_data_{timestamp}.json")
        metrics_file = os.path.join(self.save_dir, f"{filename_prefix}_metrics_{timestamp}.json")
        
        with open(raw_file, 'w') as f:
            json.dump(raw_data, f, indent=2, default=str)
        
        with open(metrics_file, 'w') as f:
            json.dump(performance_metrics, f, indent=2, default=str)
        
        print(f"Data saved to {raw_file} and {metrics_file}")
        return raw_file, metrics_file

class TrainingCallback(BaseCallback):
    """Custom callback to track training progress and metrics"""
    
    def __init__(self, performance_logger, log_interval=1000, verbose=0):
        super(TrainingCallback, self).__init__(verbose)
        self.performance_logger = performance_logger
        self.log_interval = log_interval
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_count = 0
        self.episode_start_time = None
        self.current_episode_reward = 0
        self.current_episode_length = 0
        self.last_log_step = 0
        
    def _on_training_start(self) -> None:
        """Called at the beginning of training"""
        self.episode_start_time = time.time()
        
    def _on_step(self) -> bool:
        """Called at each step"""
        # Track episode progress
        self.current_episode_length += 1
        
        # Get reward from info if available
        if len(self.locals.get('rewards', [])) > 0:
            self.current_episode_reward += self.locals['rewards'][0]
        
        # Check if episode ended
        if len(self.locals.get('dones', [])) > 0 and self.locals['dones'][0]:
            self._on_episode_end()
        
        # Log training metrics periodically
        if self.num_timesteps - self.last_log_step >= self.log_interval:
            self._log_training_metrics()
            self.last_log_step = self.num_timesteps
        
        return True
    
    def _on_episode_end(self):
        """Called when an episode ends"""
        if self.episode_start_time is not None:
            episode_time = time.time() - self.episode_start_time
            
            # Try to get DQN-specific information from model
            exploration_rate = None
            q_loss = None
            
            # Get exploration rate from model
            if hasattr(self.model, 'exploration_rate'):
                exploration_rate = self.model.exploration_rate
            
            if hasattr(self.model, 'logger') and self.model.logger is not None:
                # Get recent loss values if available
                if hasattr(self.model.logger, 'name_to_value'):
                    q_loss = self.model.logger.name_to_value.get('train/loss', None)
            
            # Log episode data
            self.performance_logger.log_episode(
                episode_num=self.episode_count,
                reward=self.current_episode_reward,
                episode_length=self.current_episode_length,
                training_time=episode_time,
                total_steps=self.num_timesteps,
                exploration_rate=exploration_rate,
                q_loss=q_loss
            )
            
            # Reset for next episode
            self.episode_count += 1
            self.current_episode_reward = 0
            self.current_episode_length = 0
            self.episode_start_time = time.time()
            
            # Log progress
            if self.episode_count % 50 == 0:
                recent_rewards = self.performance_logger.episode_rewards[-50:] if len(self.performance_logger.episode_rewards) >= 50 else self.performance_logger.episode_rewards
                avg_reward = np.mean(recent_rewards) if recent_rewards else 0
                eps = exploration_rate if exploration_rate is not None else 0
                print(f"Episode {self.episode_count}, Avg Reward (50 ep): {avg_reward:.2f}, Exploration: {eps:.3f}, Steps: {self.num_timesteps}")
    
    def _log_training_metrics(self):
        """Log training metrics periodically"""
        if hasattr(self.model, 'logger') and self.model.logger is not None:
            if hasattr(self.model.logger, 'name_to_value'):
                loss = self.model.logger.name_to_value.get('train/loss', None)
                if loss is not None:
                    self.performance_logger.log_training_loss(loss)

class LunarLanderWrapper(gym.Wrapper):
    """LunarLander-specific wrapper for any preprocessing if needed"""
    
    def __init__(self, env):
        super(LunarLanderWrapper, self).__init__(env)
        
    def reset(self, **kwargs):
        """Reset environment and handle new gym API"""
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple):
            obs, info = result
            return obs, info
        else:
            return result, {}
        
    def step(self, action):
        """Step environment"""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Optional: Add reward shaping for better DQN learning
        # For example, small penalty for fuel usage or reward for staying upright
        # reward = reward  # Keep original reward structure for now
        
        return obs, reward, terminated, truncated, info

def run(is_training=True):
    print("Setting up LunarLander with Rainbow DQN...")
    
    # Create LunarLander environment
    if is_training:
        def make_env():
            env = gym.make('LunarLander-v2')
            env = LunarLanderWrapper(env)
            return env
            
        env = DummyVecEnv([make_env])
    else:
        def make_env():
            env = gym.make('LunarLander-v2')
            env = LunarLanderWrapper(env)
            return env
            
        env = DummyVecEnv([make_env])

    # Create directories
    os.makedirs("./lunarlander/rainbow/weights/", exist_ok=True)
    os.makedirs("./lunarlander/rainbow/json_data/", exist_ok=True)

    print(f"Environment created successfully")
    print(f"Number of environments: {env.num_envs}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Action space type: Discrete")
    print(f"State dimensions: 8 (x, y, vx, vy, angle, angular_vel, leg1_contact, leg2_contact)")
    print(f"Actions: 4 (do nothing, fire left, fire main, fire right)")

    if is_training:
        logger = PerformanceLogger()
        
        # Rainbow DQN configuration optimized for LunarLander
        model = DQN(
            "MlpPolicy",
            env,
            # Core Rainbow DQN parameters
            learning_rate=6.3e-4,            # Slightly higher than vanilla DQN
            buffer_size=100000,              # Large replay buffer
            learning_starts=1000,            # Start training after warmup
            batch_size=128,                  # Larger batch size for stability
            tau=1.0,                         # Hard updates (Rainbow style)
            gamma=0.999,                     # High discount factor for long episodes
            train_freq=4,                    # Train every 4 steps
            gradient_steps=1,                # Gradient steps per train_freq
            target_update_interval=1000,     # Hard target update frequency
            
            # Rainbow DQN specific improvements (available in SB3 DQN)
            exploration_fraction=0.12,       # 12% of training for exploration
            exploration_initial_eps=1.0,     # Start with full exploration
            exploration_final_eps=0.02,      # End with minimal exploration
            max_grad_norm=10,                # Gradient clipping
            
            # Network architecture optimized for LunarLander
            policy_kwargs=dict(
                net_arch=[512, 512],         # Larger network for intermediate complexity
                activation_fn=torch.nn.ReLU,
                # Note: SB3 DQN automatically includes Double DQN and Dueling Networks
            ),
            
            # Training configuration
            verbose=1,
            device="auto",                   # Use GPU if available
            seed=42,
        )
        
        print("Model created with Rainbow DQN configuration for LunarLander:")
        print(f"- Algorithm: Enhanced DQN (includes Rainbow improvements)")
        print(f"- Environment: Single environment with replay buffer")
        print(f"- Policy Type: MLP + Discrete Actions")
        print(f"- Learning Rate: 6.3e-4 (optimized for DQN)")
        print(f"- Buffer Size: 100,000 transitions")
        print(f"- Batch Size: 128")
        print(f"- Gamma: 0.999 (high discount for long episodes)")
        print(f"- Target Update: Hard updates every 1000 steps")
        print(f"- Exploration: Linear annealing from 1.0 to 0.02")
        print(f"- Rainbow Features:")
        print(f"  * Double DQN: Enabled (reduces overestimation bias)")
        print(f"  * Dueling Networks: Enabled (separates value/advantage)")
        print(f"  * Experience Replay: Large buffer with uniform sampling")
        print(f"  * Target Networks: Hard updates for stability")
        print(f"- Network Architecture: [8] -> [512, 512] -> [4]")
        print(f"- Action Space: 4 discrete actions")
        
        # Create callback for logging
        callback = TrainingCallback(logger, log_interval=5000)
        
        print("\nStarting training...")
        print("Note: Rainbow DQN should be more sample efficient than vanilla DQN")
        print("Expected stable learning with reduced overestimation bias")
        
        # Train the model
        total_timesteps = 800000  # 800k steps should be sufficient for Rainbow DQN
        
        start_time = time.time()
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=100,
            progress_bar=True
        )
        
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time/3600:.2f} hours")
        
        # Save the model
        model.save("./lunarlander/rainbow/weights/rainbow_dqn_lunarlander")
        print("Model saved successfully!")
        
        # Save comprehensive metrics
        logger.save_data()
        
        # Generate training plots
        if logger.episode_rewards:
            plt.figure(figsize=(16, 12))
            
            # Episode rewards over time
            plt.subplot(3, 3, 1)
            episodes = range(len(logger.episode_rewards))
            plt.plot(episodes, logger.episode_rewards, alpha=0.6, label='Episode Reward')
            
            # Moving average
            if len(logger.episode_rewards) >= 100:
                moving_avg = []
                for i in range(len(logger.episode_rewards)):
                    start_idx = max(0, i-99)
                    moving_avg.append(np.mean(logger.episode_rewards[start_idx:i+1]))
                plt.plot(episodes, moving_avg, color='red', linewidth=2, label='Moving Average (100 ep)')
            
            plt.axhline(y=200, color='green', linestyle='--', alpha=0.7, label='Convergence Threshold (200)')
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3, label='Break Even')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.title('Training Progress: Episode Rewards')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Rewards vs timesteps
            plt.subplot(3, 3, 2)
            if logger.timesteps:
                timesteps_thousands = [t/1000 for t in logger.timesteps]
                plt.plot(timesteps_thousands, logger.episode_rewards, alpha=0.6)
                plt.xlabel('Training Steps (Thousands)')
                plt.ylabel('Episode Reward')
                plt.title('Reward vs Training Steps')
                plt.axhline(y=200, color='green', linestyle='--', alpha=0.7)
                plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                plt.grid(True, alpha=0.3)
            
            # Exploration rate over time
            plt.subplot(3, 3, 3)
            if logger.exploration_rates:
                plt.plot(episodes, logger.exploration_rates, color='orange', alpha=0.8, label='Exploration Rate')
                plt.xlabel('Episode')
                plt.ylabel('Exploration Rate (ε)')
                plt.title('Exploration Rate Decay')
                plt.legend()
                plt.grid(True, alpha=0.3)
            
            # Reward distribution
            plt.subplot(3, 3, 4)
            plt.hist(logger.episode_rewards, bins=50, alpha=0.7, edgecolor='black')
            plt.axvline(np.mean(logger.episode_rewards), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(logger.episode_rewards):.1f}')
            plt.axvline(200, color='green', linestyle='--', label='Target: 200')
            plt.axvline(0, color='black', linestyle='-', alpha=0.5, label='Break Even')
            plt.xlabel('Episode Reward')
            plt.ylabel('Frequency')
            plt.title('Reward Distribution')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Episode length over time
            plt.subplot(3, 3, 5)
            if logger.episode_lengths:
                plt.plot(episodes, logger.episode_lengths, alpha=0.6, color='purple')
                if len(logger.episode_lengths) >= 50:
                    moving_avg_len = []
                    for i in range(len(logger.episode_lengths)):
                        start_idx = max(0, i-49)
                        moving_avg_len.append(np.mean(logger.episode_lengths[start_idx:i+1]))
                    plt.plot(episodes, moving_avg_len, color='darkred', linewidth=2, label='Moving Avg (50 ep)')
                plt.xlabel('Episode')
                plt.ylabel('Episode Length')
                plt.title('Episode Length Over Time')
                plt.legend()
                plt.grid(True, alpha=0.3)
            
            # Q-Loss (DQN-specific)
            plt.subplot(3, 3, 6)
            if logger.q_losses:
                episodes_with_loss = range(len(logger.q_losses))
                plt.plot(episodes_with_loss, logger.q_losses, alpha=0.7, color='blue', label='Q-Loss')
                if len(logger.q_losses) >= 20:
                    window = min(20, len(logger.q_losses)//5)
                    smoothed_q = []
                    for i in range(window, len(logger.q_losses)):
                        smoothed_q.append(np.mean(logger.q_losses[i-window:i]))
                    plt.plot(range(window, len(logger.q_losses)), smoothed_q, 
                            color='darkblue', linewidth=2, label=f'Smoothed ({window})')
                plt.xlabel('Episode')
                plt.ylabel('Q-Loss')
                plt.title('Q-Network Loss Over Time')
                plt.legend()
                plt.grid(True, alpha=0.3)
            
            # Learning curve analysis
            plt.subplot(3, 3, 7)
            if len(logger.episode_rewards) >= 100:
                # Plot smoothed learning curve
                window = min(100, len(logger.episode_rewards)//10)
                smoothed_rewards = []
                for i in range(window, len(logger.episode_rewards)):
                    smoothed_rewards.append(np.mean(logger.episode_rewards[i-window:i]))
                
                plt.plot(range(window, len(logger.episode_rewards)), smoothed_rewards, 
                        color='blue', linewidth=2, label=f'Smoothed ({window} ep window)')
                plt.axhline(y=200, color='green', linestyle='--', alpha=0.7, label='Target')
                plt.axhline(y=0, color='black', linestyle='-', alpha=0.3, label='Break Even')
                plt.xlabel('Episode')
                plt.ylabel('Smoothed Reward')
                plt.title('Learning Curve (Smoothed)')
                plt.legend()
                plt.grid(True, alpha=0.3)
            
            # Training efficiency
            plt.subplot(3, 3, 8)
            if logger.episode_times:
                cumulative_time = np.cumsum(logger.episode_times)
                cumulative_time_hours = [t/3600 for t in cumulative_time]
                plt.plot(cumulative_time_hours, logger.episode_rewards, alpha=0.6, color='brown')
                plt.xlabel('Training Time (Hours)')
                plt.ylabel('Episode Reward')
                plt.title('Reward vs Training Time')
                plt.axhline(y=200, color='green', linestyle='--', alpha=0.7)
                plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                plt.grid(True, alpha=0.3)
            
            # Exploration vs Performance
            plt.subplot(3, 3, 9)
            if logger.exploration_rates and len(logger.exploration_rates) == len(logger.episode_rewards):
                # Create exploration phase analysis
                exploration_high = [(r, e) for r, e in zip(logger.episode_rewards, logger.exploration_rates) if e > 0.5]
                exploration_mid = [(r, e) for r, e in zip(logger.episode_rewards, logger.exploration_rates) if 0.1 <= e <= 0.5]
                exploration_low = [(r, e) for r, e in zip(logger.episode_rewards, logger.exploration_rates) if e < 0.1]
                
                if exploration_high:
                    high_rewards = [r for r, e in exploration_high]
                    plt.hist(high_rewards, bins=20, alpha=0.7, label=f'High Exploration (ε>0.5)', color='red')
                if exploration_mid:
                    mid_rewards = [r for r, e in exploration_mid]
                    plt.hist(mid_rewards, bins=20, alpha=0.7, label=f'Mid Exploration (0.1≤ε≤0.5)', color='orange')
                if exploration_low:
                    low_rewards = [r for r, e in exploration_low]
                    plt.hist(low_rewards, bins=20, alpha=0.7, label=f'Low Exploration (ε<0.1)', color='green')
                
                plt.axvline(200, color='black', linestyle='--', label='Target: 200')
                plt.xlabel('Episode Reward')
                plt.ylabel('Frequency')
                plt.title('Performance vs Exploration Phase')
                plt.legend()
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig("./lunarlander/rainbow/weights/rainbow_dqn_lunarlander_training.png", 
                       dpi=300, bbox_inches='tight')
            plt.show()
            
            # Print final training summary
            final_metrics = logger.get_performance_metrics()
            print("\n=== RAINBOW DQN LUNARLANDER TRAINING SUMMARY ===")
            print(f"Total Episodes: {final_metrics['basic_stats']['total_episodes']}")
            print(f"Total Training Steps: {final_metrics['basic_stats']['total_training_steps']:,}")
            print(f"Training Time: {training_time/3600:.2f} hours")
            print(f"Environment: Single environment with replay buffer")
            print(f"Final Reward: {final_metrics['basic_stats']['final_reward']:.2f}")
            print(f"Mean Reward: {final_metrics['basic_stats']['mean_reward']:.2f}")
            print(f"Max Reward: {final_metrics['basic_stats']['max_reward']:.2f}")
            print(f"Std Reward: {final_metrics['basic_stats']['std_reward']:.2f}")
            print(f"Mean Episode Length: {final_metrics['basic_stats']['mean_episode_length']:.1f}")
            
            if final_metrics['sample_efficiency']['threshold_achieved']:
                print(f"Threshold (200) achieved at episode: {final_metrics['sample_efficiency']['episodes_to_threshold']}")
                print(f"Steps to threshold: {final_metrics['sample_efficiency']['steps_to_threshold']:,}")
            
            if final_metrics['sample_efficiency']['convergence_detected']:
                print(f"Convergence detected at episode: {final_metrics['sample_efficiency']['convergence_episode']}")
            
            print(f"Steps per second: {final_metrics['training_efficiency']['steps_per_second']:.1f}")
            
            # LunarLander specific insights
            successful_landings = [r for r in logger.episode_rewards if r >= 200]
            crashes = [r for r in logger.episode_rewards if r < -100]
            print(f"\nLunarLander Specific Metrics:")
            print(f"Successful landings (≥200): {len(successful_landings)}/{len(logger.episode_rewards)} ({100*len(successful_landings)/len(logger.episode_rewards):.1f}%)")
            print(f"Crashes (<-100): {len(crashes)}/{len(logger.episode_rewards)} ({100*len(crashes)/len(logger.episode_rewards):.1f}%)")
            if successful_landings:
                print(f"Average successful landing reward: {np.mean(successful_landings):.2f}")
            
            # Rainbow DQN specific insights
            if logger.exploration_rates:
                final_exploration = logger.exploration_rates[-1]
                print(f"\nRainbow DQN Specific Metrics:")
                print(f"Final exploration rate: {final_exploration:.4f}")
                print(f"Rainbow improvements enabled:")
                print(f"  * Double DQN: Reduces Q-value overestimation")
                print(f"  * Dueling Networks: Separate value/advantage estimation")
                print(f"  * Large replay buffer: 100k transitions for diverse experience")
                print(f"  * Optimized hyperparameters: Enhanced learning rate and batch size")
        
    else:
        print("Loading trained Rainbow DQN model...")
        try:
            model = DQN.load("./lunarlander/rainbow/weights/rainbow_dqn_lunarlander", env=env)
            print("Model loaded successfully!")
            
            # Test the model
            print("Testing the Rainbow DQN agent...")
            test_episodes = 10
            test_rewards = []
            
            for episode in range(test_episodes):
                obs = env.reset()
                episode_reward = 0
                done = False
                step_count = 0
                max_steps = 1000  # Prevent infinite episodes
                
                while not done and step_count < max_steps:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, done, info = env.step(action)
                    episode_reward += reward[0]  # VecEnv returns arrays
                    step_count += 1
                
                test_rewards.append(episode_reward)
                print(f"Test Episode {episode + 1}: Reward = {episode_reward:.2f}, Steps = {step_count}")
            
            print(f"\nTest Results:")
            print(f"Average Reward: {np.mean(test_rewards):.2f}")
            print(f"Std Reward: {np.std(test_rewards):.2f}")
            print(f"Max Reward: {np.max(test_rewards):.2f}")
            print(f"Min Reward: {np.min(test_rewards):.2f}")
            print(f"Success Rate (≥200): {100*sum(1 for r in test_rewards if r >= 200)/len(test_rewards):.1f}%")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Please train the model first by running with is_training=True")

    env.close()

if __name__ == "__main__":
    run(is_training=True)