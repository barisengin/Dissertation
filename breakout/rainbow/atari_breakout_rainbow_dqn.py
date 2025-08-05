import gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.atari_wrappers import AtariWrapper
import json
import time
import os
from datetime import datetime
from collections import deque
import torch

class PerformanceLogger:
    
    def __init__(self, save_dir="./breakout/rainbow/json_data/"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize tracking variables
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_times = []
        self.training_losses = []
        self.exploration_rates = []
        self.timesteps = []
        
        # Performance metrics (Atari Breakout specific)
        self.convergence_threshold = 300.0  # Breakout considered good performance at 300+
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
                   exploration_rate, total_steps):
        """Log data for a completed episode"""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(episode_length)
        self.episode_times.append(training_time)
        self.exploration_rates.append(exploration_rate)
        self.timesteps.append(total_steps)
        self.total_training_steps = total_steps
        
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
        
        # Calculate stability metrics every 100 episodes
        if episode_num % 100 == 0 and episode_num > 0:
            self._calculate_stability_metrics(episode_num)
    
    def log_training_loss(self, loss):
        """Log training loss values"""
        if loss is not None:
            self.training_losses.append(loss)
    
    def _calculate_stability_metrics(self, episode_num):
        """Calculate stability metrics for recent performance"""
        if len(self.episode_rewards) >= 100:
            recent_100 = self.episode_rewards[-100:]
            variance = np.var(recent_100)
            self.reward_variance_history.append({
                'episode': episode_num,
                'variance': variance,
                'mean': np.mean(recent_100)
            })
            
            # Detect performance drops (>30% decrease in moving average for Atari)
            if len(self.episode_rewards) >= 200:
                prev_avg = np.mean(self.episode_rewards[-200:-100])
                curr_avg = np.mean(recent_100)
                if curr_avg < prev_avg * 0.7:  # 30% drop
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
        if len(self.episode_rewards) < 100:
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
    
    def save_data(self, filename_prefix="rainbow_dqn_breakout"):
        """Save all logged data to JSON files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Raw data
        raw_data = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'episode_times': self.episode_times,
            'exploration_rates': self.exploration_rates,
            'training_losses': self.training_losses,
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
        
        return True
    
    def _on_episode_end(self):
        """Called when an episode ends"""
        if self.episode_start_time is not None:
            episode_time = time.time() - self.episode_start_time
            
            # Get exploration rate from model
            exploration_rate = self.model.exploration_rate if hasattr(self.model, 'exploration_rate') else 0.0
            
            # Log episode data
            self.performance_logger.log_episode(
                episode_num=self.episode_count,
                reward=self.current_episode_reward,
                episode_length=self.current_episode_length,
                training_time=episode_time,
                exploration_rate=exploration_rate,
                total_steps=self.num_timesteps
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
                print(f"Episode {self.episode_count}, Avg Reward (50 ep): {avg_reward:.2f}, Exploration: {exploration_rate:.3f}")

def run(is_training=True):
    print("Setting up Atari Breakout with Rainbow DQN...")
    
    # Create Atari environment with proper wrappers
    if is_training:
        env = make_atari_env('BreakoutNoFrameskip-v4', n_envs=1, seed=42)
        env = VecFrameStack(env, n_stack=4)
    else:
        env = make_atari_env('BreakoutNoFrameskip-v4', n_envs=1, seed=42)
        env = VecFrameStack(env, n_stack=4)

    # Create directories
    os.makedirs("./breakout/rainbow/weights/", exist_ok=True)
    os.makedirs("./breakout/rainbow/json_data/", exist_ok=True)

    print(f"Environment created successfully")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    if is_training:
        logger = PerformanceLogger()
        
        # Rainbow DQN configuration with all Rainbow improvements
        model = DQN(
            "CnnPolicy",
            env,
            # Core Rainbow parameters
            learning_rate=6.25e-5,           # Optimized for Atari
            buffer_size=1000000,             # Large replay buffer
            learning_starts=80000,           # Warm-up period
            batch_size=32,                   # Standard batch size
            tau=1.0,                         # Hard updates
            gamma=0.99,                      # Discount factor
            train_freq=4,                    # Training frequency
            gradient_steps=1,                # Steps per update
            target_update_interval=32000,    # Target network update frequency
            
            # Rainbow-specific improvements available in SB3 DQN
            exploration_fraction=0.1,        # Fraction of training for exploration
            exploration_initial_eps=1.0,     # Initial exploration rate
            exploration_final_eps=0.01,      # Final exploration rate
            max_grad_norm=10.0,              # Gradient clipping
            
            # Network architecture (automatically uses dueling and double DQN in SB3)
            policy_kwargs=dict(
                net_arch=[512],              # Hidden layer size
                activation_fn=torch.nn.ReLU,
                normalize_images=True,       # Normalize pixel values
            ),
            
            # Optimization (disabled for compatibility)
            optimize_memory_usage=False,     # Disabled to avoid SB3 buffer conflicts
            verbose=1,
            device="auto",                   # Use GPU if available
            seed=42,
        )
        
        print("Model created with Rainbow DQN configuration:")
        print(f"- Double DQN: Enabled (default in SB3)")
        print(f"- Dueling Networks: Enabled (default in SB3)")
        print(f"- Experience Replay: Large buffer (1M)")
        print(f"- Target Networks: Hard updates every 32k steps")
        print(f"- Exploration: Linear annealing")
        print(f"- Frame Stacking: 4 frames")
        print(f"- Learning Rate: 6.25e-5")
        
        # Create callback for logging
        callback = TrainingCallback(performance_logger=logger, log_interval=10000)
        
        print("\nStarting training...")
        print("Note: Rainbow DQN training on Atari takes significant time (hours to days)")
        
        # Train the model
        total_timesteps = 10000000  # 10 million steps (standard for Atari Rainbow)
        
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
        model.save("./breakout/rainbow/weights/rainbow_dqn_breakout")
        print("Model saved successfully!")
        
        # Save comprehensive metrics
        logger.save_data()
        
        # Generate training plots
        if logger.episode_rewards:
            plt.figure(figsize=(15, 10))
            
            # Episode rewards over time
            plt.subplot(2, 3, 1)
            episodes = range(len(logger.episode_rewards))
            plt.plot(episodes, logger.episode_rewards, alpha=0.6, label='Episode Reward')
            
            # Moving average
            if len(logger.episode_rewards) >= 100:
                moving_avg = []
                for i in range(len(logger.episode_rewards)):
                    start_idx = max(0, i-99)
                    moving_avg.append(np.mean(logger.episode_rewards[start_idx:i+1]))
                plt.plot(episodes, moving_avg, color='red', linewidth=2, label='Moving Average (100 ep)')
            
            plt.axhline(y=300, color='green', linestyle='--', alpha=0.7, label='Good Performance (300)')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.title('Training Progress: Episode Rewards')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Rewards vs timesteps
            plt.subplot(2, 3, 2)
            if logger.timesteps:
                timesteps_millions = [t/1000000 for t in logger.timesteps]
                plt.plot(timesteps_millions, logger.episode_rewards, alpha=0.6)
                plt.xlabel('Training Steps (Millions)')
                plt.ylabel('Episode Reward')
                plt.title('Reward vs Training Steps')
                plt.axhline(y=300, color='green', linestyle='--', alpha=0.7)
                plt.grid(True, alpha=0.3)
            
            # Exploration rate over time
            plt.subplot(2, 3, 3)
            if logger.exploration_rates:
                plt.plot(episodes, logger.exploration_rates, color='orange', alpha=0.8)
                plt.xlabel('Episode')
                plt.ylabel('Exploration Rate')
                plt.title('Exploration Rate Decay')
                plt.grid(True, alpha=0.3)
            
            # Reward distribution
            plt.subplot(2, 3, 4)
            plt.hist(logger.episode_rewards, bins=50, alpha=0.7, edgecolor='black')
            plt.axvline(np.mean(logger.episode_rewards), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(logger.episode_rewards):.1f}')
            plt.axvline(300, color='green', linestyle='--', label='Threshold: 300')
            plt.xlabel('Episode Reward')
            plt.ylabel('Frequency')
            plt.title('Reward Distribution')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Episode length over time
            plt.subplot(2, 3, 5)
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
            
            # Learning curve analysis
            plt.subplot(2, 3, 6)
            if len(logger.episode_rewards) >= 100:
                # Plot smoothed learning curve
                window = min(100, len(logger.episode_rewards)//10)
                smoothed_rewards = []
                for i in range(window, len(logger.episode_rewards)):
                    smoothed_rewards.append(np.mean(logger.episode_rewards[i-window:i]))
                
                plt.plot(range(window, len(logger.episode_rewards)), smoothed_rewards, 
                        color='blue', linewidth=2, label=f'Smoothed ({window} ep window)')
                plt.axhline(y=300, color='green', linestyle='--', alpha=0.7, label='Target')
                plt.xlabel('Episode')
                plt.ylabel('Smoothed Reward')
                plt.title('Learning Curve (Smoothed)')
                plt.legend()
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig("./breakout/rainbow/weights/rainbow_dqn_breakout_training.png", 
                       dpi=300, bbox_inches='tight')
            plt.show()
            
            # Print final training summary
            final_metrics = logger.get_performance_metrics()
            print("\n=== RAINBOW DQN TRAINING SUMMARY ===")
            print(f"Total Episodes: {final_metrics['basic_stats']['total_episodes']}")
            print(f"Total Training Steps: {final_metrics['basic_stats']['total_training_steps']:,}")
            print(f"Training Time: {training_time/3600:.2f} hours")
            print(f"Final Reward: {final_metrics['basic_stats']['final_reward']:.2f}")
            print(f"Mean Reward: {final_metrics['basic_stats']['mean_reward']:.2f}")
            print(f"Max Reward: {final_metrics['basic_stats']['max_reward']:.2f}")
            print(f"Std Reward: {final_metrics['basic_stats']['std_reward']:.2f}")
            
            if final_metrics['sample_efficiency']['threshold_achieved']:
                print(f"Threshold (300) achieved at episode: {final_metrics['sample_efficiency']['episodes_to_threshold']}")
                print(f"Steps to threshold: {final_metrics['sample_efficiency']['steps_to_threshold']:,}")
            
            if final_metrics['sample_efficiency']['convergence_detected']:
                print(f"Convergence detected at episode: {final_metrics['sample_efficiency']['convergence_episode']}")
        
    else:
        print("Loading trained Rainbow DQN model...")
        try:
            model = DQN.load("./breakout/rainbow/weights/rainbow_dqn_breakout", env=env)
            print("Model loaded successfully!")
            
            # Test the model
            print("Testing the Rainbow DQN agent...")
            test_episodes = 10
            test_rewards = []
            
            for episode in range(test_episodes):
                obs = env.reset()
                episode_reward = 0
                done = False
                
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, done, info = env.step(action)
                    episode_reward += reward[0]  # VecEnv returns arrays
                
                test_rewards.append(episode_reward)
                print(f"Test Episode {episode + 1}: Reward = {episode_reward:.2f}")
            
            print(f"\nTest Results:")
            print(f"Average Reward: {np.mean(test_rewards):.2f}")
            print(f"Std Reward: {np.std(test_rewards):.2f}")
            print(f"Max Reward: {np.max(test_rewards):.2f}")
            print(f"Min Reward: {np.min(test_rewards):.2f}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Please train the model first by running with is_training=True")

    env.close()

if __name__ == "__main__":
    run(is_training=True)