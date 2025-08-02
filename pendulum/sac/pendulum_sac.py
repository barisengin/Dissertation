import gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import json
import time
import os
from datetime import datetime
from collections import deque
import torch

class PerformanceLogger:
    
    def __init__(self, save_dir="./pendulum/sac/json_data/"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize tracking variables
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_times = []
        self.training_losses = []
        self.actor_losses = []
        self.critic_losses = []
        self.entropy_losses = []
        self.entropy_coefficients = []
        self.timesteps = []
        
        # Performance metrics (Pendulum specific)
        self.convergence_threshold = -200.0  # Pendulum considered good performance at -200+ (less negative)
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
                   total_steps, actor_loss=None, critic_loss=None, entropy_loss=None, entropy_coef=None):
        """Log data for a completed episode"""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(episode_length)
        self.episode_times.append(training_time)
        self.timesteps.append(total_steps)
        self.total_training_steps = total_steps
        
        # Log SAC-specific losses if provided
        if actor_loss is not None:
            self.actor_losses.append(actor_loss)
        if critic_loss is not None:
            self.critic_losses.append(critic_loss)
        if entropy_loss is not None:
            self.entropy_losses.append(entropy_loss)
        if entropy_coef is not None:
            self.entropy_coefficients.append(entropy_coef)
        
        self.recent_rewards.append(reward)
        
        # Check for convergence (higher reward is better for Pendulum, but rewards are negative)
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
            
            # Detect performance drops (>15% decrease in moving average for Pendulum)
            if len(self.episode_rewards) >= 200:
                prev_avg = np.mean(self.episode_rewards[-200:-100])
                curr_avg = np.mean(recent_100)
                if curr_avg < prev_avg * 0.85:  # 15% drop
                    self.performance_drops.append({
                        'episode': episode_num,
                        'drop_magnitude': (prev_avg - curr_avg) / abs(prev_avg)
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
                'coefficient_of_variation': np.std(self.episode_rewards) / abs(np.mean(self.episode_rewards)) if np.mean(self.episode_rewards) != 0 else None
            },
            'learning_curve_analysis': self._analyze_learning_curve(),
            'training_efficiency': {
                'average_episode_time': np.mean(self.episode_times),
                'total_training_time': sum(self.episode_times),
                'steps_per_second': self.total_training_steps / sum(self.episode_times) if sum(self.episode_times) > 0 else 0
            },
            'sac_specific_metrics': {
                'actor_losses': self.actor_losses,
                'critic_losses': self.critic_losses,
                'entropy_losses': self.entropy_losses,
                'entropy_coefficients': self.entropy_coefficients
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
            'improvement_ratio': late_performance / early_performance if early_performance != 0 else None,
            'monotonic_improvement': slope > 0
        }
    
    def save_data(self, filename_prefix="sac_pendulum"):
        """Save all logged data to JSON files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Raw data
        raw_data = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'episode_times': self.episode_times,
            'training_losses': self.training_losses,
            'actor_losses': self.actor_losses,
            'critic_losses': self.critic_losses,
            'entropy_losses': self.entropy_losses,
            'entropy_coefficients': self.entropy_coefficients,
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
            
            # Try to get SAC-specific loss information from model
            actor_loss = None
            critic_loss = None
            entropy_loss = None
            entropy_coef = None
            
            if hasattr(self.model, 'logger') and self.model.logger is not None:
                # Get recent loss values if available
                if hasattr(self.model.logger, 'name_to_value'):
                    actor_loss = self.model.logger.name_to_value.get('train/actor_loss', None)
                    critic_loss = self.model.logger.name_to_value.get('train/critic_loss', None)
                    entropy_loss = self.model.logger.name_to_value.get('train/ent_coef_loss', None)
                    entropy_coef = self.model.logger.name_to_value.get('train/ent_coef', None)
            
            # Log episode data
            self.performance_logger.log_episode(
                episode_num=self.episode_count,
                reward=self.current_episode_reward,
                episode_length=self.current_episode_length,
                training_time=episode_time,
                total_steps=self.num_timesteps,
                actor_loss=actor_loss,
                critic_loss=critic_loss,
                entropy_loss=entropy_loss,
                entropy_coef=entropy_coef
            )
            
            # Reset for next episode
            self.episode_count += 1
            self.current_episode_reward = 0
            self.current_episode_length = 0
            self.episode_start_time = time.time()
            
            # Log progress
            if self.episode_count % 25 == 0:
                recent_rewards = self.performance_logger.episode_rewards[-25:] if len(self.performance_logger.episode_rewards) >= 25 else self.performance_logger.episode_rewards
                avg_reward = np.mean(recent_rewards) if recent_rewards else 0
                print(f"Episode {self.episode_count}, Avg Reward (25 ep): {avg_reward:.2f}, Steps: {self.num_timesteps}")
    
    def _log_training_metrics(self):
        """Log training metrics periodically"""
        if hasattr(self.model, 'logger') and self.model.logger is not None:
            if hasattr(self.model.logger, 'name_to_value'):
                loss = self.model.logger.name_to_value.get('train/loss', None)
                if loss is not None:
                    self.performance_logger.log_training_loss(loss)

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
        
        # Optional: Add reward shaping for better learning
        # Pendulum has dense rewards already, but we could add small modifications
        # reward = reward  # Keep original reward structure for now
        
        return obs, reward, terminated, truncated, info

def run(is_training=True):
    print("Setting up Pendulum with SAC...")
    
    # Create Pendulum environment
    if is_training:
        # SAC works well with single environment due to off-policy replay buffer
        def make_env():
            env = gym.make('Pendulum-v1')
            env = PendulumWrapper(env)
            return env
            
        env = DummyVecEnv([make_env])
    else:
        def make_env():
            env = gym.make('Pendulum-v1')
            env = PendulumWrapper(env)
            return env
            
        env = DummyVecEnv([make_env])

    # Create directories
    os.makedirs("./pendulum/sac/weights/", exist_ok=True)
    os.makedirs("./pendulum/sac/json_data/", exist_ok=True)

    print(f"Environment created successfully")
    print(f"Number of environments: {env.num_envs}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Action space type: Continuous")
    print(f"State dimensions: 3 [cos(θ), sin(θ), angular_velocity]")
    print(f"Action dimensions: 1 [torque ∈ [-2, 2]]")

    if is_training:
        logger = PerformanceLogger()
        
        # SAC configuration optimized for Pendulum continuous control
        model = SAC(
            "MlpPolicy",
            env,
            # Core SAC parameters optimized for continuous control
            learning_rate=3e-4,              # Learning rate for all networks
            buffer_size=50000,               # Replay buffer size (smaller for simple Pendulum)
            learning_starts=1000,            # Start training after collecting experience
            batch_size=256,                  # Batch size for training
            tau=0.005,                       # Soft update coefficient for target networks
            gamma=0.99,                      # Discount factor
            train_freq=1,                    # Train after every step
            gradient_steps=1,                # Number of gradient steps per train_freq
            
            # SAC-specific parameters
            ent_coef='auto',                 # Automatic entropy coefficient tuning
            target_update_interval=1,        # Update target networks every step
            target_entropy='auto',           # Automatic target entropy
            use_sde=False,                   # Don't use state-dependent exploration
            sde_sample_freq=-1,
            use_sde_at_warmup=False,
            
            # Network architecture for Pendulum
            policy_kwargs=dict(
                net_arch=[64, 64],           # Small network for simple Pendulum
                activation_fn=torch.nn.ReLU,
                n_critics=2,                 # Number of critic networks (standard for SAC)
                share_features_extractor=False,  # Separate feature extractors for actor/critic
            ),
            
            # Training configuration
            verbose=1,
            device="auto",                   # Use GPU if available
            seed=42,
        )
        
        print("Model created with SAC configuration for Pendulum:")
        print(f"- Algorithm: SAC (Soft Actor-Critic)")
        print(f"- Environment: Single environment with replay buffer")
        print(f"- Policy Type: MLP + Continuous Actions")
        print(f"- Learning Rate: 3e-4")
        print(f"- Buffer Size: 50,000 (optimized for simple Pendulum)")
        print(f"- Batch Size: 256")
        print(f"- Learning Starts: 1,000 (warmup period)")
        print(f"- Entropy Coefficient: Auto-tuned")
        print(f"- Target Entropy: Auto-computed")
        print(f"- Tau: 0.005 (soft target network updates)")
        print(f"- N-Critics: 2 (double critic for reduced overestimation)")
        print(f"- Train Frequency: Every step (off-policy efficiency)")
        print(f"- Network Architecture: [3] -> [64, 64] -> [1]")
        print(f"- Maximum Entropy: Encourages exploration in continuous space")
        
        # Create callback for logging
        callback = TrainingCallback(logger, log_interval=1000)
        
        print("\nStarting training...")
        print("Note: SAC should be extremely sample efficient for Pendulum")
        print("Expected very fast convergence due to dense rewards and maximum entropy")
        
        # Train the model (SAC should converge very quickly on Pendulum)
        total_timesteps = 50000  # 50k steps should be more than sufficient for SAC
        
        start_time = time.time()
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=25,
            progress_bar=True
        )
        
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time/60:.2f} minutes")
        
        # Save the model
        model.save("./pendulum/sac/weights/sac_pendulum")
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
            if len(logger.episode_rewards) >= 25:
                moving_avg = []
                for i in range(len(logger.episode_rewards)):
                    start_idx = max(0, i-24)
                    moving_avg.append(np.mean(logger.episode_rewards[start_idx:i+1]))
                plt.plot(episodes, moving_avg, color='red', linewidth=2, label='Moving Average (25 ep)')
            
            plt.axhline(y=-200, color='green', linestyle='--', alpha=0.7, label='Good Performance (-200)')
            plt.axhline(y=-500, color='orange', linestyle='--', alpha=0.7, label='Moderate Performance (-500)')
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
                plt.axhline(y=-200, color='green', linestyle='--', alpha=0.7)
                plt.axhline(y=-500, color='orange', linestyle='--', alpha=0.7)
                plt.grid(True, alpha=0.3)
            
            # Reward distribution
            plt.subplot(3, 3, 3)
            plt.hist(logger.episode_rewards, bins=30, alpha=0.7, edgecolor='black')
            plt.axvline(np.mean(logger.episode_rewards), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(logger.episode_rewards):.1f}')
            plt.axvline(-200, color='green', linestyle='--', label='Good: -200')
            plt.axvline(-500, color='orange', linestyle='--', label='Moderate: -500')
            plt.xlabel('Episode Reward')
            plt.ylabel('Frequency')
            plt.title('Reward Distribution')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Episode length over time (Pendulum episodes are fixed length)
            plt.subplot(3, 3, 4)
            if logger.episode_lengths:
                plt.plot(episodes, logger.episode_lengths, alpha=0.6, color='purple')
                plt.axhline(y=200, color='black', linestyle='-', alpha=0.5, label='Max Length (200)')
                plt.xlabel('Episode')
                plt.ylabel('Episode Length')
                plt.title('Episode Length Over Time')
                plt.legend()
                plt.grid(True, alpha=0.3)
            
            # Actor Loss (SAC-specific)
            plt.subplot(3, 3, 5)
            if logger.actor_losses:
                episodes_with_loss = range(len(logger.actor_losses))
                plt.plot(episodes_with_loss, logger.actor_losses, alpha=0.7, color='orange', label='Actor Loss')
                if len(logger.actor_losses) >= 10:
                    window = min(10, len(logger.actor_losses)//5)
                    smoothed_actor = []
                    for i in range(window, len(logger.actor_losses)):
                        smoothed_actor.append(np.mean(logger.actor_losses[i-window:i]))
                    plt.plot(range(window, len(logger.actor_losses)), smoothed_actor, 
                            color='darkorange', linewidth=2, label=f'Smoothed ({window})')
                plt.xlabel('Episode')
                plt.ylabel('Actor Loss')
                plt.title('SAC Actor Loss Over Time')
                plt.legend()
                plt.grid(True, alpha=0.3)
            
            # Critic Loss (SAC-specific)
            plt.subplot(3, 3, 6)
            if logger.critic_losses:
                episodes_with_loss = range(len(logger.critic_losses))
                plt.plot(episodes_with_loss, logger.critic_losses, alpha=0.7, color='blue', label='Critic Loss')
                if len(logger.critic_losses) >= 10:
                    window = min(10, len(logger.critic_losses)//5)
                    smoothed_critic = []
                    for i in range(window, len(logger.critic_losses)):
                        smoothed_critic.append(np.mean(logger.critic_losses[i-window:i]))
                    plt.plot(range(window, len(logger.critic_losses)), smoothed_critic, 
                            color='darkblue', linewidth=2, label=f'Smoothed ({window})')
                plt.xlabel('Episode')
                plt.ylabel('Critic Loss')
                plt.title('SAC Critic Loss Over Time')
                plt.legend()
                plt.grid(True, alpha=0.3)
            
            # Entropy Coefficient (SAC-specific)
            plt.subplot(3, 3, 7)
            if logger.entropy_coefficients:
                episodes_with_ent = range(len(logger.entropy_coefficients))
                plt.plot(episodes_with_ent, logger.entropy_coefficients, alpha=0.7, color='green', label='Entropy Coefficient')
                if len(logger.entropy_coefficients) >= 10:
                    window = min(10, len(logger.entropy_coefficients)//5)
                    smoothed_ent = []
                    for i in range(window, len(logger.entropy_coefficients)):
                        smoothed_ent.append(np.mean(logger.entropy_coefficients[i-window:i]))
                    plt.plot(range(window, len(logger.entropy_coefficients)), smoothed_ent, 
                            color='darkgreen', linewidth=2, label=f'Smoothed ({window})')
                plt.xlabel('Episode')
                plt.ylabel('Entropy Coefficient')
                plt.title('SAC Auto-Tuned Entropy Coefficient')
                plt.legend()
                plt.grid(True, alpha=0.3)
            
            # Learning curve analysis
            plt.subplot(3, 3, 8)
            if len(logger.episode_rewards) >= 25:
                # Plot smoothed learning curve
                window = min(25, len(logger.episode_rewards)//10)
                smoothed_rewards = []
                for i in range(window, len(logger.episode_rewards)):
                    smoothed_rewards.append(np.mean(logger.episode_rewards[i-window:i]))
                
                plt.plot(range(window, len(logger.episode_rewards)), smoothed_rewards, 
                        color='blue', linewidth=2, label=f'Smoothed ({window} ep window)')
                plt.axhline(y=-200, color='green', linestyle='--', alpha=0.7, label='Good Performance')
                plt.axhline(y=-500, color='orange', linestyle='--', alpha=0.7, label='Moderate Performance')
                plt.xlabel('Episode')
                plt.ylabel('Smoothed Reward')
                plt.title('Learning Curve (Smoothed)')
                plt.legend()
                plt.grid(True, alpha=0.3)
            
            # Training efficiency
            plt.subplot(3, 3, 9)
            if logger.episode_times:
                cumulative_time = np.cumsum(logger.episode_times)
                cumulative_time_minutes = [t/60 for t in cumulative_time]
                plt.plot(cumulative_time_minutes, logger.episode_rewards, alpha=0.6, color='brown')
                plt.xlabel('Training Time (Minutes)')
                plt.ylabel('Episode Reward')
                plt.title('Reward vs Training Time')
                plt.axhline(y=-200, color='green', linestyle='--', alpha=0.7)
                plt.axhline(y=-500, color='orange', linestyle='--', alpha=0.7)
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig("./pendulum/sac/weights/sac_pendulum_training.png", 
                       dpi=300, bbox_inches='tight')
            plt.show()
            
            # Print final training summary
            final_metrics = logger.get_performance_metrics()
            print("\n=== SAC PENDULUM TRAINING SUMMARY ===")
            print(f"Total Episodes: {final_metrics['basic_stats']['total_episodes']}")
            print(f"Total Training Steps: {final_metrics['basic_stats']['total_training_steps']:,}")
            print(f"Training Time: {training_time/60:.2f} minutes")
            print(f"Environment: Single environment with replay buffer")
            print(f"Final Reward: {final_metrics['basic_stats']['final_reward']:.2f}")
            print(f"Mean Reward: {final_metrics['basic_stats']['mean_reward']:.2f}")
            print(f"Max Reward: {final_metrics['basic_stats']['max_reward']:.2f}")
            print(f"Std Reward: {final_metrics['basic_stats']['std_reward']:.2f}")
            print(f"Mean Episode Length: {final_metrics['basic_stats']['mean_episode_length']:.1f}")
            
            if final_metrics['sample_efficiency']['threshold_achieved']:
                print(f"Threshold (-200) achieved at episode: {final_metrics['sample_efficiency']['episodes_to_threshold']}")
                print(f"Steps to threshold: {final_metrics['sample_efficiency']['steps_to_threshold']:,}")
            
            if final_metrics['sample_efficiency']['convergence_detected']:
                print(f"Convergence detected at episode: {final_metrics['sample_efficiency']['convergence_episode']}")
            
            print(f"Steps per second: {final_metrics['training_efficiency']['steps_per_second']:.1f}")
            
            # Pendulum specific insights
            good_performance = [r for r in logger.episode_rewards if r >= -200]
            moderate_performance = [r for r in logger.episode_rewards if -500 <= r < -200]
            poor_performance = [r for r in logger.episode_rewards if r < -500]
            
            print(f"\nPendulum Specific Metrics:")
            print(f"Good performance episodes (≥-200): {len(good_performance)}/{len(logger.episode_rewards)} ({100*len(good_performance)/len(logger.episode_rewards):.1f}%)")
            print(f"Moderate performance episodes (-500 to -200): {len(moderate_performance)}/{len(logger.episode_rewards)} ({100*len(moderate_performance)/len(logger.episode_rewards):.1f}%)")
            print(f"Poor performance episodes (<-500): {len(poor_performance)}/{len(logger.episode_rewards)} ({100*len(poor_performance)/len(logger.episode_rewards):.1f}%)")
            
            if good_performance:
                print(f"Average good performance reward: {np.mean(good_performance):.2f}")
            
            print(f"\nPendulum Control Analysis:")
            print(f"Dense reward environment - ideal for SAC")
            print(f"Continuous control challenge - SAC's specialty")
            print(f"Episode length: 200 steps (fixed)")
            print(f"Reward range: [-16.27, 0] per step (theoretical)")
            
            # SAC specific insights
            if logger.entropy_coefficients:
                final_ent_coef = logger.entropy_coefficients[-1]
                avg_ent_coef = np.mean(logger.entropy_coefficients)
                print(f"\nSAC Specific Metrics:")
                print(f"Final entropy coefficient: {final_ent_coef:.4f}")
                print(f"Average entropy coefficient: {avg_ent_coef:.4f}")
                print(f"Maximum entropy objective balances reward and exploration")
                print(f"Auto-tuned entropy coefficient adapts exploration dynamically")
                print(f"Off-policy learning with replay buffer enables sample efficiency")
                print(f"Double critic networks reduce Q-value overestimation")
        
    else:
        print("Loading trained SAC model...")
        try:
            model = SAC.load("./pendulum/sac/weights/sac_pendulum", env=env)
            print("Model loaded successfully!")
            
            # Test the model
            print("Testing the SAC agent...")
            test_episodes = 5
            test_rewards = []
            
            for episode in range(test_episodes):
                obs = env.reset()
                episode_reward = 0
                done = False
                step_count = 0
                max_steps = 200  # Pendulum episodes are 200 steps
                
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
            print(f"Good Performance Rate (≥-200): {100*sum(1 for r in test_rewards if r >= -200)/len(test_rewards):.1f}%")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Please train the model first by running with is_training=True")

    env.close()

if __name__ == "__main__":
    run(is_training=True)