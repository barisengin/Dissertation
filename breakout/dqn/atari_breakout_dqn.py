import gym
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Permute
from keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory
from rl.processors import Processor
import json
import time
import os
from datetime import datetime
from collections import deque
from PIL import Image

class PerformanceLogger:
    
    def __init__(self, save_dir="./breakout/dqn/json_data/"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize tracking variables
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_times = []
        self.training_losses = []
        self.exploration_rates = []
        
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
    
    def save_data(self, filename_prefix="dqn_breakout"):
        """Save all logged data to JSON files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Raw data
        raw_data = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'episode_times': self.episode_times,
            'exploration_rates': self.exploration_rates,
            'training_losses': self.training_losses,
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
    
class TrainingCallback:
    """Callback to track training progress and metrics"""
    
    def __init__(self, logger):
        self.logger = logger
        self.episode_start_time = None
        self.episode_reward = 0
        self.episode_length = 0
        self.current_episode = 0
        
    def on_episode_begin(self, episode, logs=None):
        self.episode_start_time = time.time()
        self.episode_reward = 0
        self.episode_length = 0
        self.current_episode = episode
        
    def on_step_end(self, step, logs=None):
        if logs:
            self.episode_reward += logs.get('reward', 0)
            self.episode_length += 1
            
            # Log training loss if available
            if 'loss' in logs:
                self.logger.log_training_loss(logs['loss'])
    
    def on_episode_end(self, episode, logs=None):
        if self.episode_start_time:
            episode_time = time.time() - self.episode_start_time
            
            # Get exploration rate from agent if available
            exploration_rate = getattr(logs, 'eps', 0.0) if logs else 0.0
            
            self.logger.log_episode(
                episode_num=episode,
                reward=self.episode_reward,
                episode_length=self.episode_length,
                training_time=episode_time,
                exploration_rate=exploration_rate,
                total_steps=logs.get('nb_steps', 0) if logs else 0
            )

class AtariProcessor(Processor):
    """Atari-specific image preprocessing"""
    
    def __init__(self):
        super(AtariProcessor, self).__init__()
        
    def process_observation(self, observation):
        """Process Atari frames: convert to grayscale and resize to 84x84"""
        assert observation.ndim == 3  # (height, width, channel)
        
        # Convert to PIL Image for easier processing
        img = Image.fromarray(observation)
        
        # Convert to grayscale
        img = img.convert('L')
        
        # Resize to 84x84
        img = img.resize((84, 84), Image.LANCZOS)
        
        # Convert back to numpy array
        processed_observation = np.array(img, dtype=np.uint8)
        
        # Normalize to [0, 1]
        processed_observation = processed_observation.astype(np.float32) / 255.0
        
        return processed_observation
    
    def process_state_batch(self, batch):
        """Process state batch for training"""
        # Input batch shape: (batch_size, window_length, height, width)
        processed_batch = np.array(batch, dtype=np.float32)
        return processed_batch
    
    def process_reward(self, reward):
        """Clip rewards to [-1, 1] as is standard for Atari"""
        return np.clip(reward, -1.0, 1.0)
    
class GymWrapper(gym.Wrapper):
    """Wrapper to convert new gym API to old API for keras-rl compatibility"""
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Convert new 5-value API to old 4-value API
        done = terminated or truncated
        return obs, reward, done, info
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs
    
def process_training_history(history, save_dir="./breakout/dqn/json_data/"):
    """Process keras-rl training history and generate comprehensive metrics"""
    
    if not hasattr(history, 'history') or 'episode_reward' not in history.history:
        print("No episode reward data found in history")
        return
    
    # Extract basic data from history
    episode_rewards = history.history['episode_reward']
    episode_lengths = history.history.get('episode_length', [])
    nb_steps = history.history.get('nb_steps', [])
    
    # Initialize logger and populate with historical data
    logger = PerformanceLogger(save_dir)
    
    # Process each episode
    for i, reward in enumerate(episode_rewards):
        episode_length = episode_lengths[i] if i < len(episode_lengths) else 0
        total_steps = nb_steps[i] if i < len(nb_steps) else i * 1000  # Estimate if not available
        
        # Log episode (using estimated values for missing data)
        logger.log_episode(
            episode_num=i,
            reward=reward,
            episode_length=episode_length,
            training_time=1.0,  # Estimated - we don't have actual timing from keras-rl
            exploration_rate=max(0.1, 1.0 - (i / len(episode_rewards)) * 0.9),  # Estimated decay
            total_steps=total_steps
        )
    
    # Save comprehensive metrics
    logger.save_data()
    
    # Print summary
    final_metrics = logger.get_performance_metrics()
    print("\n=== TRAINING SUMMARY ===")
    print(f"Total Episodes: {final_metrics['basic_stats']['total_episodes']}")
    print(f"Final Reward: {final_metrics['basic_stats']['final_reward']:.2f}")
    print(f"Mean Reward: {final_metrics['basic_stats']['mean_reward']:.2f}")
    print(f"Max Reward: {final_metrics['basic_stats']['max_reward']:.2f}")
    
    if final_metrics['sample_efficiency']['threshold_achieved']:
        print(f"Threshold achieved at episode: {final_metrics['sample_efficiency']['episodes_to_threshold']}")
        print(f"Steps to threshold: {final_metrics['sample_efficiency']['steps_to_threshold']}")
    
    if final_metrics['sample_efficiency']['convergence_detected']:
        print(f"Convergence detected at episode: {final_metrics['sample_efficiency']['convergence_episode']}")
    
    print("Comprehensive metrics saved to JSON files for later analysis.")
    
    return final_metrics

def run(is_training=True):
    # Create Atari Breakout environment
    base_env = gym.make("ALE/Breakout-v5", render_mode="human" if not is_training else None)
    env = GymWrapper(base_env)

    # Get state and action dimensions
    action_size = env.action_space.n  # 4 for Breakout
    input_shape = (84, 84)  # After preprocessing
    window_length = 4  # Frame stacking

    print(f"Action size: {action_size}")
    print(f"Input shape: {input_shape}")
    print(f"Window length: {window_length}")
    print(f"Original observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Test the environment to make sure it works
    test_obs = env.reset()
    print(f"Test observation shape: {np.array(test_obs).shape}")
    print(f"Test observation range: [{np.min(test_obs)}, {np.max(test_obs)}]")

    # Build the Convolutional Neural Network model (Nature DQN architecture)
    model = Sequential()
    model.add(Permute((2, 3, 1), input_shape=(window_length,) + input_shape))
    model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu'))
    model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(action_size, activation='linear'))

    print("Model architecture:")
    model.summary()

    # Configure memory for experience replay (large buffer for Atari)
    memory = SequentialMemory(limit=1000000, window_length=window_length)

    # Configure epsilon-greedy policy with linear annealing (extended for Atari)
    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(),
        attr='eps',
        value_max=1.0,        # Start with 100% exploration
        value_min=0.1,        # End with 10% exploration
        value_test=0.05,      # 5% exploration during testing
        nb_steps=1000000      # Anneal over 1 million steps
    )

    # Create processor
    processor = AtariProcessor()

    # Create the DQN agent (Atari-specific configuration)
    agent = DQNAgent(
        model=model,
        nb_actions=action_size,
        memory=memory,
        nb_steps_warmup=50000,    # Large warmup for Atari
        target_model_update=10000, # Hard update every 10k steps
        policy=policy,
        processor=processor,
        gamma=0.99,              # Standard discount factor
        train_interval=4,        # Train every 4 steps
        delta_clip=1.0           # Gradient clipping
    )

    # Compile the agent (lower learning rate for Atari)
    agent.compile(Adam(learning_rate=0.00025), metrics=['mae'])

    if is_training:
        # Create weights directory
        os.makedirs("./breakout/dqn/weights/", exist_ok=True)
        
        logger = PerformanceLogger()
        print("Starting training...")
        print("Note: Atari training takes significantly longer than other environments")
        
        # Train the agent (many steps for Atari)
        history = agent.fit(
            env,
            nb_steps=5000000,      # 5 million steps (standard for Atari)
            visualize=False,       # Set to True to watch training
            verbose=1,             
            log_interval=50000,    # Log every 50k steps
        )
        print("Available keys in history:", list(history.history.keys()) if hasattr(history, 'history') else "No history attribute")
        
        process_training_history(history)

        # Plot training results
        if hasattr(history, 'history') and 'episode_reward' in history.history:
            episode_rewards = history.history['episode_reward']
            
            plt.figure(figsize=(15, 5))
            
            # Plot episode rewards
            plt.subplot(1, 3, 1)
            plt.plot(episode_rewards, alpha=0.6, label='Episode Reward')
            
            # Calculate and plot moving average
            if len(episode_rewards) >= 100:
                moving_avg = []
                for i in range(len(episode_rewards)):
                    start_idx = max(0, i-99)
                    moving_avg.append(np.mean(episode_rewards[start_idx:i+1]))
                plt.plot(moving_avg, color='red', linewidth=2, label='Moving Average (100 episodes)')
            
            # Add convergence threshold line
            plt.axhline(y=300, color='green', linestyle='--', alpha=0.7, label='Good Performance Threshold (300)')
            
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.title('Training Progress: Episode Rewards')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Plot learning curve (if available)
            plt.subplot(1, 3, 2)
            if 'nb_steps' in history.history:
                steps_in_millions = [s/1000000 for s in history.history['nb_steps']]
                plt.plot(steps_in_millions, episode_rewards, alpha=0.6)
                plt.xlabel('Training Steps (Millions)')
                plt.ylabel('Episode Reward')
                plt.title('Reward vs Training Steps')
                plt.axhline(y=300, color='green', linestyle='--', alpha=0.7)
                plt.grid(True, alpha=0.3)
            
            # Plot reward distribution
            plt.subplot(1, 3, 3)
            plt.hist(episode_rewards, bins=30, alpha=0.7, edgecolor='black')
            plt.axvline(np.mean(episode_rewards), color='red', linestyle='--', label=f'Mean: {np.mean(episode_rewards):.1f}')
            plt.axvline(300, color='green', linestyle='--', label='Threshold: 300')
            plt.xlabel('Episode Reward')
            plt.ylabel('Frequency')
            plt.title('Reward Distribution')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig("./breakout/dqn/weights/breakout_dqn_training.png", dpi=300, bbox_inches='tight')
            plt.show()

            # Save the trained model
            agent.save_weights('./breakout/dqn/weights/breakout_dqn_weights.h5f', overwrite=True)
            print("Training completed and weights saved!")
        
    else:
        print("Loading trained weights...")
        # Load trained weights
        agent.load_weights('./breakout/dqn/weights/breakout_dqn_weights.h5f')
        
        # Test the agent
        print("Testing the agent...")
        test_history = agent.test(
            env,
            nb_episodes=10,
            visualize=True
        )
        
        print(f"Average test reward: {np.mean(test_history.history['episode_reward'])}")

    env.close()

if __name__ == "__main__":
    run(is_training=True)