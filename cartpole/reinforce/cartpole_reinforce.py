import gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import json
import time
import os
from datetime import datetime
from collections import deque
import warnings
warnings.filterwarnings('ignore')

class PerformanceLogger:
    
    def __init__(self, save_dir="./cartpole/reinforce/json_data/"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize tracking variables
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_times = []
        self.policy_losses = []
        self.baseline_losses = []
        self.timesteps = []
        
        # Performance metrics (CartPole specific)
        self.convergence_threshold = 195.0  # CartPole considered solved at 195
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
                   total_steps, policy_loss=None, baseline_loss=None):
        """Log data for a completed episode"""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(episode_length)
        self.episode_times.append(training_time)
        self.timesteps.append(total_steps)
        self.total_training_steps = total_steps
        
        # Log REINFORCE-specific losses if provided
        if policy_loss is not None:
            self.policy_losses.append(policy_loss)
        if baseline_loss is not None:
            self.baseline_losses.append(baseline_loss)
        
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
            'reinforce_specific_metrics': {
                'policy_losses': self.policy_losses,
                'baseline_losses': self.baseline_losses
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
    
    def save_data(self, filename_prefix="reinforce_cartpole"):
        """Save all logged data to JSON files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Raw data
        raw_data = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'episode_times': self.episode_times,
            'policy_losses': self.policy_losses,
            'baseline_losses': self.baseline_losses,
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

class PolicyNetwork(nn.Module):
    """Simple policy network for REINFORCE"""
    
    def __init__(self, state_size, action_size, hidden_size=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)

class BaselineNetwork(nn.Module):
    """Value network for baseline subtraction"""
    
    def __init__(self, state_size, hidden_size=128):
        super(BaselineNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class REINFORCEAgent:
    """REINFORCE algorithm implementation"""
    
    def __init__(self, state_size, action_size, lr=1e-3, gamma=0.99, use_baseline=True):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.use_baseline = use_baseline
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize networks
        self.policy_net = PolicyNetwork(state_size, action_size).to(self.device)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        if use_baseline:
            self.baseline_net = BaselineNetwork(state_size).to(self.device)
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
    
    def save_model(self, filepath):
        """Save model weights"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'baseline_net_state_dict': self.baseline_net.state_dict() if self.use_baseline else None,
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'baseline_optimizer_state_dict': self.baseline_optimizer.state_dict() if self.use_baseline else None,
        }, filepath)
    
    def load_model(self, filepath):
        """Load model weights"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        if self.use_baseline and checkpoint['baseline_net_state_dict'] is not None:
            self.baseline_net.load_state_dict(checkpoint['baseline_net_state_dict'])

def run(is_training=True):
    print("Setting up CartPole with REINFORCE...")
    
    # Create CartPole environment
    env = gym.make('CartPole-v1')
    
    # Get environment info
    state_size = env.observation_space.shape[0]  # 4 for CartPole
    action_size = env.action_space.n  # 2 for CartPole

    print(f"State size: {state_size}")
    print(f"Action size: {action_size}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Create directories
    os.makedirs("./cartpole/reinforce/weights/", exist_ok=True)
    os.makedirs("./cartpole/reinforce/json_data/", exist_ok=True)

    if is_training:
        logger = PerformanceLogger()
        
        # Create REINFORCE agent
        agent = REINFORCEAgent(
            state_size=state_size,
            action_size=action_size,
            lr=1e-3,                    # Learning rate
            gamma=0.99,                 # Discount factor
            use_baseline=True           # Use baseline to reduce variance
        )
        
        print("REINFORCE Agent created with configuration:")
        print(f"- Algorithm: REINFORCE (vanilla policy gradient)")
        print(f"- Learning Rate: 1e-3")
        print(f"- Discount Factor: 0.99")
        print(f"- Baseline: Enabled (reduces variance)")
        print(f"- Network Architecture: [4] -> [128] -> [128] -> [2]")
        print(f"- Device: {agent.device}")
        print(f"- Optimization: Adam optimizer")
        
        print("\nStarting training...")
        print("Note: REINFORCE has high variance but is theoretically sound")
        print("Expected to see noisy learning but eventual convergence")
        
        # Training parameters
        max_episodes = 2000
        total_steps = 0
        
        start_time = time.time()
        
        for episode in range(max_episodes):
            episode_start_time = time.time()
            state, _ = env.reset()
            episode_reward = 0
            episode_length = 0
            
            # Run episode
            while True:
                action = agent.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                agent.store_reward(reward)
                episode_reward += reward
                episode_length += 1
                total_steps += 1
                
                if done:
                    break
                    
                state = next_state
            
            # Update policy after episode completion
            policy_loss, baseline_loss = agent.update_policy()
            
            # Log episode data
            episode_time = time.time() - episode_start_time
            logger.log_episode(
                episode_num=episode,
                reward=episode_reward,
                episode_length=episode_length,
                training_time=episode_time,
                total_steps=total_steps,
                policy_loss=policy_loss,
                baseline_loss=baseline_loss
            )
            
            # Print progress
            if episode % 100 == 0:
                recent_rewards = logger.episode_rewards[-100:] if len(logger.episode_rewards) >= 100 else logger.episode_rewards
                avg_reward = np.mean(recent_rewards) if recent_rewards else 0
                print(f"Episode {episode}, Avg Reward (100 ep): {avg_reward:.2f}, Total Steps: {total_steps}")
            
            # Check for early convergence
            if logger.convergence_detected:
                print(f"Training completed early due to convergence at episode {episode}")
                break
        
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time/60:.2f} minutes")
        
        # Save the model
        agent.save_model("./cartpole/reinforce/weights/reinforce_cartpole.pth")
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
            
            plt.axhline(y=195, color='green', linestyle='--', alpha=0.7, label='Convergence Threshold (195)')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.title('Training Progress: Episode Rewards')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Rewards vs timesteps
            plt.subplot(2, 3, 2)
            if logger.timesteps:
                timesteps_thousands = [t/1000 for t in logger.timesteps]
                plt.plot(timesteps_thousands, logger.episode_rewards, alpha=0.6)
                plt.xlabel('Training Steps (Thousands)')
                plt.ylabel('Episode Reward')
                plt.title('Reward vs Training Steps')
                plt.axhline(y=195, color='green', linestyle='--', alpha=0.7)
                plt.grid(True, alpha=0.3)
            
            # Reward distribution
            plt.subplot(2, 3, 3)
            plt.hist(logger.episode_rewards, bins=30, alpha=0.7, edgecolor='black')
            plt.axvline(np.mean(logger.episode_rewards), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(logger.episode_rewards):.1f}')
            plt.axvline(195, color='green', linestyle='--', label='Target: 195')
            plt.xlabel('Episode Reward')
            plt.ylabel('Frequency')
            plt.title('Reward Distribution')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Policy Loss (REINFORCE-specific)
            plt.subplot(2, 3, 4)
            if logger.policy_losses:
                episodes_with_loss = range(len(logger.policy_losses))
                plt.plot(episodes_with_loss, logger.policy_losses, alpha=0.7, color='orange', label='Policy Loss')
                if len(logger.policy_losses) >= 20:
                    window = min(20, len(logger.policy_losses)//5)
                    smoothed_policy = []
                    for i in range(window, len(logger.policy_losses)):
                        smoothed_policy.append(np.mean(logger.policy_losses[i-window:i]))
                    plt.plot(range(window, len(logger.policy_losses)), smoothed_policy, 
                            color='darkorange', linewidth=2, label=f'Smoothed ({window})')
                plt.xlabel('Episode')
                plt.ylabel('Policy Loss')
                plt.title('REINFORCE Policy Loss Over Time')
                plt.legend()
                plt.grid(True, alpha=0.3)
            
            # Baseline Loss (if using baseline)
            plt.subplot(2, 3, 5)
            if logger.baseline_losses and any(x is not None for x in logger.baseline_losses):
                valid_losses = [x for x in logger.baseline_losses if x is not None]
                episodes_with_baseline = range(len(valid_losses))
                plt.plot(episodes_with_baseline, valid_losses, alpha=0.7, color='blue', label='Baseline Loss')
                if len(valid_losses) >= 20:
                    window = min(20, len(valid_losses)//5)
                    smoothed_baseline = []
                    for i in range(window, len(valid_losses)):
                        smoothed_baseline.append(np.mean(valid_losses[i-window:i]))
                    plt.plot(range(window, len(valid_losses)), smoothed_baseline, 
                            color='darkblue', linewidth=2, label=f'Smoothed ({window})')
                plt.xlabel('Episode')
                plt.ylabel('Baseline Loss')
                plt.title('Baseline Network Loss Over Time')
                plt.legend()
                plt.grid(True, alpha=0.3)
            
            # Learning curve analysis
            plt.subplot(2, 3, 6)
            if len(logger.episode_rewards) >= 100:
                # Plot smoothed learning curve
                window = min(50, len(logger.episode_rewards)//10)
                smoothed_rewards = []
                for i in range(window, len(logger.episode_rewards)):
                    smoothed_rewards.append(np.mean(logger.episode_rewards[i-window:i]))
                
                plt.plot(range(window, len(logger.episode_rewards)), smoothed_rewards, 
                        color='blue', linewidth=2, label=f'Smoothed ({window} ep window)')
                plt.axhline(y=195, color='green', linestyle='--', alpha=0.7, label='Target')
                plt.xlabel('Episode')
                plt.ylabel('Smoothed Reward')
                plt.title('Learning Curve (Smoothed)')
                plt.legend()
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig("./cartpole/reinforce/weights/reinforce_cartpole_training.png", 
                       dpi=300, bbox_inches='tight')
            plt.show()
            
            # Print final training summary
            final_metrics = logger.get_performance_metrics()
            print("\n=== REINFORCE CARTPOLE TRAINING SUMMARY ===")
            print(f"Total Episodes: {final_metrics['basic_stats']['total_episodes']}")
            print(f"Total Training Steps: {final_metrics['basic_stats']['total_training_steps']:,}")
            print(f"Training Time: {training_time/60:.2f} minutes")
            print(f"Final Reward: {final_metrics['basic_stats']['final_reward']:.2f}")
            print(f"Mean Reward: {final_metrics['basic_stats']['mean_reward']:.2f}")
            print(f"Max Reward: {final_metrics['basic_stats']['max_reward']:.2f}")
            print(f"Std Reward: {final_metrics['basic_stats']['std_reward']:.2f}")
            print(f"Mean Episode Length: {final_metrics['basic_stats']['mean_episode_length']:.1f}")
            
            if final_metrics['sample_efficiency']['threshold_achieved']:
                print(f"Threshold (195) achieved at episode: {final_metrics['sample_efficiency']['episodes_to_threshold']}")
                print(f"Steps to threshold: {final_metrics['sample_efficiency']['steps_to_threshold']:,}")
            
            if final_metrics['sample_efficiency']['convergence_detected']:
                print(f"Convergence detected at episode: {final_metrics['sample_efficiency']['convergence_episode']}")
            
            print(f"Steps per second: {final_metrics['training_efficiency']['steps_per_second']:.1f}")
            
            # REINFORCE specific insights
            print(f"\nREINFORCE Specific Metrics:")
            print(f"Policy gradient algorithm with Monte Carlo returns")
            print(f"Baseline network used for variance reduction: {'Yes' if agent.use_baseline else 'No'}")
            if logger.policy_losses:
                avg_policy_loss = np.mean([x for x in logger.policy_losses if x is not None])
                print(f"Average policy loss: {avg_policy_loss:.4f}")
            print(f"Learning stability: {'High variance expected' if not agent.use_baseline else 'Reduced variance with baseline'}")
    
    else:
        print("Loading trained REINFORCE model...")
        try:
            # Create agent with same configuration
            agent = REINFORCEAgent(
                state_size=state_size,
                action_size=action_size,
                lr=1e-3,
                gamma=0.99,
                use_baseline=True
            )
            
            # Load model
            agent.load_model("./cartpole/reinforce/weights/reinforce_cartpole.pth")
            print("Model loaded successfully!")
            
            # Test the model
            print("Testing the REINFORCE agent...")
            test_episodes = 10
            test_rewards = []
            
            for episode in range(test_episodes):
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
    run(is_training=False)