import gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import os
import time

class PPONetwork(nn.Module):
    """Combined policy and value network for PPO"""
    def __init__(self, state_size, action_size, hidden_size=128):
        super(PPONetwork, self).__init__()
        
        # Shared layers
        self.shared_layers = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 64),
            nn.ReLU()
        )
        
        # Policy head (actor)
        self.policy_head = nn.Linear(64, action_size)
        
        # Value head (critic)
        self.value_head = nn.Linear(64, 1)
        
    def forward(self, state):
        x = self.shared_layers(state)
        
        # Get policy logits and value
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        
        return policy_logits, value
    
    def get_action_and_value(self, state):
        policy_logits, value = self.forward(state)
        probs = F.softmax(policy_logits, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        
        return action.item(), dist.log_prob(action), value, dist.entropy()

class PPOAgent:
    """PPO Agent implementation"""
    def __init__(self, state_size, action_size, lr=3e-4, gamma=0.99, 
                 clip_eps=0.2, entropy_coef=0.01, value_coef=0.5, 
                 max_grad_norm=0.5):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Network and optimizer
        self.network = PPONetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        # Storage for episode data
        self.reset_storage()
        
    def reset_storage(self):
        """Reset storage for new episode collection"""
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        
    def get_action(self, state):
        """Get action from current policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action, log_prob, value, entropy = self.network.get_action_and_value(state_tensor)
        
        return action, log_prob.item(), value.item()
    
    def store_transition(self, state, action, log_prob, value, reward, done):
        """Store transition data"""
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
    
    def compute_gae(self, next_value, gae_lambda=0.95):
        """Compute Generalized Advantage Estimation"""
        advantages = []
        gae = 0
        
        # Add next_value for bootstrapping
        values = self.values + [next_value]
        
        for step in reversed(range(len(self.rewards))):
            delta = self.rewards[step] + self.gamma * values[step + 1] * (1 - self.dones[step]) - values[step]
            gae = delta + self.gamma * gae_lambda * (1 - self.dones[step]) * gae
            advantages.insert(0, gae)
            
        returns = [adv + val for adv, val in zip(advantages, self.values)]
        
        return advantages, returns
    
    def update(self, next_value, ppo_epochs=4, batch_size=64):
        """Update policy using PPO"""
        if len(self.states) == 0:
            return {}
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(next_value)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        
        for _ in range(ppo_epochs):
            # Forward pass
            policy_logits, values = self.network(states)
            values = values.squeeze()
            
            # Compute probabilities and log probabilities
            probs = F.softmax(policy_logits, dim=-1)
            dist = Categorical(probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            # Compute ratio
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # Compute surrogate losses
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(values, returns)
            
            # Total loss
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy_loss += entropy.item()
        
        # Reset storage
        self.reset_storage()
        
        return {
            'policy_loss': total_policy_loss / ppo_epochs,
            'value_loss': total_value_loss / ppo_epochs,
            'entropy': total_entropy_loss / ppo_epochs
        }
    
    def save_weights(self, filepath):
        """Save model weights"""
        torch.save(self.network.state_dict(), filepath)
        
    def load_weights(self, filepath):
        """Load model weights"""
        self.network.load_state_dict(torch.load(filepath, map_location=self.device))

class GymWrapper(gym.Wrapper):
    """Wrapper to ensure compatibility and handle new gym API"""
    def step(self, action):
        if hasattr(self.env, '_max_episode_steps'):
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            return obs, reward, done, info
        else:
            # Older gym API
            return self.env.step(action)
    
    def reset(self, **kwargs):
        if hasattr(self.env, 'reset'):
            result = self.env.reset(**kwargs)
            if isinstance(result, tuple):
                obs, info = result
                return obs
            return result

def test_agent_performance(agent, env, num_test_episodes=20, max_steps=500):
    """Test agent performance and return detailed statistics"""
    test_rewards = []
    test_lengths = []
    
    for episode in range(num_test_episodes):
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        
        for step in range(max_steps):
            action, _, _ = agent.get_action(state)
            state, reward, done, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            
            if done:
                break
        
        test_rewards.append(episode_reward)
        test_lengths.append(episode_length)
    
    # Calculate statistics
    avg_reward = np.mean(test_rewards)
    std_reward = np.std(test_rewards)
    success_rate = sum(1 for r in test_rewards if r >= 475) / len(test_rewards)
    max_reward_rate = sum(1 for r in test_rewards if r >= 500) / len(test_rewards)
    
    return {
        'avg_reward': avg_reward,
        'std_reward': std_reward,
        'success_rate': success_rate,
        'max_reward_rate': max_reward_rate,
        'test_rewards': test_rewards,
        'test_lengths': test_lengths
    }

def check_convergence_confidence(agent, env, confidence_threshold=0.6, min_episodes=500):
    """
    Check if we're confident the model will achieve max reward in >50% of test runs
    
    Args:
        agent: PPO agent to test
        env: Environment
        confidence_threshold: Required success rate to consider converged
        min_episodes: Minimum training episodes before checking convergence
    
    Returns:
        dict with convergence info
    """
    # Test current performance
    test_results = test_agent_performance(agent, env, num_test_episodes=30)
    
    # Confidence criteria:
    # 1. Success rate (475+ reward) >= confidence_threshold
    # 2. Average reward >= 450 (high performance)
    # 3. Max reward rate (500 reward) >= 0.5 (50% achieve maximum)
    
    is_converged = (
        test_results['success_rate'] >= confidence_threshold and
        test_results['avg_reward'] >= 450 and
        test_results['max_reward_rate'] >= 0.5
    )
    
    return {
        'is_converged': is_converged,
        'test_results': test_results,
        'criteria_met': {
            'success_rate_met': test_results['success_rate'] >= confidence_threshold,
            'avg_reward_met': test_results['avg_reward'] >= 450,
            'max_reward_rate_met': test_results['max_reward_rate'] >= 0.5
        }
    }

def run(is_training=True):
    # Create CartPole environment
    base_env = gym.make("CartPole-v1", render_mode="human" if not is_training else None)
    env = GymWrapper(base_env)

    # Get state and action dimensions
    state_size = env.observation_space.shape[0]  # 4 for CartPole
    action_size = env.action_space.n  # 2 for CartPole

    print(f"State size: {state_size}")
    print(f"Action size: {action_size}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Test the environment
    test_obs = env.reset()
    print(f"Test observation shape: {np.array(test_obs).shape}")
    print(f"Test observation: {test_obs}")

    # Create PPO agent
    agent = PPOAgent(
        state_size=state_size,
        action_size=action_size,
        lr=3e-4,              # Learning rate
        gamma=0.99,           # Discount factor
        clip_eps=0.2,         # PPO clipping parameter
        entropy_coef=0.01,    # Entropy coefficient
        value_coef=0.5,       # Value loss coefficient
        max_grad_norm=0.5     # Gradient clipping
    )

    print("PPO Agent created with network architecture:")
    print(agent.network)
    print(f"Training device: {agent.device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name()}")
    print()
    

    # Create directory for saving data
    os.makedirs('./cartpole/ppo/generated_data', exist_ok=True)

    if is_training:
        print("Starting PPO training...")
        
        # Training parameters
        max_episodes = 2000
        max_steps_per_episode = 500
        update_frequency = 2048  # Update after collecting this many steps
        
        episode_rewards = []
        episode_lengths = []
        step_count = 0
        episode_count = 0
        
        steps_collected = 0
        
        while episode_count < max_episodes:
            state = env.reset()
            episode_reward = 0
            episode_length = 0
            
            for step in range(max_steps_per_episode):
                # Get action from agent
                action, log_prob, value = agent.get_action(state)
                
                # Take step in environment
                next_state, reward, done, info = env.step(action)
                
                # Store transition
                agent.store_transition(state, action, log_prob, value, reward, done)
                
                state = next_state
                episode_reward += reward
                episode_length += 1
                step_count += 1
                steps_collected += 1
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            episode_count += 1
            
            # Update agent if enough steps collected
            if steps_collected >= update_frequency:
                # Get next state value for bootstrapping
                if done:
                    next_value = 0
                else:
                    _, _, next_value = agent.get_action(state)
                
                # Update agent
                loss_info = agent.update(next_value)
                steps_collected = 0
                
                # Log progress
                if episode_count % 25 == 0:
                    avg_reward = np.mean(episode_rewards[-100:])
                    avg_length = np.mean(episode_lengths[-100:])
                    print(f"Episode {episode_count}, Avg Reward: {avg_reward:.2f}, "
                          f"Avg Length: {avg_length:.2f}, Steps: {step_count}")
                    
                    if 'policy_loss' in loss_info:
                        print(f"  Policy Loss: {loss_info['policy_loss']:.4f}, "
                              f"Value Loss: {loss_info['value_loss']:.4f}, "
                              f"Entropy: {loss_info['entropy']:.4f}")

        # Plot training results
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
        
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('PPO Training Progress: Episode Rewards')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot episode lengths
        plt.subplot(1, 3, 2)
        plt.plot(episode_lengths, alpha=0.6, label='Episode Length')
        
        if len(episode_lengths) >= 100:
            moving_avg_length = []
            for i in range(len(episode_lengths)):
                start_idx = max(0, i-99)
                moving_avg_length.append(np.mean(episode_lengths[start_idx:i+1]))
            plt.plot(moving_avg_length, color='red', linewidth=2, label='Moving Average (100 episodes)')
        
        plt.xlabel('Episode')
        plt.ylabel('Episode Length')
        plt.title('PPO Training Progress: Episode Lengths')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot success rate (episodes reaching 475+ reward)
        plt.subplot(1, 3, 3)
        success_rate = []
        for i in range(len(episode_rewards)):
            start_idx = max(0, i-99)
            recent_rewards = episode_rewards[start_idx:i+1]
            success_rate.append(sum(1 for r in recent_rewards if r >= 475) / len(recent_rewards) * 100)
        
        plt.plot(success_rate, color='green', linewidth=2)
        plt.xlabel('Episode')
        plt.ylabel('Success Rate (%)')
        plt.title('Success Rate (475+ reward)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("./cartpole/ppo/generated_data/cartpole_ppo_training.png", dpi=300, bbox_inches='tight')
        plt.show()

        # Save the trained model
        agent.save_weights('./cartpole/ppo/generated_data/cartpole_ppo_weights.pth')
        print("Training completed and weights saved!")
        
        # Print final statistics
        final_100_avg = np.mean(episode_rewards[-100:])
        success_episodes = sum(1 for r in episode_rewards[-100:] if r >= 475)
        print(f"\nFinal Performance (last 100 episodes):")
        print(f"Average Reward: {final_100_avg:.2f}")
        print(f"Success Rate: {success_episodes}% (episodes with 475+ reward)")
        print(f"Max Reward: {max(episode_rewards[-100:])}")
        
    else:
        print("Loading trained weights...")
        # Load trained weights
        try:
            agent.load_weights('./cartpole/ppo/generated_data/cartpole_ppo_weights.pth')
            print("Weights loaded successfully!")
        except FileNotFoundError:
            print("No trained weights found. Please train the model first.")
            return
        
        # Test the agent
        print("Testing the agent...")
        test_rewards = []
        test_lengths = []
        
        for episode in range(10):
            state = env.reset()
            episode_reward = 0
            episode_length = 0
            
            for step in range(500):
                action, _, _ = agent.get_action(state)
                state, reward, done, info = env.step(action)
                episode_reward += reward
                episode_length += 1
                
                if done:
                    break
            
            test_rewards.append(episode_reward)
            test_lengths.append(episode_length)
            print(f"Test Episode {episode + 1}: Reward = {episode_reward}, Length = {episode_length}")
        
        print(f"\nTest Results:")
        print(f"Average Reward: {np.mean(test_rewards):.2f} ± {np.std(test_rewards):.2f}")
        print(f"Average Length: {np.mean(test_lengths):.2f} ± {np.std(test_lengths):.2f}")
        print(f"Success Rate: {sum(1 for r in test_rewards if r >= 475)}0% (episodes with 475+ reward)")

    env.close()

if __name__ == "__main__":
    # Set training mode here
    run(is_training=True)  # Change to False for testing