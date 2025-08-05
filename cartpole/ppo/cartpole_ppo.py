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

# ============================================================================
# HYPERPARAMETERS - MODIFY THESE TO TUNE THE ALGORITHM
# ============================================================================
HYPERPARAMETERS = {
    # Learning parameters
    'learning_rate': 0.0001,        # Learning rate for optimizer
    'gamma': 0.99,                  # Discount factor
    'gae_lambda': 0.95,             # GAE lambda parameter
    
    # PPO specific parameters
    'clip_range': 0.2,              # PPO clipping parameter
    'ent_coef': 0.01,               # Entropy coefficient
    'vf_coef': 0.5,                 # Value function coefficient
    'max_grad_norm': 0.5,           # Gradient clipping
    
    # Training parameters
    'n_steps': 2048,                # Steps to collect before update
    'batch_size': 64,               # Mini-batch size for updates
    'n_epochs': 10,                 # Number of epochs per update
    
    # Network architecture
    'policy_arch': [64, 64],        # Policy network architecture
    'value_arch': [64, 64],         # Value network architecture
    
    # Training duration
    'total_timesteps': 100000,      # Total training timesteps
    'max_episodes': 2000,           # Maximum episodes
    'max_steps_per_episode': 500,   # Maximum steps per episode
    'update_frequency': 2048,       # Update after this many steps
}
# ============================================================================

class PPONetwork(nn.Module):
    """Combined policy and value network for PPO"""
    def __init__(self, state_size, action_size, policy_arch=[64, 64], value_arch=[64, 64]):
        super(PPONetwork, self).__init__()
        
        # Policy network (actor)
        policy_layers = []
        prev_size = state_size
        for hidden_size in policy_arch:
            policy_layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU()
            ])
            prev_size = hidden_size
        policy_layers.append(nn.Linear(prev_size, action_size))
        self.policy_net = nn.Sequential(*policy_layers)
        
        # Value network (critic)
        value_layers = []
        prev_size = state_size
        for hidden_size in value_arch:
            value_layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU()
            ])
            prev_size = hidden_size
        value_layers.append(nn.Linear(prev_size, 1))
        self.value_net = nn.Sequential(*value_layers)
        
    def forward(self, state):
        # Get policy logits and value
        policy_logits = self.policy_net(state)
        value = self.value_net(state)
        
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
                 max_grad_norm=0.5, gae_lambda=0.95, n_epochs=4, batch_size=64,
                 policy_arch=[64, 64], value_arch=[64, 64]):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.gae_lambda = gae_lambda
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Network and optimizer
        self.network = PPONetwork(state_size, action_size, policy_arch, value_arch).to(self.device)
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
    
    def compute_gae(self, next_value):
        """Compute Generalized Advantage Estimation"""
        advantages = []
        gae = 0
        
        # Add next_value for bootstrapping
        values = self.values + [next_value]
        
        for step in reversed(range(len(self.rewards))):
            delta = self.rewards[step] + self.gamma * values[step + 1] * (1 - self.dones[step]) - values[step]
            gae = delta + self.gamma * self.gae_lambda * (1 - self.dones[step]) * gae
            advantages.insert(0, gae)
            
        returns = [adv + val for adv, val in zip(advantages, self.values)]
        
        return advantages, returns
    
    def update(self, next_value):
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
        
        # PPO update for multiple epochs
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        
        dataset_size = len(states)
        
        for epoch in range(self.n_epochs):
            # Shuffle data for each epoch
            indices = torch.randperm(dataset_size)
            
            # Mini-batch updates
            for start_idx in range(0, dataset_size, self.batch_size):
                end_idx = min(start_idx + self.batch_size, dataset_size)
                batch_indices = indices[start_idx:end_idx]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Forward pass
                policy_logits, values = self.network(batch_states)
                values = values.squeeze()
                
                # Compute probabilities and log probabilities
                probs = F.softmax(policy_logits, dim=-1)
                dist = Categorical(probs)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # Compute ratio
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # Compute surrogate losses
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values, batch_returns)
                
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
        
        num_updates = self.n_epochs * ((dataset_size + self.batch_size - 1) // self.batch_size)
        
        return {
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy': total_entropy_loss / num_updates
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

def run(is_training=True):
    # Print current hyperparameters
    print("Current Hyperparameters:")
    print("=" * 50)
    for key, value in HYPERPARAMETERS.items():
        print(f"{key}: {value}")
    print("=" * 50)
    
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

    # Create PPO agent with hyperparameters
    agent = PPOAgent(
        state_size=state_size,
        action_size=action_size,
        lr=HYPERPARAMETERS['learning_rate'],
        gamma=HYPERPARAMETERS['gamma'],
        clip_eps=HYPERPARAMETERS['clip_range'],
        entropy_coef=HYPERPARAMETERS['ent_coef'],
        value_coef=HYPERPARAMETERS['vf_coef'],
        max_grad_norm=HYPERPARAMETERS['max_grad_norm'],
        gae_lambda=HYPERPARAMETERS['gae_lambda'],
        n_epochs=HYPERPARAMETERS['n_epochs'],
        batch_size=HYPERPARAMETERS['batch_size'],
        policy_arch=HYPERPARAMETERS['policy_arch'],
        value_arch=HYPERPARAMETERS['value_arch']
    )

    print("PPO Agent created with network architecture:")
    print(f"Policy: {HYPERPARAMETERS['policy_arch']}")
    print(f"Value: {HYPERPARAMETERS['value_arch']}")
    print(f"Training device: {agent.device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name()}")
    print()
    

    # Create directory for saving data
    os.makedirs('./cartpole/ppo/weights', exist_ok=True)

    if is_training:
        start_time = time.time()
        print("Starting PPO training...")
        
        # Training parameters from hyperparameters
        max_episodes = HYPERPARAMETERS['max_episodes']
        max_steps_per_episode = HYPERPARAMETERS['max_steps_per_episode']
        update_frequency = HYPERPARAMETERS['update_frequency']
        total_timesteps = HYPERPARAMETERS['total_timesteps']
        
        episode_rewards = []
        episode_lengths = []
        step_count = 0
        episode_count = 0
        
        steps_collected = 0
        
        while episode_count < max_episodes and step_count < total_timesteps:
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
                
                # Check if we've reached total timesteps
                if step_count >= total_timesteps:
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
        plt.savefig("./cartpole/ppo/weights/cartpole_ppo_training.png", dpi=300, bbox_inches='tight')
        plt.show()

        # Save the trained model
        agent.save_weights('./cartpole/ppo/weights/cartpole_ppo_weights.pth')
        print("Training completed and weights saved!")
        
        # Print final statistics
        final_100_avg = np.mean(episode_rewards[-100:])
        success_episodes = sum(1 for r in episode_rewards[-100:] if r >= 475)
        print(f"\nFinal Performance (last 100 episodes):")
        print(f"Average Reward: {final_100_avg:.2f}")
        print(f"Success Rate: {success_episodes}% (episodes with 475+ reward)")
        print(f"Max Reward: {max(episode_rewards[-100:])}")

        end_time = time.time()
        duration = end_time - start_time
        print(f"\nTotal training time: {duration:.2f} seconds")
        print(f"Total episodes: {episode_count}")
        print(f"Total timesteps: {step_count}")
        
    else:
        print("Loading trained weights...")
        # Load trained weights
        try:
            agent.load_weights('./cartpole/ppo/weights/cartpole_ppo_weights.pth')
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
    run(is_training=True)