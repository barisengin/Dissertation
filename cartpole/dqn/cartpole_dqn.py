import gym
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory
from rl.processors import Processor

class CartPoleProcessor(Processor):
    """Processor to ensure observations are in the correct format"""
    def process_observation(self, observation):
        # Ensure observation is a 1D numpy array
        processed_obs = np.array(observation, dtype=np.float32).flatten()
        return processed_obs
    
    def process_state_batch(self, batch):
        # Handle batch processing to ensure correct dimensions
        # batch might have shape (batch_size, window_length, obs_dim)
        # We want (batch_size, obs_dim) for CartPole
        batch = np.array(batch)
        if batch.ndim == 3 and batch.shape[1] == 1:
            # Remove the window dimension if it's 1
            batch = batch.squeeze(axis=1)
        return batch
    
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

    # Test the environment to make sure it works
    test_obs = env.reset()
    print(f"Test observation shape: {np.array(test_obs).shape}")
    print(f"Test observation: {test_obs}")

    # Build the neural network model
    model = Sequential()
    model.add(Dense(128, input_dim=state_size, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(action_size, activation='linear'))

    print("Model architecture:")
    model.summary()

    # Configure memory for experience replay
    memory = SequentialMemory(limit=50000, window_length=1)

    # Configure epsilon-greedy policy with linear annealing
    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(),
        attr='eps',
        value_max=1.0,        # Start with 100% exploration
        value_min=0.1,       # End with 10% exploration
        value_test=0.0,       # No exploration during testing
        nb_steps=10000        # Anneal over 10,000 steps
    )

    # Create processor
    processor = CartPoleProcessor()

    # Create the DQN agent
    agent = DQNAgent(
        model=model,
        nb_actions=action_size,
        memory=memory,
        nb_steps_warmup=1000,     # Steps before training starts
        target_model_update=0.01, # Soft update rate for target network
        policy=policy,
        processor=processor,     # Add the processor
        gamma=0.99,              # Discount factor
        train_interval=4,        # Train every 4 steps
        delta_clip=1.0           # Gradient clipping
    )

    # Compile the agent
    agent.compile(Adam(learning_rate=0.0005), metrics=['mae'])

    if is_training:
        print("Starting training...")
        # Train the agent
        history = agent.fit(
            env,
            nb_steps=50000,       # Total training steps
            visualize=False,      # Set to True to watch training
            verbose=1,            # Reduce verbosity
            log_interval=5000,     # Log every 5000 steps
        )

        # Plot training results
        if hasattr(history, 'history') and 'episode_reward' in history.history:
            episode_rewards = history.history['episode_reward']
            
            plt.figure(figsize=(12, 5))
            
            # Plot episode rewards
            plt.subplot(1, 2, 1)
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
            plt.title('Training Progress: Episode Rewards')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Plot learning curve (if available)
            plt.subplot(1, 2, 2)
            if 'nb_steps' in history.history:
                plt.plot(history.history['nb_steps'], episode_rewards, alpha=0.6)
                plt.xlabel('Training Steps')
                plt.ylabel('Episode Reward')
                plt.title('Reward vs Training Steps')
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig("./cartpole/dqn/generated_data/cartpole_dqn_training.png", dpi=300, bbox_inches='tight')
            plt.show()

            # Save the trained model
            agent.save_weights('./cartpole/dqn/generated_data/cartpole_dqn_weights_tuned.h5f', overwrite=True)
            print("Training completed and weights saved!")
        
    else:
        print("Loading trained weights...")
        # Load trained weights
        agent.load_weights('./cartpole/dqn/generated_data/cartpole_dqn_weights_tuned.h5f')
        
        # Test the agent
        print("Testing the agent...")
        test_history = agent.test(
            env,
            nb_episodes=10,
            visualize=False
        )
        
        print(f"Average test reward: {np.mean(test_history.history['episode_reward'])}")

    env.close()

if __name__ == "__main__":
    run(is_training=True)