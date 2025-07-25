import numpy as np
import random
from grid_world import GridWorld
import time
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any
import pickle
import os

class QLearningAgent:
    def __init__(self, state_size: int, action_size: int, learning_rate: float = 0.1, 
                 discount_factor: float = 0.95, epsilon: float = 0.1, epsilon_decay: float = 0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = 0.01
        
        # Initialize Q-table as dictionary for sparse storage
        self.q_table = {}
        
        # Training statistics
        self.episode_rewards = []
        self.episode_steps = []
        self.success_rates = []
        self.epsilon_history = []
    
    def get_state_key(self, state):
        """Convert state array to a hashable key"""
        return tuple(state.flatten())
    
    def get_action(self, state, training: bool = True):
        """Choose action using epsilon-greedy policy"""
        state_key = self.get_state_key(state)
        
        # Initialize Q-values for this state if not seen before
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        
        if training and random.random() < self.epsilon:
            # Exploration: random action
            return random.randint(0, self.action_size - 1)
        else:
            # Exploitation: best action
            return np.argmax(self.q_table[state_key])
    
    def update(self, state, action, reward, next_state, done):
        """Update Q-values using Q-learning update rule"""
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        
        # Initialize Q-values if not seen before
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_size)
        
        # Q-learning update
        current_q = self.q_table[state_key][action]
        if done:
            max_next_q = 0
        else:
            max_next_q = np.max(self.q_table[next_state_key])
        
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state_key][action] = new_q
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save(self, filename: str = "q_learning_agent.pkl"):
        """Save the trained agent to a file"""
        # Create agents directory if it doesn't exist
        agents_dir = "agents"
        if not os.path.exists(agents_dir):
            os.makedirs(agents_dir)
        
        # Save to agents folder
        filepath = os.path.join(agents_dir, filename)
        
        agent_data = {
            'q_table': self.q_table,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'epsilon': self.epsilon,
            'epsilon_decay': self.epsilon_decay,
            'epsilon_min': self.epsilon_min,
            'action_size': self.action_size,
            'state_size': self.state_size,
            'episode_rewards': self.episode_rewards,
            'episode_steps': self.episode_steps,
            'success_rates': self.success_rates,
            'epsilon_history': self.epsilon_history
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(agent_data, f)
        
        print(f"✅ Agent saved to {filepath}")
        print(f"   Q-table size: {len(self.q_table)} entries")
        print(f"   Training episodes: {len(self.episode_rewards)}")
    
    @classmethod
    def load(cls, filename: str = "q_learning_agent.pkl"):
        """Load a trained agent from a file"""
        # Check in agents directory first
        agents_dir = "agents"
        filepath = os.path.join(agents_dir, filename)
        
        if not os.path.exists(filepath):
            # Fallback to current directory
            if not os.path.exists(filename):
                raise FileNotFoundError(f"Agent file {filepath} not found!")
            filepath = filename
        
        with open(filepath, 'rb') as f:
            agent_data = pickle.load(f)
        
        # Create agent instance
        agent = cls(
            state_size=agent_data['state_size'],
            action_size=agent_data['action_size'],
            learning_rate=agent_data['learning_rate'],
            discount_factor=agent_data['discount_factor'],
            epsilon=agent_data['epsilon'],
            epsilon_decay=agent_data['epsilon_decay']
        )
        
        # Restore agent data
        agent.q_table = agent_data['q_table']
        agent.epsilon_min = agent_data['epsilon_min']
        agent.episode_rewards = agent_data['episode_rewards']
        agent.episode_steps = agent_data['episode_steps']
        agent.success_rates = agent_data['success_rates']
        agent.epsilon_history = agent_data['epsilon_history']
        
        print(f"✅ Agent loaded from {filepath}")
        print(f"   Q-table size: {len(agent.q_table)} entries")
        print(f"   Training episodes: {len(agent.episode_rewards)}")
        print(f"   Final epsilon: {agent.epsilon:.3f}")
        
        return agent

def train_agent(episodes: int = 1000, render_every: int = 1000, env_size: int = 5, seed: int = 42, save_every: int = 10000):
    """Train the Q-learning agent"""
    env = GridWorld(size=env_size, seed=seed)
    agent = QLearningAgent(state_size=env_size*env_size, action_size=4)
    
    print("🤖 Training Q-learning agent...")
    print(f"Episodes: {episodes}")
    print(f"Learning rate: {agent.learning_rate}")
    print(f"Discount factor: {agent.discount_factor}")
    print(f"Initial epsilon: {agent.epsilon}")
    print(f"Epsilon decay: {agent.epsilon_decay}")
    print(f"Save every: {save_every} episodes")
    print()
    
    success_count = 0
    
    for episode in range(episodes):
        observation, info = env.reset()
        total_reward = 0
        steps = 0
        
        while True:
            # Choose action
            action = agent.get_action(observation, training=True)
            
            # Take action
            next_observation, reward, terminated, truncated, info = env.step(action)
            
            # Update agent
            agent.update(observation, action, reward, next_observation, terminated)
            
            total_reward += reward
            steps += 1
            observation = next_observation
            
            # Render very rarely (only every 1000 episodes)
            if episode % render_every == 0:
                env.render()
                # No sleep delay for faster training
            
            if terminated or truncated:
                break
        
        # Track statistics
        agent.episode_rewards.append(total_reward)
        agent.episode_steps.append(steps)
        agent.epsilon_history.append(agent.epsilon)
        
        # Track success
        if terminated and info['agent_pos'] == info['goal_pos']:
            success_count += 1
        
        # Calculate success rate over last 100 episodes
        if episode >= 99:
            recent_success_rate = sum(1 for i in range(episode-99, episode+1) 
                                    if agent.episode_rewards[i] > 0) / 100
            agent.success_rates.append(recent_success_rate)
        else:
            agent.success_rates.append(success_count / (episode + 1))
        
        # Decay epsilon
        agent.decay_epsilon()
        
        # Save agent periodically
        if (episode + 1) % save_every == 0:
            checkpoint_filename = f"agent_checkpoint_{episode + 1}.pkl"
            agent.save(checkpoint_filename)
            print(f"💾 Checkpoint saved: {checkpoint_filename}")
        
        # Print progress less frequently
        if episode % 100 == 0:
            avg_reward = np.mean(agent.episode_rewards[-100:])
            avg_steps = np.mean(agent.episode_steps[-100:])
            success_rate = agent.success_rates[-1]
            print(f"Episode {episode}/{episodes}")
            print(f"  Avg Reward (last 100): {avg_reward:.2f}")
            print(f"  Avg Steps (last 100): {avg_steps:.1f}")
            print(f"  Success Rate: {success_rate:.2%}")
            print(f"  Epsilon: {agent.epsilon:.3f}")
            print(f"  Q-table size: {len(agent.q_table)}")
            print()
    
    print("✅ Training completed!")
    print(f"Final success rate: {agent.success_rates[-1]:.2%}")
    print(f"Final average reward: {np.mean(agent.episode_rewards[-100:]):.2f}")
    print(f"Final average steps: {np.mean(agent.episode_steps[-100:]):.1f}")
    
    return agent

def test_agent(agent, episodes: int = 10, env_size: int = 5, render: bool = False):
    """Test the trained agent"""
    env = GridWorld(size=env_size)
    
    print(f"\n🧪 Testing agent for {episodes} episodes...")
    print("=" * 50)
    
    success_count = 0
    total_rewards = []
    total_steps = []
    
    for episode in range(episodes):
        observation, info = env.reset()
        total_reward = 0
        steps = 0
        
        print(f"\nEpisode {episode + 1}:")
        
        while True:
            # Choose action (no exploration during testing)
            action = agent.get_action(observation, training=False)
            
            # Take action
            next_observation, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            steps += 1
            observation = next_observation
            
            # Render the episode only if requested
            if render:
                env.render()
                time.sleep(0.1)  # Reduced sleep time
            
            if terminated or truncated:
                if terminated and info['agent_pos'] == info['goal_pos']:
                    print("🎉 Success! Agent reached the goal!")
                    success_count += 1
                elif truncated:
                    print("⏰ Time's up! Agent ran out of moves.")
                else:
                    print("💥 Failed! Agent didn't reach the goal.")
                break
        
        total_rewards.append(total_reward)
        total_steps.append(steps)
        print(f"Total reward: {total_reward:.2f}, Steps: {steps}")
    
    print("\n" + "=" * 50)
    print(f"Test Results:")
    print(f"Success rate: {success_count}/{episodes} ({success_count/episodes*100:.1f}%)")
    print(f"Average reward: {np.mean(total_rewards):.2f}")
    print(f"Average steps: {np.mean(total_steps):.1f}")
    
    return success_count, total_rewards, total_steps

def plot_training_results(agent):
    """Plot training statistics"""
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # Episode rewards
        ax1.plot(agent.episode_rewards)
        ax1.set_title('Episode Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.grid(True)
        
        # Success rate
        ax2.plot(agent.success_rates)
        ax2.set_title('Success Rate (100-episode window)')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Success Rate')
        ax2.grid(True)
        
        # Episode steps
        ax3.plot(agent.episode_steps)
        ax3.set_title('Steps per Episode')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Steps')
        ax3.grid(True)
        
        # Epsilon decay
        ax4.plot(agent.epsilon_history)
        ax4.set_title('Epsilon Decay')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Epsilon')
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_results.png')
        plt.show()
        print("📊 Training plots saved as 'training_results.png'")
        
    except ImportError:
        print("📊 matplotlib not available. Skipping plots.")

def main():
    """Main training and testing function"""
    print("🎮 Grid World Q-Learning Agent")
    print("=" * 50)
    
    # Training parameters
    episodes = 100000000 # Changed to 100 million
    env_size = 5
    seed = 42
    
    # Train the agent
    agent = train_agent(episodes=episodes, render_every=1000, env_size=env_size, seed=seed, save_every=100000)
    
    # Save the trained agent
    agent.save("trained_agent.pkl")
    
    # Test the trained agent
    success_count, rewards, steps = test_agent(agent, episodes=5, render=True)
    
    # Plot results
    plot_training_results(agent)
    
    print("\n🎯 Training Summary:")
    print(f"Total episodes trained: {episodes}")
    print(f"Final success rate: {agent.success_rates[-1]:.2%}")
    print(f"Q-table entries: {len(agent.q_table)}")
    print(f"Final epsilon: {agent.epsilon:.3f}")
    print(f"Average steps to goal: {np.mean(agent.episode_steps[-100:]):.1f}")
    print(f"Best 100-episode success rate: {max(agent.success_rates):.2%}")

def watch_saved_agent(filename: str = "trained_agent.pkl", episodes: int = 10):
    """Load and watch a saved agent play"""
    print("🎬 Loading saved agent for demonstration...")
    
    try:
        # Load the trained agent
        agent = QLearningAgent.load(filename)
        
        # Watch the agent play
        success_count, rewards, steps = test_agent(agent, episodes=episodes, render=True)
        
        print(f"\n🎯 Agent Performance:")
        print(f"Success rate: {success_count}/{episodes} ({success_count/episodes*100:.1f}%)")
        print(f"Average reward: {np.mean(rewards):.2f}")
        print(f"Average steps: {np.mean(steps):.1f}")
        
    except FileNotFoundError:
        print(f"❌ Agent file '{filename}' not found!")
        print("Please train an agent first using: python q_learning_agent.py")

if __name__ == "__main__":
    main() 