#!/usr/bin/env python3
"""
Deep Q-Network (DQN) Agent for Grid World
Uses neural networks to generalize across similar states with CUDA acceleration
"""

import numpy as np
import random
import time
import os
import pickle
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from grid_world import GridWorld

# Import global configuration
try:
    from config import DEVICE, DQN_CONFIG, COLAB_CONFIG
    from tqdm import tqdm
except ImportError:
    # Fallback if config not available
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DQN_CONFIG = {
        'learning_rate': 0.0005,
        'gamma': 0.95,
        'epsilon': 1.0,
        'epsilon_min': 0.05,
        'epsilon_decay': 0.9995,
        'memory_size': 10000,
        'batch_size': 64,
        'target_update': 1000
    }
    COLAB_CONFIG = {'use_tqdm': True}

class DQNNetwork(nn.Module):
    """Neural network for DQN agent"""
    
    def __init__(self, input_size: int, output_size: int):
        super(DQNNetwork, self).__init__()
        
        # Input: flattened grid + agent_pos + goal_pos
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, output_size)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = F.relu(self.fc4(x))
        x = self.dropout(x)
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return x

class DQNAgent:
    """Deep Q-Network agent for grid world navigation with CUDA acceleration"""
    
    def __init__(self, state_size: int, action_size: int, 
                 learning_rate: float = None, gamma: float = None, 
                 epsilon: float = None, epsilon_min: float = None, 
                 epsilon_decay: float = None, memory_size: int = None,
                 batch_size: int = None, target_update: int = None):
        
        # Use config values or provided parameters
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate or DQN_CONFIG['learning_rate']
        self.gamma = gamma or DQN_CONFIG['gamma']
        self.epsilon = epsilon or DQN_CONFIG['epsilon']
        self.epsilon_min = epsilon_min or DQN_CONFIG['epsilon_min']
        self.epsilon_decay = epsilon_decay or DQN_CONFIG['epsilon_decay']
        self.memory_size = memory_size or DQN_CONFIG['memory_size']
        self.batch_size = batch_size or DQN_CONFIG['batch_size']
        self.target_update = target_update or DQN_CONFIG['target_update']
        
        # Use global device configuration
        self.device = DEVICE
        print(f"🤖 DQN Agent using device: {self.device}")
        
        # Neural networks
        self.q_network = DQNNetwork(state_size, action_size).to(self.device)
        self.target_network = DQNNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        
        # Experience replay memory
        self.memory = deque(maxlen=self.memory_size)
        
        # Training statistics
        self.episode_rewards = []
        self.episode_steps = []
        self.success_rates = []
        self.epsilon_history = []
        self.loss_history = []
        
        # Update target network
        self.update_target_network()
        
    def get_state_representation(self, grid_state, agent_pos, goal_pos):
        """Convert environment state to neural network input"""
        # Flatten grid and add position information
        grid_flat = grid_state.flatten().astype(np.float32)
        
        # Normalize positions to [0, 1]
        size = int(np.sqrt(len(grid_flat)))
        agent_norm = [agent_pos[0] / size, agent_pos[1] / size]
        goal_norm = [goal_pos[0] / size, goal_pos[1] / size]
        
        # Combine all information
        state_vector = np.concatenate([
            grid_flat,
            agent_norm,
            goal_norm
        ]).astype(np.float32)
        
        return state_vector
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, training: bool = True):
        """Choose action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        # Convert state to tensor and move to device
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Get Q-values
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        
        # Return best action
        return q_values.argmax().item()
    
    def replay(self):
        """Train the network on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to numpy arrays first, then to tensors and move to device
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q-values (from target network)
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Store loss for monitoring
        self.loss_history.append(loss.item())
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_network(self):
        """Update target network with current network weights"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def calculate_reward(self, action: int, new_pos: list, old_pos: list, goal_pos: list) -> float:
        """Calculate reward for goal-seeking behavior"""
        # Goal reward (highest priority)
        if new_pos == goal_pos:
            return 10.0
        
        # Base step penalty
        reward = -0.1
        
        # Progress toward goal
        old_distance = abs(old_pos[0] - goal_pos[0]) + abs(old_pos[1] - goal_pos[1])
        new_distance = abs(new_pos[0] - goal_pos[0]) + abs(new_pos[1] - goal_pos[1])
        
        if new_distance < old_distance:
            reward += 0.1
        
        return reward
    
    def save(self, filename: str = "dqn_agent.pkl"):
        """Save the trained agent"""
        agents_dir = "agents"
        if not os.path.exists(agents_dir):
            os.makedirs(agents_dir)
        
        filepath = os.path.join(agents_dir, filename)
        
        agent_data = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'epsilon_min': self.epsilon_min,
            'epsilon_decay': self.epsilon_decay,
            'action_size': self.action_size,
            'state_size': self.state_size,
            'episode_rewards': self.episode_rewards,
            'episode_steps': self.episode_steps,
            'success_rates': self.success_rates,
            'epsilon_history': self.epsilon_history,
            'loss_history': self.loss_history
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(agent_data, f)
        
        print(f"✅ DQN Agent saved to {filepath}")
        print(f"   Training episodes: {len(self.episode_rewards)}")
        print(f"   Memory size: {len(self.memory)}")
    
    @classmethod
    def load(cls, filename: str = "dqn_agent.pkl"):
        """Load a trained agent"""
        agents_dir = "agents"
        filepath = os.path.join(agents_dir, filename)
        
        if not os.path.exists(filepath):
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
            gamma=agent_data['gamma'],
            epsilon=agent_data['epsilon'],
            epsilon_min=agent_data['epsilon_min'],
            epsilon_decay=agent_data['epsilon_decay']
        )
        
        # Restore networks and optimizer
        agent.q_network.load_state_dict(agent_data['q_network_state_dict'])
        agent.target_network.load_state_dict(agent_data['target_network_state_dict'])
        agent.optimizer.load_state_dict(agent_data['optimizer_state_dict'])
        
        # Restore training data
        agent.episode_rewards = agent_data['episode_rewards']
        agent.episode_steps = agent_data['episode_steps']
        agent.success_rates = agent_data['success_rates']
        agent.epsilon_history = agent_data['epsilon_history']
        agent.loss_history = agent_data['loss_history']
        
        print(f"✅ DQN Agent loaded from {filepath}")
        print(f"   Training episodes: {len(agent.episode_rewards)}")
        print(f"   Final epsilon: {agent.epsilon:.3f}")
        
        return agent

def train_dqn_agent(episodes: int = None, render_every: int = None, env_size: int = None, 
                   seed: int = None, save_every: int = None):
    """Train the DQN agent with CUDA acceleration"""
    # Use config values or provided parameters
    episodes = episodes or DQN_CONFIG['episodes']
    render_every = render_every or DQN_CONFIG['render_every']
    env_size = env_size or DQN_CONFIG['env_size']
    seed = seed or DQN_CONFIG['seed']
    save_every = save_every or DQN_CONFIG['save_every']
    
    env = GridWorld(size=env_size, seed=seed)
    
    # Calculate state size (grid + positions)
    grid_size = env_size * env_size
    state_size = grid_size + 4  # grid + agent_pos(2) + goal_pos(2)
    
    agent = DQNAgent(state_size=state_size, action_size=4)
    
    print("🤖 Training DQN agent with CUDA acceleration...")
    print(f"Episodes: {episodes}")
    print(f"Learning rate: {agent.learning_rate}")
    print(f"Gamma: {agent.gamma}")
    print(f"Initial epsilon: {agent.epsilon}")
    print(f"Epsilon decay: {agent.epsilon_decay}")
    print(f"Batch size: {agent.batch_size}")
    print(f"Save every: {save_every} episodes")
    print(f"Device: {agent.device}")
    print()
    
    success_count = 0
    start_time = time.time()
    
    # Use progress bar if available and enabled
    episode_range = range(episodes)
    if COLAB_CONFIG.get('use_tqdm', False):
        try:
            episode_range = tqdm(episode_range, desc="Training DQN")
        except ImportError:
            pass
    
    for episode in episode_range:
        observation, info = env.reset()
        total_reward = 0
        steps = 0
        
        # Get initial state representation
        grid_state = env.grid
        agent_pos = info['agent_pos']
        goal_pos = info['goal_pos']
        state = agent.get_state_representation(grid_state, agent_pos, goal_pos)
        old_pos = agent_pos.copy()
        
        while True:
            # Choose action
            action = agent.act(state, training=True)
            
            # Take action
            next_observation, env_reward, terminated, truncated, info = env.step(action)
            
            # Get new state representation
            new_grid_state = env.grid
            new_agent_pos = info['agent_pos']
            new_goal_pos = info['goal_pos']
            next_state = agent.get_state_representation(new_grid_state, new_agent_pos, new_goal_pos)
            
            # Calculate reward
            reward = agent.calculate_reward(action, new_agent_pos, old_pos, new_goal_pos)
            
            # Store experience
            done = terminated or truncated
            agent.remember(state, action, reward, next_state, done)
            
            # Train the network
            agent.replay()
            
            total_reward += reward
            steps += 1
            state = next_state
            old_pos = new_agent_pos.copy()
            
            # Render occasionally
            if episode % render_every == 0:
                env.render()
            
            if terminated or truncated:
                break
        
        # Update target network periodically
        if episode % agent.target_update == 0:
            agent.update_target_network()
        
        # Track statistics
        agent.episode_rewards.append(total_reward)
        agent.episode_steps.append(steps)
        agent.epsilon_history.append(agent.epsilon)
        
        # Track success
        if terminated and info['agent_pos'] == info['goal_pos']:
            success_count += 1
        
        # Calculate success rate
        if episode >= 99:
            recent_success_rate = sum(1 for i in range(episode-99, episode+1) 
                                    if agent.episode_rewards[i] > 0) / 100
            agent.success_rates.append(recent_success_rate)
        else:
            agent.success_rates.append(success_count / (episode + 1))
        
        # Save agent periodically
        if (episode + 1) % save_every == 0:
            checkpoint_filename = f"dqn_checkpoint_{episode + 1}.pkl"
            agent.save(checkpoint_filename)
            print(f"💾 Checkpoint saved: {checkpoint_filename}")
        
        # Print progress
        if episode % 1000 == 0:  # Less frequent reporting for long training
            elapsed_time = time.time() - start_time
            avg_reward = np.mean(agent.episode_rewards[-100:])
            avg_steps = np.mean(agent.episode_steps[-100:])
            success_rate = agent.success_rates[-1]
            avg_loss = np.mean(agent.loss_history[-100:]) if agent.loss_history else 0
            
            print(f"Episode {episode}/{episodes} ({episode/episodes*100:.1f}%)")
            print(f"  Time elapsed: {elapsed_time/3600:.1f}h")
            print(f"  Avg Reward (last 100): {avg_reward:.2f}")
            print(f"  Avg Steps (last 100): {avg_steps:.1f}")
            print(f"  Success Rate: {success_rate:.2%}")
            print(f"  Epsilon: {agent.epsilon:.3f}")
            print(f"  Avg Loss: {avg_loss:.4f}")
            print(f"  Memory Size: {len(agent.memory)}")
            print()
    
    print("✅ DQN Training completed!")
    print(f"Final success rate: {agent.success_rates[-1]:.2%}")
    print(f"Final average reward: {np.mean(agent.episode_rewards[-100:]):.2f}")
    print(f"Final average steps: {np.mean(agent.episode_steps[-100:]):.1f}")
    print(f"Total training time: {(time.time() - start_time)/3600:.1f} hours")
    
    return agent

def test_dqn_agent(agent, episodes: int = 10, env_size: int = 5, render: bool = False):
    """Test the trained DQN agent"""
    env = GridWorld(size=env_size)
    
    print(f"\n🧪 Testing DQN agent for {episodes} episodes...")
    print("=" * 50)
    
    success_count = 0
    total_rewards = []
    total_steps = []
    
    for episode in range(episodes):
        observation, info = env.reset()
        total_reward = 0
        steps = 0
        
        # Get initial state
        grid_state = env.grid
        agent_pos = info['agent_pos']
        goal_pos = info['goal_pos']
        state = agent.get_state_representation(grid_state, agent_pos, goal_pos)
        old_pos = agent_pos.copy()
        
        print(f"\nEpisode {episode + 1}:")
        
        while True:
            # Choose action (no exploration during testing)
            action = agent.act(state, training=False)
            
            # Take action
            next_observation, env_reward, terminated, truncated, info = env.step(action)
            
            # Get new state
            new_grid_state = env.grid
            new_agent_pos = info['agent_pos']
            new_goal_pos = info['goal_pos']
            next_state = agent.get_state_representation(new_grid_state, new_agent_pos, new_goal_pos)
            
            # Calculate reward
            reward = agent.calculate_reward(action, new_agent_pos, old_pos, new_goal_pos)
            
            total_reward += reward
            steps += 1
            state = next_state
            old_pos = new_agent_pos.copy()
            
            # Render if requested
            if render:
                env.render()
                time.sleep(0.1)
            
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

def plot_dqn_results(agent):
    """Plot DQN training statistics"""
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
        
        # Loss history
        if agent.loss_history:
            ax4.plot(agent.loss_history)
            ax4.set_title('Training Loss')
            ax4.set_xlabel('Training Step')
            ax4.set_ylabel('Loss')
            ax4.grid(True)
        else:
            ax4.text(0.5, 0.5, 'No loss data', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Training Loss')
        
        plt.tight_layout()
        plt.savefig('dqn_training_results.png')
        plt.show()
        print("📊 DQN training plots saved as 'dqn_training_results.png'")
        
    except ImportError:
        print("📊 matplotlib not available. Skipping plots.")

def main():
    """Main training and testing function"""
    print("🎮 Grid World DQN Agent")
    print("=" * 50)
    
    # Training parameters
    episodes = 1000000  # Increased to 1 million episodes
    env_size = 5
    seed = 42
    
    # Train the agent
    agent = train_dqn_agent(episodes=episodes, render_every=10000, env_size=env_size, seed=seed, save_every=50000)
    
    # Save the trained agent
    agent.save("trained_dqn_agent.pkl")
    
    # Test the trained agent
    success_count, rewards, steps = test_dqn_agent(agent, episodes=5, render=True)
    
    # Plot results
    plot_dqn_results(agent)
    
    print("\n🎯 DQN Training Summary:")
    print(f"Total episodes trained: {episodes}")
    print(f"Final success rate: {agent.success_rates[-1]:.2%}")
    print(f"Final epsilon: {agent.epsilon:.3f}")
    print(f"Average steps to goal: {np.mean(agent.episode_steps[-100:]):.1f}")
    print(f"Best 100-episode success rate: {max(agent.success_rates):.2%}")

def watch_saved_dqn_agent(filename: str = "trained_dqn_agent.pkl", episodes: int = 10):
    """Load and watch a saved DQN agent play"""
    print("🎬 Loading saved DQN agent for demonstration...")
    
    try:
        # Load the trained agent
        agent = DQNAgent.load(filename)
        
        # Watch the agent play
        success_count, rewards, steps = test_dqn_agent(agent, episodes=episodes, render=True)
        
        print(f"\n🎯 DQN Agent Performance:")
        print(f"Success rate: {success_count}/{episodes} ({success_count/episodes*100:.1f}%)")
        print(f"Average reward: {np.mean(rewards):.2f}")
        print(f"Average steps: {np.mean(steps):.1f}")
        
    except FileNotFoundError:
        print(f"❌ DQN Agent file '{filename}' not found!")
        print("Please train a DQN agent first using: python dqn_agent.py")

if __name__ == "__main__":
    main() 