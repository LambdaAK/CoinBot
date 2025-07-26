#!/usr/bin/env python3
"""
Improved Deep Q-Network (DQN) Agent for Grid World
Optimized for better learning performance
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

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ImprovedDQNNetwork(nn.Module):
    """Simplified neural network for DQN agent"""
    
    def __init__(self, input_size: int, output_size: int):
        super(ImprovedDQNNetwork, self).__init__()
        
        # Simpler, more appropriate network for this problem size
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output_size)
        
        # Reduced dropout for simpler problem
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class ImprovedDQNAgent:
    """Improved Deep Q-Network agent with better learning dynamics"""
    
    def __init__(self, state_size: int, action_size: int):
        
        # Optimized hyperparameters
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = 0.001  # Slightly higher learning rate
        self.gamma = 0.99  # Higher discount factor for longer-term planning
        self.epsilon = 1.0
        self.epsilon_min = 0.01  # Lower minimum epsilon for more exploration
        self.epsilon_decay = 0.995  # Faster initial decay
        self.memory_size = 10000
        self.batch_size = 64  # Smaller batch size for frequent updates
        self.target_update = 500  # More frequent target updates
        
        self.device = DEVICE
        print(f"🤖 Improved DQN Agent using device: {self.device}")
        
        # Neural networks
        self.q_network = ImprovedDQNNetwork(state_size, action_size).to(self.device)
        self.target_network = ImprovedDQNNetwork(state_size, action_size).to(self.device)
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
        
    def get_state_representation(self, grid_state, agent_pos, goal_pos, enemy_pos):
        """Improved state representation with more informative features"""
        size = grid_state.shape[0]
        
        # Create a more informative state representation
        state_vector = []
        
        # 1. Agent position (normalized)
        state_vector.extend([agent_pos[0] / size, agent_pos[1] / size])
        
        # 2. Goal position (normalized)
        state_vector.extend([goal_pos[0] / size, goal_pos[1] / size])
        
        # 3. Enemy position (normalized)
        state_vector.extend([enemy_pos[0] / size, enemy_pos[1] / size])
        
        # 4. Distance to goal (normalized)
        goal_distance = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
        state_vector.append(goal_distance / (2 * size))
        
        # 5. Distance to enemy (normalized)
        enemy_distance = abs(agent_pos[0] - enemy_pos[0]) + abs(agent_pos[1] - enemy_pos[1])
        state_vector.append(enemy_distance / (2 * size))
        
        # 6. Direction to goal (unit vectors)
        goal_dx = goal_pos[0] - agent_pos[0]
        goal_dy = goal_pos[1] - agent_pos[1]
        goal_mag = max(1, abs(goal_dx) + abs(goal_dy))
        state_vector.extend([goal_dx / goal_mag, goal_dy / goal_mag])
        
        # 7. Direction to enemy (unit vectors)
        enemy_dx = enemy_pos[0] - agent_pos[0]
        enemy_dy = enemy_pos[1] - agent_pos[1]
        enemy_mag = max(1, abs(enemy_dx) + abs(enemy_dy))
        state_vector.extend([enemy_dx / enemy_mag, enemy_dy / enemy_mag])
        
        # 8. Local grid information (3x3 around agent)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                x, y = agent_pos[0] + dx, agent_pos[1] + dy
                if 0 <= x < size and 0 <= y < size:
                    # Obstacle presence
                    state_vector.append(1.0 if grid_state[x, y] == 3 else 0.0)
                    # Goal presence
                    state_vector.append(1.0 if [x, y] == goal_pos else 0.0)
                    # Enemy presence
                    state_vector.append(1.0 if [x, y] == enemy_pos else 0.0)
                else:
                    # Out of bounds
                    state_vector.extend([1.0, 0.0, 0.0])  # Treat as obstacle
        
        return np.array(state_vector, dtype=np.float32)
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, training: bool = True):
        """Choose action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        
        return q_values.argmax().item()
    
    def replay(self):
        """Train the network on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        self.loss_history.append(loss.item())
    
    def update_target_network(self):
        """Update target network with current network weights"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def calculate_reward(self, action: int, new_pos: list, old_pos: list, 
                        goal_pos: list, enemy_pos: list = None, 
                        enemy_collision: bool = False, steps: int = 0) -> float:
        """Improved reward function with better learning signals"""
        
        # Death penalty (immediate termination)
        if enemy_collision:
            return -50.0
        
        # Goal reward (big positive reward)
        if new_pos == goal_pos:
            # Bonus for reaching goal quickly
            step_bonus = max(0, 50 - steps)
            return 100.0 + step_bonus
        
        # Calculate distances
        old_goal_dist = abs(old_pos[0] - goal_pos[0]) + abs(old_pos[1] - goal_pos[1])
        new_goal_dist = abs(new_pos[0] - goal_pos[0]) + abs(new_pos[1] - goal_pos[1])
        
        # Progress reward - encourage moving toward goal
        progress_reward = (old_goal_dist - new_goal_dist) * 2.0
        
        # Enemy avoidance reward
        enemy_reward = 0.0
        if enemy_pos:
            old_enemy_dist = abs(old_pos[0] - enemy_pos[0]) + abs(old_pos[1] - enemy_pos[1])
            new_enemy_dist = abs(new_pos[0] - enemy_pos[0]) + abs(new_pos[1] - enemy_pos[1])
            
            # Penalty for getting closer to enemy
            if new_enemy_dist < old_enemy_dist:
                enemy_reward = -1.0
            # Small reward for moving away from enemy
            elif new_enemy_dist > old_enemy_dist:
                enemy_reward = 0.5
            
            # Additional penalty for being very close to enemy
            if new_enemy_dist <= 2:
                enemy_reward -= 2.0
        
        # Step penalty to encourage efficiency
        step_penalty = -0.1
        
        # Wall bump penalty (if agent didn't move when it should have)
        wall_penalty = -0.5 if new_pos == old_pos else 0.0
        
        total_reward = progress_reward + enemy_reward + step_penalty + wall_penalty
        
        return total_reward
    
    def save(self, filename: str = "improved_dqn_agent.pkl"):
        """Save the trained agent"""
        agents_dir = "agents"
        if not os.path.exists(agents_dir):
            os.makedirs(agents_dir)
        
        filepath = os.path.join(agents_dir, filename)
        
        agent_data = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'hyperparameters': {
                'learning_rate': self.learning_rate,
                'gamma': self.gamma,
                'epsilon': self.epsilon,
                'epsilon_min': self.epsilon_min,
                'epsilon_decay': self.epsilon_decay,
                'action_size': self.action_size,
                'state_size': self.state_size,
            },
            'training_data': {
                'episode_rewards': self.episode_rewards,
                'episode_steps': self.episode_steps,
                'success_rates': self.success_rates,
                'epsilon_history': self.epsilon_history,
                'loss_history': self.loss_history
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(agent_data, f)
        
        print(f"✅ Improved DQN Agent saved to {filepath}")
    
    @classmethod
    def load(cls, filename: str = "improved_dqn_agent.pkl"):
        """Load a trained agent"""
        agents_dir = "agents"
        filepath = os.path.join(agents_dir, filename)
        
        if not os.path.exists(filepath):
            if not os.path.exists(filename):
                raise FileNotFoundError(f"Agent file {filepath} not found!")
            filepath = filename
        
        with open(filepath, 'rb') as f:
            agent_data = pickle.load(f)
        
        hp = agent_data['hyperparameters']
        agent = cls(state_size=hp['state_size'], action_size=hp['action_size'])
        
        # Restore hyperparameters
        for key, value in hp.items():
            setattr(agent, key, value)
        
        # Restore networks
        agent.q_network.load_state_dict(agent_data['q_network_state_dict'])
        agent.target_network.load_state_dict(agent_data['target_network_state_dict'])
        agent.optimizer.load_state_dict(agent_data['optimizer_state_dict'])
        
        # Restore training data
        td = agent_data['training_data']
        for key, value in td.items():
            setattr(agent, key, value)
        
        print(f"✅ Improved DQN Agent loaded from {filepath}")
        return agent

def train_improved_dqn_agent(episodes: int = None, render_every: int = 1000, 
                           env_size: int = 10, seed: int = 42, 
                           save_every: int = 1000):
    """Train the improved DQN agent"""
    # Import GridWorld from your existing file
    from grid_world import GridWorld
    
    env = GridWorld(size=env_size, seed=seed)
    
    # Calculate state size based on improved representation
    # 2 (agent) + 2 (goal) + 2 (enemy) + 1 (goal_dist) + 1 (enemy_dist) 
    # + 2 (goal_dir) + 2 (enemy_dir) + 27 (3x3 local grid * 3 features)
    state_size = 39
    
    agent = ImprovedDQNAgent(state_size=state_size, action_size=4)
    
    print("🚀 Training Improved DQN agent...")
    print(f"Episodes: Unlimited (train until interrupted)")
    print(f"State size: {state_size}")
    print(f"Device: {agent.device}")
    print("Press Ctrl+C to stop training")
    print()
    
    success_count = 0
    start_time = time.time()
    episode = 0
    
    try:
        while True:  # Train indefinitely
            observation, info = env.reset()
            total_reward = 0
            steps = 0
            
            # Get initial state
            grid_state = env.grid
            agent_pos = info['agent_pos']
            goal_pos = info['goal_pos']
            enemy_pos = info.get('enemy_pos', [env.size//2, env.size//2])  # Fallback if not in info
            state = agent.get_state_representation(grid_state, agent_pos, goal_pos, enemy_pos)
            old_pos = agent_pos.copy()
            
            while True:
                action = agent.act(state, training=True)
                next_observation, env_reward, terminated, truncated, info = env.step(action)
                
                # Get new state
                new_grid_state = env.grid
                new_agent_pos = info['agent_pos']
                new_goal_pos = info['goal_pos']
                new_enemy_pos = info.get('enemy_pos', [env.size//2, env.size//2])  # Fallback if not in info
                next_state = agent.get_state_representation(new_grid_state, new_agent_pos, new_goal_pos, new_enemy_pos)
                
                # Calculate reward using improved function
                enemy_collision = info.get('enemy_collision', False)
                reward = agent.calculate_reward(action, new_agent_pos, old_pos, 
                                               new_goal_pos, new_enemy_pos, 
                                               enemy_collision, steps)
                
                done = terminated or truncated
                agent.remember(state, action, reward, next_state, done)
                agent.replay()
                
                total_reward += reward
                steps += 1
                state = next_state
                old_pos = new_agent_pos.copy()
                
                if episode % render_every == 0 and episode > 0:
                    # env.render()  # Commented out to disable rendering
                    # time.sleep(0.1)  # Commented out to disable rendering
                    pass
                
                if done:
                    break
            
            # Update target network
            if episode % agent.target_update == 0:
                agent.update_target_network()
            
            # Decay epsilon
            if agent.epsilon > agent.epsilon_min:
                agent.epsilon *= agent.epsilon_decay
            
            # Track statistics
            agent.episode_rewards.append(total_reward)
            agent.episode_steps.append(steps)
            agent.epsilon_history.append(agent.epsilon)
            
            # Track success
            if terminated and info['agent_pos'] == info['goal_pos'] and not info.get('enemy_collision', False):
                success_count += 1
            
            # Calculate rolling success rate
            window_size = min(100, episode + 1)
            recent_successes = 0
            for i in range(max(0, episode - window_size + 1), episode + 1):
                ep_reward = agent.episode_rewards[i]
                if ep_reward > 50:  # Threshold for success (goal reward is 100+)
                    recent_successes += 1
            
            success_rate = recent_successes / window_size
            agent.success_rates.append(success_rate)
            
            # Save periodically
            if (episode + 1) % save_every == 0:
                checkpoint_filename = f"improved_dqn_checkpoint_{episode + 1}.pkl"
                agent.save(checkpoint_filename)
            
            # Print progress
            if episode % 100 == 0 and episode > 0:
                elapsed_time = time.time() - start_time
                avg_reward = np.mean(agent.episode_rewards[-100:])
                avg_steps = np.mean(agent.episode_steps[-100:])
                avg_loss = np.mean(agent.loss_history[-100:]) if agent.loss_history else 0
                
                print(f"Episode {episode}")
                print(f"  Avg Reward: {avg_reward:.2f}")
                print(f"  Avg Steps: {avg_steps:.1f}")
                print(f"  Success Rate: {success_rate:.1%}")
                print(f"  Epsilon: {agent.epsilon:.3f}")
                print(f"  Avg Loss: {avg_loss:.4f}")
                print(f"  Time: {elapsed_time/60:.1f}m")
                print()
            
            episode += 1
            
    except KeyboardInterrupt:
        print("\n🛑 Training stopped by user")
        print(f"Total episodes trained: {episode}")
    
    print("✅ Training completed!")
    print(f"Final success rate: {agent.success_rates[-1]:.1%}")
    print(f"Training time: {(time.time() - start_time)/60:.1f} minutes")
    
    return agent

def test_improved_agent(agent, episodes: int = 10, render: bool = True):
    """Test the improved DQN agent"""
    from grid_world import GridWorld
    
    env = GridWorld(size=10)
    
    print(f"\n🧪 Testing improved DQN agent for {episodes} episodes...")
    
    success_count = 0
    total_rewards = []
    total_steps = []
    
    for episode in range(episodes):
        observation, info = env.reset()
        total_reward = 0
        steps = 0
        
        grid_state = env.grid
        agent_pos = info['agent_pos']
        goal_pos = info['goal_pos']
        enemy_pos = info.get('enemy_pos', [env.size//2, env.size//2])  # Fallback if not in info
        state = agent.get_state_representation(grid_state, agent_pos, goal_pos, enemy_pos)
        old_pos = agent_pos.copy()
        
        print(f"\nEpisode {episode + 1}:")
        
        while True:
            action = agent.act(state, training=False)
            next_observation, env_reward, terminated, truncated, info = env.step(action)
            
            new_grid_state = env.grid
            new_agent_pos = info['agent_pos']
            new_goal_pos = info['goal_pos']
            new_enemy_pos = info.get('enemy_pos', [env.size//2, env.size//2])  # Fallback if not in info
            next_state = agent.get_state_representation(new_grid_state, new_agent_pos, new_goal_pos, new_enemy_pos)
            
            enemy_collision = info.get('enemy_collision', False)
            reward = agent.calculate_reward(action, new_agent_pos, old_pos, 
                                          new_goal_pos, new_enemy_pos, 
                                          enemy_collision, steps)
            
            total_reward += reward
            steps += 1
            state = next_state
            old_pos = new_agent_pos.copy()
            
            if render:
                env.render()
                time.sleep(0.3)
            
            if terminated or truncated:
                if terminated and info['agent_pos'] == info['goal_pos'] and not enemy_collision:
                    print("🎉 Success! Agent reached the goal!")
                    success_count += 1
                elif enemy_collision:
                    print("💀 Game Over! Enemy collision!")
                else:
                    print("⏰ Time's up!")
                break
        
        total_rewards.append(total_reward)
        total_steps.append(steps)
        print(f"Total reward: {total_reward:.2f}, Steps: {steps}")
    
    print(f"\n📊 Test Results:")
    print(f"Success rate: {success_count}/{episodes} ({success_count/episodes*100:.1f}%)")
    print(f"Average reward: {np.mean(total_rewards):.2f}")
    print(f"Average steps: {np.mean(total_steps):.1f}")
    
    return success_count, total_rewards, total_steps

def main():
    """Main function to train and test the improved agent"""
    print("🎮 Improved Grid World DQN Agent")
    print("=" * 50)
    
    # Train the agent
    agent = train_improved_dqn_agent(render_every=1000, env_size=10)
    
    # Save the trained agent
    agent.save("improved_dqn_agent.pkl")
    
    # Test the agent
    test_improved_agent(agent, episodes=5, render=True)
    
    # Plot results if matplotlib is available
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        ax1.plot(agent.episode_rewards)
        ax1.set_title('Episode Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.grid(True)
        
        ax2.plot(agent.success_rates)
        ax2.set_title('Success Rate (Rolling Window)')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Success Rate')
        ax2.grid(True)
        
        ax3.plot(agent.episode_steps)
        ax3.set_title('Steps per Episode')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Steps')
        ax3.grid(True)
        
        if agent.loss_history:
            ax4.plot(agent.loss_history)
            ax4.set_title('Training Loss')
            ax4.set_xlabel('Training Step')
            ax4.set_ylabel('Loss')
            ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig('improved_dqn_results.png')
        plt.show()
        
    except ImportError:
        print("📊 matplotlib not available. Skipping plots.")

if __name__ == "__main__":
    main()