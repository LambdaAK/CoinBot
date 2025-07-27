#!/usr/bin/env python3
"""
Coin Collection Agent

Specialized agent for the coin collection grid world environment.
Updated state representation and reward function for coin collection.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import pickle
import os
from typing import List, Tuple, Dict, Any
import time

class CoinCollectionDQN(nn.Module):
    """Simpler neural network for coin collection DQN agent"""
    
    def __init__(self, state_size: int, action_size: int):
        super(CoinCollectionDQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, action_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CoinCollectionAgent:
    """DQN agent specialized for coin collection"""
    
    def __init__(self, state_size: int, action_size: int, learning_rate: float = 0.001,
                 gamma: float = 0.95, epsilon: float = 1.0, epsilon_decay: float = 0.9995,
                 epsilon_min: float = 0.01, memory_size: int = 10000, batch_size: int = 32):
        
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.memory_size = memory_size
        self.batch_size = batch_size
        
        # Device (GPU if available, else CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.q_network = CoinCollectionDQN(state_size, action_size).to(self.device)
        self.target_network = CoinCollectionDQN(state_size, action_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Experience replay memory
        self.memory = []
        
        # Training statistics
        self.episode_rewards = []
        self.episode_steps = []
        self.success_rates = []
        self.epsilon_history = []
        self.loss_history = []
        self.coins_collected_history = []
    
    def get_state_representation(self, grid_state, agent_pos, coins, enemy_positions, weapon_powerups=None, has_weapon=False, weapon_turns_remaining=0):
        """Enhanced state representation for coin collection with weapon powerups"""
        size = grid_state.shape[0]
        
        # Create a comprehensive state vector
        state_vector = []
        
        # 1. Agent position (normalized)
        state_vector.extend([agent_pos[0] / size, agent_pos[1] / size])
        
        # 2. Find closest coin
        closest_coin = None
        closest_coin_distance = float('inf')
        
        if coins:
            for coin_pos in coins:
                distance = abs(agent_pos[0] - coin_pos[0]) + abs(agent_pos[1] - coin_pos[1])
                if distance < closest_coin_distance:
                    closest_coin_distance = distance
                    closest_coin = coin_pos
        
        # If no coins, use a default position
        if closest_coin is None:
            closest_coin = [size//2, size//2]
            closest_coin_distance = abs(agent_pos[0] - closest_coin[0]) + abs(agent_pos[1] - closest_coin[1])
        
        # 3. Closest coin position (normalized)
        state_vector.extend([closest_coin[0] / size, closest_coin[1] / size])
        
        # 4. Find closest enemy
        closest_enemy = None
        closest_enemy_distance = float('inf')
        
        if enemy_positions:
            for enemy_pos in enemy_positions:
                distance = abs(agent_pos[0] - enemy_pos[0]) + abs(agent_pos[1] - enemy_pos[1])
                if distance < closest_enemy_distance:
                    closest_enemy_distance = distance
                    closest_enemy = enemy_pos
        
        # If no enemies, use a default position
        if closest_enemy is None:
            closest_enemy = [size//2, size//2]
            closest_enemy_distance = abs(agent_pos[0] - closest_enemy[0]) + abs(agent_pos[1] - closest_enemy[1])
        
        # 5. Closest enemy position (normalized)
        state_vector.extend([closest_enemy[0] / size, closest_enemy[1] / size])
        
        # 6. Distance to closest coin (normalized)
        state_vector.append(closest_coin_distance / (2 * size))
        
        # 7. Distance to closest enemy (normalized)
        state_vector.append(closest_enemy_distance / (2 * size))
        
        # 8. Direction to closest coin (unit vectors)
        coin_dx = closest_coin[0] - agent_pos[0]
        coin_dy = closest_coin[1] - agent_pos[1]
        coin_mag = max(1, abs(coin_dx) + abs(coin_dy))
        state_vector.extend([coin_dx / coin_mag, coin_dy / coin_mag])
        
        # 9. Direction to closest enemy (unit vectors)
        enemy_dx = closest_enemy[0] - agent_pos[0]
        enemy_dy = closest_enemy[1] - agent_pos[1]
        enemy_mag = max(1, abs(enemy_dx) + abs(enemy_dy))
        state_vector.extend([enemy_dx / enemy_mag, enemy_dy / enemy_mag])
        
        # 10. Number of coins remaining (normalized)
        num_coins = len(coins) if coins else 0
        state_vector.append(num_coins / 10.0)  # Normalize by max expected coins
        
        # 11. Number of enemies (normalized)
        num_enemies = len(enemy_positions) if enemy_positions else 0
        state_vector.append(num_enemies / 5.0)  # Normalize by max expected enemies
        
        # 12. Coin density in local area (5x5 around agent)
        coin_density = 0
        total_cells = 0
        for dx in [-2, -1, 0, 1, 2]:
            for dy in [-2, -1, 0, 1, 2]:
                x, y = agent_pos[0] + dx, agent_pos[1] + dy
                if 0 <= x < size and 0 <= y < size:
                    total_cells += 1
                    if [x, y] in coins:
                        coin_density += 1
        
        state_vector.append(coin_density / max(1, total_cells))
        
        # 13. Enemy density in local area (5x5 around agent)
        enemy_density = 0
        for dx in [-2, -1, 0, 1, 2]:
            for dy in [-2, -1, 0, 1, 2]:
                x, y = agent_pos[0] + dx, agent_pos[1] + dy
                if 0 <= x < size and 0 <= y < size:
                    if any([x, y] == enemy_pos for enemy_pos in enemy_positions):
                        enemy_density += 1
        
        state_vector.append(enemy_density / max(1, total_cells))
        
        # 14. Local grid information (5x5 around agent)
        for dx in [-2, -1, 0, 1, 2]:
            for dy in [-2, -1, 0, 1, 2]:
                x, y = agent_pos[0] + dx, agent_pos[1] + dy
                if 0 <= x < size and 0 <= y < size:
                    # Obstacle presence
                    state_vector.append(1.0 if grid_state[x, y] == 3 else 0.0)
                    # Coin presence
                    state_vector.append(1.0 if [x, y] in coins else 0.0)
                    # Enemy presence (any enemy)
                    has_enemy = any([x, y] == enemy_pos for enemy_pos in enemy_positions) if enemy_positions else False
                    state_vector.append(1.0 if has_enemy else 0.0)
                else:
                    # Out of bounds
                    state_vector.extend([1.0, 0.0, 0.0])  # Treat as obstacle
        
        # 15. Weapon powerup information (4 new dimensions)
        # Distance to nearest powerup
        closest_powerup = None
        closest_powerup_distance = float('inf')
        
        if weapon_powerups:
            for powerup_pos in weapon_powerups:
                distance = abs(agent_pos[0] - powerup_pos[0]) + abs(agent_pos[1] - powerup_pos[1])
                if distance < closest_powerup_distance:
                    closest_powerup_distance = distance
                    closest_powerup = powerup_pos
        
        # If no powerups, use default values
        if closest_powerup is None:
            closest_powerup_distance = 2 * size  # Max distance
            closest_powerup = [size//2, size//2]
        
        # Distance to nearest powerup (normalized)
        state_vector.append(closest_powerup_distance / (2 * size))
        
        # Unit vector toward nearest powerup
        powerup_dx = closest_powerup[0] - agent_pos[0]
        powerup_dy = closest_powerup[1] - agent_pos[1]
        powerup_mag = max(1, abs(powerup_dx) + abs(powerup_dy))
        state_vector.extend([powerup_dx / powerup_mag, powerup_dy / powerup_mag])
        
        # Remaining weapon duration (normalized)
        state_vector.append(weapon_turns_remaining / 10.0)
        
        # Weapon status binary
        state_vector.append(1.0 if has_weapon else 0.0)
        
        return np.array(state_vector, dtype=np.float32)
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
        
        # Keep memory size limited
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)
    
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
                        coins: list, enemy_positions: list = None, 
                        enemy_collision: bool = False, coin_collected: bool = False,
                        weapon_collected: bool = False, enemy_died: bool = False,
                        has_weapon: bool = False, weapon_turns_remaining: int = 0,
                        steps: int = 0) -> float:
        """Enhanced reward function for coin collection with weapon powerups"""
        
        # Death penalty (immediate termination)
        if enemy_collision:
            return -50.0
        
        # Weapon collection reward (big reward)
        if weapon_collected:
            return 15.0
        
        # Enemy defeat reward (when armed) - higher than coin collection
        if enemy_died:
            return 50.0  # Increased from 10.0 - now higher than coin collection (25-44)
        
        # Coin collection reward
        if coin_collected:
            # Bonus for collecting coins quickly
            step_bonus = max(0, 20 - steps)
            return 25.0 + step_bonus
        
        # All coins collected reward (completion bonus)
        if len(coins) == 0:
            # Big bonus for completing the level
            step_bonus = max(0, 100 - steps)
            return 100.0 + step_bonus
        
        # Calculate distances to nearest coin
        if coins:
            old_coin_dist = float('inf')
            new_coin_dist = float('inf')
            
            for coin_pos in coins:
                old_dist = abs(old_pos[0] - coin_pos[0]) + abs(old_pos[1] - coin_pos[1])
                new_dist = abs(new_pos[0] - coin_pos[0]) + abs(new_pos[1] - coin_pos[1])
                old_coin_dist = min(old_coin_dist, old_dist)
                new_coin_dist = min(new_coin_dist, new_dist)
            
            # Progress reward - encourage moving toward coins
            progress_reward = (old_coin_dist - new_coin_dist) * 3.0
        else:
            progress_reward = 0.0
        
        # Enemy interaction reward (depends on weapon status)
        enemy_reward = 0.0
        if enemy_positions:
            for enemy_pos in enemy_positions:
                old_enemy_dist = abs(old_pos[0] - enemy_pos[0]) + abs(old_pos[1] - enemy_pos[1])
                new_enemy_dist = abs(new_pos[0] - enemy_pos[0]) + abs(new_pos[1] - enemy_pos[1])
                
                if has_weapon and weapon_turns_remaining > 0:
                    # Armed: much stronger reward for approaching enemies
                    if new_enemy_dist < old_enemy_dist:
                        enemy_reward += 15.0  # Increased from 5.0 - much bigger reward for moving toward enemies when armed
                    elif new_enemy_dist > old_enemy_dist:
                        enemy_reward -= 3.0  # Increased penalty from -1.0 - stronger discouragement for moving away when armed
                    
                    # Stronger proximity rewards when armed
                    if new_enemy_dist <= 1:
                        enemy_reward += 8.0  # Increased from 2.0 - much bigger bonus for being adjacent when armed
                    elif new_enemy_dist <= 2:
                        enemy_reward += 3.0  # New bonus for being close when armed
                    elif new_enemy_dist <= 3:
                        enemy_reward += 1.0  # Small bonus for being within 3 steps when armed
                else:
                    # Unarmed: penalty for approaching enemies
                    if new_enemy_dist < old_enemy_dist:
                        enemy_reward -= 0.5
                    elif new_enemy_dist > old_enemy_dist:
                        enemy_reward += 0.2
                    
                    # Additional penalties for being close to enemies when unarmed
                    if new_enemy_dist <= 1:
                        enemy_reward -= 2.0  # Penalty for being adjacent when unarmed
                    elif new_enemy_dist <= 2:
                        enemy_reward -= 0.5  # Moderate penalty for being close when unarmed
        
        # Step penalty to encourage efficiency
        step_penalty = -0.1
        
        # Wall bump penalty (if agent didn't move when it should have)
        wall_penalty = -0.5 if new_pos == old_pos else 0.0
        
        # Coin proximity bonus (small reward for being near coins)
        coin_proximity_bonus = 0.0
        if coins:
            min_coin_dist = min(abs(new_pos[0] - coin_pos[0]) + abs(new_pos[1] - coin_pos[1]) 
                              for coin_pos in coins)
            if min_coin_dist <= 1:
                coin_proximity_bonus = 0.1  # Small bonus for being adjacent to a coin
        
        total_reward = progress_reward + enemy_reward + step_penalty + wall_penalty + coin_proximity_bonus
        
        return total_reward
    
    def save(self, filename: str = "coin_collection_agent.pkl"):
        """Save the trained agent"""
        # Get the project root directory (parent of src)
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))  # src directory
        project_root = os.path.dirname(current_dir)  # project root
        agents_dir = os.path.join(project_root, "agents")
        
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
                'loss_history': self.loss_history,
                'coins_collected_history': self.coins_collected_history
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(agent_data, f)
        
        print(f"✅ Coin Collection Agent saved to {filepath}")
    
    @classmethod
    def load(cls, filename: str):
        """Load a trained agent"""
        # Get the project root directory (parent of src)
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))  # src directory
        project_root = os.path.dirname(current_dir)  # project root
        filepath = os.path.join(project_root, "agents", filename)
        
        with open(filepath, 'rb') as f:
            agent_data = pickle.load(f)
        
        # Create agent with same hyperparameters
        hyperparams = agent_data['hyperparameters']
        agent = cls(
            state_size=hyperparams['state_size'],
            action_size=hyperparams['action_size'],
            learning_rate=hyperparams['learning_rate'],
            gamma=hyperparams['gamma'],
            epsilon=hyperparams['epsilon'],
            epsilon_decay=hyperparams['epsilon_decay'],
            epsilon_min=hyperparams['epsilon_min']
        )
        
        # Load network weights
        agent.q_network.load_state_dict(agent_data['q_network_state_dict'])
        agent.target_network.load_state_dict(agent_data['target_network_state_dict'])
        agent.optimizer.load_state_dict(agent_data['optimizer_state_dict'])
        
        # Load training data
        training_data = agent_data['training_data']
        agent.episode_rewards = training_data['episode_rewards']
        agent.episode_steps = training_data['episode_steps']
        agent.success_rates = training_data['success_rates']
        agent.epsilon_history = training_data['epsilon_history']
        agent.loss_history = training_data['loss_history']
        agent.coins_collected_history = training_data.get('coins_collected_history', [])
        
        print(f"✅ Coin Collection Agent loaded from {filepath}")
        return agent

def train_coin_collection_agent(episodes: int = None, render_every: int = 1000, 
                               env_size: int = 10, seed: int = 42, 
                               save_every: int = 1000):
    """Train the coin collection DQN agent with weapon powerups"""
    
    # Try relative import first, fall back to absolute import
    try:
        from .grid_world import GridWorld
    except ImportError:
        from grid_world import GridWorld
    
    # Create environment
    env = GridWorld(size=env_size, seed=seed)
    
    # Calculate state size based on enhanced representation with weapon powerups
    # 2 (agent) + 2 (closest_coin) + 2 (closest_enemy) + 1 (coin_dist) + 1 (enemy_dist) 
    # + 2 (coin_dir) + 2 (enemy_dir) + 1 (num_coins) + 1 (num_enemies) + 1 (coin_density) 
    # + 1 (enemy_density) + 75 (5x5 local grid * 3 features) + 4 (weapon powerup info)
    # Total: 2+2+2+1+1+2+2+1+1+1+1+75+4 = 96
    state_size = 96
    
    # Create agent
    agent = CoinCollectionAgent(
        state_size=state_size,
        action_size=4,  # 4 actions: up, right, down, left

    )
    
    print(f"🎮 Training Coin Collection Agent with Weapon Powerups")
    print(f"   Environment: {env_size}x{env_size} grid")
    print(f"   State size: {state_size}")
    print(f"   Episodes: {'Infinite' if episodes is None else episodes}")
    print("=" * 50)
    
    episode = 0
    success_count = 0
    
    try:
        while episodes is None or episode < episodes:
            observation, info = env.reset()
            total_reward = 0
            steps = 0
            
            # Get initial state
            grid_state = env.grid
            agent_pos = info['agent_pos']
            coins = info['coins']
            enemy_positions = info.get('enemy_pos', [])
            weapon_powerups = info.get('weapon_powerups', [])
            has_weapon = info.get('has_weapon', False)
            weapon_turns_remaining = info.get('weapon_turns_remaining', 0)
            state = agent.get_state_representation(grid_state, agent_pos, coins, enemy_positions, 
                                                 weapon_powerups, has_weapon, weapon_turns_remaining)
            old_pos = agent_pos.copy()
            
            while True:
                action = agent.act(state, training=True)
                next_observation, env_reward, terminated, truncated, info = env.step(action)
                
                # Get new state
                new_grid_state = env.grid
                new_agent_pos = info['agent_pos']
                new_coins = info['coins']
                new_enemy_positions = info.get('enemy_pos', [])
                new_weapon_powerups = info.get('weapon_powerups', [])
                new_has_weapon = info.get('has_weapon', False)
                new_weapon_turns_remaining = info.get('weapon_turns_remaining', 0)
                next_state = agent.get_state_representation(new_grid_state, new_agent_pos, new_coins, 
                                                          new_enemy_positions, new_weapon_powerups,
                                                          new_has_weapon, new_weapon_turns_remaining)
                
                # Calculate reward using enhanced function
                enemy_collision = info.get('enemy_collision', False)
                coin_collected = info.get('coin_collected', False)
                weapon_collected = info.get('weapon_collected', False)
                enemy_died = info.get('enemy_died', False)
                reward = agent.calculate_reward(action, new_agent_pos, old_pos, 
                                              new_coins, new_enemy_positions, 
                                              enemy_collision, coin_collected, weapon_collected,
                                              enemy_died, new_has_weapon, new_weapon_turns_remaining, steps)
                
                done = terminated or truncated
                agent.remember(state, action, reward, next_state, done)
                agent.replay()
                
                total_reward += reward
                steps += 1
                state = next_state
                old_pos = new_agent_pos.copy()
                
                if terminated or truncated:
                    break
            
            # Track statistics
            agent.episode_rewards.append(total_reward)
            agent.episode_steps.append(steps)
            agent.epsilon_history.append(agent.epsilon)
            
            # Track success (all coins collected)
            if terminated and info['all_coins_collected'] and not info.get('enemy_collision', False):
                success_count += 1
            
            # Track coins collected
            coins_collected = info['coins_collected']
            agent.coins_collected_history.append(coins_collected)
            
            # Calculate success rate over last 100 episodes
            if episode >= 99:
                recent_success_rate = sum(1 for i in range(episode-99, episode+1) 
                                        if agent.episode_rewards[i] > 0) / 100
                agent.success_rates.append(recent_success_rate)
            else:
                agent.success_rates.append(success_count / (episode + 1))
            
            # Decay epsilon
            agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
            
            # Update target network every 100 episodes
            if episode % 100 == 0:
                agent.update_target_network()
            
            # Print progress
            if episode % 100 == 0:
                avg_reward = np.mean(agent.episode_rewards[-100:])
                avg_steps = np.mean(agent.episode_steps[-100:])
                avg_coins = np.mean(agent.coins_collected_history[-100:])
                success_rate = agent.success_rates[-1] if agent.success_rates else 0
                
                print(f"Episode {episode:4d} | "
                      f"Avg Reward: {avg_reward:6.2f} | "
                      f"Avg Steps: {avg_steps:4.1f} | "
                      f"Avg Coins: {avg_coins:3.1f} | "
                      f"Success Rate: {success_rate:.3f} | "
                      f"Epsilon: {agent.epsilon:.3f}")
            
            # Save agent periodically
            if save_every and episode % save_every == 0 and episode > 0:
                agent.save(f"coin_collection_agent_episode_{episode}.pkl")
            
            episode += 1
            
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
    
    # Save final agent
    agent.save("coin_collection_agent_final.pkl")
    
    print(f"\n✅ Training completed!")
    print(f"   Total episodes: {episode}")
    print(f"   Final success rate: {agent.success_rates[-1] if agent.success_rates else 0:.3f}")
    print(f"   Final epsilon: {agent.epsilon:.3f}")
    
    return agent

def test_coin_collection_agent(agent, episodes: int = 10, render: bool = True):
    """Test the coin collection DQN agent with weapon powerups"""
    
    # Try relative import first, fall back to absolute import
    try:
        from .grid_world import GridWorld
    except ImportError:
        from grid_world import GridWorld
    
    env = GridWorld(size=10)
    
    print(f"\n🧪 Testing Coin Collection Agent with Weapon Powerups for {episodes} episodes...")
    print("=" * 60)
    
    success_count = 0
    total_rewards = []
    total_steps = []
    total_coins_collected = []
    
    for episode in range(episodes):
        observation, info = env.reset()
        total_reward = 0
        steps = 0
        
        # Get initial state
        grid_state = env.grid
        agent_pos = info['agent_pos']
        coins = info['coins']
        enemy_positions = info.get('enemy_pos', [])
        weapon_powerups = info.get('weapon_powerups', [])
        has_weapon = info.get('has_weapon', False)
        weapon_turns_remaining = info.get('weapon_turns_remaining', 0)
        state = agent.get_state_representation(grid_state, agent_pos, coins, enemy_positions,
                                             weapon_powerups, has_weapon, weapon_turns_remaining)
        old_pos = agent_pos.copy()
        
        print(f"\nEpisode {episode + 1}:")
        
        while True:
            action = agent.act(state, training=False)
            next_observation, env_reward, terminated, truncated, info = env.step(action)
            
            new_grid_state = env.grid
            new_agent_pos = info['agent_pos']
            new_coins = info['coins']
            new_enemy_positions = info.get('enemy_pos', [])
            new_weapon_powerups = info.get('weapon_powerups', [])
            new_has_weapon = info.get('has_weapon', False)
            new_weapon_turns_remaining = info.get('weapon_turns_remaining', 0)
            next_state = agent.get_state_representation(new_grid_state, new_agent_pos, new_coins,
                                                      new_enemy_positions, new_weapon_powerups,
                                                      new_has_weapon, new_weapon_turns_remaining)
            
            enemy_collision = info.get('enemy_collision', False)
            coin_collected = info.get('coin_collected', False)
            weapon_collected = info.get('weapon_collected', False)
            enemy_died = info.get('enemy_died', False)
            reward = agent.calculate_reward(action, new_agent_pos, old_pos, 
                                          new_coins, new_enemy_positions, 
                                          enemy_collision, coin_collected, weapon_collected,
                                          enemy_died, new_has_weapon, new_weapon_turns_remaining, steps)
            
            total_reward += reward
            steps += 1
            state = next_state
            old_pos = new_agent_pos.copy()
            
            if render:
                env.render()
                time.sleep(0.3)
            
            if terminated or truncated:
                if terminated and info['all_coins_collected'] and not enemy_collision:
                    print("🎉 Success! All coins collected!")
                    success_count += 1
                elif enemy_collision:
                    print("💀 Game Over! Enemy collision!")
                else:
                    print("⏰ Time's up!")
                break
        
        total_rewards.append(total_reward)
        total_steps.append(steps)
        total_coins_collected.append(info['coins_collected'])
        print(f"Total reward: {total_reward:.2f}, Steps: {steps}, Coins: {info['coins_collected']}")
    
    print(f"\n📊 Test Results:")
    print(f"Success rate: {success_count}/{episodes} ({success_count/episodes*100:.1f}%)")
    print(f"Average reward: {np.mean(total_rewards):.2f}")
    print(f"Average steps: {np.mean(total_steps):.1f}")
    print(f"Average coins collected: {np.mean(total_coins_collected):.1f}")
    
    return success_count, total_rewards, total_steps

if __name__ == "__main__":
    # Train the agent
    agent = train_coin_collection_agent(episodes=5000, render_every=1000)
    
    # Test the agent
    test_coin_collection_agent(agent, episodes=5, render=True) 