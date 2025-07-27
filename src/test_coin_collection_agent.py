#!/usr/bin/env python3
"""
Test Coin Collection Agent

Loads a trained coin collection agent and evaluates it on the GridWorld environment.
"""
import time
import numpy as np
from DQN_agent import CoinCollectionAgent
from grid_world import GridWorld
import os
import random
from typing import Optional


class CustomGridWorld(GridWorld):
    """Custom GridWorld with configurable ranges for obstacles, coins, enemies, and weapon powerups"""
    
    def __init__(self, size: int = 10, max_steps: int = 50, seed: Optional[int] = None,
                 obstacle_range: tuple = (2, 4), coin_range: tuple = (3, 6), 
                 enemy_range: tuple = (1, 2), weapon_powerup_range: tuple = (1, 2)):
        self.obstacle_range = obstacle_range
        self.coin_range = coin_range
        self.enemy_range = enemy_range
        self.weapon_powerup_range = weapon_powerup_range
        super().__init__(size, max_steps, seed)
    
    def _initialize_grid(self):
        """Initialize the grid with agent, coins, enemies, obstacles, and weapon powerups"""
        self.grid = np.zeros((self.size, self.size), dtype=int)
        
        # Place agent
        self.grid[self.agent_pos[0], self.agent_pos[1]] = 1
        
        # Place coins at random positions
        self._place_coins()
        
        # Place enemies at random positions (not too close to agent or coins)
        self._place_enemies()
        
        # Place weapon powerups
        self._place_weapon_powerups()
        
        # Add obstacles with DFS validation
        max_attempts = 100  # Prevent infinite loops
        attempts = 0
        
        while attempts < max_attempts:
            # Clear previous obstacles
            self.obstacles = []
            self.grid = np.zeros((self.size, self.size), dtype=int)
            self.grid[self.agent_pos[0], self.agent_pos[1]] = 1
            
            # Re-place coins, enemies, and weapon powerups
            self._place_coins()
            self._place_enemies()
            self._place_weapon_powerups()
            
            # Add random obstacles using custom range
            num_obstacles = random.randint(self.obstacle_range[0], self.obstacle_range[1])
            for _ in range(num_obstacles):
                while True:
                    obstacle_pos = [random.randint(0, self.size-1), random.randint(0, self.size-1)]
                    if (obstacle_pos != self.agent_pos and 
                        obstacle_pos not in self.coins and
                        obstacle_pos not in self.enemy_positions and 
                        obstacle_pos not in self.weapon_powerups and
                        obstacle_pos not in self.obstacles):
                        self.obstacles.append(obstacle_pos)
                        self.grid[obstacle_pos[0], obstacle_pos[1]] = 3
                        break
            
            # Check if at least one coin is reachable
            if self._is_coin_reachable():
                break
            
            attempts += 1
        
        # If we couldn't find a valid configuration, create a simple path
        if attempts >= max_attempts:
            self._create_simple_path()
    
    def _place_coins(self):
        """Place coins at random positions using custom range"""
        num_coins = random.randint(self.coin_range[0], self.coin_range[1])
        self.coins = []
        
        for _ in range(num_coins):
            max_attempts = 50
            attempts = 0
            
            while attempts < max_attempts:
                coin_row = random.randint(0, self.size - 1)
                coin_col = random.randint(0, self.size - 1)
                coin_pos = [coin_row, coin_col]
                
                # Check distance from agent
                agent_dist = abs(coin_pos[0] - self.agent_pos[0]) + abs(coin_pos[1] - self.agent_pos[1])
                
                # Coin should be at least 1 step away from agent
                # Also check distance from other coins
                too_close_to_other_coin = False
                for existing_coin in self.coins:
                    coin_dist = abs(coin_pos[0] - existing_coin[0]) + abs(coin_pos[1] - existing_coin[1])
                    if coin_dist < 1:  # Keep coins at least 1 step apart
                        too_close_to_other_coin = True
                        break
                
                if (coin_pos != self.agent_pos and 
                    agent_dist >= 1 and
                    not too_close_to_other_coin):
                    self.coins.append(coin_pos)
                    self.grid[coin_pos[0], coin_pos[1]] = 2
                    break
                
                attempts += 1
            
            # Fallback placement if we couldn't find a good spot
            if attempts >= max_attempts:
                # Try to place in a corner or edge
                fallback_positions = [
                    [1, 1], [1, self.size-2], [self.size-2, 1], [self.size-2, self.size-2],
                    [0, self.size//2], [self.size//2, 0], [self.size-1, self.size//2], [self.size//2, self.size-1]
                ]
                
                for pos in fallback_positions:
                    if (pos != self.agent_pos and 
                        pos not in self.coins):
                        self.coins.append(pos)
                        self.grid[pos[0], pos[1]] = 2
                        break
                else:
                    # Last resort: place somewhere random
                    self.coins.append([1, 1])
                    self.grid[1, 1] = 2
    
    def _place_enemies(self):
        """Place enemies at safe distances from agent and coins using custom range"""
        # Randomly choose number of enemies using custom range
        num_enemies = random.randint(self.enemy_range[0], self.enemy_range[1])
        self.enemy_positions = []
        
        for enemy_id in range(num_enemies):
            max_attempts = 50
            attempts = 0
            
            while attempts < max_attempts:
                enemy_row = random.randint(0, self.size - 1)
                enemy_col = random.randint(0, self.size - 1)
                enemy_pos = [enemy_row, enemy_col]
                
                # Check distance from agent and coins
                agent_dist = abs(enemy_pos[0] - self.agent_pos[0]) + abs(enemy_pos[1] - self.agent_pos[1])
                
                # Enemy should be at least 2 steps away from agent
                # Also check distance from other enemies
                too_close_to_other_enemy = False
                for existing_enemy in self.enemy_positions:
                    enemy_dist = abs(enemy_pos[0] - existing_enemy[0]) + abs(enemy_pos[1] - existing_enemy[1])
                    if enemy_dist < 2:  # Keep enemies at least 2 steps apart
                        too_close_to_other_enemy = True
                        break
                
                if (enemy_pos != self.agent_pos and 
                    enemy_pos not in self.coins and
                    agent_dist >= 2 and
                    not too_close_to_other_enemy):
                    self.enemy_positions.append(enemy_pos)
                    self.grid[enemy_pos[0], enemy_pos[1]] = 5
                    break
                
                attempts += 1
            
            # Fallback placement if we couldn't find a good spot
            if attempts >= max_attempts:
                # Try to place in a corner or edge
                fallback_positions = [
                    [1, 1], [1, self.size-2], [self.size-2, 1], [self.size-2, self.size-2],
                    [0, self.size//2], [self.size//2, 0], [self.size-1, self.size//2], [self.size//2, self.size-1]
                ]
                
                for pos in fallback_positions:
                    if (pos != self.agent_pos and 
                        pos not in self.coins and
                        pos not in self.enemy_positions):
                        self.enemy_positions.append(pos)
                        self.grid[pos[0], pos[1]] = 5
                        break
                else:
                    # Last resort: place somewhere random
                    self.enemy_positions.append([1, 1])
                    self.grid[1, 1] = 5
    
    def _place_weapon_powerups(self):
        """Place weapon powerups at random positions using custom range"""
        num_weapons = random.randint(self.weapon_powerup_range[0], self.weapon_powerup_range[1])
        self.weapon_powerups = []
        
        for _ in range(num_weapons):
            max_attempts = 50
            attempts = 0
            
            while attempts < max_attempts:
                weapon_row = random.randint(0, self.size - 1)
                weapon_col = random.randint(0, self.size - 1)
                weapon_pos = [weapon_row, weapon_col]
                
                # Simple placement: just avoid agent position and other weapons
                if (weapon_pos != self.agent_pos and 
                    weapon_pos not in self.weapon_powerups):
                    self.weapon_powerups.append(weapon_pos)
                    self.grid[weapon_pos[0], weapon_pos[1]] = 6
                    break
                
                attempts += 1
            
            # Fallback placement if we couldn't find a good spot
            if attempts >= max_attempts:
                # Try to place in a corner or edge
                fallback_positions = [
                    [1, 1], [1, self.size-2], [self.size-2, 1], [self.size-2, self.size-2],
                    [0, self.size//2], [self.size//2, 0], [self.size-1, self.size//2], [self.size//2, self.size-1]
                ]
                
                for pos in fallback_positions:
                    if (pos != self.agent_pos and 
                        pos not in self.weapon_powerups):
                        self.weapon_powerups.append(pos)
                        self.grid[pos[0], pos[1]] = 6
                        break
                else:
                    # Last resort: place somewhere random
                    self.weapon_powerups.append([2, 2])
                    self.grid[2, 2] = 6

    def _create_simple_path(self):
        """Create a simple guaranteed path with coins"""
        self.obstacles = []
        self.grid = np.zeros((self.size, self.size), dtype=int)
        self.grid[self.agent_pos[0], self.agent_pos[1]] = 1
        
        # Re-place coins, enemies, and weapon powerups
        self._place_coins()
        self._place_enemies()
        self._place_weapon_powerups()
        
        # Add minimal obstacles that don't block the path
        # For a 5x5 grid, we can add obstacles in corners or edges
        safe_positions = [
            [1, 1], [1, 3], [3, 1], [3, 3],  # Corner areas
            [0, 2], [2, 0], [2, 4], [4, 2]   # Edge areas
        ]
        
        num_obstacles = random.randint(1, 3)
        selected_obstacles = random.sample(safe_positions, num_obstacles)
        
        for pos in selected_obstacles:
            if pos not in self.coins and pos not in self.enemy_positions and pos not in self.weapon_powerups:
                self.obstacles.append(pos)
                self.grid[pos[0], pos[1]] = 3


def get_range_input(prompt, default_min, default_max):
    """Helper function to get range input from user"""
    while True:
        try:
            user_input = input(prompt).strip()
            if not user_input:
                return (default_min, default_max)
            
            if ',' in user_input:
                parts = user_input.split(',')
                if len(parts) == 2:
                    min_val = int(parts[0].strip())
                    max_val = int(parts[1].strip())
                    if min_val <= max_val and min_val >= 0:
                        return (min_val, max_val)
            
            print(f"Please enter a range as 'min,max' (e.g., '{default_min},{default_max}')")
        except ValueError:
            print("Please enter valid numbers.")


def main():
    print("\n=== Coin Collection Agent Tester ===")
    
    # List available agents (look in project root, not src directory)
    current_dir = os.path.dirname(os.path.abspath(__file__))  # src directory
    project_root = os.path.dirname(current_dir)  # project root
    agents_dir = os.path.join(project_root, "agents")
    
    if not os.path.exists(agents_dir):
        print(f"No agents directory found at: {agents_dir}")
        return
        
    agent_files = [f for f in os.listdir(agents_dir) if f.endswith('.pkl')]
    if not agent_files:
        print("No agent files found in 'agents/' directory.")
        return
    
    print("Available agents:")
    for idx, fname in enumerate(agent_files):
        print(f"  [{idx+1}] {fname}")
    
    agent_idx = input(f"Select agent [1-{len(agent_files)}] (default 1): ").strip()
    try:
        agent_idx = int(agent_idx) - 1 if agent_idx else 0
        if not (0 <= agent_idx < len(agent_files)):
            print("Invalid selection.")
            return
    except ValueError:
        agent_idx = 0
    
    agent_file = agent_files[agent_idx]

    # Get test parameters
    try:
        episodes = int(input("Number of test episodes [10]: ") or 10)
    except ValueError:
        episodes = 10
    
    try:
        grid_size = int(input("Grid size [10]: ") or 10)
    except ValueError:
        grid_size = 10
    
    # Get ranges for obstacles, coins, and enemies
    print("\n--- Environment Configuration ---")
    obstacle_range = get_range_input("Obstacle range (min,max) [2,4]: ", 2, 4)
    coin_range = get_range_input("Coin range (min,max) [3,6]: ", 3, 6)
    enemy_range = get_range_input("Enemy range (min,max) [1,2]: ", 1, 2)
    weapon_powerup_range = get_range_input("Weapon powerup range (min,max) [1,2]: ", 1, 2)
    
    try:
        max_steps = int(input("Max steps per episode [50]: ") or 50)
    except ValueError:
        max_steps = 50
    
    render_input = input("Render environment? [y/N]: ").strip().lower()
    render = render_input == 'y'

    print(f"\n🔍 Loading agent from: agents/{agent_file}")
    agent = CoinCollectionAgent.load(agent_file)

    print(f"\n🧪 Testing agent for {episodes} episodes on {grid_size}x{grid_size} grid...")
    print(f"Environment settings:")
    print(f"  Obstacles: {obstacle_range[0]}-{obstacle_range[1]}")
    print(f"  Coins: {coin_range[0]}-{coin_range[1]}")
    print(f"  Enemies: {enemy_range[0]}-{enemy_range[1]}")
    print(f"  Weapon Powerups: {weapon_powerup_range[0]}-{weapon_powerup_range[1]}")
    print(f"  Max steps: {max_steps}")
    
    # Create custom environment with specified parameters
    env = CustomGridWorld(
        size=grid_size, 
        max_steps=max_steps,
        obstacle_range=obstacle_range,
        coin_range=coin_range,
        enemy_range=enemy_range,
        weapon_powerup_range=weapon_powerup_range
    )

    success_count = 0
    total_rewards = []
    total_steps = []
    total_coins_collected = []

    for episode in range(episodes):
        observation, info = env.reset()
        total_reward = 0
        steps = 0
        grid_state = env.grid
        agent_pos = info['agent_pos']
        coins = info['coins']
        enemy_positions = info.get('enemy_pos', [])
        weapon_powerups = info.get('weapon_powerups', [])
        has_weapon = info.get('has_weapon', False)
        weapon_turns_remaining = info.get('weapon_turns_remaining', 0)
        weapon_inventory = info.get('weapon_inventory', [])
        state = agent.get_state_representation(grid_state, agent_pos, coins, enemy_positions,
                                             weapon_powerups, has_weapon, weapon_turns_remaining, weapon_inventory)
        old_pos = agent_pos.copy()
        
        print(f"\nEpisode {episode + 1}:")
        
        while True:
            action = agent.act(state, training=False)
            
            # Handle weapon activation (action 4)
            if action == 4:
                weapon_activated = env.activate_weapon()
                if weapon_activated:
                    print("⚔️  Weapon activated!")
                else:
                    print("⚠️  No weapons to activate or weapon already active!")
                
                # Update state even when weapon activation fails to prevent infinite loop
                new_grid_state = env.grid
                new_agent_pos = info['agent_pos']
                new_coins = info['coins']
                new_enemy_positions = info.get('enemy_pos', [])
                new_weapon_powerups = info.get('weapon_powerups', [])
                new_has_weapon = info.get('has_weapon', False)
                new_weapon_turns_remaining = info.get('weapon_turns_remaining', 0)
                new_weapon_inventory = info.get('weapon_inventory', [])
                next_state = agent.get_state_representation(new_grid_state, new_agent_pos, new_coins, new_enemy_positions,
                                                         new_weapon_powerups, new_has_weapon, new_weapon_turns_remaining, new_weapon_inventory)
                
                # Calculate reward for weapon activation attempt
                reward = agent.calculate_reward(action, new_agent_pos, old_pos, 
                                              new_coins, new_enemy_positions, 
                                              False, False, False, False,
                                              new_has_weapon, new_weapon_turns_remaining, steps, weapon_activated)
                
                total_reward += reward
                steps += 1
                state = next_state
                old_pos = new_agent_pos.copy()
                continue
            
            next_observation, env_reward, terminated, truncated, info = env.step(action)
            
            new_grid_state = env.grid
            new_agent_pos = info['agent_pos']
            new_coins = info['coins']
            new_enemy_positions = info.get('enemy_pos', [])
            new_weapon_powerups = info.get('weapon_powerups', [])
            new_has_weapon = info.get('has_weapon', False)
            new_weapon_turns_remaining = info.get('weapon_turns_remaining', 0)
            new_weapon_inventory = info.get('weapon_inventory', [])
            next_state = agent.get_state_representation(new_grid_state, new_agent_pos, new_coins, new_enemy_positions,
                                                         new_weapon_powerups, new_has_weapon, new_weapon_turns_remaining, new_weapon_inventory)
            
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
                time.sleep(0.2)
            
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


if __name__ == "__main__":
    main() 