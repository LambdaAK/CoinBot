import numpy as np
import random
from typing import Tuple, Dict, Any, Optional
import time
import os

class GridWorld:
    def __init__(self, size: int = 10, max_steps: int = 50, seed: Optional[int] = None):
        self.size = size
        self.grid = np.zeros((size, size), dtype=int)
        self.agent_pos = [0, 0]  # Start at top-left
        self.coins = []  # List of coin positions
        self.enemy_positions = []  # Enemy positions
        self.obstacles = []
        self.steps = 0
        self.max_steps = max_steps
        self.seed = seed
        self.coins_collected = 0  # Track collected coins
        
        # RL environment properties
        self.action_space = 4  # 0=up, 1=right, 2=down, 3=left
        self.observation_space = size * size  # flattened grid
        self.reward_range = (-10.0, 10.0)  # min/max possible rewards (updated for enemy penalty)
        
        # Emoji mappings
        self.emoji_map = {
            0: "·",  # Empty space
            1: "A",  # Agent
            2: "C",  # Coin
            3: "X",  # Obstacle
            4: "★",  # Reward
            5: "E",  # Enemy
        }
        
        # Set seed for reproducibility
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Initialize the grid
        self._initialize_grid()
    
    def _is_coin_reachable(self) -> bool:
        """Check if at least one coin is reachable from start using DFS"""
        visited = set()
        stack = [(self.agent_pos[0], self.agent_pos[1])]
        
        while stack:
            row, col = stack.pop()
            
            # Check if we reached any coin
            if [row, col] in self.coins:
                return True
            
            # Mark as visited
            if (row, col) in visited:
                continue
            visited.add((row, col))
            
            # Check all 4 directions
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                
                # Check bounds
                if 0 <= new_row < self.size and 0 <= new_col < self.size:
                    # Check if not visited and not an obstacle
                    if ((new_row, new_col) not in visited and 
                        self.grid[new_row, new_col] != 3):
                        stack.append((new_row, new_col))
        
        return False
    
    def _initialize_grid(self):
        """Initialize the grid with agent, coins, enemies, and some obstacles"""
        self.grid = np.zeros((self.size, self.size), dtype=int)
        
        # Place agent
        self.grid[self.agent_pos[0], self.agent_pos[1]] = 1
        
        # Place coins at random positions
        self._place_coins()
        
        # Place enemies at random positions (not too close to agent or coins)
        self._place_enemies()
        
        # Add obstacles with DFS validation
        max_attempts = 100  # Prevent infinite loops
        attempts = 0
        
        while attempts < max_attempts:
            # Clear previous obstacles
            self.obstacles = []
            self.grid = np.zeros((self.size, self.size), dtype=int)
            self.grid[self.agent_pos[0], self.agent_pos[1]] = 1
            
            # Re-place coins and enemies
            self._place_coins()
            self._place_enemies()
            
            # Add random obstacles
            num_obstacles = random.randint(2, 4)
            for _ in range(num_obstacles):
                while True:
                    obstacle_pos = [random.randint(0, self.size-1), random.randint(0, self.size-1)]
                    if (obstacle_pos != self.agent_pos and 
                        obstacle_pos not in self.coins and
                        obstacle_pos not in self.enemy_positions and 
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
    
    def _create_simple_path(self):
        """Create a simple guaranteed path with coins"""
        self.obstacles = []
        self.grid = np.zeros((self.size, self.size), dtype=int)
        self.grid[self.agent_pos[0], self.agent_pos[1]] = 1
        
        # Re-place coins and enemies
        self._place_coins()
        self._place_enemies()
        
        # Add minimal obstacles that don't block the path
        # For a 5x5 grid, we can add obstacles in corners or edges
        safe_positions = [
            [1, 1], [1, 3], [3, 1], [3, 3],  # Corner areas
            [0, 2], [2, 0], [2, 4], [4, 2]   # Edge areas
        ]
        
        num_obstacles = random.randint(1, 3)
        selected_obstacles = random.sample(safe_positions, num_obstacles)
        
        for pos in selected_obstacles:
            if pos not in self.coins and pos not in self.enemy_positions:
                self.obstacles.append(pos)
                self.grid[pos[0], pos[1]] = 3
    
    def _place_coins(self):
        """Place 3-6 coins at random positions"""
        num_coins = random.randint(3, 6)
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
        """Place 1-2 enemies at safe distances from agent and coins"""
        # Randomly choose number of enemies (1-2)
        num_enemies = random.randint(1, 2)
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
    
    def _move_enemies(self):
        """Move all enemies randomly to adjacent cells (50% chance each to move)"""
        for i, enemy_pos in enumerate(self.enemy_positions):
            # Only move 50% of the time
            if random.random() < 0.5:
                # Clear current enemy position
                self.grid[enemy_pos[0], enemy_pos[1]] = 0
                
                # Get possible moves
                possible_moves = []
                directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
                
                for dr, dc in directions:
                    new_row = enemy_pos[0] + dr
                    new_col = enemy_pos[1] + dc
                    
                    # Check bounds
                    if (0 <= new_row < self.size and 0 <= new_col < self.size):
                        # Check if not obstacle or coin (enemy can move through agent space)
                        if self.grid[new_row, new_col] not in [2, 3]:  # not coin or obstacle
                            # Check if not too close to other enemies
                            too_close = False
                            for j, other_enemy in enumerate(self.enemy_positions):
                                if i != j:  # Don't check against self
                                    dist = abs(new_row - other_enemy[0]) + abs(new_col - other_enemy[1])
                                    if dist < 1:  # Don't move on top of other enemies
                                        too_close = True
                                        break
                            
                            if not too_close:
                                possible_moves.append([new_row, new_col])
                
                # If no valid moves, stay in place
                if possible_moves:
                    self.enemy_positions[i] = random.choice(possible_moves)
                
                # Place enemy in new position (may overwrite agent temporarily)
                new_pos = self.enemy_positions[i]
                if self.grid[new_pos[0], new_pos[1]] == 1:
                    # Enemy moved to agent position - will be handled in collision check
                    pass
                else:
                    self.grid[new_pos[0], new_pos[1]] = 5
    
    def _check_enemy_collision(self) -> bool:
        """Check if agent is adjacent to or on any enemy"""
        agent_row, agent_col = self.agent_pos
        
        for enemy_pos in self.enemy_positions:
            enemy_row, enemy_col = enemy_pos
            
            # Check if on same position
            if agent_row == enemy_row and agent_col == enemy_col:
                return True
            
            # Check if adjacent (4-directional)
            distance = abs(agent_row - enemy_row) + abs(agent_col - enemy_col)
            if distance == 1:
                return True
        
        return False
    
    def _check_coin_collection(self) -> bool:
        """Check if agent collected a coin"""
        agent_row, agent_col = self.agent_pos
        
        for i, coin_pos in enumerate(self.coins):
            if coin_pos[0] == agent_row and coin_pos[1] == agent_col:
                # Remove coin
                self.coins.pop(i)
                self.coins_collected += 1
                return True
        
        return False
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state
        
        Returns:
            (observation, info)
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        self.agent_pos = [0, 0]
        self.steps = 0
        self.coins_collected = 0
        self.obstacles = []
        self._initialize_grid()
        
        # Check initial conditions
        enemy_collision = self._check_enemy_collision()
        
        info = {
            'agent_pos': self.agent_pos,
            'coins': self.coins,
            'enemy_pos': self.enemy_positions,
            'coins_collected': self.coins_collected,
            'total_coins': len(self.coins),
            'steps': self.steps,
            'enemy_collision': enemy_collision
        }
        
        return self._get_state(), info
    
    def _get_state(self) -> np.ndarray:
        """Get current state representation"""
        return self.grid.copy()
    
    def _get_observation(self) -> np.ndarray:
        """Get observation for RL agent (flattened grid)"""
        return self.grid.flatten()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment
        
        Args:
            action: 0=up, 1=right, 2=down, 3=left
            
        Returns:
            (observation, reward, terminated, truncated, info)
        """
        if not 0 <= action < self.action_space:
            raise ValueError(f"Invalid action {action}. Must be 0-{self.action_space-1}")
        
        self.steps += 1
        
        # Store old position for reward calculation
        old_pos = self.agent_pos.copy()
        
        # Calculate new position
        new_pos = self.agent_pos.copy()
        if action == 0:  # Up
            new_pos[0] = max(0, new_pos[0] - 1)
        elif action == 1:  # Right
            new_pos[1] = min(self.size - 1, new_pos[1] + 1)
        elif action == 2:  # Down
            new_pos[0] = min(self.size - 1, new_pos[0] + 1)
        elif action == 3:  # Left
            new_pos[1] = max(0, new_pos[1] - 1)
        
        # Check if new position is valid (not an obstacle)
        if self.grid[new_pos[0], new_pos[1]] != 3:  # 3 is obstacle
            # Update grid
            self.grid[self.agent_pos[0], self.agent_pos[1]] = 0
            self.agent_pos = new_pos
            self.grid[self.agent_pos[0], self.agent_pos[1]] = 1
        
        # Check for coin collection
        coin_collected = self._check_coin_collection()
        
        # Move enemies after agent moves
        self._move_enemies()
        
        # Update grid display for all enemies
        for enemy_pos in self.enemy_positions:
            if enemy_pos != self.agent_pos:
                self.grid[enemy_pos[0], enemy_pos[1]] = 5
        
        # Always ensure agent is visible on grid
        self.grid[self.agent_pos[0], self.agent_pos[1]] = 1
        
        # Check for enemy collision (game over condition)
        enemy_collision = self._check_enemy_collision()
        
        # Check if all coins collected or enemy collision
        terminated = False
        all_coins_collected = len(self.coins) == 0
        if all_coins_collected or enemy_collision:
            terminated = True
        
        # Check if max steps reached
        truncated = False
        if self.steps >= self.max_steps:
            truncated = True
        
        info = {
            'steps': self.steps,
            'agent_pos': self.agent_pos,
            'coins': self.coins,
            'enemy_pos': self.enemy_positions,
            'coins_collected': self.coins_collected,
            'total_coins': len(self.coins) + self.coins_collected,
            'coin_collected': coin_collected,
            'enemy_collision': enemy_collision,
            'all_coins_collected': all_coins_collected,
            'action': action
        }
        
        return self._get_observation(), 0.0, terminated, truncated, info
    
    def get_valid_actions(self) -> list:
        """Get list of valid actions from current position"""
        valid_actions = []
        for action in range(self.action_space):
            new_pos = self.agent_pos.copy()
            if action == 0:  # Up
                new_pos[0] = max(0, new_pos[0] - 1)
            elif action == 1:  # Right
                new_pos[1] = min(self.size - 1, new_pos[1] + 1)
            elif action == 2:  # Down
                new_pos[0] = min(self.size - 1, new_pos[0] + 1)
            elif action == 3:  # Left
                new_pos[1] = max(0, new_pos[1] - 1)
            
            if self.grid[new_pos[0], new_pos[1]] != 3:  # Not an obstacle
                valid_actions.append(action)
        
        return valid_actions
    
    def get_distance_to_nearest_coin(self) -> int:
        """Get Manhattan distance from agent to nearest coin"""
        if not self.coins:
            return 0
        
        min_distance = float('inf')
        for coin_pos in self.coins:
            distance = abs(self.agent_pos[0] - coin_pos[0]) + abs(self.agent_pos[1] - coin_pos[1])
            min_distance = min(min_distance, distance)
        
        return int(min_distance)
    
    def render(self):
        """Render the current state of the environment"""
        os.system('clear' if os.name == 'posix' else 'cls')
        print(f"Grid World ({self.size}x{self.size}) - Step: {self.steps}")
        print("=" * (self.size * 4 + 2))
        
        for i in range(self.size):
            row = "|"
            for j in range(self.size):
                cell_value = self.grid[i, j]
                emoji = self.emoji_map[cell_value]
                # Center the emoji in a fixed-width cell
                row += f" {emoji} "
            row += "|"
            print(row)
        
        print("=" * (self.size * 4 + 2))
        print("Legend: A=Agent, C=Coin, X=Obstacle, E=Enemy, ·=Empty")
        print(f"Agent Position: {self.agent_pos}")
        print(f"Coins Remaining: {len(self.coins)}")
        print(f"Coins Collected: {self.coins_collected}")
        print(f"Enemy Positions: {self.enemy_positions}")
        print(f"Number of Enemies: {len(self.enemy_positions)}")
        
        # Calculate distances
        distance_to_nearest_coin = self.get_distance_to_nearest_coin()
        
        # Find closest enemy distance
        closest_enemy_distance = float('inf')
        for enemy_pos in self.enemy_positions:
            enemy_dist = abs(self.agent_pos[0] - enemy_pos[0]) + abs(self.agent_pos[1] - enemy_pos[1])
            closest_enemy_distance = min(closest_enemy_distance, enemy_dist)
        
        print(f"Distance to Nearest Coin: {distance_to_nearest_coin}")
        print(f"Distance to Closest Enemy: {closest_enemy_distance}")
        
        # Warning if close to any enemy
        if closest_enemy_distance <= 2:
            print("⚠️  WARNING: Enemy nearby!")
        if closest_enemy_distance <= 1:
            print("💥 DANGER: Enemy can catch you!")
        print(f"Valid Actions: {self.get_valid_actions()}")
        print()

def manual_play():
    """Allow manual play of the environment"""
    env = GridWorld(10)
    state, info = env.reset()
    
    print("Welcome to Grid World - Coin Collection!")
    print("Use WASD keys to move:")
    print("W = Up, A = Left, S = Down, D = Right")
    print("Q = Quit")
    print("Objective: Collect as many coins as possible while avoiding enemies!")
    print()
    
    while True:
        env.render()
        
        # Get user input
        action = input("Enter move (W/A/S/D/Q): ").upper()
        
        if action == 'Q':
            print("Thanks for playing!")
            break
        elif action == 'W':
            action_code = 0
        elif action == 'D':
            action_code = 1
        elif action == 'S':
            action_code = 2
        elif action == 'A':
            action_code = 3
        else:
            print("Invalid input! Use W/A/S/D to move or Q to quit.")
            continue
        
        observation, reward, terminated, truncated, info = env.step(action_code)
        
        if terminated or truncated:
            env.render()
            if info['all_coins_collected']:
                print("🎉 Congratulations! You collected all coins!")
            elif info['enemy_collision']:
                print("💀 Game Over! Enemy collision!")
            elif truncated:
                print("⏰ Time's up! You ran out of moves.")
            else:
                print("💥 Game Over!")
            print(f"Final Score: {info['coins_collected']} coins collected!")
            break

if __name__ == "__main__":
    manual_play() 