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
        
        # Weapon powerup system
        self.weapon_powerups = []  # List of weapon powerup positions
        self.has_weapon = False
        self.weapon_turns_remaining = 0
        self.weapon_duration = 10  # Number of turns weapon lasts
        
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
            6: "W", # Weapon powerup
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
    
    def _place_weapon_powerups(self):
        """Place 1-2 weapon powerups at random positions"""
        num_weapons = random.randint(1, 2)
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
        """Check if agent is adjacent to any enemy (within 1 cell horizontally or vertically)"""
        agent_row, agent_col = self.agent_pos
        
        for enemy_pos in self.enemy_positions:
            enemy_row, enemy_col = enemy_pos
            
            # Check if adjacent (4-directional, Manhattan distance = 1)
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
    
    def _check_weapon_collection(self) -> bool:
        """Check if agent collected a weapon powerup"""
        agent_row, agent_col = self.agent_pos
        
        for i, weapon_pos in enumerate(self.weapon_powerups):
            if weapon_pos[0] == agent_row and weapon_pos[1] == agent_col:
                # Remove weapon powerup
                self.weapon_powerups.pop(i)
                # Activate weapon
                self.has_weapon = True
                self.weapon_turns_remaining = self.weapon_duration
                return True
        
        return False
    
    def _check_enemy_combat(self) -> Tuple[bool, list]:
        """Check if agent fights enemies (when armed) and return (enemy_died, killed_enemies)"""
        if not self.has_weapon or self.weapon_turns_remaining <= 0:
            return False, []
        
        agent_row, agent_col = self.agent_pos
        killed_enemies = []
        
        # Check for enemies adjacent to agent
        for i, enemy_pos in enumerate(self.enemy_positions):
            enemy_row, enemy_col = enemy_pos
            
            # Check if adjacent (4-directional)
            distance = abs(agent_row - enemy_row) + abs(agent_col - enemy_col)
            if distance == 1:
                killed_enemies.append(i)
        
        # Remove killed enemies (in reverse order to maintain indices)
        for i in reversed(killed_enemies):
            self.enemy_positions.pop(i)
        
        # Note: Weapon duration is decreased in the step function, not here
        return len(killed_enemies) > 0, killed_enemies
    
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
        self.weapon_powerups = []
        self.has_weapon = False
        self.weapon_turns_remaining = 0
        self._initialize_grid()
        
        # Check initial conditions
        enemy_collision = self._check_enemy_collision()
        
        info = {
            'agent_pos': self.agent_pos,
            'coins': self.coins,
            'enemy_pos': self.enemy_positions,
            'weapon_powerups': self.weapon_powerups,
            'has_weapon': self.has_weapon,
            'weapon_turns_remaining': self.weapon_turns_remaining,
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
        
        # Check for weapon collection
        weapon_collected = self._check_weapon_collection()

        # Check for enemy combat
        enemy_combat_result = self._check_enemy_combat()
        enemy_died, killed_enemies = enemy_combat_result

        # Decrease weapon duration with each move (if agent has weapon)
        if self.has_weapon and self.weapon_turns_remaining > 0:
            self.weapon_turns_remaining -= 1
            # Check if weapon expired
            if self.weapon_turns_remaining <= 0:
                self.has_weapon = False

        # Move enemies after agent moves
        self._move_enemies()
        
        # Clear the grid of all enemies first
        for i in range(self.size):
            for j in range(self.size):
                if self.grid[i, j] == 5:  # Clear all enemy positions
                    self.grid[i, j] = 0
        
        # Update grid display for remaining enemies only
        for enemy_pos in self.enemy_positions:
            if enemy_pos != self.agent_pos:
                self.grid[enemy_pos[0], enemy_pos[1]] = 5
        
        # Always ensure agent is visible on grid
        self.grid[self.agent_pos[0], self.agent_pos[1]] = 1
        
        # Check for enemy collision (game over condition) - only if not armed
        enemy_collision = False
        if not self.has_weapon or self.weapon_turns_remaining <= 0:
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
            'weapon_powerups': self.weapon_powerups,
            'has_weapon': self.has_weapon,
            'weapon_turns_remaining': self.weapon_turns_remaining,
            'coins_collected': self.coins_collected,
            'total_coins': len(self.coins) + self.coins_collected,
            'coin_collected': coin_collected,
            'weapon_collected': weapon_collected,
            'enemy_died': enemy_died,
            'killed_enemies': killed_enemies,
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
        """Render the current state of the environment with colors and ASCII icons only (no emojis)"""
        os.system('clear' if os.name == 'posix' else 'cls')
        
        # Color codes
        colors = {
            'reset': '\033[0m',
            'bold': '\033[1m',
            'red': '\033[91m',
            'green': '\033[92m',
            'yellow': '\033[93;1m',  # Use bright yellow for coins
            'blue': '\033[94m',
            'magenta': '\033[95m',
            'cyan': '\033[96m',
            'white': '\033[97m',
            'gray': '\033[90m',
        }
        
        # ASCII and color mappings for grid elements
        grid_display = {
            0: (".", colors['white']),      # Empty space
            1: ("A", colors['green']),      # Agent
            2: ("C", colors['yellow']),    # Coin (bright yellow)
            3: ("X", colors['gray']),      # Obstacle
            4: ("*", colors['magenta']),   # Reward (not used, but kept for completeness)
            5: ("E", colors['red']),       # Enemy
            6: ("W", colors['blue']),      # Weapon powerup
        }
        
        # Header
        print(f"{colors['bold']}{colors['cyan']}╔{'═' * (self.size * 4 + 2)}╗{colors['reset']}")
        print(f"{colors['bold']}{colors['cyan']}║{colors['reset']} {colors['bold']}{colors['white']}Grid World ({self.size}x{self.size}) - Step: {self.steps}{colors['reset']} {colors['cyan']}║{colors['reset']}")
        print(f"{colors['bold']}{colors['cyan']}╚{'═' * (self.size * 4 + 2)}╝{colors['reset']}")
        
        # Grid
        for i in range(self.size):
            row = f"{colors['cyan']}║{colors['reset']}"
            for j in range(self.size):
                cell_value = self.grid[i, j]
                icon, color = grid_display[cell_value]
                row += f" {color}{icon}{colors['reset']} "
            row += f"{colors['cyan']}║{colors['reset']}"
            print(row)
        
        # Footer
        print(f"{colors['bold']}{colors['cyan']}╔{'═' * (self.size * 4 + 2)}╗{colors['reset']}")
        
        # Legend
        print(f"{colors['cyan']}║{colors['reset']} {colors['bold']}{colors['white']}Legend:{colors['reset']}")
        legend_items = [
            (f"{colors['green']}A{colors['reset']} Agent", colors['green']),
            (f"{colors['yellow']}C{colors['reset']} Coin", colors['yellow']),  # Ensure bright yellow
            (f"{colors['gray']}X{colors['reset']} Obstacle", colors['gray']),
            (f"{colors['red']}E{colors['reset']} Enemy", colors['red']),
            (f"{colors['blue']}W{colors['reset']} Weapon", colors['blue']),
            (f"{colors['white']}.{colors['reset']} Empty", colors['white'])
        ]
        legend_line = " ".join([f"{item[0]}" for item in legend_items])
        print(f"{colors['cyan']}║{colors['reset']} {legend_line}")
        print(f"{colors['cyan']}║{colors['reset']}")
        
        # Game info
        print(f"{colors['cyan']}║{colors['reset']} {colors['bold']}{colors['blue']}Agent Position:{colors['reset']} {colors['white']}{self.agent_pos}{colors['reset']}")
        print(f"{colors['cyan']}║{colors['reset']} {colors['bold']}{colors['yellow']}Coins Remaining:{colors['reset']} {colors['white']}{len(self.coins)}{colors['reset']}")
        print(f"{colors['cyan']}║{colors['reset']} {colors['bold']}{colors['yellow']}Coins Collected:{colors['reset']} {colors['white']}{self.coins_collected}{colors['reset']}")
        print(f"{colors['cyan']}║{colors['reset']} {colors['bold']}{colors['red']}Enemy Positions:{colors['reset']} {colors['white']}{self.enemy_positions}{colors['reset']}")
        print(f"{colors['cyan']}║{colors['reset']} {colors['bold']}{colors['red']}Number of Enemies:{colors['reset']} {colors['white']}{len(self.enemy_positions)}{colors['reset']}")
        print(f"{colors['cyan']}║{colors['reset']} {colors['bold']}{colors['blue']}Weapon Powerups:{colors['reset']} {colors['white']}{self.weapon_powerups}{colors['reset']}")
        
        # Weapon status
        if self.has_weapon and self.weapon_turns_remaining > 0:
            print(f"{colors['cyan']}║{colors['reset']} {colors['bold']}{colors['blue']}Weapon Active:{colors['reset']} {colors['white']}{self.weapon_turns_remaining} turns remaining{colors['reset']}")
        else:
            print(f"{colors['cyan']}║{colors['reset']} {colors['bold']}{colors['gray']}Weapon Status:{colors['reset']} {colors['white']}None{colors['reset']}")
        
        # Calculate distances
        distance_to_nearest_coin = self.get_distance_to_nearest_coin()
        closest_enemy_distance = float('inf')
        for enemy_pos in self.enemy_positions:
            enemy_dist = abs(self.agent_pos[0] - enemy_pos[0]) + abs(self.agent_pos[1] - enemy_pos[1])
            closest_enemy_distance = min(closest_enemy_distance, enemy_dist)
        print(f"{colors['cyan']}║{colors['reset']} {colors['bold']}{colors['yellow']}Distance to Nearest Coin:{colors['reset']} {colors['white']}{distance_to_nearest_coin}{colors['reset']}")
        print(f"{colors['cyan']}║{colors['reset']} {colors['bold']}{colors['red']}Distance to Closest Enemy:{colors['reset']} {colors['white']}{closest_enemy_distance}{colors['reset']}")
        if closest_enemy_distance <= 2:
            print(f"{colors['cyan']}║{colors['reset']} {colors['bold']}{colors['yellow']}⚠️  WARNING: Enemy nearby!{colors['reset']}")
        if closest_enemy_distance <= 1:
            if self.has_weapon and self.weapon_turns_remaining > 0:
                print(f"{colors['cyan']}║{colors['reset']} {colors['bold']}{colors['blue']}⚔️  ATTACK: You can defeat this enemy!{colors['reset']}")
            else:
                print(f"{colors['cyan']}║{colors['reset']} {colors['bold']}{colors['red']}💥 DANGER: Enemy can catch you!{colors['reset']}")
        print(f"{colors['cyan']}║{colors['reset']} {colors['bold']}{colors['blue']}Valid Actions:{colors['reset']} {colors['white']}{self.get_valid_actions()}{colors['reset']}")
        print(f"{colors['bold']}{colors['cyan']}╚{'═' * (self.size * 4 + 2)}╝{colors['reset']}")
        print()

def manual_play():
    """Allow manual play of the environment with colored output"""
    env = GridWorld(10)
    state, info = env.reset()
    
    # Color codes
    colors = {
        'reset': '\033[0m',
        'bold': '\033[1m',
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'magenta': '\033[95m',
        'cyan': '\033[96m',
        'white': '\033[97m',
    }
    
    print(f"{colors['bold']}{colors['cyan']}╔{'═' * 60}╗{colors['reset']}")
    print(f"{colors['bold']}{colors['cyan']}║{colors['reset']} {colors['bold']}{colors['white']}Welcome to Grid World - Coin Collection with Weapons!{colors['reset']} {colors['cyan']}║{colors['reset']}")
    print(f"{colors['bold']}{colors['cyan']}║{colors['reset']} {colors['yellow']}Use WASD keys to move:{colors['reset']} {colors['cyan']}║{colors['reset']}")
    print(f"{colors['bold']}{colors['cyan']}║{colors['reset']} {colors['white']}W = Up, A = Left, S = Down, D = Right{colors['reset']} {colors['cyan']}║{colors['reset']}")
    print(f"{colors['bold']}{colors['cyan']}║{colors['reset']} {colors['white']}Q = Quit{colors['reset']} {colors['cyan']}║{colors['reset']}")
    print(f"{colors['bold']}{colors['cyan']}║{colors['reset']} {colors['green']}Objective: Collect coins and weapons to defeat enemies!{colors['reset']} {colors['cyan']}║{colors['reset']}")
    print(f"{colors['bold']}{colors['cyan']}║{colors['reset']} {colors['blue']}Weapons (W): Give you 10 moves to defeat enemies!{colors['reset']} {colors['cyan']}║{colors['reset']}")
    print(f"{colors['bold']}{colors['cyan']}╚{'═' * 60}╝{colors['reset']}")
    print()
    
    while True:
        env.render()
        
        # Get user input
        action = input(f"{colors['bold']}{colors['blue']}Enter move (W/A/S/D/Q):{colors['reset']} ").upper()
        
        if action == 'Q':
            print(f"{colors['bold']}{colors['green']}Thanks for playing!{colors['reset']}")
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
            print(f"{colors['bold']}{colors['red']}Invalid input! Use W/A/S/D to move or Q to quit.{colors['reset']}")
            continue
        
        observation, reward, terminated, truncated, info = env.step(action_code)
        
        # Show feedback for weapon collection
        if info.get('weapon_collected', False):
            print(f"{colors['bold']}{colors['blue']}⚔️  Weapon collected! You can now defeat enemies!{colors['reset']}")
        
        # Show feedback for enemy defeat
        if info.get('enemy_died', False):
            print(f"{colors['bold']}{colors['green']}💀 Enemy defeated! {info.get('weapon_turns_remaining', 0)} weapon turns remaining.{colors['reset']}")
        
        if terminated or truncated:
            env.render()
            if info['all_coins_collected']:
                print(f"{colors['bold']}{colors['green']}🎉 Congratulations! You collected all coins!{colors['reset']}")
            elif info['enemy_collision']:
                print(f"{colors['bold']}{colors['red']}💀 Game Over! Enemy collision!{colors['reset']}")
            elif truncated:
                print(f"{colors['bold']}{colors['yellow']}⏰ Time's up! You ran out of moves.{colors['reset']}")
            else:
                print(f"{colors['bold']}{colors['red']}💥 Game Over!{colors['reset']}")
            print(f"{colors['bold']}{colors['yellow']}Final Score: {info['coins_collected']} coins collected!{colors['reset']}")
            break

if __name__ == "__main__":
    manual_play() 