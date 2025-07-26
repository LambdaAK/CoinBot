import numpy as np
import random
from typing import Tuple, Dict, Any, Optional
import time
import os

class GridWorld:
    def __init__(self, size: int = 8, max_steps: int = 50, seed: Optional[int] = None):
        self.size = size
        self.grid = np.zeros((size, size), dtype=int)
        self.agent_pos = [0, 0]  # Start at top-left
        self.goal_pos = [size-1, size-1]  # Goal at bottom-right
        self.enemy_pos = [0, 0]  # Enemy position
        self.obstacles = []
        self.steps = 0
        self.max_steps = max_steps
        self.seed = seed
        
        # RL environment properties
        self.action_space = 4  # 0=up, 1=right, 2=down, 3=left
        self.observation_space = size * size  # flattened grid
        self.reward_range = (-10.0, 10.0)  # min/max possible rewards (updated for enemy penalty)
        
        # Emoji mappings
        self.emoji_map = {
            0: "·",  # Empty space
            1: "A",  # Agent
            2: "G",  # Goal
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
    
    def _is_goal_reachable(self) -> bool:
        """Check if goal is reachable from start using DFS"""
        visited = set()
        stack = [(self.agent_pos[0], self.agent_pos[1])]
        
        while stack:
            row, col = stack.pop()
            
            # Check if we reached the goal
            if (row, col) == (self.goal_pos[0], self.goal_pos[1]):
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
        """Initialize the grid with agent, goal, enemy, and some obstacles"""
        self.grid = np.zeros((self.size, self.size), dtype=int)
        
        # Place agent
        self.grid[self.agent_pos[0], self.agent_pos[1]] = 1
        
        # Place goal
        self.grid[self.goal_pos[0], self.goal_pos[1]] = 2
        
        # Place enemy at random position (not too close to agent or goal)
        self._place_enemy()
        
        # Add obstacles with DFS validation
        max_attempts = 100  # Prevent infinite loops
        attempts = 0
        
        while attempts < max_attempts:
            # Clear previous obstacles
            self.obstacles = []
            self.grid = np.zeros((self.size, self.size), dtype=int)
            self.grid[self.agent_pos[0], self.agent_pos[1]] = 1
            self.grid[self.goal_pos[0], self.goal_pos[1]] = 2
            self._place_enemy()
            
            # Add random obstacles
            num_obstacles = random.randint(2, 4)
            for _ in range(num_obstacles):
                while True:
                    obstacle_pos = [random.randint(0, self.size-1), random.randint(0, self.size-1)]
                    if (obstacle_pos != self.agent_pos and 
                        obstacle_pos != self.goal_pos and
                        obstacle_pos != self.enemy_pos and 
                        obstacle_pos not in self.obstacles):
                        self.obstacles.append(obstacle_pos)
                        self.grid[obstacle_pos[0], obstacle_pos[1]] = 3
                        break
            
            # Check if goal is reachable
            if self._is_goal_reachable():
                break
            
            attempts += 1
        
        # If we couldn't find a valid configuration, create a simple path
        if attempts >= max_attempts:
            self._create_simple_path()
    
    def _create_simple_path(self):
        """Create a simple guaranteed path from start to goal"""
        self.obstacles = []
        self.grid = np.zeros((self.size, self.size), dtype=int)
        self.grid[self.agent_pos[0], self.agent_pos[1]] = 1
        self.grid[self.goal_pos[0], self.goal_pos[1]] = 2
        self._place_enemy()
        
        # Add minimal obstacles that don't block the path
        # For a 5x5 grid, we can add obstacles in corners or edges
        safe_positions = [
            [1, 1], [1, 3], [3, 1], [3, 3],  # Corner areas
            [0, 2], [2, 0], [2, 4], [4, 2]   # Edge areas
        ]
        
        num_obstacles = random.randint(1, 3)
        selected_obstacles = random.sample(safe_positions, num_obstacles)
        
        for pos in selected_obstacles:
            self.obstacles.append(pos)
            self.grid[pos[0], pos[1]] = 3
    
    def _place_enemy(self):
        """Place enemy at a safe distance from agent and goal"""
        max_attempts = 50
        attempts = 0
        
        while attempts < max_attempts:
            enemy_row = random.randint(0, self.size - 1)
            enemy_col = random.randint(0, self.size - 1)
            enemy_pos = [enemy_row, enemy_col]
            
            # Check distance from agent and goal
            agent_dist = abs(enemy_pos[0] - self.agent_pos[0]) + abs(enemy_pos[1] - self.agent_pos[1])
            goal_dist = abs(enemy_pos[0] - self.goal_pos[0]) + abs(enemy_pos[1] - self.goal_pos[1])
            
            # Enemy should be at least 2 steps away from agent and goal
            if (enemy_pos != self.agent_pos and 
                enemy_pos != self.goal_pos and
                agent_dist >= 2 and 
                goal_dist >= 2):
                self.enemy_pos = enemy_pos
                self.grid[enemy_pos[0], enemy_pos[1]] = 5
                return
            
            attempts += 1
        
        # Fallback: place enemy in center if possible
        center = self.size // 2
        if ([center, center] != self.agent_pos and 
            [center, center] != self.goal_pos):
            self.enemy_pos = [center, center]
            self.grid[center, center] = 5
        else:
            # Last resort: place somewhere random
            self.enemy_pos = [1, 1]
            self.grid[1, 1] = 5
    
    def _move_enemy(self):
        """Move enemy randomly to adjacent cell (50% chance to move)"""
        # Only move 50% of the time
        if random.random() < 0.5:
            # Clear current enemy position
            self.grid[self.enemy_pos[0], self.enemy_pos[1]] = 0
            
            # Get possible moves
            possible_moves = []
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
            
            for dr, dc in directions:
                new_row = self.enemy_pos[0] + dr
                new_col = self.enemy_pos[1] + dc
                
                # Check bounds
                if (0 <= new_row < self.size and 0 <= new_col < self.size):
                    # Check if not obstacle or goal (enemy can move through agent space)
                    if self.grid[new_row, new_col] not in [2, 3]:  # not goal or obstacle
                        possible_moves.append([new_row, new_col])
            
            # If no valid moves, stay in place
            if possible_moves:
                self.enemy_pos = random.choice(possible_moves)
            
            # Place enemy in new position (may overwrite agent temporarily)
            if self.grid[self.enemy_pos[0], self.enemy_pos[1]] == 1:
                # Enemy moved to agent position - will be handled in collision check
                pass
            else:
                self.grid[self.enemy_pos[0], self.enemy_pos[1]] = 5
    
    def _check_enemy_collision(self) -> bool:
        """Check if agent is adjacent to or on enemy"""
        agent_row, agent_col = self.agent_pos
        enemy_row, enemy_col = self.enemy_pos
        
        # Check if on same position
        if agent_row == enemy_row and agent_col == enemy_col:
            return True
        
        # Check if adjacent (4-directional)
        distance = abs(agent_row - enemy_row) + abs(agent_col - enemy_col)
        return distance == 1
    
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
        self.obstacles = []
        self._initialize_grid()
        
        # Check initial conditions
        goal_reached = (self.agent_pos == self.goal_pos)
        enemy_collision = self._check_enemy_collision()
        
        info = {
            'agent_pos': self.agent_pos,
            'goal_pos': self.goal_pos,
            'enemy_pos': self.enemy_pos,
            'steps': self.steps,
            'goal_reached': goal_reached,
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
        
        # Move enemy after agent moves
        self._move_enemy()
        
        # Update grid display (enemy might overwrite agent temporarily)
        if self.enemy_pos != self.agent_pos:
            self.grid[self.enemy_pos[0], self.enemy_pos[1]] = 5
        
        # Always ensure agent is visible on grid
        self.grid[self.agent_pos[0], self.agent_pos[1]] = 1
        
        # Check for enemy collision (game over condition)
        enemy_collision = self._check_enemy_collision()
        
        # Check if goal reached or enemy collision
        terminated = False
        goal_reached = (self.agent_pos == self.goal_pos)
        if goal_reached or enemy_collision:
            terminated = True
        
        # Check if max steps reached
        truncated = False
        if self.steps >= self.max_steps:
            truncated = True
        
        info = {
            'steps': self.steps,
            'agent_pos': self.agent_pos,
            'goal_pos': self.goal_pos,
            'enemy_pos': self.enemy_pos,
            'enemy_collision': enemy_collision,
            'goal_reached': goal_reached,
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
    
    def get_distance_to_goal(self) -> int:
        """Get Manhattan distance from agent to goal"""
        return abs(self.agent_pos[0] - self.goal_pos[0]) + abs(self.agent_pos[1] - self.goal_pos[1])
    
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
        print("Legend: A=Agent, G=Goal, X=Obstacle, E=Enemy, ·=Empty")
        print(f"Agent Position: {self.agent_pos}")
        print(f"Goal Position: {self.goal_pos}")
        print(f"Enemy Position: {self.enemy_pos}")
        
        # Calculate distances
        distance_to_goal = abs(self.agent_pos[0] - self.goal_pos[0]) + abs(self.agent_pos[1] - self.goal_pos[1])
        distance_to_enemy = abs(self.agent_pos[0] - self.enemy_pos[0]) + abs(self.agent_pos[1] - self.enemy_pos[1])
        
        print(f"Distance to Goal: {distance_to_goal}")
        print(f"Distance to Enemy: {distance_to_enemy}")
        
        # Warning if close to enemy
        if distance_to_enemy <= 2:
            print("⚠️  WARNING: Enemy nearby!")
        if distance_to_enemy <= 1:
            print("💥 DANGER: Enemy can catch you!")
        print(f"Valid Actions: {self.get_valid_actions()}")
        print()

def manual_play():
    """Allow manual play of the environment"""
    env = GridWorld(5)
    state, info = env.reset()
    
    print("Welcome to Grid World!")
    print("Use WASD keys to move:")
    print("W = Up, A = Left, S = Down, D = Right")
    print("Q = Quit")
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
            if terminated and info['agent_pos'] == info['goal_pos']:
                print("🎉 Congratulations! You reached the goal!")
            elif truncated:
                print("⏰ Time's up! You ran out of moves.")
            else:
                print("💥 Game Over!")
            break

if __name__ == "__main__":
    manual_play() 