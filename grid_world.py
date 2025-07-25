import numpy as np
import random
from typing import Tuple, Dict, Any, Optional
import time
import os

class GridWorld:
    def __init__(self, size: int = 5, max_steps: int = 25, seed: Optional[int] = None):
        self.size = size
        self.grid = np.zeros((size, size), dtype=int)
        self.agent_pos = [0, 0]  # Start at top-left
        self.goal_pos = [size-1, size-1]  # Goal at bottom-right
        self.obstacles = []
        self.steps = 0
        self.max_steps = max_steps
        self.seed = seed
        
        # RL environment properties
        self.action_space = 4  # 0=up, 1=right, 2=down, 3=left
        self.observation_space = size * size  # flattened grid
        self.reward_range = (-5.0, 10.0)  # min/max possible rewards
        
        # Emoji mappings
        self.emoji_map = {
            0: "·",  # Empty space
            1: "A",  # Agent
            2: "G",  # Goal
            3: "X",  # Obstacle
            4: "★",  # Reward
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
        """Initialize the grid with agent, goal, and some obstacles"""
        self.grid = np.zeros((self.size, self.size), dtype=int)
        
        # Place agent
        self.grid[self.agent_pos[0], self.agent_pos[1]] = 1
        
        # Place goal
        self.grid[self.goal_pos[0], self.goal_pos[1]] = 2
        
        # Add obstacles with DFS validation
        max_attempts = 100  # Prevent infinite loops
        attempts = 0
        
        while attempts < max_attempts:
            # Clear previous obstacles
            self.obstacles = []
            self.grid = np.zeros((self.size, self.size), dtype=int)
            self.grid[self.agent_pos[0], self.agent_pos[1]] = 1
            self.grid[self.goal_pos[0], self.goal_pos[1]] = 2
            
            # Add random obstacles
            num_obstacles = random.randint(2, 4)
            for _ in range(num_obstacles):
                while True:
                    obstacle_pos = [random.randint(0, self.size-1), random.randint(0, self.size-1)]
                    if (obstacle_pos != self.agent_pos and 
                        obstacle_pos != self.goal_pos and 
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
        
        info = {
            'agent_pos': self.agent_pos,
            'goal_pos': self.goal_pos,
            'steps': self.steps
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
        
        # Reward calculation is now handled by the agent
        
        # Check if goal reached
        terminated = False
        if self.agent_pos == self.goal_pos:
            terminated = True
        
        # Check if max steps reached
        truncated = False
        if self.steps >= self.max_steps:
            truncated = True
        
        info = {
            'steps': self.steps,
            'agent_pos': self.agent_pos,
            'goal_pos': self.goal_pos,
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
        print("Legend: A=Agent, G=Goal, X=Obstacle, ·=Empty")
        print(f"Agent Position: {self.agent_pos}")
        print(f"Goal Position: {self.goal_pos}")
        print(f"Distance to Goal: {self.get_distance_to_goal()}")
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