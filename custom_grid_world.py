#!/usr/bin/env python3
"""
Custom GridWorld Environment
Loads and plays custom levels designed with the level editor
"""

import numpy as np
import random
import json
import os
from typing import Tuple, Dict, Any, Optional
import time

class CustomGridWorld:
    def __init__(self, level_file: str = None, max_steps: int = 50, seed: Optional[int] = None):
        self.max_steps = max_steps
        self.seed = seed
        
        # RL environment properties
        self.action_space = 4  # 0=up, 1=right, 2=down, 3=left
        self.observation_space = None  # Will be set after loading level
        self.reward_range = (-50.0, 150.0)  # min/max possible rewards
        
        # Emoji mappings
        self.emoji_map = {
            0: "·",  # Empty space
            1: "A",  # Agent
            2: "G",  # Goal
            3: "X",  # Obstacle
            4: "★",  # Reward (future)
            5: "E",  # Enemy
        }
        
        # Set seed for reproducibility
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Load level if provided
        if level_file:
            self.load_level(level_file)
        else:
            # Default initialization
            self.size = 10
            self.grid = np.zeros((self.size, self.size), dtype=int)
            self.agent_pos = [0, 0]
            self.goal_pos = [self.size-1, self.size-1]
            self.enemy_positions = []
            self.obstacles = []
            self.steps = 0
            self.observation_space = self.size * self.size
    
    def load_level(self, level_file: str):
        """Load a custom level from JSON file"""
        levels_dir = "levels"
        filepath = os.path.join(levels_dir, level_file)
        
        if not filepath.endswith('.json'):
            filepath += '.json'
            
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Level file {filepath} not found!")
            
        with open(filepath, 'r') as f:
            level_data = json.load(f)
            
        self.size = level_data['grid_size']
        self.agent_pos = level_data['agent_pos'].copy()
        self.goal_pos = level_data['goal_pos'].copy()
        self.enemy_positions = [pos.copy() for pos in level_data['enemy_positions']]
        self.obstacles = [pos.copy() for pos in level_data['obstacles']]
        self.grid = np.array(level_data['grid'])
        self.steps = 0
        
        # Set observation space
        self.observation_space = self.size * self.size
        
        print(f"✅ Loaded level: {level_file}")
        print(f"   Grid size: {self.size}x{self.size}")
        print(f"   Enemies: {len(self.enemy_positions)}")
        print(f"   Obstacles: {len(self.obstacles)}")
        
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state
        
        Returns:
            (observation, info)
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        # Reset to initial positions from level file
        # Note: We don't reload the file, just reset positions
        self.steps = 0
        
        # Check initial conditions
        goal_reached = (self.agent_pos == self.goal_pos)
        enemy_collision = self._check_enemy_collision()
        
        info = {
            'agent_pos': self.agent_pos.copy(),
            'goal_pos': self.goal_pos.copy(),
            'enemy_pos': [pos.copy() for pos in self.enemy_positions],
            'steps': self.steps,
            'goal_reached': goal_reached,
            'enemy_collision': enemy_collision
        }
        
        return self._get_observation(), info
    
    def _get_observation(self) -> np.ndarray:
        """Get observation for RL agent (flattened grid)"""
        return self.grid.flatten()
    
    def _move_enemy(self):
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
                        # Check if not obstacle or goal (enemy can move through agent space)
                        if self.grid[new_row, new_col] not in [2, 3]:  # not goal or obstacle
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
        
        # Update grid display for all enemies
        for enemy_pos in self.enemy_positions:
            if enemy_pos != self.agent_pos:
                self.grid[enemy_pos[0], enemy_pos[1]] = 5
        
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
            'agent_pos': self.agent_pos.copy(),
            'goal_pos': self.goal_pos.copy(),
            'enemy_pos': [pos.copy() for pos in self.enemy_positions],
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
        print(f"Custom Grid World ({self.size}x{self.size}) - Step: {self.steps}")
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
        print(f"Enemy Positions: {self.enemy_positions}")
        print(f"Number of Enemies: {len(self.enemy_positions)}")
        
        # Calculate distances
        distance_to_goal = abs(self.agent_pos[0] - self.goal_pos[0]) + abs(self.agent_pos[1] - self.goal_pos[1])
        
        # Find closest enemy distance
        closest_enemy_distance = float('inf')
        for enemy_pos in self.enemy_positions:
            enemy_dist = abs(self.agent_pos[0] - enemy_pos[0]) + abs(self.agent_pos[1] - enemy_pos[1])
            closest_enemy_distance = min(closest_enemy_distance, enemy_dist)
        
        print(f"Distance to Goal: {distance_to_goal}")
        print(f"Distance to Closest Enemy: {closest_enemy_distance}")
        
        # Warning if close to any enemy
        if closest_enemy_distance <= 2:
            print("⚠️  WARNING: Enemy nearby!")
        if closest_enemy_distance <= 1:
            print("💥 DANGER: Enemy can catch you!")
        print(f"Valid Actions: {self.get_valid_actions()}")
        print()

def test_custom_level(level_file: str, episodes: int = 5):
    """Test a custom level with manual play"""
    try:
        env = CustomGridWorld(level_file)
        state, info = env.reset()
        
        print(f"🎮 Testing Custom Level: {level_file}")
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
                
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Make sure to create the level first using the level editor!")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        level_file = sys.argv[1]
        test_custom_level(level_file)
    else:
        print("Usage: python custom_grid_world.py <level_file>")
        print("Example: python custom_grid_world.py my_level.json") 