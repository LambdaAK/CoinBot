#!/usr/bin/env python3
"""
Level Editor for Grid World
Allows users to design custom levels for the AI agents to play
"""

import json
import os
from typing import List, Dict, Any, Tuple
import numpy as np

class LevelEditor:
    def __init__(self, grid_size: int = 10):
        self.grid_size = grid_size
        self.grid = np.zeros((grid_size, grid_size), dtype=int)
        self.agent_pos = [0, 0]
        self.goal_pos = [grid_size-1, grid_size-1]
        self.enemy_positions = []
        self.obstacles = []
        self.power_ups = []  # Future feature
        self.keys = []       # Future feature
        
        # Element mappings
        self.element_map = {
            'A': 1,  # Agent
            'G': 2,  # Goal
            'X': 3,  # Obstacle
            'E': 5,  # Enemy
            '·': 0,  # Empty
        }
        
        self.reverse_map = {v: k for k, v in self.element_map.items()}
        
    def clear_grid(self):
        """Clear the entire grid"""
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.agent_pos = [0, 0]
        self.goal_pos = [self.grid_size-1, self.grid_size-1]
        self.enemy_positions = []
        self.obstacles = []
        self.power_ups = []
        self.keys = []
        
        # Place default agent and goal positions
        self.grid[0, 0] = 1  # Agent at top-left
        self.grid[self.grid_size-1, self.grid_size-1] = 2  # Goal at bottom-right
        
    def place_element(self, row: int, col: int, element: str):
        """Place an element on the grid"""
        if not (0 <= row < self.grid_size and 0 <= col < self.grid_size):
            print(f"❌ Position ({row}, {col}) is out of bounds!")
            return False
            
        if element not in self.element_map:
            print(f"❌ Unknown element: {element}")
            return False
            
        element_id = self.element_map[element]
        
        # Handle special cases
        if element == 'A':  # Agent
            # Remove old agent position from grid
            if self.agent_pos != [0, 0] or self.grid[0, 0] == 1:
                old_agent_row, old_agent_col = self.agent_pos
                self.grid[old_agent_row, old_agent_col] = 0
            self.agent_pos = [row, col]
            
        elif element == 'G':  # Goal
            # Remove old goal position from grid
            if self.goal_pos != [0, 0] or self.grid[self.grid_size-1, self.grid_size-1] == 2:
                old_goal_row, old_goal_col = self.goal_pos
                self.grid[old_goal_row, old_goal_col] = 0
            self.goal_pos = [row, col]
            
        elif element == 'E':  # Enemy
            # Add to enemy positions list
            if [row, col] not in self.enemy_positions:
                self.enemy_positions.append([row, col])
                
        elif element == 'X':  # Obstacle
            # Add to obstacles list
            if [row, col] not in self.obstacles:
                self.obstacles.append([row, col])
                
        elif element == '·':  # Empty
            # Remove from special lists
            if [row, col] in self.enemy_positions:
                self.enemy_positions.remove([row, col])
            if [row, col] in self.obstacles:
                self.obstacles.remove([row, col])
        
        # Update grid
        self.grid[row, col] = element_id
        
        # Ensure agent and goal are always visible (they take priority)
        if self.agent_pos != [0, 0]:
            self.grid[self.agent_pos[0], self.agent_pos[1]] = 1
        if self.goal_pos != [0, 0]:
            self.grid[self.goal_pos[0], self.goal_pos[1]] = 2
        
        return True
        
    def render_grid(self):
        """Display the current grid"""
        print(f"\nGrid World Level Editor ({self.grid_size}x{self.grid_size})")
        print("=" * (self.grid_size * 4 + 2))
        
        # Column headers
        header = "   "
        for j in range(self.grid_size):
            header += f" {j:2} "
        print(header)
        
        for i in range(self.grid_size):
            row = f"{i:2} |"
            for j in range(self.grid_size):
                cell_value = self.grid[i, j]
                element = self.reverse_map.get(cell_value, "?")
                row += f" {element} "
            row += "|"
            print(row)
        
        print("=" * (self.grid_size * 4 + 2))
        print("Legend: A=Agent, G=Goal, X=Obstacle, E=Enemy, ·=Empty")
        print(f"Agent Position: {self.agent_pos}")
        print(f"Goal Position: {self.goal_pos}")
        print(f"Enemy Positions: {self.enemy_positions}")
        print(f"Obstacles: {self.obstacles}")
        
    def validate_level(self) -> Tuple[bool, str]:
        """Validate that the level is playable"""
        # Check if agent is placed (should be at agent_pos and visible on grid)
        if self.agent_pos == [0, 0] or self.grid[self.agent_pos[0], self.agent_pos[1]] != 1:
            return False, "Agent not properly placed"
            
        # Check if goal is placed (should be at goal_pos and visible on grid)
        if self.goal_pos == [self.grid_size-1, self.grid_size-1] or self.grid[self.goal_pos[0], self.goal_pos[1]] != 2:
            return False, "Goal not properly placed"
            
        # Check if goal is blocked by obstacle
        if self.grid[self.goal_pos[0], self.goal_pos[1]] == 3:
            return False, "Goal is blocked by obstacle"
            
        # Check if agent is blocked
        if self.grid[self.agent_pos[0], self.agent_pos[1]] == 3:
            return False, "Agent is blocked by obstacle"
            
        # Check that agent and goal are not at the same position
        if self.agent_pos == self.goal_pos:
            return False, "Agent and goal cannot be at the same position"
            
        return True, "Level is valid"
        
    def save_level(self, filename: str):
        """Save the level to a JSON file"""
        level_data = {
            'grid_size': self.grid_size,
            'agent_pos': self.agent_pos,
            'goal_pos': self.goal_pos,
            'enemy_positions': self.enemy_positions,
            'obstacles': self.obstacles,
            'power_ups': self.power_ups,
            'keys': self.keys,
            'grid': self.grid.tolist()
        }
        
        # Create levels directory if it doesn't exist
        levels_dir = "levels"
        if not os.path.exists(levels_dir):
            os.makedirs(levels_dir)
            
        filepath = os.path.join(levels_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(level_data, f, indent=2)
            
        print(f"✅ Level saved to {filepath}")
        
    def load_level(self, filename: str):
        """Load a level from a JSON file"""
        levels_dir = "levels"
        filepath = os.path.join(levels_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"❌ Level file {filepath} not found!")
            return False
            
        with open(filepath, 'r') as f:
            level_data = json.load(f)
            
        self.grid_size = level_data['grid_size']
        self.agent_pos = level_data['agent_pos']
        self.goal_pos = level_data['goal_pos']
        self.enemy_positions = level_data['enemy_positions']
        self.obstacles = level_data['obstacles']
        self.power_ups = level_data.get('power_ups', [])
        self.keys = level_data.get('keys', [])
        self.grid = np.array(level_data['grid'])
        
        print(f"✅ Level loaded from {filepath}")
        return True
        
    def list_levels(self):
        """List all available levels"""
        levels_dir = "levels"
        if not os.path.exists(levels_dir):
            print("No levels directory found.")
            return []
            
        levels = []
        for filename in os.listdir(levels_dir):
            if filename.endswith('.json'):
                levels.append(filename)
                
        if levels:
            print("Available levels:")
            for i, level in enumerate(levels, 1):
                print(f"  {i}. {level}")
        else:
            print("No levels found.")
            
        return levels

def interactive_editor():
    """Interactive level editor"""
    print("🎮 Grid World Level Editor")
    print("=" * 40)
    
    # Get grid size
    while True:
        try:
            size = int(input("Enter grid size (5-20): "))
            if 5 <= size <= 20:
                break
            else:
                print("Please enter a size between 5 and 20.")
        except ValueError:
            print("Please enter a valid number.")
    
    editor = LevelEditor(size)
    
    while True:
        editor.render_grid()
        
        print("\nCommands:")
        print("  place <row> <col> <element> - Place element (A/G/X/E/·)")
        print("  clear - Clear the grid")
        print("  save <filename> - Save level")
        print("  load <filename> - Load level")
        print("  list - List available levels")
        print("  validate - Check if level is valid")
        print("  quit - Exit editor")
        
        command = input("\nEnter command: ").strip().split()
        
        if not command:
            continue
            
        cmd = command[0].lower()
        
        if cmd == 'quit':
            print("Goodbye!")
            break
            
        elif cmd == 'clear':
            editor.clear_grid()
            print("Grid cleared!")
            
        elif cmd == 'place':
            if len(command) != 4:
                print("Usage: place <row> <col> <element>")
                continue
            try:
                row = int(command[1])
                col = int(command[2])
                element = command[3].upper()
                if editor.place_element(row, col, element):
                    print(f"✅ Placed {element} at ({row}, {col})")
                else:
                    print("❌ Failed to place element")
            except ValueError:
                print("Row and column must be numbers")
                
        elif cmd == 'save':
            if len(command) != 2:
                print("Usage: save <filename>")
                continue
            filename = command[1]
            if not filename.endswith('.json'):
                filename += '.json'
            editor.save_level(filename)
            
        elif cmd == 'load':
            if len(command) != 2:
                print("Usage: load <filename>")
                continue
            filename = command[1]
            if not filename.endswith('.json'):
                filename += '.json'
            editor.load_level(filename)
            
        elif cmd == 'list':
            editor.list_levels()
            
        elif cmd == 'validate':
            is_valid, message = editor.validate_level()
            if is_valid:
                print(f"✅ {message}")
            else:
                print(f"❌ {message}")
                
        else:
            print(f"Unknown command: {cmd}")

if __name__ == "__main__":
    interactive_editor() 