#!/usr/bin/env python3
"""
Watch Saved DQN Agent

This script loads a previously trained DQN agent and watches it play
the grid world game. DQN agents are stored in the 'agents' folder.
Now supports custom levels from the level editor!
"""

from dqn_agent import ImprovedDQNAgent
import argparse
import os
import glob
import time
from custom_grid_world import CustomGridWorld

def list_available_dqn_agents():
    """List all available DQN agents in the agents folder"""
    agents_dir = "agents"
    if not os.path.exists(agents_dir):
        print("❌ No agents folder found!")
        return []
    
    # Look for DQN agent files (they might have 'dqn' in the filename)
    agent_files = glob.glob(os.path.join(agents_dir, "*dqn*.pkl"))
    
    # If no DQN-specific files found, show all .pkl files
    if not agent_files:
        agent_files = glob.glob(os.path.join(agents_dir, "*.pkl"))
    
    if not agent_files:
        print("❌ No agent files found in agents folder!")
        return []
    
    print("📁 Available agents:")
    for i, filepath in enumerate(sorted(agent_files), 1):
        filename = os.path.basename(filepath)
        # Highlight DQN agents
        if 'dqn' in filename.lower():
            print(f"  {i}. {filename} 🤖 (DQN)")
        else:
            print(f"  {i}. {filename} 📊 (Q-Learning)")
    
    return sorted(agent_files)

def list_available_levels():
    """List all available custom levels"""
    levels_dir = "levels"
    if not os.path.exists(levels_dir):
        print("❌ No levels folder found!")
        return []
        
    levels = []
    for filename in os.listdir(levels_dir):
        if filename.endswith('.json'):
            levels.append(filename)
            
    if levels:
        print("🎮 Available custom levels:")
        for i, level in enumerate(levels, 1):
            print(f"  {i}. {level}")
    else:
        print("❌ No custom levels found.")
        
    return levels

def select_agent_interactively():
    """Let user select an agent interactively"""
    agent_files = list_available_dqn_agents()
    
    if not agent_files:
        return None
    
    while True:
        try:
            choice = input(f"\nSelect agent (1-{len(agent_files)}) or 'q' to quit: ").strip()
            
            if choice.lower() == 'q':
                return None
            
            choice_num = int(choice)
            if 1 <= choice_num <= len(agent_files):
                selected_file = os.path.basename(agent_files[choice_num - 1])
                print(f"✅ Selected: {selected_file}")
                return selected_file
            else:
                print(f"❌ Please enter a number between 1 and {len(agent_files)}")
                
        except ValueError:
            print("❌ Please enter a valid number or 'q' to quit")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            return None

def select_level_interactively():
    """Let user select a level interactively"""
    print("\n🎮 Level Selection:")
    print("  1. Random level (generated automatically)")
    
    levels = list_available_levels()
    
    if levels:
        print("  Custom levels:")
        for i, level in enumerate(levels, 1):
            print(f"    {i+1}. {level}")
    
    while True:
        try:
            if levels:
                choice = input(f"\nSelect level (1-{len(levels)+1}) or 'q' to quit: ").strip()
            else:
                choice = input(f"\nSelect level (1) or 'q' to quit: ").strip()
            
            if choice.lower() == 'q':
                return None
            
            choice_num = int(choice)
            if choice_num == 1:
                # Get grid size for random level
                while True:
                    try:
                        size_input = input("Enter grid size for random level (5-20, default 10): ").strip()
                        if size_input == "":
                            grid_size = 10
                            break
                        grid_size = int(size_input)
                        if 5 <= grid_size <= 20:
                            break
                        else:
                            print("Please enter a size between 5 and 20.")
                    except ValueError:
                        print("Please enter a valid number.")
                
                print(f"✅ Selected: Random level ({grid_size}x{grid_size})")
                return f"random_{grid_size}"
            elif 2 <= choice_num <= len(levels) + 1:
                selected_level = levels[choice_num - 2]
                print(f"✅ Selected: {selected_level}")
                return selected_level
            else:
                print(f"❌ Please enter a number between 1 and {len(levels)+1}")
                
        except ValueError:
            print("❌ Please enter a valid number or 'q' to quit")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            return None

def test_dqn_agent_on_custom_level(agent, level_file, episodes=10, render=True):
    """Test a DQN agent on a custom level"""
    try:
        print(f"🎮 Loading custom level: {level_file}")
        env = CustomGridWorld(level_file)
        
        print(f"🧪 Testing DQN agent on custom level for {episodes} episodes...")
        print("=" * 60)
        
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
            enemy_positions = info.get('enemy_pos', [])
            state = agent.get_state_representation(grid_state, agent_pos, goal_pos, enemy_positions)
            old_pos = agent_pos.copy()
            
            print(f"\nEpisode {episode + 1}:")
            
            while True:
                action = agent.act(state, training=False)
                next_observation, env_reward, terminated, truncated, info = env.step(action)
                
                # Get new state
                new_grid_state = env.grid
                new_agent_pos = info['agent_pos']
                new_goal_pos = info['goal_pos']
                new_enemy_positions = info.get('enemy_pos', [])
                next_state = agent.get_state_representation(new_grid_state, new_agent_pos, new_goal_pos, new_enemy_positions)
                
                # Calculate reward using agent's reward function
                enemy_collision = info.get('enemy_collision', False)
                reward = agent.calculate_reward(action, new_agent_pos, old_pos, 
                                              new_goal_pos, new_enemy_positions, 
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
        print(f"Average reward: {sum(total_rewards)/len(total_rewards):.2f}")
        print(f"Average steps: {sum(total_steps)/len(total_steps):.1f}")
        
        return success_count, total_rewards, total_steps
        
    except Exception as e:
        print(f"❌ Error testing agent on custom level: {e}")
        return 0, [], []

def test_dqn_agent_on_random_level(agent, grid_size=10, episodes=10, render=True):
    """Test a DQN agent on a randomly generated level"""
    try:
        from grid_world import GridWorld
        
        print(f"🎲 Using randomly generated {grid_size}x{grid_size} level...")
        env = GridWorld(size=grid_size, seed=None)  # Random seed
        
        print(f"🧪 Testing DQN agent on random level for {episodes} episodes...")
        print("=" * 60)
        
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
            enemy_positions = info.get('enemy_pos', [])
            state = agent.get_state_representation(grid_state, agent_pos, goal_pos, enemy_positions)
            old_pos = agent_pos.copy()
            
            print(f"\nEpisode {episode + 1}:")
            
            while True:
                action = agent.act(state, training=False)
                next_observation, env_reward, terminated, truncated, info = env.step(action)
                
                # Get new state
                new_grid_state = env.grid
                new_agent_pos = info['agent_pos']
                new_goal_pos = info['goal_pos']
                new_enemy_positions = info.get('enemy_pos', [])
                next_state = agent.get_state_representation(new_grid_state, new_agent_pos, new_goal_pos, new_enemy_positions)
                
                # Calculate reward using agent's reward function
                enemy_collision = info.get('enemy_collision', False)
                reward = agent.calculate_reward(action, new_agent_pos, old_pos, 
                                              new_goal_pos, new_enemy_positions, 
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
        print(f"Average reward: {sum(total_rewards)/len(total_rewards):.2f}")
        print(f"Average steps: {sum(total_steps)/len(total_steps):.1f}")
        
        return success_count, total_rewards, total_steps
        
    except Exception as e:
        print(f"❌ Error testing agent on random level: {e}")
        return 0, [], []

def watch_saved_dqn_agent(filename: str, level_choice: str = "random_10", episodes: int = 10):
    """Load and watch a saved DQN agent play"""
    print("🎬 Loading saved DQN agent for demonstration...")
    
    try:
        # Load the trained agent
        agent = ImprovedDQNAgent.load(filename)
        
        # Test the agent on the selected level
        if level_choice.startswith("random_"):
            # Extract grid size from random level choice
            try:
                grid_size = int(level_choice.split("_")[1])
            except (IndexError, ValueError):
                grid_size = 10  # Default fallback
            success_count, rewards, steps = test_dqn_agent_on_random_level(agent, grid_size=grid_size, episodes=episodes, render=True)
        else:
            success_count, rewards, steps = test_dqn_agent_on_custom_level(agent, level_choice, episodes=episodes, render=True)
        
        print(f"\n🎯 DQN Agent Performance:")
        print(f"Success rate: {success_count}/{episodes} ({success_count/episodes*100:.1f}%)")
        print(f"Average reward: {sum(rewards)/len(rewards):.2f}")
        print(f"Average steps: {sum(steps)/len(steps):.1f}")
        
    except FileNotFoundError:
        print(f"❌ DQN Agent file '{filename}' not found!")
        print("Please train a DQN agent first using: python dqn_agent.py")
    except Exception as e:
        print(f"❌ Error loading agent: {e}")
        print("Make sure the agent file is compatible with the current version.")

def main():
    parser = argparse.ArgumentParser(description='Watch a saved DQN agent play')
    parser.add_argument('--agent', '-a', default=None, 
                       help='Agent filename in agents folder')
    parser.add_argument('--level', '-l', default=None,
                       help='Level filename in levels folder (or "random_<size>" for random level, e.g., "random_15" for 15x15)')
    parser.add_argument('--episodes', '-e', type=int, default=10,
                       help='Number of episodes to watch (default: 10)')
    parser.add_argument('--list', '-L', action='store_true',
                       help='List all available agents')
    
    args = parser.parse_args()
    
    print("🎬 Grid World DQN Agent Viewer")
    print("=" * 40)
    
    # List available agents if requested
    if args.list:
        list_available_dqn_agents()
        return
    
    # Determine which agent to use
    agent_filename = args.agent
    
    if agent_filename is None:
        # Interactive selection
        agent_filename = select_agent_interactively()
        if agent_filename is None:
            return
    
    # Determine which level to use
    level_choice = args.level
    
    if level_choice is None:
        # Interactive selection
        level_choice = select_level_interactively()
        if level_choice is None:
            return
    
    # Watch the selected agent on the selected level
    watch_saved_dqn_agent(agent_filename, level_choice, args.episodes)

if __name__ == "__main__":
    main() 