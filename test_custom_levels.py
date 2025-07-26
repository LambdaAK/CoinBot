#!/usr/bin/env python3
"""
Test AI Agents on Custom Levels
Loads trained agents and tests them on custom levels
"""

import os
import sys
import argparse
from typing import List, Tuple
import time

def list_available_levels() -> List[str]:
    """List all available custom levels"""
    levels_dir = "levels"
    if not os.path.exists(levels_dir):
        print("No levels directory found.")
        return []
        
    levels = []
    for filename in os.listdir(levels_dir):
        if filename.endswith('.json'):
            levels.append(filename)
            
    if levels:
        print("Available custom levels:")
        for i, level in enumerate(levels, 1):
            print(f"  {i}. {level}")
    else:
        print("No custom levels found.")
        
    return levels

def list_available_agents() -> List[str]:
    """List all available trained agents"""
    agents_dir = "agents"
    if not os.path.exists(agents_dir):
        print("No agents directory found.")
        return []
        
    agents = []
    for filename in os.listdir(agents_dir):
        if filename.endswith('.pkl'):
            agents.append(filename)
            
    if agents:
        print("Available trained agents:")
        for i, agent in enumerate(agents, 1):
            print(f"  {i}. {agent}")
    else:
        print("No trained agents found.")
        
    return agents

def test_dqn_agent_on_level(agent_file: str, level_file: str, episodes: int = 5, render: bool = True):
    """Test a DQN agent on a custom level"""
    try:
        from dqn_agent import ImprovedDQNAgent
        from custom_grid_world import CustomGridWorld
        
        print(f"🤖 Loading DQN agent: {agent_file}")
        agent = ImprovedDQNAgent.load(agent_file)
        
        print(f"🎮 Loading custom level: {level_file}")
        env = CustomGridWorld(level_file)
        
        # Calculate state size for the custom level
        state_size = 88  # Same as in dqn_agent.py
        
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
        
    except ImportError as e:
        print(f"❌ Error importing required modules: {e}")
        return 0, [], []
    except Exception as e:
        print(f"❌ Error testing agent: {e}")
        return 0, [], []

def test_qlearning_agent_on_level(agent_file: str, level_file: str, episodes: int = 5, render: bool = True):
    """Test a Q-learning agent on a custom level"""
    try:
        from q_learning_agent import QLearningAgent
        from custom_grid_world import CustomGridWorld
        
        print(f"🤖 Loading Q-learning agent: {agent_file}")
        agent = QLearningAgent.load(agent_file)
        
        print(f"🎮 Loading custom level: {level_file}")
        env = CustomGridWorld(level_file)
        
        print(f"🧪 Testing Q-learning agent on custom level for {episodes} episodes...")
        print("=" * 60)
        
        success_count = 0
        total_rewards = []
        total_steps = []
        
        for episode in range(episodes):
            observation, info = env.reset()
            total_reward = 0
            steps = 0
            old_pos = info['agent_pos'].copy()
            
            print(f"\nEpisode {episode + 1}:")
            
            while True:
                # Choose action (no exploration during testing)
                action = agent.get_action(observation, training=False)
                
                # Take action
                next_observation, env_reward, terminated, truncated, info = env.step(action)
                
                # Calculate reward using agent's reward function
                new_pos = info['agent_pos']
                enemy_pos = info.get('enemy_pos')
                enemy_collision = info.get('enemy_collision', False)
                reward = agent.calculate_reward(action, new_pos, old_pos, enemy_pos, enemy_collision)
                
                total_reward += reward
                steps += 1
                observation = next_observation
                old_pos = new_pos.copy()
                
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
        
    except ImportError as e:
        print(f"❌ Error importing required modules: {e}")
        return 0, [], []
    except Exception as e:
        print(f"❌ Error testing agent: {e}")
        return 0, [], []

def interactive_test():
    """Interactive testing interface"""
    print("🎮 Custom Level Testing Interface")
    print("=" * 40)
    
    # List available levels
    levels = list_available_levels()
    if not levels:
        print("Please create some levels first using the level editor!")
        return
    
    # List available agents
    agents = list_available_agents()
    if not agents:
        print("Please train some agents first!")
        return
    
    # Get user selection
    print("\nSelect a level:")
    for i, level in enumerate(levels, 1):
        print(f"  {i}. {level}")
    
    while True:
        try:
            level_choice = int(input(f"Enter level number (1-{len(levels)}): ")) - 1
            if 0 <= level_choice < len(levels):
                level_file = levels[level_choice]
                break
            else:
                print("Invalid choice!")
        except ValueError:
            print("Please enter a number!")
    
    print(f"\nSelect an agent:")
    for i, agent in enumerate(agents, 1):
        print(f"  {i}. {agent}")
    
    while True:
        try:
            agent_choice = int(input(f"Enter agent number (1-{len(agents)}): ")) - 1
            if 0 <= agent_choice < len(agents):
                agent_file = agents[agent_choice]
                break
            else:
                print("Invalid choice!")
        except ValueError:
            print("Please enter a number!")
    
    # Determine agent type and test
    if "dqn" in agent_file.lower():
        print(f"\n🤖 Testing DQN agent on custom level...")
        test_dqn_agent_on_level(agent_file, level_file)
    else:
        print(f"\n🤖 Testing Q-learning agent on custom level...")
        test_qlearning_agent_on_level(agent_file, level_file)

def main():
    parser = argparse.ArgumentParser(description='Test AI agents on custom levels')
    parser.add_argument('--agent', '-a', help='Agent filename')
    parser.add_argument('--level', '-l', help='Level filename')
    parser.add_argument('--episodes', '-e', type=int, default=5, help='Number of episodes to test')
    parser.add_argument('--no-render', action='store_true', help='Disable rendering')
    parser.add_argument('--list-levels', action='store_true', help='List available levels')
    parser.add_argument('--list-agents', action='store_true', help='List available agents')
    parser.add_argument('--interactive', '-i', action='store_true', help='Interactive mode')
    
    args = parser.parse_args()
    
    if args.list_levels:
        list_available_levels()
        return
    
    if args.list_agents:
        list_available_agents()
        return
    
    if args.interactive:
        interactive_test()
        return
    
    if not args.agent or not args.level:
        print("Please provide both --agent and --level arguments, or use --interactive")
        return
    
    # Determine agent type and test
    if "dqn" in args.agent.lower():
        test_dqn_agent_on_level(args.agent, args.level, args.episodes, not args.no_render)
    else:
        test_qlearning_agent_on_level(args.agent, args.level, args.episodes, not args.no_render)

if __name__ == "__main__":
    main() 