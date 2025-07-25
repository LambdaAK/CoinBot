#!/usr/bin/env python3
"""
Watch Saved Q-Learning Agent

This script loads a previously trained Q-learning agent and watches it play
the grid world game. Agents are stored in the 'agents' folder.
"""

from q_learning_agent import QLearningAgent, watch_saved_agent
import argparse
import os
import glob

def list_available_agents():
    """List all available agents in the agents folder"""
    agents_dir = "agents"
    if not os.path.exists(agents_dir):
        print("❌ No agents folder found!")
        return []
    
    agent_files = glob.glob(os.path.join(agents_dir, "*.pkl"))
    if not agent_files:
        print("❌ No agent files found in agents folder!")
        return []
    
    print("📁 Available agents:")
    for i, filepath in enumerate(sorted(agent_files), 1):
        filename = os.path.basename(filepath)
        print(f"  {i}. {filename}")
    
    return sorted(agent_files)

def select_agent_interactively():
    """Let user select an agent interactively"""
    agent_files = list_available_agents()
    
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

def main():
    parser = argparse.ArgumentParser(description='Watch a saved Q-learning agent play')
    parser.add_argument('--agent', '-a', default=None, 
                       help='Agent filename in agents folder')
    parser.add_argument('--episodes', '-e', type=int, default=10,
                       help='Number of episodes to watch (default: 10)')
    parser.add_argument('--list', '-l', action='store_true',
                       help='List all available agents')
    
    args = parser.parse_args()
    
    print("🎬 Grid World Agent Viewer")
    print("=" * 40)
    
    # List available agents if requested
    if args.list:
        list_available_agents()
        return
    
    # Determine which agent to use
    agent_filename = args.agent
    
    if agent_filename is None:
        # Interactive selection
        agent_filename = select_agent_interactively()
        if agent_filename is None:
            return
    
    # Watch the selected agent
    watch_saved_agent(agent_filename, args.episodes)

if __name__ == "__main__":
    main() 