#!/usr/bin/env python3
"""
Watch Saved DQN Agent

This script loads a previously trained DQN agent and watches it play
the grid world game. DQN agents are stored in the 'agents' folder.
"""

from dqn_agent import DQNAgent, watch_saved_dqn_agent
import argparse
import os
import glob

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

def main():
    parser = argparse.ArgumentParser(description='Watch a saved DQN agent play')
    parser.add_argument('--agent', '-a', default=None, 
                       help='Agent filename in agents folder')
    parser.add_argument('--episodes', '-e', type=int, default=10,
                       help='Number of episodes to watch (default: 10)')
    parser.add_argument('--list', '-l', action='store_true',
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
    
    # Watch the selected agent
    watch_saved_dqn_agent(agent_filename, args.episodes)

if __name__ == "__main__":
    main() 