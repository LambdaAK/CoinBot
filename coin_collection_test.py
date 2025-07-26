#!/usr/bin/env python3
"""
Coin Collection Test Suite

A comprehensive testing framework for the coin collection grid world game
with configurable parameters for board dimensions, enemy/coin/obstacle ranges,
and maximum moves.
"""

import argparse
import time
import numpy as np
import os
from grid_world import GridWorld
from coin_collection_agent import CoinCollectionAgent

class CoinCollectionTester:
    def __init__(self):
        self.test_results = []
        
    def run_test(self, board_dim=10, enemy_range=(1, 3), coin_range=(3, 6), 
                 obstacle_range=(2, 5), max_moves=50, episodes=10, render=False, 
                 agent_file=None, test_name="Custom Test"):
        """
        Run a comprehensive test with specified parameters
        
        Args:
            board_dim: Grid size (e.g., 10 for 10x10)
            enemy_range: Tuple of (min, max) enemies
            coin_range: Tuple of (min, max) coins  
            obstacle_range: Tuple of (min, max) obstacles
            max_moves: Maximum moves per episode
            episodes: Number of test episodes
            render: Whether to render the game
            agent_file: Path to AI agent file (None for manual testing)
            test_name: Name for this test configuration
        """
        
        print(f"\n{'='*60}")
        print(f"🧪 {test_name}")
        print(f"{'='*60}")
        print(f"Board Size: {board_dim}x{board_dim}")
        print(f"Enemies: {enemy_range[0]}-{enemy_range[1]}")
        print(f"Coins: {coin_range[0]}-{coin_range[1]}")
        print(f"Obstacles: {obstacle_range[0]}-{obstacle_range[1]}")
        print(f"Max Moves: {max_moves}")
        print(f"Episodes: {episodes}")
        print(f"Mode: {'AI Agent' if agent_file else 'Manual'}")
        print(f"{'='*60}")
        
        # Load agent if specified
        agent = None
        if agent_file:
            try:
                print(f"🤖 Loading agent: {agent_file}")
                agent = CoinCollectionAgent.load(agent_file)
                print("✅ Agent loaded successfully")
            except Exception as e:
                print(f"❌ Failed to load agent: {e}")
                return
        
        # Test statistics
        success_count = 0
        total_rewards = []
        total_steps = []
        total_coins_collected = []
        total_enemies_spawned = []
        total_obstacles_spawned = []
        
        for episode in range(episodes):
            print(f"\n📊 Episode {episode + 1}/{episodes}")
            
            # Create environment with custom parameters
            env = self.create_custom_environment(
                board_dim=board_dim,
                enemy_range=enemy_range,
                coin_range=coin_range,
                obstacle_range=obstacle_range,
                max_moves=max_moves
            )
            
            # Run episode
            episode_result = self.run_episode(env, agent, render)
            
            # Collect statistics
            success_count += episode_result['success']
            total_rewards.append(episode_result['reward'])
            total_steps.append(episode_result['steps'])
            total_coins_collected.append(episode_result['coins_collected'])
            total_enemies_spawned.append(episode_result['enemies_spawned'])
            total_obstacles_spawned.append(episode_result['obstacles_spawned'])
            
            # Episode summary
            status = "✅ SUCCESS" if episode_result['success'] else "❌ FAILED"
            print(f"   {status} | Reward: {episode_result['reward']:.2f} | "
                  f"Steps: {episode_result['steps']} | Coins: {episode_result['coins_collected']}")
            
            if render and episode < episodes - 1:  # Don't pause after last episode
                input("Press Enter to continue...")
        
        # Calculate and display final statistics
        self.display_test_results(
            success_count, episodes, total_rewards, total_steps, 
            total_coins_collected, total_enemies_spawned, total_obstacles_spawned,
            test_name
        )
        
        # Save test results
        test_result = {
            'test_name': test_name,
            'parameters': {
                'board_dim': board_dim,
                'enemy_range': enemy_range,
                'coin_range': coin_range,
                'obstacle_range': obstacle_range,
                'max_moves': max_moves,
                'episodes': episodes
            },
            'results': {
                'success_rate': success_count / episodes,
                'avg_reward': np.mean(total_rewards),
                'avg_steps': np.mean(total_steps),
                'avg_coins': np.mean(total_coins_collected),
                'avg_enemies': np.mean(total_enemies_spawned),
                'avg_obstacles': np.mean(total_obstacles_spawned)
            }
        }
        
        self.test_results.append(test_result)
        return test_result
    
    def create_custom_environment(self, board_dim, enemy_range, coin_range, 
                                 obstacle_range, max_moves):
        """Create a GridWorld environment with custom parameters"""
        env = GridWorld(size=board_dim, max_steps=max_moves)
        
        # Override the default placement with custom ranges
        env._customize_placement(enemy_range, coin_range, obstacle_range)
        
        return env
    
    def run_episode(self, env, agent, render):
        """Run a single episode and return results"""
        observation, info = env.reset()
        total_reward = 0
        steps = 0
        
        # Get initial state for AI agent
        if agent:
            grid_state = env.grid
            agent_pos = info['agent_pos']
            coins = info['coins']
            enemy_positions = info.get('enemy_pos', [])
            state = agent.get_state_representation(grid_state, agent_pos, coins, enemy_positions)
            old_pos = agent_pos.copy()
        
        while True:
            if render:
                env.render()
                time.sleep(0.3)
            
            # Choose action
            if agent:
                action = agent.act(state, training=False)
            else:
                # Manual play - you can implement this or just use AI
                action = agent.act(state, training=False) if agent else 0
            
            # Take action
            next_observation, env_reward, terminated, truncated, info = env.step(action)
            
            # Update state for AI agent
            if agent:
                new_grid_state = env.grid
                new_agent_pos = info['agent_pos']
                new_coins = info['coins']
                new_enemy_positions = info.get('enemy_pos', [])
                next_state = agent.get_state_representation(new_grid_state, new_agent_pos, new_coins, new_enemy_positions)
                
                enemy_collision = info.get('enemy_collision', False)
                coin_collected = info.get('coin_collected', False)
                reward = agent.calculate_reward(action, new_agent_pos, old_pos, 
                                              new_coins, new_enemy_positions, 
                                              enemy_collision, coin_collected, steps)
                
                total_reward += reward
                state = next_state
                old_pos = new_agent_pos.copy()
            else:
                total_reward += env_reward
            
            steps += 1
            
            if terminated or truncated:
                break
        
        # Determine success
        success = (terminated and info['all_coins_collected'] and 
                  not info.get('enemy_collision', False))
        
        return {
            'success': success,
            'reward': total_reward,
            'steps': steps,
            'coins_collected': info['coins_collected'],
            'enemies_spawned': len(info.get('enemy_pos', [])),
            'obstacles_spawned': len(env.obstacles) if hasattr(env, 'obstacles') else 0,
            'termination_reason': 'success' if success else 'failure'
        }
    
    def display_test_results(self, success_count, episodes, total_rewards, total_steps,
                           total_coins_collected, total_enemies_spawned, total_obstacles_spawned,
                           test_name):
        """Display comprehensive test results"""
        print(f"\n{'='*60}")
        print(f"📊 {test_name} - Final Results")
        print(f"{'='*60}")
        print(f"Success Rate: {success_count}/{episodes} ({success_count/episodes*100:.1f}%)")
        print(f"Average Reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
        print(f"Average Steps: {np.mean(total_steps):.1f} ± {np.std(total_steps):.1f}")
        print(f"Average Coins Collected: {np.mean(total_coins_collected):.1f} ± {np.std(total_coins_collected):.1f}")
        print(f"Average Enemies Spawned: {np.mean(total_enemies_spawned):.1f}")
        print(f"Average Obstacles Spawned: {np.mean(total_obstacles_spawned):.1f}")
        
        # Additional statistics
        print(f"\n📈 Performance Metrics:")
        print(f"Best Episode Reward: {max(total_rewards):.2f}")
        print(f"Worst Episode Reward: {min(total_rewards):.2f}")
        print(f"Fastest Completion: {min(total_steps)} steps")
        print(f"Most Coins Collected: {max(total_coins_collected)}")
        print(f"{'='*60}")
    
    def run_preset_tests(self):
        """Run a series of preset tests with different configurations"""
        print("🎯 Running Preset Test Suite")
        
        # Test 1: Easy level
        self.run_test(
            board_dim=8, enemy_range=(1, 2), coin_range=(3, 4), 
            obstacle_range=(1, 3), max_moves=30, episodes=5,
            test_name="Easy Level Test"
        )
        
        # Test 2: Medium level
        self.run_test(
            board_dim=10, enemy_range=(2, 3), coin_range=(4, 6), 
            obstacle_range=(3, 5), max_moves=50, episodes=5,
            test_name="Medium Level Test"
        )
        
        # Test 3: Hard level
        self.run_test(
            board_dim=12, enemy_range=(3, 4), coin_range=(5, 7), 
            obstacle_range=(4, 6), max_moves=60, episodes=5,
            test_name="Hard Level Test"
        )
        
        # Test 4: Extreme level
        self.run_test(
            board_dim=15, enemy_range=(4, 5), coin_range=(6, 8), 
            obstacle_range=(5, 8), max_moves=80, episodes=5,
            test_name="Extreme Level Test"
        )
        
        # Display overall results
        self.display_overall_results()
    
    def display_overall_results(self):
        """Display results from all tests"""
        if not self.test_results:
            return
            
        print(f"\n{'='*80}")
        print(f"🏆 OVERALL TEST SUITE RESULTS")
        print(f"{'='*80}")
        
        for result in self.test_results:
            params = result['parameters']
            stats = result['results']
            print(f"\n📋 {result['test_name']}:")
            print(f"   Board: {params['board_dim']}x{params['board_dim']} | "
                  f"Enemies: {params['enemy_range']} | Coins: {params['coin_range']}")
            print(f"   Success Rate: {stats['success_rate']:.1%} | "
                  f"Avg Reward: {stats['avg_reward']:.2f} | "
                  f"Avg Coins: {stats['avg_coins']:.1f}")

def main():
    parser = argparse.ArgumentParser(description="Coin Collection Test Suite")
    parser.add_argument('--mode', choices=['preset', 'custom', 'interactive'], 
                       default='interactive', help='Test mode')
    parser.add_argument('--board-dim', type=int, default=10, help='Board dimension')
    parser.add_argument('--enemy-min', type=int, default=1, help='Minimum enemies')
    parser.add_argument('--enemy-max', type=int, default=3, help='Maximum enemies')
    parser.add_argument('--coin-min', type=int, default=3, help='Minimum coins')
    parser.add_argument('--coin-max', type=int, default=6, help='Maximum coins')
    parser.add_argument('--obstacle-min', type=int, default=2, help='Minimum obstacles')
    parser.add_argument('--obstacle-max', type=int, default=5, help='Maximum obstacles')
    parser.add_argument('--max-moves', type=int, default=50, help='Maximum moves')
    parser.add_argument('--episodes', type=int, default=10, help='Number of episodes')
    parser.add_argument('--render', action='store_true', help='Render games')
    parser.add_argument('--agent', type=str, help='Path to agent file')
    
    args = parser.parse_args()
    
    tester = CoinCollectionTester()
    
    if args.mode == 'preset':
        tester.run_preset_tests()
    elif args.mode == 'custom':
        tester.run_test(
            board_dim=args.board_dim,
            enemy_range=(args.enemy_min, args.enemy_max),
            coin_range=(args.coin_min, args.coin_max),
            obstacle_range=(args.obstacle_min, args.obstacle_max),
            max_moves=args.max_moves,
            episodes=args.episodes,
            render=args.render,
            agent_file=args.agent,
            test_name="Custom Configuration Test"
        )
    else:  # interactive
        print("🎮 Coin Collection Test Suite - Interactive Mode")
        print("=" * 50)
        
        # Get test parameters from user
        board_dim = int(input("Board dimension (5-20, default 10): ") or 10)
        enemy_min = int(input("Minimum enemies (0-5, default 1): ") or 1)
        enemy_max = int(input("Maximum enemies (1-8, default 3): ") or 3)
        coin_min = int(input("Minimum coins (1-10, default 3): ") or 3)
        coin_max = int(input("Maximum coins (2-15, default 6): ") or 6)
        obstacle_min = int(input("Minimum obstacles (0-10, default 2): ") or 2)
        obstacle_max = int(input("Maximum obstacles (1-15, default 5): ") or 5)
        max_moves = int(input("Maximum moves (20-100, default 50): ") or 50)
        episodes = int(input("Number of episodes (1-20, default 5): ") or 5)
        
        render = input("Render games? (y/N): ").lower() == 'y'
        
        # Agent selection
        agent_file = None
        if input("Use AI agent? (y/N): ").lower() == 'y':
            agents_dir = "agents"
            if os.path.exists(agents_dir):
                agent_files = [f for f in os.listdir(agents_dir) if f.endswith('.pkl')]
                if agent_files:
                    print("Available agents:")
                    for i, fname in enumerate(agent_files):
                        print(f"  [{i+1}] {fname}")
                    agent_idx = input(f"Select agent [1-{len(agent_files)}] (default 1): ").strip()
                    try:
                        agent_idx = int(agent_idx) - 1 if agent_idx else 0
                        if 0 <= agent_idx < len(agent_files):
                            agent_file = agent_files[agent_idx]
                    except ValueError:
                        agent_file = agent_files[0]
                else:
                    print("No agent files found in agents/ directory")
        
        tester.run_test(
            board_dim=board_dim,
            enemy_range=(enemy_min, enemy_max),
            coin_range=(coin_min, coin_max),
            obstacle_range=(obstacle_min, obstacle_max),
            max_moves=max_moves,
            episodes=episodes,
            render=render,
            agent_file=agent_file,
            test_name="Interactive Test"
        )

if __name__ == "__main__":
    main() 