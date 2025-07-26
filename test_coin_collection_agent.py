#!/usr/bin/env python3
"""
Test Coin Collection Agent

Loads a trained coin collection agent and evaluates it on the GridWorld environment.
"""
import time
import numpy as np
from coin_collection_agent import CoinCollectionAgent
from grid_world import GridWorld
import os


def main():
    print("\n=== Coin Collection Agent Tester ===")
    # List available agents
    agents_dir = "agents"
    agent_files = [f for f in os.listdir(agents_dir) if f.endswith('.pkl')]
    if not agent_files:
        print("No agent files found in 'agents/' directory.")
        return
    print("Available agents:")
    for idx, fname in enumerate(agent_files):
        print(f"  [{idx+1}] {fname}")
    agent_idx = input(f"Select agent [1-{len(agent_files)}] (default 1): ").strip()
    try:
        agent_idx = int(agent_idx) - 1 if agent_idx else 0
        if not (0 <= agent_idx < len(agent_files)):
            print("Invalid selection.")
            return
    except ValueError:
        agent_idx = 0
    agent_file = agent_files[agent_idx]

    try:
        episodes = int(input("Number of test episodes [10]: ") or 10)
    except ValueError:
        episodes = 10
    try:
        grid_size = int(input("Grid size [10]: ") or 10)
    except ValueError:
        grid_size = 10
    render_input = input("Render environment? [y/N]: ").strip().lower()
    render = render_input == 'y'

    print(f"\n🔍 Loading agent from: agents/{agent_file}")
    agent = CoinCollectionAgent.load(agent_file)

    print(f"\n🧪 Testing agent for {episodes} episodes on {grid_size}x{grid_size} grid...")
    env = GridWorld(size=grid_size)

    success_count = 0
    total_rewards = []
    total_steps = []
    total_coins_collected = []

    for episode in range(episodes):
        observation, info = env.reset()
        total_reward = 0
        steps = 0
        grid_state = env.grid
        agent_pos = info['agent_pos']
        coins = info['coins']
        enemy_positions = info.get('enemy_pos', [])
        state = agent.get_state_representation(grid_state, agent_pos, coins, enemy_positions)
        old_pos = agent_pos.copy()
        print(f"\nEpisode {episode + 1}:")
        while True:
            action = agent.act(state, training=False)
            next_observation, env_reward, terminated, truncated, info = env.step(action)
            new_grid_state = env.grid
            new_agent_pos = info['agent_pos']
            new_coins = info['coins']
            new_enemy_positions = info.get('enemy_pos', [])
            next_state = agent.get_state_representation(new_grid_state, new_agent_pos, new_coins, new_enemy_positions)
            enemy_collision = info.get('enemy_collision', False)
            coin_collected = info.get('coin_collected', False)
            reward = agent.calculate_reward(action, new_agent_pos, old_pos, new_coins, new_enemy_positions, enemy_collision, coin_collected, steps)
            total_reward += reward
            steps += 1
            state = next_state
            old_pos = new_agent_pos.copy()
            if render:
                env.render()
                time.sleep(0.2)
            if terminated or truncated:
                if terminated and info['all_coins_collected'] and not enemy_collision:
                    print("🎉 Success! All coins collected!")
                    success_count += 1
                elif enemy_collision:
                    print("💀 Game Over! Enemy collision!")
                else:
                    print("⏰ Time's up!")
                break
        total_rewards.append(total_reward)
        total_steps.append(steps)
        total_coins_collected.append(info['coins_collected'])
        print(f"Total reward: {total_reward:.2f}, Steps: {steps}, Coins: {info['coins_collected']}")

    print(f"\n📊 Test Results:")
    print(f"Success rate: {success_count}/{episodes} ({success_count/episodes*100:.1f}%)")
    print(f"Average reward: {np.mean(total_rewards):.2f}")
    print(f"Average steps: {np.mean(total_steps):.1f}")
    print(f"Average coins collected: {np.mean(total_coins_collected):.1f}")

if __name__ == "__main__":
    main() 