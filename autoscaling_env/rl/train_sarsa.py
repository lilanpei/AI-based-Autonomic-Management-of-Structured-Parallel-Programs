"""
Train SARSA Agent for OpenFaaS Autoscaling

This script trains a SARSA agent to learn optimal autoscaling policies.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import argparse
from datetime import datetime
import json

from sarsa_agent import SARSAAgent
from openfaas_autoscaling_env import OpenFaaSAutoscalingEnv


def train_sarsa(episodes=100,
                max_steps=30,
                max_workers=32,
                step_duration=20,
                learning_rate=0.1,
                gamma=0.99,
                epsilon=1.0,
                epsilon_min=0.01,
                epsilon_decay=0.995,
                save_freq=10,
                save_dir='models'):
    """
    Train SARSA agent

    Args:
        episodes: Number of training episodes
        max_steps: Maximum steps per episode
        max_workers: Maximum number of workers
        step_duration: Step duration in seconds
        learning_rate: Learning rate (alpha)
        gamma: Discount factor
        epsilon: Initial exploration rate
        epsilon_min: Minimum exploration rate
        epsilon_decay: Epsilon decay rate per episode
        save_freq: Save model every N episodes
        save_dir: Directory to save models
    """
    print("\n" + "="*70)
    print("SARSA TRAINING")
    print("="*70)
    print(f"Episodes:       {episodes}")
    print(f"Max Steps:      {max_steps}")
    print(f"Max Workers:    {max_workers}")
    print(f"Step Duration:  {step_duration}s")
    print(f"Learning Rate:  {learning_rate}")
    print(f"Gamma:          {gamma}")
    print(f"Epsilon:        {epsilon} → {epsilon_min} (decay: {epsilon_decay})")
    print("="*70 + "\n")

    # Create environment
    print("[1/4] Initializing environment...")

    # Calculate phase_duration to cover episode
    episode_duration = max_steps * step_duration
    phase_duration = (episode_duration + 60) // 4  # Divide by 4 phases, add buffer

    env = OpenFaaSAutoscalingEnv(
        max_workers=max_workers,
        min_workers=1,
        observation_window=30,
        step_duration=step_duration,
        max_steps=max_steps,
        initial_workers=3,
        initialize_workflow=True
    )

    # Update phase_duration
    env.config['phase_duration'] = phase_duration
    print(f"✓ Environment initialized (phase_duration: {phase_duration}s)")

    # Create agent
    print("\n[2/4] Creating SARSA agent...")
    agent = SARSAAgent(
        state_dim=9,
        action_dim=5,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        num_tilings=8,
        tiles_per_dim=8,
        use_replay=False  # SARSA typically doesn't use replay
    )
    print(f"✓ SARSA agent created")

    # Training loop
    print("\n[3/4] Starting training...")
    print("-"*70)

    episode_rewards = []
    episode_steps = []
    episode_qos = []

    for episode in range(episodes):
        print(f"\n{'='*70}")
        print(f"EPISODE {episode + 1}/{episodes}")
        print(f"{'='*70}")
        print(f"Epsilon: {agent.epsilon:.4f}")

        # Reset environment
        state = env.reset()

        # Select first action
        action = agent.select_action(state, training=True)

        episode_reward = 0
        episode_qos_rates = []
        step = 0

        done = False
        while not done and step < max_steps:
            # Take action
            next_state, reward, done, info = env.step(action)

            # Select next action (SARSA: use actual next action)
            next_action = agent.select_action(next_state, training=True)

            # Update Q-table
            td_error = agent.update(state, action, reward, next_state, next_action, done)

            # Track metrics
            episode_reward += reward
            episode_qos_rates.append(info['qos_rate'])

            # Move to next state
            state = next_state
            action = next_action
            step += 1

        # Episode finished
        agent.decay_epsilon()

        mean_qos = np.mean(episode_qos_rates) if episode_qos_rates else 0
        episode_rewards.append(episode_reward)
        episode_steps.append(step)
        episode_qos.append(mean_qos)

        print(f"\n{'='*70}")
        print(f"EPISODE {episode + 1} SUMMARY")
        print(f"{'='*70}")
        print(f"Steps:          {step}")
        print(f"Total Reward:   {episode_reward:.2f}")
        print(f"Mean QoS:       {mean_qos*100:.2f}%")
        print(f"Q-table Size:   {len(agent.q_table)}")
        print(f"Epsilon:        {agent.epsilon:.4f}")
        print(f"{'='*70}")

        # Save model periodically
        if (episode + 1) % save_freq == 0:
            os.makedirs(save_dir, exist_ok=True)
            filepath = os.path.join(save_dir, f'sarsa_episode_{episode+1}.pkl')
            agent.save(filepath)

            # Save training stats (convert numpy types to Python types for JSON)
            stats = {
                'episode': int(episode + 1),
                'episode_rewards': [float(r) for r in episode_rewards],
                'episode_steps': [int(s) for s in episode_steps],
                'episode_qos': [float(q) for q in episode_qos],
                'agent_stats': agent.get_stats()
            }
            stats_file = os.path.join(save_dir, f'training_stats_{episode+1}.json')
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
            print(f"\n✓ Model and stats saved (episode {episode+1})")

    # Training complete
    print("\n" + "="*70)
    print("[4/4] Training Complete!")
    print("="*70)
    print(f"Total Episodes:     {episodes}")
    print(f"Total Steps:        {agent.total_steps}")
    print(f"Q-table Size:       {len(agent.q_table)}")
    print(f"Final Epsilon:      {agent.epsilon:.4f}")
    print(f"Mean Reward (last 10): {np.mean(episode_rewards[-10:]):.2f}")
    print(f"Mean QoS (last 10):    {np.mean(episode_qos[-10:])*100:.2f}%")
    print("="*70)

    # Save final model
    os.makedirs(save_dir, exist_ok=True)
    final_path = os.path.join(save_dir, 'sarsa_final.pkl')
    agent.save(final_path)

    # Save final stats (convert numpy types to Python types for JSON)
    stats = {
        'episodes': int(episodes),
        'episode_rewards': [float(r) for r in episode_rewards],
        'episode_steps': [int(s) for s in episode_steps],
        'episode_qos': [float(q) for q in episode_qos],
        'agent_stats': agent.get_stats(),
        'config': {
            'max_steps': int(max_steps),
            'max_workers': int(max_workers),
            'step_duration': int(step_duration),
            'learning_rate': float(learning_rate),
            'gamma': float(gamma),
            'epsilon_decay': float(epsilon_decay),
        }
    }
    stats_file = os.path.join(save_dir, 'training_stats_final.json')
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\n✓ Final model saved to {final_path}")
    print(f"✓ Training stats saved to {stats_file}")

    # Close environment
    env.close()
    print("\n✓ Done!\n")

    return agent, episode_rewards, episode_qos


def main():
    parser = argparse.ArgumentParser(description='Train SARSA agent for autoscaling')

    # Training parameters
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of training episodes')
    parser.add_argument('--max-steps', type=int, default=30,
                        help='Maximum steps per episode')
    parser.add_argument('--max-workers', type=int, default=32,
                        help='Maximum number of workers')
    parser.add_argument('--step-duration', type=int, default=20,
                        help='Step duration in seconds')

    # SARSA hyperparameters
    parser.add_argument('--learning-rate', type=float, default=0.1,
                        help='Learning rate (alpha)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--epsilon', type=float, default=1.0,
                        help='Initial exploration rate')
    parser.add_argument('--epsilon-min', type=float, default=0.01,
                        help='Minimum exploration rate')
    parser.add_argument('--epsilon-decay', type=float, default=0.995,
                        help='Epsilon decay rate per episode')

    # Saving
    parser.add_argument('--save-freq', type=int, default=10,
                        help='Save model every N episodes')
    parser.add_argument('--save-dir', type=str, default='models/sarsa',
                        help='Directory to save models')

    args = parser.parse_args()

    # Train agent
    train_sarsa(
        episodes=args.episodes,
        max_steps=args.max_steps,
        max_workers=args.max_workers,
        step_duration=args.step_duration,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        save_freq=args.save_freq,
        save_dir=args.save_dir
    )


if __name__ == '__main__':
    main()
