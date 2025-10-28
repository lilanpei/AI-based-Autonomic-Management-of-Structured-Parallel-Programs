"""
Test/Evaluate SARSA Agent for OpenFaaS Autoscaling

This script evaluates a trained SARSA agent and compares it with baselines.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import argparse
from datetime import datetime
import matplotlib.pyplot as plt

from sarsa_agent import SARSAAgent
from openfaas_autoscaling_env import OpenFaaSAutoscalingEnv
from baselines.reactive_baselines import ReactiveAverage, ReactiveMaximum


def test_agent(agent, env, max_steps, agent_name="SARSA"):
    """
    Test an agent in the environment

    Args:
        agent: Agent to test (SARSA or baseline)
        env: Environment
        max_steps: Maximum steps
        agent_name: Name for logging

    Returns:
        Dictionary with results
    """
    print(f"\n{'='*70}")
    print(f"TESTING {agent_name}")
    print(f"{'='*70}")

    state = env.reset()

    rewards = []
    qos_rates = []
    worker_counts = []
    queue_lengths = []

    done = False
    step = 0

    while not done and step < max_steps:
        # Select action (no exploration during testing)
        if hasattr(agent, 'select_action'):
            action = agent.select_action(state, training=False)
        else:
            # Baseline agent
            action = agent.get_action(state)

        # Take action
        next_state, reward, done, info = env.step(action)

        # Track metrics
        rewards.append(reward)
        qos_rates.append(info['qos_rate'])
        worker_counts.append(info['workers'])
        queue_lengths.append(info['queue_length'])

        state = next_state
        step += 1

    # Calculate statistics
    results = {
        'agent_name': agent_name,
        'rewards': rewards,
        'qos_rates': qos_rates,
        'worker_counts': worker_counts,
        'queue_lengths': queue_lengths,
        'total_reward': sum(rewards),
        'mean_reward': np.mean(rewards),
        'mean_qos': np.mean(qos_rates),
        'mean_workers': np.mean(worker_counts),
        'total_steps': len(rewards)
    }

    print(f"\n{agent_name} Results:")
    print(f"  Total Steps:    {results['total_steps']}")
    print(f"  Total Reward:   {results['total_reward']:.2f}")
    print(f"  Mean Reward:    {results['mean_reward']:.2f}")
    print(f"  Mean QoS:       {results['mean_qos']*100:.2f}%")
    print(f"  Mean Workers:   {results['mean_workers']:.2f}")

    return results


def plot_comparison(results_dict, save_dir='plots'):
    """
    Plot comparison between agents

    Args:
        results_dict: Dictionary mapping agent names to results
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('SARSA vs Baselines Comparison', fontsize=16, fontweight='bold')

    colors = {
        'SARSA': 'green',
        'ReactiveAverage': 'blue',
        'ReactiveMaximum': 'red'
    }
    markers = {
        'SARSA': '^',
        'ReactiveAverage': 'o',
        'ReactiveMaximum': 's'
    }

    # 1. Cumulative Reward
    ax = axes[0, 0]
    for agent_name, results in results_dict.items():
        cumulative = np.cumsum(results['rewards'])
        steps = range(1, len(cumulative) + 1)
        ax.plot(steps, cumulative, color=colors.get(agent_name, 'gray'),
                linewidth=2, marker=markers.get(agent_name, 'o'),
                markersize=4, label=agent_name, alpha=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel('Cumulative Reward')
    ax.set_title('Cumulative Reward Over Time')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 2. QoS Rate
    ax = axes[0, 1]
    for agent_name, results in results_dict.items():
        steps = range(1, len(results['qos_rates']) + 1)
        qos_percent = [q*100 for q in results['qos_rates']]
        ax.plot(steps, qos_percent, color=colors.get(agent_name, 'gray'),
                linewidth=2, marker=markers.get(agent_name, 'o'),
                markersize=4, label=agent_name, alpha=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel('QoS Rate (%)')
    ax.set_title('QoS Rate Over Time')
    ax.set_ylim([0, 105])
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 3. Worker Count
    ax = axes[1, 0]
    for agent_name, results in results_dict.items():
        steps = range(1, len(results['worker_counts']) + 1)
        ax.plot(steps, results['worker_counts'], color=colors.get(agent_name, 'gray'),
                linewidth=2, marker=markers.get(agent_name, 'o'),
                markersize=4, label=agent_name, alpha=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel('Number of Workers')
    ax.set_title('Worker Count Over Time')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 4. Queue Length
    ax = axes[1, 1]
    for agent_name, results in results_dict.items():
        steps = range(1, len(results['queue_lengths']) + 1)
        ax.plot(steps, results['queue_lengths'], color=colors.get(agent_name, 'gray'),
                linewidth=2, marker=markers.get(agent_name, 'o'),
                markersize=4, label=agent_name, alpha=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel('Queue Length')
    ax.set_title('Worker Queue Length Over Time')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    filename = f'sarsa_comparison_{timestamp}.png'
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"\n[PLOT] Saved to: {filepath}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Test SARSA agent')

    # Test parameters
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained SARSA model')
    parser.add_argument('--max-steps', type=int, default=30,
                        help='Maximum steps per episode')
    parser.add_argument('--max-workers', type=int, default=32,
                        help='Maximum number of workers')
    parser.add_argument('--step-duration', type=int, default=20,
                        help='Step duration in seconds')
    parser.add_argument('--compare-baselines', action='store_true',
                        help='Compare with reactive baselines')

    args = parser.parse_args()

    print("\n" + "="*70)
    print("SARSA EVALUATION")
    print("="*70)
    print(f"Model:          {args.model}")
    print(f"Max Steps:      {args.max_steps}")
    print(f"Max Workers:    {args.max_workers}")
    print(f"Step Duration:  {args.step_duration}s")
    print("="*70)

    # Load SARSA agent
    print("\n[1/3] Loading SARSA agent...")
    agent = SARSAAgent(state_dim=9, action_dim=5)
    agent.load(args.model)
    print("✓ SARSA agent loaded")

    # Create environment
    print("\n[2/3] Initializing environment...")
    episode_duration = args.max_steps * args.step_duration
    phase_duration = (episode_duration + 60) // 4

    env = OpenFaaSAutoscalingEnv(
        max_workers=args.max_workers,
        min_workers=1,
        observation_window=30,
        step_duration=args.step_duration,
        max_steps=args.max_steps,
        initial_workers=3,
        initialize_workflow=True
    )
    env.config['phase_duration'] = phase_duration
    print(f"✓ Environment initialized")

    # Test SARSA agent
    print("\n[3/3] Testing agents...")
    results = {}

    sarsa_results = test_agent(agent, env, args.max_steps, "SARSA")
    results['SARSA'] = sarsa_results

    # Compare with baselines
    if args.compare_baselines:
        print("\n" + "-"*70)
        print("COMPARING WITH BASELINES")
        print("-"*70)

        # Test ReactiveAverage
        baseline_avg = ReactiveAverage(action_dim=5, max_workers=args.max_workers)
        avg_results = test_agent(baseline_avg, env, args.max_steps, "ReactiveAverage")
        results['ReactiveAverage'] = avg_results

        # Test ReactiveMaximum
        baseline_max = ReactiveMaximum(action_dim=5, max_workers=args.max_workers)
        max_results = test_agent(baseline_max, env, args.max_steps, "ReactiveMaximum")
        results['ReactiveMaximum'] = max_results

    # Print comparison table
    print("\n" + "="*70)
    print("COMPARISON TABLE")
    print("="*70)
    print(f"{'Metric':<25} {'SARSA':<15} {'Avg':<15} {'Max':<15}")
    print("-"*70)

    metrics = [
        ('Total Reward', 'total_reward', '.2f'),
        ('Mean Reward', 'mean_reward', '.2f'),
        ('Mean QoS', 'mean_qos', '.2%'),
        ('Mean Workers', 'mean_workers', '.2f'),
        ('Total Steps', 'total_steps', 'd'),
    ]

    for metric_name, metric_key, fmt in metrics:
        sarsa_val = results['SARSA'][metric_key]
        avg_val = results.get('ReactiveAverage', {}).get(metric_key, 0)
        max_val = results.get('ReactiveMaximum', {}).get(metric_key, 0)

        print(f"{metric_name:<25} {sarsa_val:{fmt}:<15} {avg_val:{fmt}:<15} {max_val:{fmt}:<15}")

    print("="*70)

    # Determine winner
    if args.compare_baselines:
        print("\nKEY INSIGHTS:")

        # QoS comparison
        sarsa_qos = results['SARSA']['mean_qos']
        avg_qos = results['ReactiveAverage']['mean_qos']
        max_qos = results['ReactiveMaximum']['mean_qos']

        best_qos = max(sarsa_qos, avg_qos, max_qos)
        best_qos_agent = [k for k, v in results.items() if v['mean_qos'] == best_qos][0]
        print(f"✓ {best_qos_agent} has best QoS ({best_qos*100:.1f}%)")

        # Reward comparison
        sarsa_reward = results['SARSA']['total_reward']
        avg_reward = results['ReactiveAverage']['total_reward']
        max_reward = results['ReactiveMaximum']['total_reward']

        best_reward = max(sarsa_reward, avg_reward, max_reward)
        best_reward_agent = [k for k, v in results.items() if v['total_reward'] == best_reward][0]
        print(f"✓ {best_reward_agent} has best reward ({best_reward:.2f})")

        # Worker efficiency
        sarsa_workers = results['SARSA']['mean_workers']
        avg_workers = results['ReactiveAverage']['mean_workers']
        max_workers = results['ReactiveMaximum']['mean_workers']

        min_workers = min(sarsa_workers, avg_workers, max_workers)
        most_efficient = [k for k, v in results.items() if v['mean_workers'] == min_workers][0]
        print(f"✓ {most_efficient} is most efficient ({min_workers:.1f} workers)")

    # Generate plots
    if args.compare_baselines:
        print("\n[PLOTTING] Generating visualizations...")
        plot_comparison(results)
        print("✓ Plots generated")

    # Close environment
    env.close()
    print("\n✓ Done!\n")


if __name__ == '__main__':
    main()
