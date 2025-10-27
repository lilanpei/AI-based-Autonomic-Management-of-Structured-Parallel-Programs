#!/usr/bin/env python3
"""
Test Reactive Baselines with OpenFaaS Autoscaling Environment

This script tests both reactive baseline policies:
1. ReactiveAverage (conservative)
2. ReactiveMaximum (aggressive)

Usage:
    python test_reactive_baselines.py --agent average --steps 15
    python test_reactive_baselines.py --agent maximum --steps 15
    python test_reactive_baselines.py --agent both --steps 15
"""

import os
import sys
import argparse
import numpy as np
import time
import matplotlib.pyplot as plt
from datetime import datetime

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from autoscaling_env.openfaas_autoscaling_env import OpenFaaSAutoscalingEnv
from autoscaling_env.baselines.reactive_policies import ReactiveAverage, ReactiveMaximum


def test_single_agent(agent_name, agent, env, num_steps=20):
    """
    Test a single reactive baseline agent

    Args:
        agent_name: Name of the agent
        agent: Agent instance
        env: Environment instance
        num_steps: Number of steps to run
    """
    print(f"\n{'='*70}")
    print(f"TESTING {agent_name.upper()}")
    print(f"{'='*70}")

    # Reset environment
    print("\n[1/3] Resetting environment...")
    state = env.reset()
    print(f"✓ Initial state: {state}")

    # Track statistics
    rewards = []
    qos_rates = []
    worker_counts = []
    actions_taken = []
    queue_lengths = []
    avg_times = []
    max_times = []
    arrival_rates = []

    print(f"\n[2/3] Running {num_steps} steps...")
    print("-" * 70)

    for step in range(num_steps):
        # Select action
        action = agent.select_action(state, training=False)
        action_map = {0: "-2", 1: "-1", 2: "0", 3: "+1", 4: "+2"}

        # Execute action
        next_state, reward, done, info = env.step(action)

        # Track statistics
        rewards.append(reward)
        qos_rates.append(info['qos_rate'])
        worker_counts.append(info['workers'])
        actions_taken.append(action)
        queue_lengths.append(next_state[1])  # worker_queue
        avg_times.append(next_state[5])
        max_times.append(next_state[6])
        arrival_rates.append(next_state[7])

        # Print step info
        queue_length = next_state[1]  # worker_queue
        workers = next_state[4]
        qos_rate = next_state[8]

        print(f"{'Step':<6} {'Action':<8} {'Input_Q':<8} {'Worker_Q':<8} {'Result_Q':<8} {'Output_Q':<8} {'Workers':<8} {'QoS':<8} {'AVG_T':<8} {'MAX_T':<8} {'ARR.':<8} {'Reward':<10}")
        print(f"{step+1:<6} {action_map[action]:<8} {next_state[0]:<8.0f} {queue_length:<8.0f} {next_state[2]:<8.0f} {next_state[3]:<8.0f} {workers:<8.0f} {qos_rate:<8.2%} {avg_times[-1]:<8.2f} {max_times[-1]:<8.2f} {arrival_rates[-1]:<8.2f} {reward:<10.2f}")

        # Update state
        state = next_state

        if done:
            print(f"\n[INFO] Episode finished at step {step+1}")
            break

    # Print summary
    print(f"\n[3/3] Summary Statistics")
    print("-" * 70)
    print(f"Total Steps:        {len(rewards)}")
    print(f"Total Reward:       {sum(rewards):.2f}")
    print(f"Mean Reward:        {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Mean QoS Rate:      {np.mean(qos_rates):.2%}")
    print(f"Mean Workers:       {np.mean(worker_counts):.1f}")
    print(f"Scaling Actions:    {sum(1 for a in actions_taken if a != 2)}")
    print(f"No-op Actions:      {sum(1 for a in actions_taken if a == 2)}")
    print("="*70 + "\n")

    return {
        'rewards': rewards,
        'qos_rates': qos_rates,
        'worker_counts': worker_counts,
        'actions': actions_taken,
        'queue_lengths': queue_lengths,
        'avg_times': avg_times,
        'max_times': max_times,
        'arrival_rates': arrival_rates
    }


def plot_results(results, agent_name, save_dir='plots'):
    """
    Plot detailed results for a single agent

    Args:
        results: Dictionary with metrics
        agent_name: Name of the agent
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    steps = range(1, len(results['rewards']) + 1)

    # Create figure with 6 subplots
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle(f'{agent_name} Performance Metrics', fontsize=16, fontweight='bold')

    # 1. Rewards over time
    ax = axes[0, 0]
    ax.plot(steps, results['rewards'], 'b-', linewidth=2, marker='o', markersize=4)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Zero line')
    ax.set_xlabel('Step')
    ax.set_ylabel('Reward')
    ax.set_title('Reward per Step')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Add cumulative reward as text
    cumulative = np.cumsum(results['rewards'])
    ax.text(0.02, 0.98, f'Total: {cumulative[-1]:.1f}',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 2. QoS Rate over time
    ax = axes[0, 1]
    ax.plot(steps, [q*100 for q in results['qos_rates']], 'g-', linewidth=2, marker='o', markersize=4)
    ax.axhline(y=90, color='orange', linestyle='--', alpha=0.5, label='90% target')
    ax.axhline(y=80, color='r', linestyle='--', alpha=0.5, label='80% threshold')
    ax.set_xlabel('Step')
    ax.set_ylabel('QoS Rate (%)')
    ax.set_title('QoS Success Rate')
    ax.set_ylim([0, 105])
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Add mean QoS as text
    mean_qos = np.mean(results['qos_rates']) * 100
    ax.text(0.02, 0.98, f'Mean: {mean_qos:.1f}%',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    # 3. Worker count over time
    ax = axes[1, 0]
    ax.plot(steps, results['worker_counts'], 'purple', linewidth=2, marker='s', markersize=4)
    ax.set_xlabel('Step')
    ax.set_ylabel('Number of Workers')
    ax.set_title('Worker Count')
    ax.grid(True, alpha=0.3)

    # Add mean workers as text
    mean_workers = np.mean(results['worker_counts'])
    ax.text(0.02, 0.98, f'Mean: {mean_workers:.1f}',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='plum', alpha=0.5))

    # 4. Queue length over time
    ax = axes[1, 1]
    ax.plot(steps, results['queue_lengths'], 'orange', linewidth=2, marker='o', markersize=4)
    ax.set_xlabel('Step')
    ax.set_ylabel('Queue Length')
    ax.set_title('Worker Queue Length')
    ax.grid(True, alpha=0.3)

    # Add mean queue as text
    mean_queue = np.mean(results['queue_lengths'])
    ax.text(0.02, 0.98, f'Mean: {mean_queue:.1f}',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='peachpuff', alpha=0.5))

    # 5. Processing times
    ax = axes[2, 0]
    ax.plot(steps, results['avg_times'], 'b-', linewidth=2, marker='o', markersize=4, label='Avg Time')
    ax.plot(steps, results['max_times'], 'r-', linewidth=2, marker='s', markersize=4, label='Max Time')
    ax.set_xlabel('Step')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Processing Times')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 6. Arrival rate
    ax = axes[2, 1]
    ax.plot(steps, results['arrival_rates'], 'cyan', linewidth=2, marker='o', markersize=4)
    ax.set_xlabel('Step')
    ax.set_ylabel('Tasks/second')
    ax.set_title('Task Arrival Rate')
    ax.grid(True, alpha=0.3)

    # Add mean arrival rate as text
    mean_arrival = np.mean(results['arrival_rates'])
    ax.text(0.02, 0.98, f'Mean: {mean_arrival:.2f} tasks/s',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.5))

    plt.tight_layout()

    # Save plot
    filename = f'{agent_name.lower().replace(" ", "_")}_{timestamp}.png'
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"[PLOT] Saved to: {filepath}")

    plt.close()


def plot_comparison(results_dict, save_dir='plots'):
    """
    Plot comparison between multiple agents

    Args:
        results_dict: Dictionary mapping agent names to results
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Baseline Policy Comparison', fontsize=16, fontweight='bold')

    colors = {'ReactiveAverage': 'blue', 'ReactiveMaximum': 'red'}
    markers = {'ReactiveAverage': 'o', 'ReactiveMaximum': 's'}

    # 1. Cumulative Reward
    ax = axes[0, 0]
    for agent_name, results in results_dict.items():
        cumulative = np.cumsum(results['rewards'])
        steps = range(1, len(cumulative) + 1)
        ax.plot(steps, cumulative, color=colors[agent_name], linewidth=2,
                marker=markers[agent_name], markersize=4, label=agent_name, alpha=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel('Cumulative Reward')
    ax.set_title('Cumulative Reward Over Time')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 2. QoS Rate
    ax = axes[0, 1]
    for agent_name, results in results_dict.items():
        steps = range(1, len(results['qos_rates']) + 1)
        qos_pct = [q*100 for q in results['qos_rates']]
        ax.plot(steps, qos_pct, color=colors[agent_name], linewidth=2,
                marker=markers[agent_name], markersize=4, label=agent_name, alpha=0.7)
    ax.axhline(y=90, color='orange', linestyle='--', alpha=0.5, label='90% target')
    ax.set_xlabel('Step')
    ax.set_ylabel('QoS Rate (%)')
    ax.set_title('QoS Success Rate Comparison')
    ax.set_ylim([0, 105])
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 3. Worker Count
    ax = axes[1, 0]
    for agent_name, results in results_dict.items():
        steps = range(1, len(results['worker_counts']) + 1)
        ax.plot(steps, results['worker_counts'], color=colors[agent_name], linewidth=2,
                marker=markers[agent_name], markersize=4, label=agent_name, alpha=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel('Number of Workers')
    ax.set_title('Worker Count Comparison')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 4. Queue Length
    ax = axes[1, 1]
    for agent_name, results in results_dict.items():
        steps = range(1, len(results['queue_lengths']) + 1)
        ax.plot(steps, results['queue_lengths'], color=colors[agent_name], linewidth=2,
                marker=markers[agent_name], markersize=4, label=agent_name, alpha=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel('Queue Length')
    ax.set_title('Worker Queue Length Comparison')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    # Save plot
    filename = f'baseline_comparison_{timestamp}.png'
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"[PLOT] Comparison saved to: {filepath}")

    plt.close()


def compare_agents(env, num_steps=20):
    """
    Compare both reactive baseline agents

    Args:
        env: Environment instance
        num_steps: Number of steps to run
    """
    print("\n" + "="*70)
    print("COMPARING REACTIVE BASELINES")
    print("="*70)

    # Initialize agents
    agents = {
        'ReactiveAverage': ReactiveAverage(),
        'ReactiveMaximum': ReactiveMaximum()
    }

    # Test each agent
    results = {}
    for agent_name, agent in agents.items():
        results[agent_name] = test_single_agent(agent_name, agent, env, num_steps)

    # Print comparison
    print("\n" + "="*70)
    print("COMPARISON TABLE")
    print("="*70)
    print(f"{'Metric':<25} {'Average':<20} {'Maximum':<20}")
    print("-" * 70)

    metrics = [
        ('Total Reward', lambda r: sum(r['rewards'])),
        ('Mean Reward', lambda r: np.mean(r['rewards'])),
        ('Mean QoS Rate', lambda r: np.mean(r['qos_rates'])),
        ('Mean Workers', lambda r: np.mean(r['worker_counts'])),
        ('Scaling Actions', lambda r: sum(1 for a in r['actions'] if a != 2))
    ]

    for metric_name, metric_fn in metrics:
        avg_val = metric_fn(results['ReactiveAverage'])
        max_val = metric_fn(results['ReactiveMaximum'])

        if 'QoS' in metric_name:
            print(f"{metric_name:<25} {avg_val:<20.2%} {max_val:<20.2%}")
        else:
            print(f"{metric_name:<25} {avg_val:<20.2f} {max_val:<20.2f}")

    print("="*70 + "\n")

    # Insights
    print("KEY INSIGHTS:")
    avg_qos = np.mean(results['ReactiveAverage']['qos_rates'])
    max_qos = np.mean(results['ReactiveMaximum']['qos_rates'])
    avg_workers = np.mean(results['ReactiveAverage']['worker_counts'])
    max_workers = np.mean(results['ReactiveMaximum']['worker_counts'])

    if max_qos > avg_qos:
        print(f"✓ ReactiveMaximum has better QoS ({max_qos:.1%} vs {avg_qos:.1%})")
    else:
        print(f"✓ ReactiveAverage has better QoS ({avg_qos:.1%} vs {max_qos:.1%})")

    if avg_workers < max_workers:
        print(f"✓ ReactiveAverage uses fewer workers ({avg_workers:.1f} vs {max_workers:.1f})")
    else:
        print(f"✓ ReactiveMaximum uses fewer workers ({max_workers:.1f} vs {avg_workers:.1f})")

    print()

    # Generate plots
    print("\n[PLOTTING] Generating visualizations...")
    plot_comparison(results)
    for agent_name, agent_results in results.items():
        plot_results(agent_results, agent_name)
    print("✓ All plots generated\n")


def main():
    parser = argparse.ArgumentParser(description='Test reactive baseline policies')
    parser.add_argument('--agent', choices=['average', 'maximum', 'both'], default='both',
                        help='Which agent to test')
    parser.add_argument('--steps', type=int, default=30,
                        help='Maximum number of steps (episode ends early if all tasks complete)')
    parser.add_argument('--max-workers', type=int, default=32,
                        help='Maximum number of workers')
    parser.add_argument('--step-duration', type=int, default=20,
                        help='Total step duration in seconds (includes scaling ~10s + observation ~10s)')

    args = parser.parse_args()

    print("\n" + "="*70)
    print("REACTIVE BASELINE TEST")
    print("="*70)
    print(f"Agent:         {args.agent}")
    print(f"Steps:         {args.steps}")
    print(f"Max Workers:   {args.max_workers}")
    print(f"Step Duration: {args.step_duration}s")
    print("="*70)

    # Initialize environment
    print("\n[SETUP] Initializing environment...")
    print("[INFO] This will deploy emitter, collector, and initial workers...")
    print("[INFO] This may take 1-2 minutes on first run...")
    try:
        env = OpenFaaSAutoscalingEnv(
            max_workers=args.max_workers,
            min_workers=1,
            observation_window=10,
            step_duration=args.step_duration,
            max_steps=args.steps,
            initial_workers=1,
            initialize_workflow=True  # Deploy emitter/collector/workers
        )
        print("✓ Environment initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize environment: {e}")
        print("\nPlease ensure:")
        print("  1. Kubernetes cluster is running")
        print("  2. Redis is accessible")
        print("  3. OpenFaaS is deployed")
        print("  4. OpenFaaS gateway is accessible (port 8080)")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Run tests
    try:
        if args.agent == 'both':
            compare_agents(env, args.steps)
        else:
            if args.agent == 'average':
                agent = ReactiveAverage()
                agent_name = 'ReactiveAverage'
            else:  # maximum
                agent = ReactiveMaximum()
                agent_name = 'ReactiveMaximum'

            results = test_single_agent(agent_name, agent, env, args.steps)

            # Generate plot for single agent
            print("\n[PLOTTING] Generating visualization...")
            plot_results(results, agent_name)
            print("✓ Plot generated\n")

    except KeyboardInterrupt:
        print("\n\n[INFO] Test interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        print("\n[CLEANUP] Closing environment...")
        env.close()
        print("✓ Done\n")


if __name__ == "__main__":
    main()
