#!/usr/bin/env python3
"""Train a tabular SARSA agent against the OpenFaaS autoscaling environment."""

from __future__ import annotations

import argparse
import signal
from pathlib import Path
from typing import Dict, List

import numpy as np
import os
import sys

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
baselines_dir = current_dir  # We're in baselines/
autoscaling_env_dir = os.path.dirname(baselines_dir)  # Go up to autoscaling_env/
project_root = os.path.dirname(autoscaling_env_dir)  # Go up to project root
sys.path.insert(0, project_root)

from autoscaling_env.openfaas_autoscaling_env import OpenFaaSAutoscalingEnv
from autoscaling_env.rl.sarsa_agent import (
    SARSAAgent,
    SARSAHyperparams,
)
from autoscaling_env.rl.utils import (
    build_discretization_config,
    configure_logging,
    prepare_output_directory,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=100, help="Number of training episodes")
    parser.add_argument("--max-steps", type=int, default=50, help="Maximum steps per episode")
    parser.add_argument("--step-duration", type=int, default=8, help="Seconds per action step in the environment")
    parser.add_argument("--observation-window", type=int, default=8, help="Observation window size passed to the environment")
    parser.add_argument("--initial-workers", type=int, default=12, help="Initial number of worker replicas")
    parser.add_argument("--output-dir", type=Path, default=Path("runs"), help="Base directory for experiment outputs")
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.2,
        help="Initial epsilon for epsilon-greedy exploration",
    )
    parser.add_argument("--epsilon-min", type=float, default=0.05, help="Minimum epsilon after decay")
    parser.add_argument("--epsilon-decay", type=float, default=0.995, help="Episode-wise epsilon decay factor")
    parser.add_argument("--alpha", type=float, default=0.1, help="Learning rate (alpha)")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor (gamma)")
    parser.add_argument(
        "--bins",
        type=int,
        nargs=9,
        default=[10, 10, 10, 10, 16, 10, 10, 10, 10],
        metavar=(
            "input_q",
            "worker_q",
            "result_q",
            "output_q",
            "workers",
            "avg_time",
            "max_time",
            "arrival_rate",
            "qos",
        ),
        help="Number of discretization bins for each observation dimension",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help="Save intermediate checkpoints every N episodes (0 to disable)",
    )
    parser.add_argument(
        "--initialize-workflow",
        action="store_true",
        help="Deploy/initialize the OpenFaaS workflow before training begins",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def create_agent(env: OpenFaaSAutoscalingEnv, args: argparse.Namespace) -> SARSAAgent:
    observation_low = env.observation_space.low
    observation_high = env.observation_space.high

    discretization = build_discretization_config(
        observation_low=observation_low,
        observation_high=observation_high,
        bins_per_dimension=args.bins,
    )

    hyperparams = SARSAHyperparams(
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
    )

    return SARSAAgent(
        action_size=env.action_space.n,
        discretization=discretization,
        hyperparams=hyperparams,
    )


def plot_training_curves(metrics: List[Dict[str, float]], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    episodes = np.arange(1, len(metrics) + 1)
    total_rewards = np.array([m["total_reward"] for m in metrics])
    qos_rates = np.array([m["mean_qos"] for m in metrics])
    epsilons = np.array([m["epsilon"] for m in metrics])

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.set_title("SARSA Training Progress")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Reward", color="tab:blue")
    ax1.plot(episodes, total_rewards, label="Total Reward", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.set_ylabel("QoS Rate", color="tab:green")
    ax2.plot(episodes, qos_rates, label="Mean QoS", color="tab:green", linestyle="--")
    ax2.tick_params(axis="y", labelcolor="tab:green")

    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("outward", 60))
    ax3.set_ylabel("Epsilon", color="tab:red")
    ax3.plot(episodes, epsilons, label="Epsilon", color="tab:red", linestyle=":")
    ax3.tick_params(axis="y", labelcolor="tab:red")

    fig.tight_layout()
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    fig.savefig(plots_dir / "training_curves.png", dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    experiment_dir = prepare_output_directory(args.output_dir)
    logger = configure_logging(experiment_dir / "logs")
    logger.info("Experiment directory: %s", experiment_dir)

    env = OpenFaaSAutoscalingEnv(
        max_steps=args.max_steps,
        step_duration=args.step_duration,
        observation_window=args.observation_window,
        initial_workers=args.initial_workers,
        initialize_workflow=args.initialize_workflow,
    )

    agent = create_agent(env, args)
    logger.info(
        "Initialized SARSA agent | alpha=%.3f gamma=%.3f epsilon=%.3f bins=%s",
        agent.hyperparams.alpha,
        agent.hyperparams.gamma,
        agent.hyperparams.epsilon,
        args.bins,
    )

    metrics: List[Dict[str, float]] = []

    def _graceful_exit(*_args, **_kwargs):
        logger.warning("Training interrupted. Saving current model state...")
        agent.save(experiment_dir / "models" / "sarsa_interrupt.pkl")
        write_json({"episodes_completed": len(metrics)}, experiment_dir / "training_summary.json")
        env.close()
        raise SystemExit(1)

    signal.signal(signal.SIGINT, _graceful_exit)

    for episode in range(1, args.episodes + 1):
        observation = env.reset()
        state = agent.discretize(observation)
        action = agent.select_action(state)

        total_reward = 0.0
        qos_rates: List[float] = []
        step_count = 0

        for step in range(args.max_steps):
            next_observation, reward, done, info = env.step(action)
            next_state = agent.discretize(next_observation)
            next_action = agent.select_action(next_state)

            agent.update(state, action, reward, next_state, next_action)

            state = next_state
            action = next_action
            total_reward += reward
            qos_rates.append(info.get("qos_rate", 0.0))
            step_count = step + 1

            if done:
                break

        agent.decay_epsilon()

        episode_metrics = {
            "episode": episode,
            "steps": step_count,
            "total_reward": float(total_reward),
            "mean_qos": float(np.mean(qos_rates) if qos_rates else 0.0),
            "epsilon": float(agent.epsilon),
        }
        metrics.append(episode_metrics)
        logger.info(
            "Episode %d/%d | steps=%d reward=%.2f mean_qos=%.3f epsilon=%.3f",
            episode,
            args.episodes,
            step_count,
            episode_metrics["total_reward"],
            episode_metrics["mean_qos"],
            episode_metrics["epsilon"],
        )

        if args.checkpoint_every and episode % args.checkpoint_every == 0:
            checkpoint_path = experiment_dir / "models" / f"sarsa_ep{episode:04d}.pkl"
            agent.save(checkpoint_path)
            logger.info("Saved checkpoint to %s", checkpoint_path)

    # Save final artifacts
    final_model_path = experiment_dir / "models" / "sarsa_final.pkl"
    agent.save(final_model_path)
    logger.info("Saved final model to %s", final_model_path)

    write_json({"episodes": metrics}, experiment_dir / "training_metrics.json")
    plot_training_curves(metrics, experiment_dir)

    env.close()
    logger.info("Training complete. Metrics stored at %s", experiment_dir)


if __name__ == "__main__":
    main()
