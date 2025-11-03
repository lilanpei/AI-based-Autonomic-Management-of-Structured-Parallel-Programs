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
import random

# Add project root to path when script executed directly
current_dir = os.path.dirname(os.path.abspath(__file__))
autoscaling_env_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(autoscaling_env_dir)
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
from utilities.utilities import get_utc_now


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=250, help="Number of training episodes")
    parser.add_argument("--max-steps", type=int, default=31, help="Maximum steps per episode")
    parser.add_argument("--step-duration", type=int, default=8, help="Seconds per action step in the environment")
    parser.add_argument("--observation-window", type=int, default=8, help="Observation window size passed to the environment")
    parser.add_argument("--initial-workers", type=int, default=12, help="Initial number of worker replicas")
    parser.add_argument("--output-dir", type=Path, default=Path("runs"), help="Base directory for experiment outputs")
    parser.add_argument("--epsilon", type=float, default=0.35, help="Initial epsilon for epsilon-greedy exploration")
    parser.add_argument("--epsilon-min", type=float, default=0.08, help="Minimum epsilon after decay")
    parser.add_argument("--epsilon-decay", type=float, default=0.995, help="Episode-wise epsilon decay factor")
    parser.add_argument("--alpha", type=float, default=0.01, help="Learning rate (alpha)")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor (gamma)")
    parser.add_argument(
        "--bins",
        type=int,
        nargs=9,
        default=[4, 36, 10, 24, 20, 16, 16, 20, 20],
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
        "--phase-shuffle",
        action="store_true",
        help="Randomize phase order each training episode (for training only)",
    )
    parser.add_argument(
        "--phase-shuffle-seed",
        type=int,
        default=None,
        help="Optional RNG seed used when --phase-shuffle is enabled",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--task-seed",
        type=int,
        default=42,
        help="Seed passed to the task generator (defaults to --seed)",
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

    def _smooth(values: np.ndarray, window: int) -> np.ndarray:
        if values.size == 0 or window <= 1:
            return values
        window = min(window, values.size)
        kernel = np.ones(window, dtype=np.float64) / window
        return np.convolve(values, kernel, mode="same")

    smooth_window = 0
    if len(metrics) >= 3:
        smooth_window = min(15, max(3, len(metrics) // 10 or 3))

    rewards_smooth = _smooth(total_rewards, smooth_window) if smooth_window else total_rewards
    qos_smooth = _smooth(qos_rates, smooth_window) if smooth_window else qos_rates

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.set_title("SARSA Training Progress")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Reward", color="tab:blue")
    ax1.plot(episodes, total_rewards, label="Total Reward (raw)", color="tab:blue", alpha=0.35, linewidth=1)
    ax1.plot(episodes, rewards_smooth, label="Total Reward (smoothed)", color="tab:blue", linewidth=2)
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.set_ylabel("QoS Rate", color="tab:green")
    ax2.plot(episodes, qos_rates, label="Mean QoS (raw)", color="tab:green", linestyle="--", alpha=0.35, linewidth=1)
    ax2.plot(episodes, qos_smooth, label="Mean QoS (smoothed)", color="tab:green", linestyle="--", linewidth=2)
    ax2.tick_params(axis="y", labelcolor="tab:green")

    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("outward", 60))
    ax3.set_ylabel("Epsilon", color="tab:red")
    ax3.plot(episodes, epsilons, label="Epsilon", color="tab:red", linestyle=":")
    ax3.tick_params(axis="y", labelcolor="tab:red")

    handles, labels = [], []
    for axis in (ax1, ax2, ax3):
        h, l = axis.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    if handles:
        ax1.legend(handles, labels, loc="upper left", frameon=False)

    fig.tight_layout()
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    fig.savefig(plots_dir / "training_curves.png", dpi=200)
    plt.close(fig)


def _log_step_header(logger) -> None:
    logger.info(
        "%-6s %10s %10s %11s %11s %10s %10s %10s %10s %9s %9s %11s %11s %9s %11s",
        "Step",
        "Time[s]",
        "Action",
        "Scale_T[s]",
        "Step_D[s]",
        "Input_Q",
        "Worker_Q",
        "Result_Q",
        "Output_Q",
        "Workers",
        "QoS[%]",
        "AVG_T[s]",
        "MAX_T[s]",
        "ARR",
        "Reward",
    )


def _log_step_row(
    logger,
    step: int,
    time_offset: float,
    action_str: str,
    scaling_time: float,
    step_duration: float,
    observation: np.ndarray,
    workers: float,
    qos: float,
    avg_time: float,
    max_time: float,
    arrival_rate: float,
    reward: float,
) -> None:
    logger.info(
        "%-6d %10.2f %10s %11.2f %11.2f %10.0f %10.0f %10.0f %10.0f %9.0f %9.2f %11.2f %11.2f %9.2f %11.2f",
        step,
        time_offset,
        action_str,
        scaling_time,
        step_duration,
        observation[0],
        observation[1],
        observation[2],
        observation[3],
        workers,
        qos * 100.0,
        avg_time,
        max_time,
        arrival_rate,
        reward,
    )


def main() -> None:
    args = parse_args()
    base_seed = args.seed
    task_seed = args.task_seed if args.task_seed is not None else base_seed

    np.random.seed(base_seed)
    random.seed(base_seed)

    experiment_dir = prepare_output_directory(args.output_dir)
    logger = configure_logging(experiment_dir / "logs")
    logger.info("Experiment directory: %s", experiment_dir)
    logger.info("Training configuration: %s", " ".join(f"{k}={v}" for k, v in sorted(vars(args).items())))

    env = OpenFaaSAutoscalingEnv(
        max_steps=args.max_steps,
        step_duration=args.step_duration,
        observation_window=args.observation_window,
        initial_workers=args.initial_workers,
        initialize_workflow=args.initialize_workflow,
        phase_shuffle=args.phase_shuffle,
        phase_shuffle_seed=args.phase_shuffle_seed,
        task_seed=task_seed,
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
        episode_seed = base_seed + episode - 1
        np.random.seed(episode_seed)
        random.seed(episode_seed)
        observation = env.reset(seed=episode_seed)
        state = agent.discretize(observation)
        action = agent.select_action(state)

        total_reward = 0.0
        qos_rates: List[float] = []
        step_count = 0
        header_logged = False

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

            scaling_time = float(info.get("scaling_time", 0.0))
            step_duration = float(info.get("step_duration", 0.0))
            program_start = info.get("program_start_time")
            task_start = info.get("task_generation_start_time")
            if program_start and task_start is not None:
                time_offset = (get_utc_now() - program_start).total_seconds() - task_start
            else:
                time_offset = step_count * env.step_duration

            if not header_logged:
                _log_step_header(logger)
                header_logged = True

            action_map = {0: "-1", 1: "0", 2: "+1"}
            _log_step_row(
                logger,
                step_count,
                time_offset,
                action_map.get(action, str(action)),
                scaling_time,
                step_duration,
                next_observation,
                info.get("workers", next_observation[4]),
                info.get("qos_rate", next_observation[8]),
                next_observation[5],
                next_observation[6],
                next_observation[7],
                reward,
            )

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
