#!/usr/bin/env python3
"""Train a tabular SARSA agent against the OpenFaaS autoscaling environment."""

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
from autoscaling_env.rl.test_sarsa import evaluate_episode
from autoscaling_env.rl.utils import (
    build_discretization_config,
    configure_logging,
    prepare_output_directory,
    write_json,
)
from utilities.utilities import get_utc_now


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=100, help="Number of training episodes")
    parser.add_argument("--max-steps", type=int, default=30, help="Maximum steps per episode")
    parser.add_argument("--step-duration", type=int, default=8, help="Seconds per action step in the environment")
    parser.add_argument("--observation-window", type=int, default=8, help="Observation window size passed to the environment")
    parser.add_argument("--initial-workers", type=int, default=12, help="Initial number of worker replicas")
    parser.add_argument("--output-dir", type=Path, default=Path("runs"), help="Base directory for experiment outputs")
    parser.add_argument("--epsilon", type=float, default=0.35, help="Initial epsilon for epsilon-greedy policy")
    parser.add_argument("--epsilon-min", type=float, default=0.15, help="Minimum epsilon after decay")
    parser.add_argument("--epsilon-decay", type=float, default=0.995, help="Episode-wise epsilon decay factor")
    parser.add_argument("--alpha", type=float, default=0.025, help="Learning rate (alpha)")
    parser.add_argument("--gamma", type=float, default=0.98, help="Discount factor (gamma)")
    parser.add_argument(
        "--bins",
        type=int,
        nargs=9,
        default=[4, 12, 3, 15, 8, 8, 8, 6, 10],
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
        "--eval-episodes",
        type=int,
        default=1,
        help="Number of evaluation episodes to run at each checkpoint (0 to disable)",
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


def plot_training_curves(
    training_metrics: List[Dict[str, float]],
    eval_metrics: List[Dict[str, float]],
    output_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    episodes = np.arange(1, len(training_metrics) + 1)
    total_rewards = np.array([m["total_reward"] for m in training_metrics])
    qos_rates = np.array([m["mean_qos"] for m in training_metrics])
    epsilons = np.array([m["epsilon"] for m in training_metrics])

    def _smooth(values: np.ndarray, window: int) -> np.ndarray:
        if values.size == 0 or window <= 1:
            return values
        window = min(window, values.size)
        kernel = np.ones(window, dtype=np.float64) / window
        return np.convolve(values, kernel, mode="same")

    smooth_window = 0
    if len(training_metrics) >= 3:
        smooth_window = min(15, max(3, len(training_metrics) // 10 or 3))

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

    if eval_metrics:
        eval_episodes = np.array([m["episode"] for m in eval_metrics])
        eval_rewards = np.array([m["mean_total_reward"] for m in eval_metrics])
        ax1.plot(
            eval_episodes,
            eval_rewards,
            marker="o",
            color="tab:orange",
            linestyle="-",
            linewidth=1.5,
            label="Evaluation Reward (mean)",
        )

    ax2 = ax1.twinx()
    ax2.set_ylabel("QoS Rate", color="tab:green")
    ax2.plot(episodes, qos_rates, label="Mean QoS (raw)", color="tab:green", linestyle="--", alpha=0.35, linewidth=1)
    ax2.plot(episodes, qos_smooth, label="Mean QoS (smoothed)", color="tab:green", linestyle="--", linewidth=2)
    ax2.tick_params(axis="y", labelcolor="tab:green")

    if eval_metrics:
        eval_episodes = np.array([m["episode"] for m in eval_metrics])
        eval_qos = np.array([m["mean_final_qos"] for m in eval_metrics]) * 100.0
        ax2.plot(
            eval_episodes,
            eval_qos,
            marker="s",
            color="tab:olive",
            linestyle="--",
            linewidth=1.5,
            label="Evaluation Final QoS (mean)",
        )

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


def evaluate_checkpoint(
    train_env: OpenFaaSAutoscalingEnv,
    agent: SARSAAgent,
    args: argparse.Namespace,
    checkpoint_episode: int,
    base_seed: int,
    logger,
) -> Dict[str, float] | None:
    """Evaluate the current agent policy across several episodes."""

    if args.eval_episodes <= 0:
        return None

    eval_env = OpenFaaSAutoscalingEnv(
        max_steps=args.max_steps,
        step_duration=args.step_duration,
        observation_window=args.observation_window,
        initial_workers=args.initial_workers,
        initialize_workflow=args.initialize_workflow,
        task_seed=train_env.default_task_seed,
    )

    original_discretization = agent.discretization
    temp_discretization = build_discretization_config(
        observation_low=eval_env.observation_space.low,
        observation_high=eval_env.observation_space.high,
        bins_per_dimension=agent.discretization.bins_per_dimension,
    )
    agent.discretization = temp_discretization

    records: List[Dict[str, List[float]]] = []

    try:
        for eval_idx in range(1, args.eval_episodes + 1):
            episode_seed = base_seed + checkpoint_episode * 1000 + eval_idx
            np.random.seed(episode_seed)
            random.seed(episode_seed)

            record = evaluate_episode(
                eval_env,
                agent,
                max_steps=args.max_steps,
                episode_idx=eval_idx,
                total_episodes=args.eval_episodes,
                seed=episode_seed,
            )
            records.append(record)

    except Exception as exc:  # noqa: BLE001
        logger.exception("Evaluation failed at episode %d: %s", checkpoint_episode, exc)
        return None
    finally:
        agent.discretization = original_discretization
        eval_env.close()

    if not records:
        return None

    def _collect(metric_key: str) -> np.ndarray:
        values = []
        for record in records:
            summary = record.get("summary", {})
            value = summary.get(metric_key)
            if value is not None:
                values.append(float(value))
        return np.array(values, dtype=float)

    total_rewards = _collect("total_reward")
    final_qos = _collect("final_qos_total")
    scaling_actions = _collect("scaling_actions")
    max_workers = _collect("max_workers")

    def _mean_std(values: np.ndarray) -> tuple[float, float]:
        if values.size == 0:
            return 0.0, 0.0
        mean = float(np.mean(values))
        std = float(np.std(values)) if values.size > 1 else 0.0
        return mean, std

    mean_reward, std_reward = _mean_std(total_rewards)
    mean_qos, std_qos = _mean_std(final_qos)
    mean_scaling, std_scaling = _mean_std(scaling_actions)
    mean_workers, std_workers = _mean_std(max_workers)

    logger.info(
        "[EVAL] Episode %d | reward=%.2f±%.2f qos=%.2f%%±%.2f%% scaling=%.1f±%.1f workers=%.1f±%.1f",
        checkpoint_episode,
        mean_reward,
        std_reward,
        mean_qos * 100.0,
        std_qos * 100.0,
        mean_scaling,
        std_scaling,
        mean_workers,
        std_workers,
    )

    return {
        "episode": checkpoint_episode,
        "mean_total_reward": mean_reward,
        "std_total_reward": std_reward,
        "mean_final_qos": mean_qos,
        "std_final_qos": std_qos,
        "mean_scaling_actions": mean_scaling,
        "std_scaling_actions": std_scaling,
        "mean_max_workers": mean_workers,
        "std_max_workers": std_workers,
        "eval_episodes": args.eval_episodes,
    }


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
    eval_history: List[Dict[str, float]] = []

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
        last_step_info: Dict[str, float] | None = None

        for step in range(args.max_steps):
            executed_action = action
            next_observation, reward, done, info = env.step(executed_action)
            next_state = agent.discretize(next_observation)
            next_action = agent.select_action(next_state)

            agent.update(state, executed_action, reward, next_state, next_action)

            state = next_state
            action = next_action
            total_reward += reward
            qos_rates.append(info.get("qos_rate"))
            step_count = step + 1
            last_step_info = info

            scaling_time = float(info.get("scaling_time"))
            step_duration = float(info.get("step_duration"))
            program_start = info.get("program_start_time")
            task_start = info.get("task_generation_start_time")
            if program_start and task_start is not None:
                time_offset = (get_utc_now() - program_start).total_seconds() - task_start
            else:
                time_offset = step_count * env.step_duration

            if not header_logged:
                _log_step_header(logger)
                header_logged = True

            applied_delta = info.get("applied_delta")
            if applied_delta is not None:
                delta_int = int(applied_delta)
                action_str = f"{delta_int:+d}" if delta_int != 0 else "0"
            else:
                action_map = {0: "-1", 1: "0", 2: "+1"}
                action_str = action_map.get(executed_action, str(executed_action))
            _log_step_row(
                logger,
                step_count,
                time_offset,
                action_str,
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

        final_qos = env.get_episode_final_qos()
        if final_qos is None and last_step_info is not None:
            final_qos = last_step_info.get("episode_final_qos")
        if final_qos is None:
            final_qos = float(np.mean(qos_rates) if qos_rates else 0.0)

        processed_tasks = 0
        if last_step_info and last_step_info.get("processed_tasks") is not None:
            processed_tasks = int(last_step_info.get("processed_tasks"))
        else:
            processed_tasks = len(getattr(env, "task_history", []))

        episode_metrics = {
            "episode": episode,
            "steps": step_count,
            "total_reward": float(total_reward),
            "mean_qos": float(np.mean(qos_rates) if qos_rates else 0.0),
            "final_qos": float(final_qos),
            "processed_tasks": processed_tasks,
            "epsilon": float(agent.epsilon),
        }
        metrics.append(episode_metrics)
        logger.info(
            "Episode %d/%d | steps=%d reward=%.2f mean_qos=%.3f final_qos=%.3f tasks=%d epsilon=%.3f",
            episode,
            args.episodes,
            step_count,
            episode_metrics["total_reward"],
            episode_metrics["mean_qos"],
            episode_metrics["final_qos"],
            episode_metrics["processed_tasks"],
            episode_metrics["epsilon"],
        )

        if args.checkpoint_every and episode % args.checkpoint_every == 0:
            checkpoint_path = experiment_dir / "models" / f"sarsa_ep{episode:04d}.pkl"
            agent.save(checkpoint_path)
            logger.info("Saved checkpoint to %s", checkpoint_path)

            if args.eval_episodes > 0:
                eval_summary = evaluate_checkpoint(env, agent, args, episode, base_seed, logger)
                if eval_summary:
                    eval_history.append(eval_summary)

    # Save final artifacts
    final_model_path = experiment_dir / "models" / "sarsa_final.pkl"
    agent.save(final_model_path)
    logger.info("Saved final model to %s", final_model_path)

    if args.eval_episodes > 0:
        if not eval_history or eval_history[-1]["episode"] != args.episodes:
            eval_summary = evaluate_checkpoint(env, agent, args, args.episodes, base_seed, logger)
            if eval_summary:
                eval_history.append(eval_summary)

    write_json({"episodes": metrics}, experiment_dir / "training_metrics.json")
    if eval_history:
        write_json({"checkpoints": eval_history}, experiment_dir / "evaluation_metrics.json")
    plot_training_curves(metrics, eval_history, experiment_dir)

    env.close()
    logger.info("Training complete. Metrics stored at %s", experiment_dir)


if __name__ == "__main__":
    main()
