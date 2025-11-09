#!/usr/bin/env python3
"""Evaluate a trained DQN agent with detailed logging and plots."""

from __future__ import annotations

import argparse
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Ensure project root on path when script run directly
current_dir = os.path.dirname(os.path.abspath(__file__))
autoscaling_env_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(autoscaling_env_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from autoscaling_env.openfaas_autoscaling_env import OpenFaaSAutoscalingEnv
from autoscaling_env.rl.dqn_agent import DQNAgent
from autoscaling_env.rl.utils import write_json
from utilities.utilities import get_utc_now


class Tee:
    """Duplicate writes to stdout and file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def prepare_run(output: Path) -> Tuple[Path, Path, Path, Path]:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if output.suffix:
        base_dir = output.parent
        summary_name = output.name
    else:
        base_dir = output
        summary_name = "eval_results.json"

    run_dir = base_dir / f"dqn_eval_{timestamp}"
    logs_dir = run_dir / "logs"
    plots_dir = run_dir / "plots"
    run_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)

    summary_path = run_dir / summary_name
    log_path = logs_dir / "dqn_eval.log"
    return run_dir, plots_dir, summary_path, log_path


def plot_episode(episode_idx: int, data: Dict[str, List[float]], plots_dir: Path) -> None:
    if not data["rewards"]:
        return

    steps = np.arange(1, len(data["rewards"]) + 1)
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle(f"DQN Evaluation Episode {episode_idx}", fontsize=16, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(steps, data["rewards"], "b-", linewidth=2, marker="o")
    ax.axhline(0, color="r", linestyle="--", alpha=0.5)
    ax.set_title("Reward per Step")
    ax.grid(True, alpha=0.3)
    ax.text(
        0.02,
        0.98,
        f"Total: {np.sum(data['rewards']):.1f}",
        transform=ax.transAxes,
        va="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    ax = axes[0, 1]
    qos_pct = np.array(data["qos_rates"]) * 100
    ax.plot(steps, qos_pct, "g-", linewidth=2, marker="o")
    ax.axhline(90, color="orange", linestyle="--", alpha=0.5)
    ax.axhline(80, color="r", linestyle="--", alpha=0.5)
    ax.set_ylim([0, 110])
    ax.set_title("QoS Success Rate")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(steps, data["worker_counts"], "purple", linewidth=2, marker="s")
    ax.set_title("Worker Count")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(steps, data["queue_lengths"], "orange", linewidth=2, marker="o")
    ax.set_title("Worker Queue Length")
    ax.grid(True, alpha=0.3)

    ax = axes[2, 0]
    ax.plot(steps, data["avg_times"], "b-", marker="o", label="Avg")
    ax.plot(steps, data["max_times"], "r-", marker="s", label="Max")
    ax.set_title("Processing Times")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[2, 1]
    ax.plot(steps, data["arrival_rates"], "cyan", linewidth=2, marker="o")
    ax.set_title("Arrival Rate")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(plots_dir / f"episode_{episode_idx:02d}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_overview(records: List[Dict[str, object]], plots_dir: Path) -> None:
    if not records:
        return

    episodes = np.arange(1, len(records) + 1)
    rewards = [r["summary"]["total_reward"] for r in records]
    mean_qos = [r["summary"]["mean_qos"] * 100 for r in records]
    workers = [r["summary"]["mean_workers"] for r in records]
    steps = [r["summary"]["steps"] for r in records]
    scaling_actions = [r["summary"]["scaling_actions"] for r in records]

    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle("DQN Evaluation Overview", fontsize=16, fontweight="bold")

    axes[0, 0].plot(episodes, rewards, "tab:blue", marker="o")
    axes[0, 0].set_title("Total Reward per Episode")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(episodes, mean_qos, "tab:green", marker="o")
    axes[0, 1].axhline(90, color="orange", linestyle="--", alpha=0.5)
    axes[0, 1].set_ylim([0, 110])
    axes[0, 1].set_title("Mean QoS per Episode")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(episodes, workers, "tab:purple", marker="o")
    axes[1, 0].set_title("Mean Workers per Episode")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(episodes, steps, "tab:orange", marker="o")
    axes[1, 1].set_title("Steps per Episode")
    axes[1, 1].grid(True, alpha=0.3)

    axes[2, 0].plot(episodes, scaling_actions, "tab:red", marker="o")
    axes[2, 0].set_title("Scaling Actions per Episode")
    axes[2, 0].grid(True, alpha=0.3)

    fig.delaxes(axes[2, 1])
    fig.tight_layout()
    fig.savefig(plots_dir / "overview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _print_step_header(_log) -> None:
    header = (
        f"{'Step':<6} "
        f"{'Time[s]':>10} "
        f"{'Action':>10} "
        f"{'Scale_T[s]':>11} "
        f"{'Step_D[s]':>11} "
        f"{'Input_Q':>10} "
        f"{'Worker_Q':>10} "
        f"{'Result_Q':>10} "
        f"{'Output_Q':>10} "
        f"{'Workers':>9} "
        f"{'QoS[%]':>9} "
        f"{'AVG_T[s]':>11} "
        f"{'MAX_T[s]':>11} "
        f"{'ARR':>9} "
        f"{'Reward':>11}"
    )
    _log(header)


def _print_step_row(
    _log,
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
    qos_pct = qos * 100.0
    row = (
        f"{step:<6} "
        f"{time_offset:>10.2f} "
        f"{action_str:>10} "
        f"{scaling_time:>11.2f} "
        f"{step_duration:>11.2f} "
        f"{observation[0]:>10.0f} "
        f"{observation[1]:>10.0f} "
        f"{observation[2]:>10.0f} "
        f"{observation[3]:>10.0f} "
        f"{workers:>9.0f} "
        f"{qos_pct:>9.2f} "
        f"{avg_time:>11.2f} "
        f"{max_time:>11.2f} "
        f"{arrival_rate:>9.2f} "
        f"{reward:>11.2f}"
    )
    _log(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Path to DQN model (.pt)")
    parser.add_argument("--episodes", type=int, default=5, help="Evaluation episodes")
    parser.add_argument("--max-steps", type=int, default=30, help="Max steps per episode")
    parser.add_argument("--step-duration", type=int, default=8, help="Seconds per action step")
    parser.add_argument("--observation-window", type=int, default=8, help="Observation window size")
    parser.add_argument("--initial-workers", type=int, default=12, help="Initial workers")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/dqn_eval_results.json"),
        help="Output file or directory",
    )
    parser.add_argument("--initialize-workflow", action="store_true", help="Deploy OpenFaaS workflow first")
    parser.add_argument("--seed", type=int, default=42, help="Global RNG seed for reproducible evaluations")
    parser.add_argument("--task-seed", type=int, default=42, help="Seed passed to task generator (defaults to --seed)")
    return parser.parse_args()


def evaluate_episode(
    env: OpenFaaSAutoscalingEnv,
    agent: DQNAgent,
    max_steps: int,
    episode_idx: int,
    total_episodes: int,
    seed: int | None = None,
    logger=None,
) -> Dict[str, object]:
    def _log(msg: str) -> None:
        if logger is not None:
            logger.info(msg)
        else:
            print(msg)

    _log("\n")
    _log("=" * 70)
    _log(f"EVALUATING DQN (Episode {episode_idx}/{total_episodes})")
    _log("=" * 70)
    _log("\n")
    _log("[1/3] Resetting environment...")
    observation = env.reset(seed=seed)
    _log(f"✓ Initial state: {np.array2string(observation, precision=2)}")

    agent.set_eval_mode()

    rewards: List[float] = []
    qos_rates: List[float] = []
    worker_counts: List[float] = []
    actions: List[int] = []
    queue_lengths: List[float] = []
    avg_times: List[float] = []
    max_times: List[float] = []
    arrival_rates: List[float] = []

    _log(f"[2/3] Running up to {max_steps} steps...")
    header_printed = False

    for step in range(1, max_steps + 1):
        action = agent.greedy_action(observation)
        next_obs, reward, done, info = env.step(action)

        rewards.append(float(reward))
        qos = float(info.get("qos_rate", next_obs[8]))
        qos_rates.append(qos)
        workers = float(info.get("workers", next_obs[4]))
        worker_counts.append(workers)
        actions.append(action)
        queue_lengths.append(float(next_obs[1]))
        avg_times.append(float(next_obs[5]))
        max_times.append(float(next_obs[6]))
        arrival_rates.append(float(next_obs[7]))

        scaling_time = float(info.get("scaling_time", 0.0))
        step_duration = float(info.get("step_duration", 0.0))
        program_start = info.get("program_start_time")
        task_start = info.get("task_generation_start_time")
        if program_start and task_start is not None:
            time_offset = (get_utc_now() - program_start).total_seconds() - task_start
        else:
            time_offset = step * env.step_duration

        if not header_printed:
            _print_step_header(_log)
            header_printed = True

        action_map = {0: "-1", 1: "0", 2: "+1"}
        _print_step_row(
            _log,
            step,
            time_offset,
            action_map.get(action, str(action)),
            scaling_time,
            step_duration,
            next_obs,
            workers,
            qos,
            avg_times[-1],
            max_times[-1],
            arrival_rates[-1],
            reward,
        )

        observation = next_obs

        if done:
            _log("\n")
            _log(f"[INFO] Episode finished at step {step}")
            break

    steps_taken = len(rewards)
    total_reward = float(np.sum(rewards)) if rewards else 0.0
    mean_reward = float(np.mean(rewards)) if rewards else 0.0
    std_reward = float(np.std(rewards)) if len(rewards) > 1 else 0.0
    mean_qos = float(np.mean(qos_rates)) if qos_rates else 0.0
    mean_workers = float(np.mean(worker_counts)) if worker_counts else 0.0
    scaling_actions = int(sum(1 for a in actions if a != 1))
    noop_actions = int(sum(1 for a in actions if a == 1))

    task_history = getattr(env, "task_history", [])
    if task_history:
        qos_successes = sum(1 for _, _, success in task_history if success)
        final_qos_total = qos_successes / len(task_history)
    else:
        final_qos_total = 1.0

    max_workers = max(worker_counts) if worker_counts else float(getattr(env, "initial_workers", 0))

    _log("\n")
    _log("[3/3] Summary Statistics")
    _log("-" * 70)
    _log(f"Total Steps:        {steps_taken}")
    _log(f"Total Reward:       {total_reward:.2f}")
    _log(f"Mean Reward:        {mean_reward:.2f} ± {std_reward:.2f}")
    _log(f"Mean QoS Rate:      {mean_qos:.2%}")
    _log(f"Final QoS:  {final_qos_total:.2%}")
    _log(f"Mean Workers:       {mean_workers:.1f}")
    _log(f"Max Workers:        {max_workers:.1f}")
    _log(f"Scaling Actions:    {scaling_actions}")
    _log(f"No-op Actions:      {noop_actions}")
    _log("=" * 70)
    _log("\n")

    return {
        "rewards": rewards,
        "qos_rates": qos_rates,
        "worker_counts": worker_counts,
        "actions": actions,
        "queue_lengths": queue_lengths,
        "avg_times": avg_times,
        "max_times": max_times,
        "arrival_rates": arrival_rates,
        "summary": {
            "steps": steps_taken,
            "total_reward": total_reward,
            "mean_reward": mean_reward,
            "mean_qos": mean_qos,
            "final_qos_total": final_qos_total,
            "mean_workers": mean_workers,
            "max_workers": max_workers,
            "scaling_actions": scaling_actions,
            "noop_actions": noop_actions,
        },
    }


def build_summary(records: List[Dict[str, object]]) -> Dict[str, object]:
    episodes = []
    totals = []
    qos_values = []
    worker_values = []
    step_values = []

    for idx, record in enumerate(records, start=1):
        summary = record["summary"]
        episodes.append(
            {
                "episode": idx,
                "steps": summary["steps"],
                "total_reward": summary["total_reward"],
                "mean_reward": summary["mean_reward"],
                "mean_qos": summary["mean_qos"],
                "mean_workers": summary["mean_workers"],
                "scaling_actions": summary["scaling_actions"],
                "noop_actions": summary["noop_actions"],
            }
        )
        totals.append(summary["total_reward"])
        qos_values.append(summary["mean_qos"])
        worker_values.append(summary["mean_workers"])
        step_values.append(summary["steps"])

    return {
        "episodes": episodes,
        "mean_reward": float(np.mean(totals)) if totals else 0.0,
        "mean_qos": float(np.mean(qos_values)) if qos_values else 0.0,
        "mean_workers": float(np.mean(worker_values)) if worker_values else 0.0,
        "mean_steps": float(np.mean(step_values)) if step_values else 0.0,
    }


def main() -> None:
    args = parse_args()

    run_dir, plots_dir, summary_path, log_path = prepare_run(args.output)
    with log_path.open("w", encoding="utf-8") as log_file:
        tee = Tee(sys.stdout, log_file)
        original_stdout = sys.stdout
        sys.stdout = tee
        try:
            print(f"Evaluation output directory: {run_dir}")
            arg_string = " ".join(f"{key}={str(value)}" for key, value in sorted(vars(args).items()))
            print(f"[ARGS] {arg_string}")
            env = OpenFaaSAutoscalingEnv(
                max_steps=args.max_steps,
                step_duration=args.step_duration,
                observation_window=args.observation_window,
                initial_workers=args.initial_workers,
                initialize_workflow=args.initialize_workflow,
                task_seed=args.task_seed or args.seed,
            )
            agent = DQNAgent.load(args.model)
            agent.observation_low = env.observation_space.low
            agent.observation_high = env.observation_space.high
            agent._setup_normalization()

            np.random.seed(args.seed)
            random.seed(args.seed)

            episode_records: List[Dict[str, object]] = []
            for episode in range(1, args.episodes + 1):
                episode_seed = args.seed + episode - 1
                record = evaluate_episode(env, agent, args.max_steps, episode, args.episodes, episode_seed)
                episode_records.append(record)
                plot_episode(episode, record, plots_dir)

            plot_overview(episode_records, plots_dir)
            summary = build_summary(episode_records)
            summary["run_directory"] = str(run_dir)
            write_json(summary, summary_path)
            print(f"\nEvaluation summary written to {summary_path}")
        finally:
            sys.stdout = original_stdout
            env.close()


if __name__ == "__main__":
    main()
