#!/usr/bin/env python3
"""Compare a trained SARSA agent with reactive baselines on a single episode."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import random
import os
import sys
import matplotlib.patches as mpatches

# Ensure project root on path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from autoscaling_env.openfaas_autoscaling_env import OpenFaaSAutoscalingEnv
from autoscaling_env.rl.sarsa_agent import SARSAAgent
from autoscaling_env.rl.utils import build_discretization_config
from autoscaling_env.rl.test_sarsa import evaluate_episode  # reuse existing helpers
from autoscaling_env.baselines.reactive_policies import ReactiveAverage, ReactiveMaximum
from utilities.utilities import get_utc_now


class Tee:
    """Duplicate writes to stdout/stderr and a log file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def run_sarsa_episode(
    env: OpenFaaSAutoscalingEnv,
    agent: SARSAAgent,
    max_steps: int,
    episode_idx: int,
    total_episodes: int,
    seed: int | None,
) -> Dict[str, List[float]]:
    return evaluate_episode(
        env,
        agent,
        max_steps=max_steps,
        episode_idx=episode_idx,
        total_episodes=total_episodes,
        seed=seed,
    )


def _print_step_header() -> None:
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
    print(header)


def _print_step_row(
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
    print(row)


def run_reactive_episode(
    env: OpenFaaSAutoscalingEnv,
    policy,
    max_steps: int,
    label: str | None = None,
    episode_idx: int = 1,
    total_episodes: int = 1,
    seed: int | None = None,
) -> Dict[str, List[float]]:
    policy_label = label or getattr(policy, "name", policy.__class__.__name__)

    print("\n" + "=" * 70)
    print(f"EVALUATING {policy_label.upper()} (Episode {episode_idx}/{total_episodes})")
    if label and label.lower() == "sarsa" and hasattr(policy, "model_path"):
        print(f"[MODEL] Loaded SARSA agent from: {policy.model_path}")
    print("=" * 70)

    print("\n[1/3] Resetting environment...")
    observation = env.reset(seed=seed)
    print(f"✓ Initial state: {np.array2string(observation, precision=2)}")

    print(f"\n[2/3] Running up to {max_steps} steps...")
    rewards: List[float] = []
    qos_rates: List[float] = []
    worker_counts: List[float] = []
    queue_lengths: List[float] = []
    avg_times: List[float] = []
    max_times: List[float] = []
    arrival_rates: List[float] = []
    actions: List[int] = []

    header_printed = False

    for step in range(1, max_steps + 1):
        action = policy.select_action(observation, training=False)
        next_obs, reward, done, info = env.step(action)

        rewards.append(reward)
        qos_rates.append(info.get("qos_rate", next_obs[8]))
        worker_counts.append(info.get("workers", next_obs[4]))
        queue_lengths.append(next_obs[1])
        avg_times.append(next_obs[5])
        max_times.append(next_obs[6])
        arrival_rates.append(next_obs[7])
        actions.append(action)

        scaling_time = float(info.get("scaling_time", 0.0))
        step_duration = float(info.get("step_duration", 0.0))
        program_start = info.get("program_start_time")
        task_start = info.get("task_generation_start_time")
        if program_start and task_start is not None:
            time_offset = (get_utc_now() - program_start).total_seconds() - task_start
        else:
            time_offset = step * env.step_duration

        if not header_printed:
            _print_step_header()
            header_printed = True

        action_map = {0: "-1", 1: "0", 2: "+1"}
        _print_step_row(
            step,
            time_offset,
            action_map.get(action, str(action)),
            scaling_time,
            step_duration,
            next_obs,
            worker_counts[-1],
            qos_rates[-1],
            avg_times[-1],
            max_times[-1],
            arrival_rates[-1],
            reward,
        )

        observation = next_obs
        if done:
            break

    print("\n[3/3] Summary Statistics")
    print("-" * 70)
    total_steps = len(rewards)
    total_reward = float(np.sum(rewards)) if rewards else 0.0
    mean_reward = float(np.mean(rewards)) if rewards else 0.0
    std_reward = float(np.std(rewards)) if len(rewards) > 1 else 0.0
    mean_qos = float(np.mean(qos_rates)) if qos_rates else 0.0
    mean_workers = float(np.mean(worker_counts)) if worker_counts else 0.0
    scaling_actions = int(sum(1 for a in actions if a != 1))
    noop_actions = int(sum(1 for a in actions if a == 1))

    print(f"Total Steps:        {total_steps}")
    print(f"Total Reward:       {total_reward:.2f}")
    print(f"Mean Reward:        {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Mean QoS Rate:      {mean_qos:.2%}")
    print(f"Mean Workers:       {mean_workers:.1f}")
    print(f"Scaling Actions:    {scaling_actions}")
    print(f"No-op Actions:      {noop_actions}")
    print("=" * 70)

    return {
        "rewards": rewards,
        "qos_rates": qos_rates,
        "worker_counts": worker_counts,
        "queue_lengths": queue_lengths,
        "avg_times": avg_times,
        "max_times": max_times,
        "arrival_rates": arrival_rates,
        "actions": actions,
        "summary": {
            "steps": total_steps,
            "total_reward": total_reward,
            "mean_reward": mean_reward,
            "mean_qos": mean_qos,
            "mean_workers": mean_workers,
            "scaling_actions": scaling_actions,
            "noop_actions": noop_actions,
        },
    }


def collect_step_series(
    records: List[Dict[str, List[float]]],
    metric_key: str,
    transform,
) -> Dict[int, List[float]]:
    step_data: Dict[int, List[float]] = {}
    for record in records:
        series = record.get(metric_key, [])
        for idx, value in enumerate(series):
            step_data.setdefault(idx, []).append(transform(value))
    return step_data


def compute_agent_summary(records: List[Dict[str, List[float]]]) -> Dict[str, float]:
    total_rewards = []
    mean_rewards = []
    mean_qos_rates = []
    final_qos_rates = []

    for record in records:
        rewards = record.get("rewards", [])
        qos_rates = record.get("qos_rates", [])

        if rewards:
            total_rewards.append(np.sum(rewards))
            mean_rewards.append(np.mean(rewards))
        if qos_rates:
            mean_qos_rates.append(np.mean(qos_rates))
            final_qos_rates.append(qos_rates[-1])

    return {
        "mean_total_reward": float(np.mean(total_rewards)) if total_rewards else 0.0,
        "mean_reward": float(np.mean(mean_rewards)) if mean_rewards else 0.0,
        "mean_qos": float(np.mean(mean_qos_rates)) if mean_qos_rates else 0.0,
        "mean_final_qos": float(np.mean(final_qos_rates)) if final_qos_rates else 0.0,
    }


def plot_step_boxplots(
    records_by_agent: Dict[str, List[Dict[str, List[float]]]],
    output_path: Path,
) -> None:
    metrics = {
        "rewards": ("Reward per Step", "Reward", lambda x: x),
        "qos_rates": ("QoS Success Rate", "QoS (%)", lambda x: x * 100.0),
        "worker_counts": ("Worker Count", "Workers", lambda x: x),
        "queue_lengths": ("Worker Queue Length", "Tasks in queue", lambda x: x),
    }
    agents = list(records_by_agent.keys())
    colors = plt.get_cmap("tab10")
    color_cycle = {agent: colors(idx % 10) for idx, agent in enumerate(agents)}

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    fig.suptitle("Policy Comparison Across Episodes", fontsize=16, fontweight="bold")

    metric_items = list(metrics.items())

    for ax, (metric_key, (title, y_label, transform)) in zip(axes.flat, metric_items):
        step_maps = {
            agent: collect_step_series(records_by_agent[agent], metric_key, transform)
            for agent in agents
        }
        max_steps = 0
        for step_map in step_maps.values():
            if step_map:
                max_steps = max(max_steps, max(step_map.keys()) + 1)

        if max_steps == 0:
            ax.set_title(title)
            ax.set_xlabel("Step")
            ax.set_ylabel(y_label)
            continue

        handles = []
        base_positions = np.arange(max_steps)
        width = max(0.15, 0.6 / max(1, len(agents)))

        for idx, agent in enumerate(agents):
            step_map = step_maps[agent]
            if not step_map:
                continue

            positions = base_positions + (idx - (len(agents) - 1) / 2) * width
            data = [np.array(step_map.get(step_idx, [])) for step_idx in range(max_steps)]

            filtered = [(pos, arr) for pos, arr in zip(positions, data) if arr.size]
            if not filtered:
                continue

            positions, data = zip(*filtered)
            bp = ax.boxplot(
                data,
                positions=positions,
                widths=width * 0.8,
                patch_artist=True,
                manage_ticks=False,
            )

            for box in bp["boxes"]:
                box.set(facecolor=color_cycle[agent], alpha=0.45)
            for median in bp["medians"]:
                median.set(color=color_cycle[agent], linewidth=1.2)
            for whisker in bp["whiskers"]:
                whisker.set(color=color_cycle[agent], linewidth=0.8)
            for cap in bp["caps"]:
                cap.set(color=color_cycle[agent], linewidth=0.8)

            means = [np.mean(arr) for arr in data]
            stds = [np.std(arr) for arr in data]
            ax.errorbar(
                positions,
                means,
                yerr=stds,
                fmt="o",
                color=color_cycle[agent],
                capsize=2,
                markersize=3,
                linewidth=1,
            )

            handles.append(mpatches.Patch(color=color_cycle[agent], label=agent))

        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.set_ylabel(y_label)
        ax.set_xticks(base_positions)
        ax.set_xticklabels([str(step + 1) for step in base_positions], rotation=45)
        ax.grid(True, alpha=0.3)
        if handles:
            unique_handles = {handle.get_label(): handle for handle in handles}
            ax.legend(unique_handles.values(), unique_handles.keys(), loc="upper right", frameon=False)

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Path to trained SARSA model (.pkl)")
    parser.add_argument("--max-steps", type=int, default=31, help="Max steps per episode")
    parser.add_argument("--step-duration", type=int, default=8, help="Seconds per environment step")
    parser.add_argument("--initial-workers", type=int, default=12, help="Initial workers (match training/eval setup)")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/comparison"), help="Directory for outputs")
    parser.add_argument("--phase-shuffle", action="store_true", help="Enable phase shuffling during comparison")
    parser.add_argument("--phase-shuffle-seed", type=int, default=123, help="RNG seed when shuffling phases")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes per policy")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed for reproducibility")
    parser.add_argument("--task-seed", type=int, default=42, help="Seed passed to task generator (defaults to --seed)")
    args = parser.parse_args()

    timestamp = get_utc_now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir / f"compare_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "comparison_boxplots.png"
    log_path = output_dir / "comparison.log"

    with log_path.open("w", encoding="utf-8") as log_file:
        tee = Tee(sys.stdout, log_file)
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = tee
        sys.stderr = tee

        env = None
        try:
            print(f"[INFO] Comparison output directory: {output_dir}")
            print(f"[INFO] Log file: {log_path}")
            arg_string = " ".join(f"{key}={str(value)}" for key, value in sorted(vars(args).items()))
            print(f"[ARGS] {arg_string}")

            base_seed = args.seed
            task_seed = args.task_seed if args.task_seed is not None else base_seed

            np.random.seed(base_seed)
            random.seed(base_seed)

            def create_env() -> OpenFaaSAutoscalingEnv:
                return OpenFaaSAutoscalingEnv(
                    max_steps=args.max_steps,
                    step_duration=args.step_duration,
                    observation_window=args.step_duration,
                    initial_workers=args.initial_workers,
                    initialize_workflow=False,
                    phase_shuffle=args.phase_shuffle,
                    phase_shuffle_seed=args.phase_shuffle_seed,
                    task_seed=task_seed,
                )

            aggregated: Dict[str, Dict[str, object]] = {}

            # Evaluate SARSA agent
            env = create_env()
            try:
                agent = SARSAAgent.load(args.model)
                setattr(agent, "model_path", str(args.model))
                discretization = build_discretization_config(
                    observation_low=env.observation_space.low,
                    observation_high=env.observation_space.high,
                    bins_per_dimension=agent.discretization.bins_per_dimension,
                )
                agent.discretization = discretization

                sarsa_records: List[Dict[str, List[float]]] = []
                for episode in range(1, args.episodes + 1):
                    episode_seed = base_seed + episode - 1 if base_seed is not None else None
                    np.random.seed(episode_seed)
                    random.seed(episode_seed)
                    record = run_sarsa_episode(env, agent, args.max_steps, episode, args.episodes, episode_seed)
                    sarsa_records.append(record)

                aggregated["SARSA"] = {
                    "records": sarsa_records,
                    "summary": compute_agent_summary(sarsa_records),
                }
            finally:
                env.close()

            # Evaluate reactive baselines
            baseline_configs = [
                ("ReactiveAverage", ReactiveAverage, "Reactive Average"),
                ("ReactiveMaximum", ReactiveMaximum, "Reactive Maximum"),
            ]

            for idx, (key, policy_cls, label) in enumerate(baseline_configs):
                env = create_env()
                try:
                    records: List[Dict[str, List[float]]] = []
                    for episode in range(1, args.episodes + 1):
                        episode_offset = idx * args.episodes
                        episode_seed = base_seed + episode_offset + episode - 1 if base_seed is not None else None
                        np.random.seed(episode_seed)
                        random.seed(episode_seed)
                        policy = policy_cls()
                        record = run_reactive_episode(
                            env,
                            policy,
                            args.max_steps,
                            label=label,
                            episode_idx=episode,
                            total_episodes=args.episodes,
                            seed=episode_seed,
                        )
                        records.append(record)

                    aggregated[key] = {
                        "records": records,
                        "summary": compute_agent_summary(records),
                    }
                finally:
                    env.close()

            for agent_name, data in aggregated.items():
                summary = data["summary"]
                print(
                    f"[SUMMARY] {agent_name}: mean_total_reward={summary['mean_total_reward']:.2f}, "
                    f"mean_reward={summary['mean_reward']:.2f}, "
                    f"mean_qos={summary['mean_qos']:.2%}, "
                    f"final_qos={summary['mean_final_qos']:.2%}"
                )

            plot_step_boxplots({name: data["records"] for name, data in aggregated.items()}, plot_path)
            print(f"[PLOT] Saved comparison to {plot_path}")
        finally:
            if env is not None:
                env.close()
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            print(f"[INFO] Comparison log written to {log_path}")


if __name__ == "__main__":
    main()