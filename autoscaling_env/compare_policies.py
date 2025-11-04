#!/usr/bin/env python3
"""Compare a trained SARSA agent with reactive baselines on a single episode."""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

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


AGGREGATED_FILENAME = "aggregated_results.json"

METRIC_LABELS = {
    "total_reward": "Total Reward",
    "final_qos_total": "Final QoS (tasks)",
    "scaling_actions": "Scaling Actions",
    "max_workers": "Max Workers",
}

AGENT_ALIAS_MAP = {
    "sarsa": "SARSA",
    "agent": "SARSA",
    "reactiveaverage": "ReactiveAverage",
    "reactiveavg": "ReactiveAverage",
    "reactiveaveragepolicy": "ReactiveAverage",
    "reactivemaximum": "ReactiveMaximum",
    "reactivemax": "ReactiveMaximum",
    "reactivemaximumpolicy": "ReactiveMaximum",
}


def _normalize_agent_token(name: str) -> str:
    return name.strip().lower().replace(" ", "").replace("-", "").replace("_", "")


def normalize_agent_names(requested: Optional[List[str]], available: Iterable[str]) -> List[str]:
    available_list = list(available)
    if not requested:
        return available_list

    available_set = set(available_list)
    resolved: List[str] = []
    for raw in requested:
        token = _normalize_agent_token(raw)
        mapped = AGENT_ALIAS_MAP.get(token)
        if mapped is None:
            # Try exact case-insensitive match against available agents
            for candidate in available_list:
                if token == _normalize_agent_token(candidate):
                    mapped = candidate
                    break
        if mapped is None:
            raise ValueError(
                f"Unknown agent name '{raw}'. Available options: {', '.join(available_list)}"
            )
        if mapped not in resolved:
            resolved.append(mapped)

    missing = [name for name in resolved if name not in available_set]
    if missing:
        raise ValueError(
            f"Requested agents not present in results: {', '.join(missing)}"
        )

    return resolved


def _ensure_serializable(value):
    if isinstance(value, dict):
        return {key: _ensure_serializable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_ensure_serializable(item) for item in value]
    if isinstance(value, tuple):
        return [_ensure_serializable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def save_aggregated_results(path: Path, aggregated: Dict[str, Dict[str, object]]) -> None:
    payload = {"agents": _ensure_serializable(aggregated)}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_aggregated_results(path: Path) -> Dict[str, Dict[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "agents" in data:
        data = data["agents"]
    if not isinstance(data, dict):
        raise ValueError("Aggregated results JSON must be an object containing agent entries")
    return data


def print_agent_summaries(aggregated: Dict[str, Dict[str, object]], agent_order: List[str]) -> None:
    for agent_name in agent_order:
        agent_data = aggregated.get(agent_name)
        if not agent_data:
            continue
        summary = agent_data.get("summary", {})
        print(f"[SUMMARY] {agent_name}")
        for key, label in METRIC_LABELS.items():
            metric = summary.get(key, {"mean": 0.0, "std": 0.0})
            mean_value = float(metric.get("mean", 0.0))
            std_value = float(metric.get("std", 0.0))
            if key == "final_qos_total":
                mean_str = f"{mean_value:.2%}"
                std_str = f"{std_value:.2%}"
            else:
                mean_str = f"{mean_value:.2f}"
                std_str = f"{std_value:.2f}"
            print(f"  - {label}: {mean_str} ± {std_str}")


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

    task_history = getattr(env, "task_history", [])
    if task_history:
        qos_successes = sum(1 for _, _, success in task_history if success)
        final_qos_total = qos_successes / len(task_history)
    else:
        final_qos_total = 1.0

    max_workers = max(worker_counts) if worker_counts else float(getattr(env, "initial_workers", 0))

    print(f"Total Steps:        {total_steps}")
    print(f"Total Reward:       {total_reward:.2f}")
    print(f"Mean Reward:        {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Mean QoS Rate:      {mean_qos:.2%}")
    print(f"Final QoS (tasks):  {final_qos_total:.2%}")
    print(f"Mean Workers:       {mean_workers:.1f}")
    print(f"Max Workers:        {max_workers:.1f}")
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
            "final_qos_total": final_qos_total,
            "mean_workers": mean_workers,
            "max_workers": max_workers,
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


def _compute_mean_std(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0}
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)) if len(values) > 1 else 0.0,
    }


def compute_agent_summary(records: List[Dict[str, List[float]]]) -> Dict[str, Dict[str, float]]:
    metrics = {
        "total_reward": [],
        "final_qos_total": [],
        "scaling_actions": [],
        "max_workers": [],
    }

    for record in records:
        summary = record.get("summary", {})
        for key in metrics:
            value = summary.get(key)
            if value is not None:
                metrics[key].append(float(value))

    return {key: _compute_mean_std(values) for key, values in metrics.items()}


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
            ax.legend(unique_handles.values(), unique_handles.keys(), loc="best", frameon=False)

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        help="Path to trained SARSA model (.pkl). Required unless --plot-only is used.",
    )
    parser.add_argument("--max-steps", type=int, default=30, help="Max steps per episode")
    parser.add_argument("--step-duration", type=int, default=8, help="Seconds per environment step")
    parser.add_argument("--initial-workers", type=int, default=12, help="Initial workers (match training/eval setup)")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/comparison"), help="Directory for outputs")
    parser.add_argument("--phase-shuffle", action="store_true", help="Enable phase shuffling during comparison")
    parser.add_argument("--phase-shuffle-seed", type=int, default=42, help="RNG seed when shuffling phases")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes per policy")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed for reproducibility")
    parser.add_argument("--task-seed", type=int, default=42, help="Seed passed to task generator (defaults to --seed)")
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip new simulations and regenerate plots/summaries from an existing comparison directory",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Existing comparison directory containing comparison.log and aggregated results (used with --plot-only)",
    )
    parser.add_argument(
        "--agents",
        nargs="+",
        default=None,
        help="Subset of agents to include in plots/summaries (e.g. 'agent reactiveaverage')",
    )
    args = parser.parse_args()

    if args.plot_only:
        run_dir = args.input_dir
        if run_dir is None:
            parser.error("--plot-only requires --input-dir pointing to an existing comparison directory")
        if not run_dir.exists():
            raise FileNotFoundError(f"Comparison directory not found: {run_dir}")

        aggregated_path = run_dir / AGGREGATED_FILENAME
        if not aggregated_path.exists():
            raise FileNotFoundError(
                f"Aggregated results not found at {aggregated_path}. Rerun comparison with the updated script to create it."
            )

        aggregated = load_aggregated_results(aggregated_path)
        if not aggregated:
            raise ValueError(f"Aggregated results file {aggregated_path} does not contain any agent data")

        selected_agents = normalize_agent_names(args.agents, aggregated.keys())
        if not selected_agents:
            raise ValueError("No agents selected for plotting")

        records_by_agent = {
            agent: aggregated[agent].get("records", []) for agent in selected_agents if agent in aggregated
        }
        if not any(records_by_agent.values()):
            raise ValueError("Aggregated results do not contain step records for the requested agents")

        plot_path = run_dir / "comparison_boxplots.png"
        plot_step_boxplots(records_by_agent, plot_path)
        print_agent_summaries(aggregated, selected_agents)

        log_path = run_dir / "comparison.log"
        if log_path.exists():
            print(f"[INFO] Using existing log at {log_path}")
        print(f"[PLOT] Saved comparison to {plot_path}")
        print(f"[INFO] Loaded aggregated results from {aggregated_path}")
        return

    if args.model is None:
        parser.error("--model is required unless --plot-only is specified")

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
        aggregated: Dict[str, Dict[str, object]] = {}

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

            metric_labels = {
                "total_reward": "Total Reward",
                "final_qos_total": "Final QoS (tasks)",
                "scaling_actions": "Scaling Actions",
                "max_workers": "Max Workers",
            }

            for agent_name, data in aggregated.items():
                summary = data["summary"]
                print(f"[SUMMARY] {agent_name}")
                for key, label in metric_labels.items():
                    metric = summary.get(key, {"mean": 0.0, "std": 0.0})
                    mean_value = metric.get("mean", 0.0)
                    std_value = metric.get("std", 0.0)

                    if key == "final_qos_total":
                        mean_str = f"{mean_value:.2%}"
                        std_str = f"{std_value:.2%}"
                    else:
                        mean_str = f"{mean_value:.2f}"
                        std_str = f"{std_value:.2f}"

                    print(f"  - {label}: {mean_str} ± {std_str}")

            save_aggregated_results(output_dir / AGGREGATED_FILENAME, aggregated)

            selected_agents = normalize_agent_names(args.agents, aggregated.keys())
            if not selected_agents:
                raise ValueError("No agents selected for plotting")

            records_by_agent = {name: aggregated[name]["records"] for name in selected_agents}
            plot_step_boxplots(records_by_agent, plot_path)
            print_agent_summaries(aggregated, selected_agents)
            print(f"[INFO] Aggregated results saved to {output_dir / AGGREGATED_FILENAME}")
            print(f"[PLOT] Saved comparison to {plot_path}")
        finally:
            if env is not None:
                env.close()
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            print(f"[INFO] Comparison log written to {log_path}")


if __name__ == "__main__":
    main()