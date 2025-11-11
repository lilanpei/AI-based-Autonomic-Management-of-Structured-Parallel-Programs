#!/usr/bin/env python3
"""Train a tabular SARSA agent against the OpenFaaS autoscaling environment."""

import argparse
import signal
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

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
from autoscaling_env.rl.plot_training import render_training_curves
from autoscaling_env.rl.test_sarsa import evaluate_episode
from autoscaling_env.rl.utils import (
    build_discretization_config,
    configure_logging,
    prepare_output_directory,
    write_json,
)
from utilities.utilities import get_utc_now


DEFAULT_DISCRETIZATION_BINS = (4, 7, 1, 1, 8, 8, 1, 5, 6)
DEFAULT_DISCRETIZATION_EDGES = (
    [0.5, 1.5, 2.5],  # input_q
    # [0.5, 3.5, 7.5, 15.5, 31.5, 63.5],  # worker_q
    [0.5, 2.5, 5.5, 9.5, 15.5, 30.5],  # worker_q
    None,  # result_q (unused)
    None,  # output_q (unused)
    [3.5, 7.5, 11.5, 12.5, 16.5, 20.5, 24.5],  # workers
    [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0],  # avg_time
    None,  # max_time (unused)
    # None,  # arrival_rate (linear bins)
    [0.15, 0.5, 1.5, 3.5],  # arrival_rate
    [0.5, 0.8, 0.9, 0.95, 0.999],  # qos (percent-style thresholds -> fraction)
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=100, help="Number of training episodes")
    parser.add_argument("--max-steps", type=int, default=30, help="Maximum steps per episode")
    parser.add_argument("--step-duration", type=int, default=8, help="Seconds per action step in the environment")
    parser.add_argument("--observation-window", type=int, default=8, help="Observation window size passed to the environment")
    parser.add_argument("--initial-workers", type=int, default=12, help="Initial number of worker replicas")
    parser.add_argument("--output-dir", type=Path, default=Path("runs"), help="Base directory for experiment outputs")
    parser.add_argument(
        "--resume-model",
        type=Path,
        default=None,
        help="Path to a previously trained SARSA model to continue training",
    )
    parser.add_argument(
        "--resume-reset-hyperparams",
        action="store_true",
        help="When resuming, overwrite saved hyperparameters with the CLI values",
    )
    parser.add_argument(
        "--resume-reset-epsilon",
        action="store_true",
        help="When resuming, reset epsilon to the CLI value (applied after any hyperparameter reset)",
    )
    parser.add_argument("--epsilon", type=float, default=0.40, help="Initial epsilon for epsilon-greedy policy")
    parser.add_argument("--epsilon-min", type=float, default=0.10, help="Minimum epsilon after decay")
    parser.add_argument("--epsilon-decay", type=float, default=0.995, help="Episode-wise epsilon decay factor")
    parser.add_argument("--alpha", type=float, default=0.025, help="Learning rate (alpha)")
    parser.add_argument("--gamma", type=float, default=0.985, help="Discount factor (gamma)")
    parser.add_argument(
        "--trace-lambda",
        type=float,
        default=0.5,
        help="Eligibility trace decay parameter λ (0 disables traces)",
    )
    parser.add_argument(
        "--trace-threshold",
        type=float,
        default=1e-5,
        help="Prune eligibility traces whose absolute value falls below this threshold",
    )
    parser.add_argument(
        "--bins",
        type=int,
        nargs=9,
        default=list(DEFAULT_DISCRETIZATION_BINS),
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
        default=3,
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

    if args.resume_model is not None:
        agent = SARSAAgent.load(args.resume_model)

        if agent.action_size != env.action_space.n:
            raise ValueError(
                f"Loaded agent action space ({agent.action_size}) does not match environment ({env.action_space.n})"
            )

        existing_bins = agent.discretization.bins_per_dimension
        existing_edges = agent.discretization.edges_per_dimension
        discretization = build_discretization_config(
            observation_low=observation_low,
            observation_high=observation_high,
            bins_per_dimension=existing_bins,
            edges_per_dimension=existing_edges,
        )
        agent.discretization = discretization
        agent._build_bins()

        if args.resume_reset_hyperparams:
            agent.hyperparams = SARSAHyperparams(
                alpha=args.alpha,
                gamma=args.gamma,
                epsilon=args.epsilon,
                epsilon_min=args.epsilon_min,
                epsilon_decay=args.epsilon_decay,
                trace_lambda=args.trace_lambda,
                trace_threshold=args.trace_threshold,
            )
            args.resume_reset_epsilon = True

        if args.resume_reset_epsilon:
            agent.hyperparams.epsilon = args.epsilon
            agent.hyperparams.epsilon_min = args.epsilon_min
            agent.hyperparams.epsilon_decay = args.epsilon_decay
            agent._epsilon = args.epsilon

        return agent

    use_custom_edges = tuple(args.bins) == DEFAULT_DISCRETIZATION_BINS
    edges_per_dimension = DEFAULT_DISCRETIZATION_EDGES if use_custom_edges else [None] * len(args.bins)

    discretization = build_discretization_config(
        observation_low=observation_low,
        observation_high=observation_high,
        bins_per_dimension=args.bins,
        edges_per_dimension=edges_per_dimension,
    )

    hyperparams = SARSAHyperparams(
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        trace_lambda=args.trace_lambda,
        trace_threshold=args.trace_threshold,
    )

    return SARSAAgent(
        action_size=env.action_space.n,
        discretization=discretization,
        hyperparams=hyperparams,
    )


def plot_state_visit_distribution(
    visit_counts: Counter[Tuple[int, ...]],
    output_dir: Path,
) -> None:
    if not visit_counts:
        return

    import matplotlib.pyplot as plt

    counts = np.array(list(visit_counts.values()), dtype=np.int64)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("State Visit Count Distribution")
    ax.set_xlabel("Visit count per state")
    ax.set_ylabel("Number of states")

    bins = np.histogram_bin_edges(counts, bins="auto")
    ax.hist(counts, bins=bins, color="tab:blue", alpha=0.7, edgecolor="black")
    ax.set_xscale("log")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(plots_dir / "state_visit_distribution.png", dpi=200)
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
                logger=logger,
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
    mean_rewards = _collect("mean_reward")
    final_qos = _collect("final_qos_total")
    mean_qos_values = _collect("mean_qos")
    scaling_actions = _collect("scaling_actions")
    noop_actions = _collect("noop_actions")
    max_workers = _collect("max_workers")
    mean_workers = _collect("mean_workers")

    def _mean_std(values: np.ndarray) -> tuple[float, float]:
        if values.size == 0:
            return 0.0, 0.0
        mean = float(np.mean(values))
        std = float(np.std(values)) if values.size > 1 else 0.0
        return mean, std

    mean_total_reward, std_total_reward = _mean_std(total_rewards)
    mean_mean_reward, std_mean_reward = _mean_std(mean_rewards)
    mean_final_qos, std_final_qos = _mean_std(final_qos)
    mean_mean_qos, std_mean_qos = _mean_std(mean_qos_values)
    mean_scaling, std_scaling = _mean_std(scaling_actions)
    mean_noop, std_noop = _mean_std(noop_actions)
    mean_max_workers, std_max_workers = _mean_std(max_workers)
    mean_mean_workers, std_mean_workers = _mean_std(mean_workers)

    logger.info(
        (
            "[EVAL] Episode %d | total_reward=%.2f±%.2f mean_reward=%.2f±%.2f "
            "final_qos=%.2f%%±%.2f%% mean_qos=%.2f%%±%.2f%% max_workers=%.1f±%.1f "
            "mean_workers=%.1f±%.1f scaling=%.1f±%.1f noop=%.1f±%.1f"
        ),
        checkpoint_episode,
        mean_total_reward,
        std_total_reward,
        mean_mean_reward,
        std_mean_reward,
        mean_final_qos * 100.0,
        std_final_qos * 100.0,
        mean_mean_qos * 100.0,
        std_mean_qos * 100.0,
        mean_max_workers,
        std_max_workers,
        mean_mean_workers,
        std_mean_workers,
        mean_scaling,
        std_scaling,
        mean_noop,
        std_noop,
    )

    return {
        "episode": checkpoint_episode,
        "mean_total_reward": mean_total_reward,
        "std_total_reward": std_total_reward,
        "mean_mean_reward": mean_mean_reward,
        "std_mean_reward": std_mean_reward,
        "mean_final_qos": mean_final_qos,
        "std_final_qos": std_final_qos,
        "mean_mean_qos": mean_mean_qos,
        "std_mean_qos": std_mean_qos,
        "mean_scaling_actions": mean_scaling,
        "std_scaling_actions": std_scaling,
        "mean_noop_actions": mean_noop,
        "std_noop_actions": std_noop,
        "mean_max_workers": mean_max_workers,
        "std_max_workers": std_max_workers,
        "mean_mean_workers": mean_mean_workers,
        "std_mean_workers": std_mean_workers,
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
        "%s SARSA agent | alpha=%.3f gamma=%.3f epsilon=%.3f (current=%.3f) bins=%s",
        "Resumed" if args.resume_model else "Initialized",
        agent.hyperparams.alpha,
        agent.hyperparams.gamma,
        agent.hyperparams.epsilon,
        agent.epsilon,
        agent.discretization.bins_per_dimension,
    )

    metrics: List[Dict[str, float]] = []
    eval_history: List[Dict[str, float]] = []
    state_visit_counts: Counter[Tuple[int, ...]] = Counter()

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
        agent.reset_traces()
        observation = env.reset(seed=episode_seed)
        state = agent.discretize(observation)
        action = agent.select_action(state)

        total_reward = 0.0
        qos_rates: List[float] = []
        step_count = 0
        header_logged = False
        last_step_info: Dict[str, float] | None = None
        scale_up_count = 0
        scale_down_count = 0
        noop_count = 0
        max_workers_seen = float(observation[4]) if observation.size > 4 else 0.0
        worker_counts: List[float] = []
        last_qos_violations = 0
        last_unfinished_tasks = 0

        for step in range(args.max_steps):
            state_visit_counts[state] += 1
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
            applied_delta = info.get("applied_delta")
            if applied_delta is not None:
                if applied_delta > 0:
                    scale_up_count += 1
                elif applied_delta < 0:
                    scale_down_count += 1
                else:
                    noop_count += 1
            max_workers_seen = max(max_workers_seen, float(info.get("workers", next_observation[4])))
            worker_counts.append(float(info.get("workers", next_observation[4])))
            last_qos_violations = int(info.get("qos_violations", last_qos_violations))
            last_unfinished_tasks = int(info.get("unfinished_tasks", last_unfinished_tasks))

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

        state_visit_counts[state] += 1
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
            "scale_up_count": scale_up_count,
            "scale_down_count": scale_down_count,
            "noop_count": noop_count,
            "max_workers": max_workers_seen,
            "mean_workers": float(np.mean(worker_counts)) if worker_counts else max_workers_seen,
            "qos_violations": last_qos_violations,
            "unfinished_tasks": last_unfinished_tasks,
        }
        metrics.append(episode_metrics)
        logger.info(
            "Episode %d/%d | steps=%d reward=%.2f mean_qos=%.3f final_qos=%.3f tasks=%d"
            " epsilon=%.3f scale_up=%d scale_down=%d noop=%d mean_workers=%.2f max_workers=%.0f"
            " qos_violations=%d unfinished=%d",
            episode,
            args.episodes,
            step_count,
            episode_metrics["total_reward"],
            episode_metrics["mean_qos"],
            episode_metrics["final_qos"],
            episode_metrics["processed_tasks"],
            episode_metrics["epsilon"],
            episode_metrics["scale_up_count"],
            episode_metrics["scale_down_count"],
            episode_metrics["noop_count"],
            episode_metrics["mean_workers"],
            episode_metrics["max_workers"],
            episode_metrics["qos_violations"],
            episode_metrics["unfinished_tasks"],
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
    visit_summary_path = experiment_dir / "training_state_visits.json"
    total_state_space = int(np.prod(agent.discretization.bins_per_dimension))
    visit_payload = {
        "total_unique_states": len(state_visit_counts),
        "total_visits": sum(state_visit_counts.values()),
        "total_state_space": total_state_space,
        "coverage": (len(state_visit_counts) / total_state_space) if total_state_space > 0 else 0.0,
        "visit_stats": {
            "mean": float(np.mean(list(state_visit_counts.values()))) if state_visit_counts else 0.0,
            "median": float(np.median(list(state_visit_counts.values()))) if state_visit_counts else 0.0,
            "max": int(max(state_visit_counts.values())) if state_visit_counts else 0,
            "min": int(min(state_visit_counts.values())) if state_visit_counts else 0,
        },
        "counts": [
            {
                "state": list(state),
                "visits": count,
                "q_values": agent.q_values(state).tolist(),
            }
            for state, count in state_visit_counts.most_common()
        ],
    }
    write_json(visit_payload, visit_summary_path)
    plot_state_visit_distribution(state_visit_counts, experiment_dir)
    logger.info(
        "Recorded %d unique states with %d total visits (details in %s)",
        visit_payload["total_unique_states"],
        visit_payload["total_visits"],
        visit_summary_path,
    )
    if eval_history:
        write_json({"checkpoints": eval_history}, experiment_dir / "evaluation_metrics.json")

    plots_dir = experiment_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    episodes_for_plot = metrics
    try:
        render_training_curves(
            episodes_for_plot,
            eval_history,
            plots_dir / "training_curves.png",
            title="SARSA Training Progress",
        )
    except ValueError as exc:
        logger.warning("Skipping training plot generation: %s", exc)

    env.close()
    logger.info("Training complete. Metrics stored at %s", experiment_dir)


if __name__ == "__main__":
    main()
