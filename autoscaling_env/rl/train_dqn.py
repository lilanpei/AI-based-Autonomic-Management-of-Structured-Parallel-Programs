#!/usr/bin/env python3
"""Train a lightweight DQN agent against the OpenFaaS autoscaling environment."""

from __future__ import annotations

import argparse
import random
import signal
from pathlib import Path
from typing import Dict, List

import numpy as np
import os
import sys

# Add project root to path when script executed directly
current_dir = os.path.dirname(os.path.abspath(__file__))
autoscaling_env_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(autoscaling_env_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from autoscaling_env.openfaas_autoscaling_env import OpenFaaSAutoscalingEnv
from autoscaling_env.rl.dqn_agent import DQNAgent, DQNHyperparams
from autoscaling_env.rl.plot_training import render_training_curves
from autoscaling_env.rl.test_dqn import evaluate_episode
from autoscaling_env.rl.utils import (
    configure_logging,
    prepare_output_directory,
    write_json,
)
from utilities.utilities import get_utc_now, get_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=120, help="Number of training episodes")
    parser.add_argument("--max-steps", type=int, default=30, help="Maximum steps per episode")
    parser.add_argument("--step-duration", type=int, default=8, help="Seconds per action step in the environment")
    parser.add_argument("--observation-window", type=int, default=8, help="Observation window size passed to the environment")
    parser.add_argument("--initial-workers", type=int, default=12, help="Initial number of worker replicas")
    parser.add_argument("--output-dir", type=Path, default=Path("runs"), help="Base directory for experiment outputs")
    parser.add_argument("--resume-model", type=Path, default=None, help="Path to a previously trained DQN checkpoint (.pt) to resume")

    # Hyperparameters
    parser.add_argument("--gamma", type=float, default=0.97, help="Discount factor")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Adam learning rate")
    parser.add_argument("--epsilon-start", type=float, default=0.8, help="Initial epsilon for exploration")
    parser.add_argument("--epsilon-end", type=float, default=0.05, help="Minimum epsilon after decay")
    parser.add_argument("--epsilon-decay", type=float, default=0.98, help="Episode-wise epsilon decay factor")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size for SGD updates")
    parser.add_argument("--replay-capacity", type=int, default=75_000, help="Replay buffer capacity")
    parser.add_argument("--target-update", type=int, default=600, help="Steps between target network updates")
    parser.add_argument(
        "--target-tau",
        type=float,
        default=0.01,
        help="Soft-update interpolation factor (1.0 falls back to hard updates)",
    )
    parser.add_argument("--warmup-steps", type=int, default=400, help="Steps to collect before training begins")
    parser.add_argument("--max-grad-norm", type=float, default=3.0, help="Gradient clipping norm (0 disables)")
    parser.add_argument("--hidden-layers", type=int, nargs="+", default=[128, 64], help="Hidden layer sizes for the Q-network")

    parser.add_argument("--checkpoint-every", type=int, default=10, help="Save intermediate checkpoints every N episodes (0 to disable)")
    parser.add_argument("--eval-episodes", type=int, default=5, help="Evaluation episodes to run per checkpoint (0 to disable)")

    parser.add_argument("--initialize-workflow", action="store_true", help="Deploy/initialize the OpenFaaS workflow before training begins")
    parser.add_argument("--phase-shuffle", action="store_true", help="Randomize phase order each training episode (training only)")
    parser.add_argument("--phase-shuffle-seed", type=int, default=None, help="Optional RNG seed used when --phase-shuffle is enabled")

    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--task-seed", type=int, default=42, help="Seed passed to the task generator (defaults to --seed)")

    return parser.parse_args()


def _build_hyperparams(args: argparse.Namespace) -> DQNHyperparams:
    return DQNHyperparams(
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        batch_size=args.batch_size,
        replay_capacity=args.replay_capacity,
        target_update_interval=args.target_update,
        target_tau=args.target_tau,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm if args.max_grad_norm > 0 else None,
        hidden_layers=tuple(args.hidden_layers),
    )


def create_agent(env: OpenFaaSAutoscalingEnv, args: argparse.Namespace) -> DQNAgent:
    observation_low = env.observation_space.low
    observation_high = env.observation_space.high
    hyperparams = _build_hyperparams(args)

    if args.resume_model is not None:
        agent = DQNAgent.load(args.resume_model)
        if agent.action_size != env.action_space.n:
            raise ValueError(
                f"Loaded agent action space ({agent.action_size}) does not match environment ({env.action_space.n})"
            )
        if agent.observation_size != observation_low.size:
            raise ValueError(
                "Loaded agent observation dimensionality does not match environment observations"
            )
        agent.observation_low = observation_low
        agent.observation_high = observation_high
        agent._setup_normalization()
        return agent

    return DQNAgent(
        observation_size=observation_low.size,
        action_size=env.action_space.n,
        hyperparams=hyperparams,
        observation_low=observation_low,
        observation_high=observation_high,
    )


def evaluate_checkpoint(
    train_env: OpenFaaSAutoscalingEnv,
    agent: DQNAgent,
    args: argparse.Namespace,
    checkpoint_episode: int,
    base_seed: int,
    logger,
) -> Dict[str, float] | None:
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

    records: List[Dict[str, List[float]]] = []
    prev_mode = agent.policy_net.training
    agent.set_eval_mode()

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
        eval_env.close()
        if prev_mode:
            agent.set_train_mode()
        else:
            agent.set_eval_mode()

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
    mean_workers_values = _collect("mean_workers")

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
    mean_mean_workers, std_mean_workers = _mean_std(mean_workers_values)

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

    experiment_dir = prepare_output_directory(args.output_dir, prefix="dqn_run")
    logger = configure_logging(experiment_dir / "logs", name="dqn")
    logger.info("Experiment directory: %s", experiment_dir)
    logger.info("Training configuration: %s", " ".join(f"{k}={v}" for k, v in sorted(vars(args).items())))
    try:
        full_config = get_config()
        reward_cfg = full_config.get("reward", {})
    except Exception as exc:
        logger.warning("Unable to load reward configuration: %s", exc)
    else:
        logger.info("Reward configuration: %s", reward_cfg)

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
        "%s DQN agent | lr=%.4f gamma=%.3f epsilon=%.3f (current=%.3f) hidden_layers=%s",
        "Resumed" if args.resume_model else "Initialized",
        agent.hyperparams.learning_rate,
        agent.hyperparams.gamma,
        agent.hyperparams.epsilon_start,
        agent.epsilon,
        agent.hyperparams.hidden_layers,
    )

    metrics: List[Dict[str, float]] = []
    eval_history: List[Dict[str, float]] = []

    def _graceful_exit(*_args, **_kwargs):
        logger.warning("Training interrupted. Saving current model state...")
        agent.save(experiment_dir / "models" / "dqn_interrupt.pt")
        write_json({"episodes_completed": len(metrics)}, experiment_dir / "training_summary.json")
        env.close()
        raise SystemExit(1)

    signal.signal(signal.SIGINT, _graceful_exit)

    total_steps = 0

    for episode in range(1, args.episodes + 1):
        episode_seed = base_seed + episode - 1
        np.random.seed(episode_seed)
        random.seed(episode_seed)
        agent.set_train_mode()
        observation = env.reset(seed=episode_seed)

        total_reward = 0.0
        qos_rates: List[float] = []
        step_count = 0
        last_step_info: Dict[str, float] | None = None
        scale_up_count = 0
        scale_down_count = 0
        noop_count = 0
        max_workers_seen = float(observation[4]) if observation.size > 4 else float(env.initial_workers)
        worker_counts: List[float] = []
        last_qos_violations = 0
        last_unfinished_tasks = 0
        losses: List[float] = []
        header_logged = False

        for step in range(args.max_steps):
            action = agent.select_action(observation)
            next_observation, reward, done, info = env.step(action)
            agent.push_transition(observation, action, reward, next_observation, done)
            update_loss = agent.update()
            if update_loss is not None:
                losses.append(update_loss)

            observation = next_observation
            total_reward += reward
            qos_rates.append(info.get("qos_rate", next_observation[8]))
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

            if applied_delta is not None:
                delta_int = int(applied_delta)
                action_str = f"{delta_int:+d}" if delta_int != 0 else "0"
            else:
                action_map = {0: "-1", 1: "0", 2: "+1"}
                action_str = action_map.get(action, str(action))
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

            total_steps += 1
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
            "scale_up_count": scale_up_count,
            "scale_down_count": scale_down_count,
            "noop_count": noop_count,
            "max_workers": max_workers_seen,
            "mean_workers": float(np.mean(worker_counts)) if worker_counts else max_workers_seen,
            "qos_violations": last_qos_violations,
            "unfinished_tasks": last_unfinished_tasks,
            "mean_loss": float(np.mean(losses)) if losses else 0.0,
        }
        metrics.append(episode_metrics)

        logger.info(
            "Episode %d/%d | steps=%d reward=%.2f mean_qos=%.3f final_qos=%.3f tasks=%d"
            " loss=%.4f epsilon=%.3f scale_up=%d scale_down=%d noop=%d mean_workers=%.2f max_workers=%.0f"
            " qos_violations=%d unfinished=%d",
            episode,
            args.episodes,
            step_count,
            episode_metrics["total_reward"],
            episode_metrics["mean_qos"],
            episode_metrics["final_qos"],
            episode_metrics["processed_tasks"],
            episode_metrics["mean_loss"],
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
            checkpoint_path = experiment_dir / "models" / f"dqn_ep{episode:04d}.pt"
            agent.save(checkpoint_path)
            logger.info("Saved checkpoint to %s", checkpoint_path)

            eval_summary = evaluate_checkpoint(env, agent, args, episode, base_seed, logger)
            if eval_summary:
                eval_history.append(eval_summary)

    # Save final artifacts
    final_model_path = experiment_dir / "models" / "dqn_final.pt"
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

    plots_dir = experiment_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    try:
        render_training_curves(
            metrics,
            eval_history,
            plots_dir / "training_curves.png",
            title="DQN Training Curves",
        )
    except ValueError as exc:
        logger.warning("Skipping training plot generation: %s", exc)

    summary_payload = {
        "episodes_completed": len(metrics),
        "total_steps": total_steps,
        "config": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        },
    }
    write_json(summary_payload, experiment_dir / "training_summary.json")

    env.close()
    logger.info("Training complete. Metrics stored at %s", experiment_dir)


if __name__ == "__main__":
    main()
