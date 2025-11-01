#!/usr/bin/env python3
"""Evaluate a trained SARSA agent in the OpenFaaS autoscaling environment."""

from __future__ import annotations

import argparse
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
from autoscaling_env.rl.sarsa_agent import SARSAAgent
from autoscaling_env.rl.utils import build_discretization_config, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Path to a trained SARSA model (.pkl)")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--max-steps", type=int, default=30, help="Maximum steps per episode")
    parser.add_argument("--step-duration", type=int, default=10, help="Seconds per action step in the environment")
    parser.add_argument("--observation-window", type=int, default=10, help="Observation window size passed to the environment")
    parser.add_argument("--initial-workers", type=int, default=3, help="Initial number of worker replicas")
    parser.add_argument("--output", type=Path, default=Path("runs/eval_results.json"), help="Path to write summary JSON")
    parser.add_argument(
        "--initialize-workflow",
        action="store_true",
        help="Deploy/initialize the OpenFaaS workflow before evaluation",
    )
    return parser.parse_args()


def evaluate_episode(env: OpenFaaSAutoscalingEnv, agent: SARSAAgent, max_steps: int) -> Dict[str, float]:
    observation = env.reset()
    state = agent.discretize(observation)
    action = int(np.argmax(agent.q_values(state)))

    total_reward = 0.0
    qos_rates: List[float] = []
    actions_taken: List[int] = []
    worker_counts: List[float] = []

    for step in range(max_steps):
        next_observation, reward, done, info = env.step(action)
        next_state = agent.discretize(next_observation)
        next_action = int(np.argmax(agent.q_values(next_state)))

        total_reward += reward
        qos_rates.append(info.get("qos_rate", 0.0))
        actions_taken.append(action)
        worker_counts.append(next_observation[4])

        state = next_state
        action = next_action

        if done:
            break

    return {
        "steps": len(actions_taken),
        "total_reward": float(total_reward),
        "mean_qos": float(np.mean(qos_rates) if qos_rates else 0.0),
        "mean_workers": float(np.mean(worker_counts) if worker_counts else 0.0),
    }


def main() -> None:
    args = parse_args()

    env = OpenFaaSAutoscalingEnv(
        max_steps=args.max_steps,
        step_duration=args.step_duration,
        observation_window=args.observation_window,
        initial_workers=args.initial_workers,
        initialize_workflow=args.initialize_workflow,
    )

    agent = SARSAAgent.load(args.model)

    results: List[Dict[str, float]] = []
    for episode in range(1, args.episodes + 1):
        episode_result = evaluate_episode(env, agent, args.max_steps)
        print(
            f"Episode {episode}/{args.episodes} | steps={episode_result['steps']} "
            f"reward={episode_result['total_reward']:.2f} mean_qos={episode_result['mean_qos']:.3f} "
            f"mean_workers={episode_result['mean_workers']:.2f}"
        )
        results.append(episode_result)

    summary = {
        "episodes": results,
        "mean_reward": float(np.mean([r["total_reward"] for r in results])),
        "mean_qos": float(np.mean([r["mean_qos"] for r in results])),
        "mean_workers": float(np.mean([r["mean_workers"] for r in results])),
    }

    write_json(summary, args.output)
    print(f"\nEvaluation summary written to {args.output}")

    env.close()


if __name__ == "__main__":
    main()
