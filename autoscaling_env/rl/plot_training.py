#!/usr/bin/env python3
"""Plot SARSA training metrics stored in training_metrics.json."""

from __future__ import annotations

import argparse
from pathlib import Path

import json

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path, help="Path to training_metrics.json produced by train_sarsa.py")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path for the generated plot (defaults to sibling plots/training_curves.png)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.metrics.exists():
        raise FileNotFoundError(f"Metrics file not found: {args.metrics}")

    data = args.metrics.read_text(encoding="utf-8")
    payload = json.loads(data)
    episodes = payload.get("episodes", [])

    if not episodes:
        raise ValueError("Metrics file does not contain any episode data")

    episode_numbers = np.array([ep["episode"] for ep in episodes])
    total_rewards = np.array([ep["total_reward"] for ep in episodes])
    mean_qos = np.array([ep["mean_qos"] for ep in episodes])
    epsilons = np.array([ep["epsilon"] for ep in episodes])

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.set_title("SARSA Training Curves")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Reward", color="tab:blue")
    ax1.plot(episode_numbers, total_rewards, color="tab:blue", label="Total Reward")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.set_ylabel("QoS Rate", color="tab:green")
    ax2.plot(episode_numbers, mean_qos, color="tab:green", linestyle="--", label="Mean QoS")
    ax2.tick_params(axis="y", labelcolor="tab:green")

    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("outward", 60))
    ax3.set_ylabel("Epsilon", color="tab:red")
    ax3.plot(episode_numbers, epsilons, color="tab:red", linestyle=":", label="Epsilon")
    ax3.tick_params(axis="y", labelcolor="tab:red")

    fig.tight_layout()

    if args.output:
        output_path = args.output
    else:
        output_path = args.metrics.parent / "plots" / "training_curves.png"
        output_path.parent.mkdir(exist_ok=True)

    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Plot saved to {output_path}")


if __name__ == "__main__":
    main()
