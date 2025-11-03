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

    def _smooth(values: np.ndarray, window: int) -> np.ndarray:
        if values.size == 0 or window <= 1:
            return values
        window = min(window, values.size)
        kernel = np.ones(window, dtype=np.float64) / window
        return np.convolve(values, kernel, mode="same")

    smooth_window = 0
    if len(episodes) >= 3:
        smooth_window = min(15, max(3, len(episodes) // 10 or 3))

    rewards_smooth = _smooth(total_rewards, smooth_window) if smooth_window else total_rewards
    qos_smooth = _smooth(mean_qos, smooth_window) if smooth_window else mean_qos

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.set_title("SARSA Training Curves")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Reward", color="tab:blue")
    ax1.plot(episode_numbers, total_rewards, color="tab:blue", alpha=0.35, linewidth=1, label="Total Reward (raw)")
    ax1.plot(episode_numbers, rewards_smooth, color="tab:blue", linewidth=2, label="Total Reward (smoothed)")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.set_ylabel("QoS Rate", color="tab:green")
    ax2.plot(episode_numbers, mean_qos, color="tab:green", linestyle="--", alpha=0.35, linewidth=1, label="Mean QoS (raw)")
    ax2.plot(episode_numbers, qos_smooth, color="tab:green", linestyle="--", linewidth=2, label="Mean QoS (smoothed)")
    ax2.tick_params(axis="y", labelcolor="tab:green")

    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("outward", 60))
    ax3.set_ylabel("Epsilon", color="tab:red")
    ax3.plot(episode_numbers, epsilons, color="tab:red", linestyle=":", label="Epsilon")
    ax3.tick_params(axis="y", labelcolor="tab:red")

    handles, labels = [], []
    for axis in (ax1, ax2, ax3):
        h, l = axis.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    if handles:
        ax1.legend(handles, labels, loc="upper left", frameon=False)

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
