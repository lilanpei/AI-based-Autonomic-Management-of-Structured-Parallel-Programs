"""Generate histograms for observation features from a SARSA training log.

Usage:
    python plot_observation_histograms.py --log autoscaling_env/rl/runs/.../logs/training.log

This script parses the step-level rows written by train_sarsa.py, computes summary
statistics for each observation dimension, and saves both JSON summaries and PNG
histograms alongside the provided log file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")  # Ensure headless environments can render plots
import matplotlib.pyplot as plt
import numpy as np

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log",
        type=Path,
        required=True,
        help="Path to training.log produced by train_sarsa.py",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=40,
        help="Maximum number of histogram bins to use (default: 40)",
    )
    return parser.parse_args()


def _iter_step_rows(log_path: Path) -> Iterable[Dict[str, float]]:
    """Yield parsed step rows from the training log.

    train_sarsa.py logs rows in a fixed-width table after the header line, e.g.

        Step      Time[s]     Action  ...  ARR      Reward
        1            8.63         -1 ...

    We split on whitespace and select the relevant observation columns.
    """

    with log_path.open() as fh:
        for raw in fh:
            if " - INFO - " not in raw:
                continue
            payload = raw.split(" - INFO - ", 1)[1].strip()
            if not payload or not payload[0].isdigit():
                continue
            parts = payload.split()
            # Expected table has 15 tokens: step, time, action, scale_t, step_d,
            # input_q, worker_q, result_q, output_q, workers, qos%, avg_t,
            # max_t, arr, reward.
            if len(parts) != 15:
                continue
            try:
                yield {
                    "Input_Q": float(parts[5]),
                    "Worker_Q": float(parts[6]),
                    "Result_Q": float(parts[7]),
                    "Output_Q": float(parts[8]),
                    "Workers": float(parts[9]),
                    "QoS": float(parts[10]) / 100.0,
                    "AVG_T": float(parts[11]),
                    "MAX_T": float(parts[12]),
                    "ARR": float(parts[13]),
                }
            except ValueError:
                continue


def _compute_summary(data: List[float]) -> Dict[str, float]:
    arr = np.asarray(data, dtype=float)
    percentiles = [0, 25, 50, 75, 90, 95, 99, 100]
    summary = {
        "count": int(arr.size),
        "min": float(arr.min()),
        "mean": float(arr.mean()),
        "max": float(arr.max()),
    }
    summary.update(
        {f"p{p}": float(np.percentile(arr, p)) for p in percentiles}
    )
    return summary


def main() -> None:
    args = parse_args()
    log_path = args.log
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    # Collect data per dimension
    observations: Dict[str, List[float]] = {
        "Input_Q": [],
        "Worker_Q": [],
        "Result_Q": [],
        "Output_Q": [],
        "Workers": [],
        "QoS": [],
        "AVG_T": [],
        "MAX_T": [],
        "ARR": [],
    }

    for row in _iter_step_rows(log_path):
        for key, value in row.items():
            observations[key].append(value)

    if not any(observations.values()):
        raise RuntimeError(
            "No observation rows were parsed. Ensure the log is a training.log "
            "produced by train_sarsa.py."
        )

    # Prepare output paths alongside the log file
    output_dir = log_path.parent
    summary_path = output_dir / "observation_summary.json"
    plots_dir = output_dir / "observation_histograms"
    plots_dir.mkdir(exist_ok=True)

    # Write summary statistics
    summary_payload = {
        key: _compute_summary(values) for key, values in observations.items() if values
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2))

    # Generate histograms
    for key, values in observations.items():
        if not values:
            continue
        data = np.asarray(values, dtype=float)
        bins = min(args.bins, max(10, int(np.sqrt(data.size))))
        plt.rcParams.update(
            {
                "font.size": 12,
                "axes.titlesize": 14,
                "axes.labelsize": 12,
                "xtick.labelsize": 11,
                "ytick.labelsize": 11,
                "legend.fontsize": 11,
                "figure.titlesize": 16,
            }
        )
        plt.figure(figsize=(6, 4))
        plt.hist(data, bins=bins, color="#1f77b4", edgecolor="black", alpha=0.75)
        plt.title(f"Distribution of {key}")
        plt.xlabel(key)
        plt.ylabel("Frequency")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        out_path = plots_dir / f"{key.lower()}_hist.png"
        plt.savefig(out_path, dpi=150)
        plt.close()

    print(f"Summary saved to {summary_path}")
    print(f"Histograms saved in {plots_dir}")


if __name__ == "__main__":
    main()
