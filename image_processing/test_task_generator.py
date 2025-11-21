#!/usr/bin/env python3
"""Visualize task-generation phases defined in utilities/configuration.yml."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utilities.utilities import get_config


def compute_phase_rate(phase_cfg, base_rate, time_in_phase):
    """Compute arrival rate for a single phase definition."""
    pattern = (phase_cfg.get("phase_pattern") or "steady").lower()
    multiplier = float(phase_cfg.get("phase_multiplier", 1.0))

    if pattern == "steady":
        return base_rate * multiplier

    if pattern.startswith("oscillation"):
        duration = max(float(phase_cfg.get("phase_duration", 60)), 1e-6)
        progress = (time_in_phase / duration) % 1.0
        min_multiplier = float(phase_cfg.get("oscillation_min", multiplier))
        max_multiplier = float(phase_cfg.get("oscillation_max", multiplier))
        cycles = float(phase_cfg.get("oscillation_cycles", 1.0))

        midpoint = (max_multiplier + min_multiplier) / 2.0
        amplitude = (max_multiplier - min_multiplier) / 2.0
        oscillation = midpoint + amplitude * np.sin(2 * np.pi * cycles * progress)
        return base_rate * oscillation

    return base_rate * multiplier


def simulate_arrival_rates(base_rate, phase_definitions):
    """Simulate arrival rates across all configured phases."""
    times = []
    rates = []
    phase_indices = []
    window_sizes = []
    phase_names = []
    phase_boundaries = [0.0]

    current_time = 0.0
    for idx, phase_cfg in enumerate(phase_definitions, start=1):
        duration = float(phase_cfg.get("phase_duration", 60))
        window = float(phase_cfg.get("window_duration", 1))
        if window <= 0:
            window = 1.0

        phase_names.append(phase_cfg.get("phase_name", f"Phase {idx}"))

        phase_time = 0.0
        while phase_time < duration - 1e-9:
            next_phase_time = min(phase_time + window, duration)
            effective_window = next_phase_time - phase_time

            rate = compute_phase_rate(phase_cfg, base_rate, phase_time)

            times.append(current_time + phase_time)
            rates.append(rate)
            phase_indices.append(idx)
            window_sizes.append(effective_window)

            phase_time = next_phase_time

        current_time += duration
        phase_boundaries.append(current_time)

    return (
        np.array(times, dtype=float),
        np.array(rates, dtype=float),
        np.array(phase_indices, dtype=int),
        np.array(window_sizes, dtype=float),
        phase_boundaries,
        phase_names,
    )


def plot_arrival_rates(sim_results, phase_definitions, base_rate, output_file="task_arrival_rates.png"):
    """Render arrival rate plot based on simulated results."""
    (
        times,
        rates,
        phase_indices,
        window_sizes,
        phase_boundaries,
        phase_names,
    ) = sim_results

    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 18,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "figure.titlesize": 20,
        }
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={"height_ratios": [3, 2]})
    fig.subplots_adjust(hspace=0.35)

    # Plot arrival rate over time
    (rate_line,) = ax1.plot(times / 60.0, rates, color="tab:blue", linewidth=2, label="Arrival Rate")

    cmap = plt.colormaps["tab20"].resampled(len(phase_names))
    colors = [cmap(i) for i in range(len(phase_names))]

    phase_handles = []
    for i, (start, end) in enumerate(zip(phase_boundaries[:-1], phase_boundaries[1:])):
        ax1.axvspan(start / 60.0, end / 60.0, alpha=0.18, color=colors[i])
        phase_handles.append(Rectangle((0, 0), 1, 1, facecolor=colors[i], alpha=0.3))

    for boundary in phase_boundaries[1:-1]:
        ax1.axvline(boundary / 60.0, color="gray", linestyle="--", linewidth=1, alpha=0.5)

    base_line = ax1.axhline(
        base_rate,
        color="tab:green",
        linestyle=":",
        linewidth=1.2,
        alpha=0.8,
        label=f"Base Rate",
    )

    max_rate = max(rates.max(), base_rate) * 1.2
    ax1.set_xlabel("Time (minutes)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Arrival Rate (tasks/min)", fontsize=12, fontweight="bold")
    ax1.set_title(
        "Configured Task Arrival Pattern",
        fontsize=14,
        fontweight="bold",
    )
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, phase_boundaries[-1] / 60.0)
    ax1.set_ylim(0, max_rate)

    legend_handles = [rate_line, base_line] + phase_handles
    legend_labels = ["Arrival Rate", f"Base Rate"] + phase_names
    ax1.legend(legend_handles, legend_labels, loc="best", fontsize=12)

    # Rate distribution by phase
    phase_data = []
    phase_labels = []
    for idx, phase_cfg in enumerate(phase_definitions, start=1):
        mask = phase_indices == idx
        phase_rates = rates[mask]
        if phase_rates.size == 0:
            phase_data.append(np.array([0.0]))
            phase_labels.append(f"{phase_names[idx-1]}\n(no samples)")
            continue

        mean_rate = np.average(phase_rates, weights=window_sizes[mask])
        min_rate = phase_rates.min()
        max_rate_phase = phase_rates.max()
        phase_data.append(phase_rates)
        phase_labels.append(
            f"{phase_names[idx-1]}\n({min_rate:.1f}–{max_rate_phase:.1f})\nμ={mean_rate:.1f}"
        )

    box = ax2.boxplot(
        phase_data,
        tick_labels=phase_labels,
        patch_artist=True,
        widths=0.6,
        showmeans=True,
    )

    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax2.axhline(base_rate, color="tab:green", linestyle=":", linewidth=1.2, alpha=0.8)
    ax2.set_xlabel("Phase", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Arrival Rate (tasks/min)", fontsize=12, fontweight="bold")
    ax2.set_title("Rate Distribution by Phase", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"\n✅ Plot saved to: {output_file}")

    return fig


def print_phase_summary(sim_results, phase_definitions, base_rate):
    """Print textual summary of configured phases."""
    (
        times,
        rates,
        phase_indices,
        window_sizes,
        phase_boundaries,
        phase_names,
    ) = sim_results

    total_duration = phase_boundaries[-1]

    print(f"\n{'='*70}")
    print("TASK ARRIVAL PATTERN SUMMARY")
    print(f"{'='*70}")
    print(f"Base Rate: {base_rate:.0f} tasks/min")
    print(f"Total Duration: {total_duration:.0f}s ({total_duration/60:.1f} min)")
    print(f"{'='*70}\n")

    for idx, phase_cfg in enumerate(phase_definitions, start=1):
        mask = phase_indices == idx
        phase_rates = rates[mask]
        phase_windows = window_sizes[mask]
        phase_start = phase_boundaries[idx - 1]
        phase_end = phase_boundaries[idx]

        if phase_rates.size == 0:
            print(f"{phase_names[idx-1]}: No samples")
            continue

        weighted_mean = np.average(phase_rates, weights=phase_windows)
        weighted_std = np.sqrt(
            np.average((phase_rates - weighted_mean) ** 2, weights=phase_windows)
        )
        estimated_tasks = np.sum(phase_rates * (phase_windows / 60.0))

        print(f"{phase_names[idx-1]} (Phase {idx})")
        print(f"  Pattern: {phase_cfg.get('phase_pattern', 'steady')}")
        print(f"  Duration: {phase_end - phase_start:.0f}s")
        print(f"  Rate Range: {phase_rates.min():.2f} - {phase_rates.max():.2f} tasks/min")
        print(f"  Mean Rate: {weighted_mean:.2f} tasks/min")
        print(f"  Std Dev: {weighted_std:.2f} tasks/min")
        print(f"  Estimated Tasks: ~{int(round(estimated_tasks))} tasks")
        print()

    total_tasks = np.sum(rates * (window_sizes / 60.0))
    avg_rate = total_tasks / (total_duration / 60.0)

    print(f"{'='*70}")
    print("OVERALL STATISTICS")
    print(f"{'='*70}")
    print(f"Total Estimated Tasks: ~{int(round(total_tasks))} tasks")
    print(f"Average Rate: {avg_rate:.2f} tasks/min")
    print(f"Min Rate: {rates.min():.2f} tasks/min")
    print(f"Max Rate: {rates.max():.2f} tasks/min")
    print(f"{'='*70}\n")


def main():
    """Entry point for phase visualization."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-rate",
        type=float,
        help="Override base rate (tasks/min) from configuration.yml",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="task_arrival_rates.png",
        help="Output path for the generated plot",
    )
    args = parser.parse_args()

    config = get_config()
    phase_definitions = config.get("phase_definitions")
    if not phase_definitions:
        raise ValueError("No phase_definitions found in utilities/configuration.yml")

    base_rate = float(args.base_rate) if args.base_rate is not None else float(config.get("base_rate", 300))

    sim_results = simulate_arrival_rates(base_rate, phase_definitions)

    print_phase_summary(sim_results, phase_definitions, base_rate)
    print("Generating plot...")
    plot_arrival_rates(sim_results, phase_definitions, base_rate, output_file=args.output)

    print("\n✅ Test complete!")
    print("\nRun with a different base rate:")
    print("  python test_task_generator.py --base-rate 400")
    print("Specify a custom output file:")
    print("  python test_task_generator.py --output plots/custom_rates.png")


if __name__ == "__main__":
    main()
