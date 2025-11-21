#!/usr/bin/env python3
"""Compare scaling behaviour across multiple experiments.

For each experiment, this script parses the controller logs to extract
per-run task runtime ("Task Run Time" = Program/Monitoring completed -
Task generation started) for each worker count. It then computes:

- Average runtime vs number of workers
- Speedup vs number of workers (relative to the smallest worker count
  available in that experiment)

and produces two comparison plots with one curve per experiment.

Experiments are identified by their experiment directory name, e.g.
"1007_1000t_30s". The script expects logs to be stored under
"<project_root>/logs_farm/<experiment>", independent of the
"log_dir" value in utilities/configuration.yml. The configuration file
is only used to read optional parameters such as
"number_of_most_recent_logs".
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


# Determine project root (one level above this script's directory) and make
# sure it is on sys.path so that local packages like `utilities` are
# importable even when running the script via an explicit path.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Try both package layouts so the script works whether utilities are
# installed as "orchestrator.utilities" or "utilities.utilities". If neither
# import works, fall back to a no-op get_config that returns an empty dict.
try:  # pragma: no cover - environment-dependent import
    from orchestrator.utilities import get_config
except ImportError:  # pragma: no cover - fallback for repo layout
    try:
        from utilities.utilities import get_config
    except ImportError:
        def get_config():  # type: ignore[no-redef]
            return {}


LOG_PATTERN = re.compile(r"logs_controller_w(\d+)_(\d+)\.txt")
TIMER_ENV_END = re.compile(r"\[TIMER\] Environment initialization completed at \[(.*?)\] seconds.")
TIMER_FARM_END = re.compile(r"\[TIMER\] Farm initialization completed at \[(.*?)\] seconds.")
TIMER_TASK_START = re.compile(r"\[TIMER\] Task generation started at \[(.*?)\] seconds.")
TIMER_PROGRAM_END = re.compile(r"\[TIMER\] Program/Monitoring completed at \[(.*?)\] seconds.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiments",
        nargs="+",
        required=True,
        help="Experiment names (subdirs under logs_farm and plots), e.g. 1007_1000t_30s",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Optional display labels for the experiments (same order as --experiments)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots/multi_run_comparison"),
        help="Directory where comparison plots will be written",
    )
    return parser.parse_args()


def _select_logs(log_dir: Path, last_n: int | None) -> List[Tuple[int, Path]]:
    """Return a list of (workers, logfile_path) pairs for the given directory.

    If last_n is not None, at most that many most recent logs (by timestamp
    encoded in filename) are kept per worker count.
    """

    grouped: Dict[int, List[Tuple[int, Path]]] = {}
    for fname in os.listdir(log_dir):
        match = LOG_PATTERN.match(fname)
        if not match:
            continue
        workers = int(match.group(1))
        ts = int(match.group(2))
        grouped.setdefault(workers, []).append((ts, log_dir / fname))

    selected: List[Tuple[int, Path]] = []
    for workers, entries in grouped.items():
        entries.sort(key=lambda x: x[0])
        if last_n is not None and last_n > 0 and len(entries) > last_n:
            entries = entries[-last_n:]
        selected.extend((workers, path) for _, path in entries)

    return selected


def _parse_single_log(path: Path) -> Tuple[float | None, float | None, float | None]:
    """Parse a single controller log.

    Returns a tuple (task_run_time, env_init_time, farm_init_time), all in
    seconds, each of which may be None if timing markers are missing.
    """

    env_end = None
    farm_end = None
    task_start = None
    prog_end = None

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if "Environment initialization completed" in line:
                m = TIMER_ENV_END.search(line)
                if m:
                    env_end = float(m.group(1))
            elif "Farm initialization completed" in line:
                m = TIMER_FARM_END.search(line)
                if m:
                    farm_end = float(m.group(1))
            elif "Task generation started" in line:
                m = TIMER_TASK_START.search(line)
                if m:
                    task_start = float(m.group(1))
            elif "Program/Monitoring completed" in line:
                m = TIMER_PROGRAM_END.search(line)
                if m:
                    prog_end = float(m.group(1))

    task_run_time: float | None = None
    env_init_time: float | None = None
    farm_init_time: float | None = None

    if task_start is not None and prog_end is not None:
        task_run_time = prog_end - task_start
    if env_end is not None:
        env_init_time = env_end
    if env_end is not None and farm_end is not None:
        farm_init_time = farm_end - env_end

    return task_run_time, env_init_time, farm_init_time


def collect_metrics_for_experiment(log_dir: Path, last_n: int | None) -> Tuple[Dict[int, List[float]], Dict[int, List[float]], Dict[int, List[float]]]:
    """Collect per-worker task, env-init, and farm-init times.

    Returns three mappings:
    - workers -> list of task run times (seconds)
    - workers -> list of Env Init Time values (seconds)
    - workers -> list of Farm Init Time values (seconds)
    """

    if not log_dir.exists() or not log_dir.is_dir():
        raise FileNotFoundError(f"Log directory not found: {log_dir}")

    per_worker_runtime: Dict[int, List[float]] = {}
    per_worker_env: Dict[int, List[float]] = {}
    per_worker_farm: Dict[int, List[float]] = {}
    selected = _select_logs(log_dir, last_n)

    for workers, log_path in selected:
        task_run_time, env_init_time, farm_init_time = _parse_single_log(log_path)
        if task_run_time is not None:
            per_worker_runtime.setdefault(workers, []).append(task_run_time)
        if env_init_time is not None:
            per_worker_env.setdefault(workers, []).append(env_init_time)
        if farm_init_time is not None:
            per_worker_farm.setdefault(workers, []).append(farm_init_time)

    if not per_worker_runtime:
        raise RuntimeError(f"No task run time data parsed from {log_dir}")

    return per_worker_runtime, per_worker_env, per_worker_farm


def compute_runtime_and_speedup(per_worker: Dict[int, List[float]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute average runtime and speedup arrays from raw per-worker data.

    Speedup is defined per experiment as T_min_workers / T_workers, where
    T_min_workers is the average runtime at the smallest worker count
    present in the experiment.
    """

    workers = np.array(sorted(per_worker.keys()), dtype=float)
    runtimes = np.array([np.mean(per_worker[w]) for w in workers], dtype=float)

    baseline = runtimes[0]
    speedup = baseline / runtimes
    return workers, runtimes, speedup


def render_comparison_plots(
    series: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    output_dir: Path,
    agg_init: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> None:
    """Render runtime vs workers and speedup vs workers comparison plots.

    `series` is a list of (label, workers, runtimes, speedup, env_inits, farm_inits) tuples.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "figure.titlesize": 18,
        }
    )

    # Use Matplotlib's tab10 palette for a familiar look, but skip the
    # orange entry (index 1) for experiment curves and reserve that orange
    # exclusively for the Ideal line to match scaling_metrics.
    tab10 = plt.get_cmap("tab10")
    experiment_colors = [
        tab10(0),  # blue
        tab10(2),  # green
        tab10(3),  # red
        tab10(4),  # purple
        tab10(5),  # brown
        tab10(6),  # pink
    ]
    marker_styles = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]

    # Ideal line color: the orange entry from tab10 (index 1).
    ideal_color = tab10(1)

    # Runtime vs workers: Task runtime (solid), Env init (dashed), Farm init (dotted)
    fig_rt, ax_rt = plt.subplots(figsize=(10, 6))
    for idx, (label, workers, runtimes, _, env_inits, farm_inits) in enumerate(series):
        color = experiment_colors[idx % len(experiment_colors)]
        marker = marker_styles[idx % len(marker_styles)]

        # Task runtime
        ax_rt.plot(
            workers,
            runtimes,
            marker=marker,
            linestyle="-",
            color=color,
            label=f"Task runtime ({label})",
        )
        # Env init time (absolute)
        ax_rt.plot(
            workers,
            env_inits,
            linestyle="--",
            linewidth=1.3,
            color=color,
            label=f"Env init ({label})",
        )
        # Farm init time (delta)
        ax_rt.plot(
            workers,
            farm_inits,
            linestyle=":",
            linewidth=1.3,
            color=color,
            label=f"Farm init ({label})",
        )
    ax_rt.set_xscale("log", base=2)
    all_workers = sorted({int(w) for _, w, _, _, _, _ in series for w in w})
    ax_rt.set_xticks(all_workers)
    ax_rt.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax_rt.set_xlabel("Number of Workers")
    ax_rt.set_ylabel("Time (s)")
    ax_rt.set_title("Task Runtime, Env Init, and Farm Init vs Number of Workers")
    ax_rt.grid(True, alpha=0.3)
    ax_rt.legend(loc="best")
    fig_rt.tight_layout()
    fig_rt.savefig(output_dir / "runtime_vs_workers.png", dpi=600, bbox_inches="tight")
    plt.close(fig_rt)

    # Speedup vs workers
    fig_sp, ax_sp = plt.subplots(figsize=(10, 6))
    for idx, (label, workers, _, speedup, _, _) in enumerate(series):
        color = experiment_colors[idx % len(experiment_colors)]
        marker = marker_styles[idx % len(marker_styles)]
        ax_sp.plot(
            workers,
            speedup,
            marker=marker,
            linestyle="-",
            color=color,
            label=label,
        )

    if all_workers:
        min_worker = min(all_workers)
        ideal_speedup = [w / min_worker for w in all_workers]
        ax_sp.plot(
            all_workers,
            ideal_speedup,
            linestyle="--",
            color=ideal_color,
            label="Ideal",
        )
    ax_sp.set_xscale("log", base=2)
    ax_sp.set_xticks(all_workers)
    ax_sp.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax_sp.set_xlabel("Number of Workers")
    ax_sp.set_ylabel("Speedup (× baseline runtime)")
    ax_sp.set_title("Speedup vs Number of Workers")
    ax_sp.grid(True, alpha=0.3)
    ax_sp.legend(loc="best")
    fig_sp.tight_layout()
    fig_sp.savefig(output_dir / "speedup_vs_workers.png", dpi=600, bbox_inches="tight")
    plt.close(fig_sp)

    # Combined figure with subplots: runtime (left) and speedup (right)
    fig_comb, (ax_rt_c, ax_sp_c) = plt.subplots(1, 2, figsize=(11, 4.2))

    # Left subplot: runtime vs workers with stacked Env/Farm init bars (pooled)
    for idx, (label, workers, runtimes, _, _, _) in enumerate(series):
        color = experiment_colors[idx % len(experiment_colors)]
        marker = marker_styles[idx % len(marker_styles)]

        ax_rt_c.plot(
            workers,
            runtimes,
            marker=marker,
            linestyle="-",
            color=color,
            label=f"Task runtime ({label})",
        )

    if agg_init is not None:
        agg_workers, env_mean_agg, env_std_agg, farm_mean_agg, farm_std_agg = agg_init
        if agg_workers.size:
            bar_width = 0.3 * agg_workers

            ax_rt_c.bar(
                agg_workers,
                env_mean_agg,
                bar_width,
                label="Env init time",
                color=experiment_colors[0],
                alpha=0.3,
                edgecolor="black",
                linewidth=0.5,
            )
            ax_rt_c.bar(
                agg_workers,
                farm_mean_agg,
                bar_width,
                bottom=env_mean_agg,
                label="Farm init time",
                color=experiment_colors[2],
                alpha=0.25,
                edgecolor="black",
                linewidth=0.5,
            )

    ax_rt_c.set_xscale("log", base=2)
    ax_rt_c.set_xticks(all_workers)
    ax_rt_c.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax_rt_c.set_xlabel("Number of Workers")
    ax_rt_c.set_ylabel("Time (s)")
    ax_rt_c.set_title("(a) Runtime and Init Times vs Number of Workers")
    ax_rt_c.grid(True, alpha=0.3)

    # Right subplot: speedup vs workers with ideal line
    for idx, (label, workers, _, speedup, _, _) in enumerate(series):
        color = experiment_colors[idx % len(experiment_colors)]
        marker = marker_styles[idx % len(marker_styles)]
        ax_sp_c.plot(
            workers,
            speedup,
            marker=marker,
            linestyle="-",
            color=color,
            label=label,
        )

    if all_workers:
        min_worker = min(all_workers)
        ideal_speedup = [w / min_worker for w in all_workers]
        ax_sp_c.plot(
            all_workers,
            ideal_speedup,
            linestyle="--",
            color=ideal_color,
            label="Ideal",
        )

    ax_sp_c.set_xscale("log", base=2)
    ax_sp_c.set_xticks(all_workers)
    ax_sp_c.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax_sp_c.set_xlabel("Number of Workers")
    ax_sp_c.set_ylabel("Speedup")
    ax_sp_c.set_title("(b) Speedup vs Number of Workers")
    ax_sp_c.grid(True, alpha=0.3)

    # Legends inside each subplot, both with the same descriptive title.
    handles_rt, labels_rt = ax_rt_c.get_legend_handles_labels()
    ax_rt_c.legend(
        handles_rt,
        labels_rt,
        loc="best",
        framealpha=0.9,
    )

    handles_sp, labels_sp = ax_sp_c.get_legend_handles_labels()
    ax_sp_c.legend(
        handles_sp,
        labels_sp,
        loc="best",
        title="Tasks / seconds:",
        framealpha=0.9,
    )

    fig_comb.tight_layout()
    fig_comb.savefig(output_dir / "multi_run_scaling_metrics.png", dpi=600, bbox_inches="tight")
    plt.close(fig_comb)


def render_init_bar_plots(
    init_series: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    output_dir: Path,
) -> None:
    # No-op stub: kept for backward compatibility; bar plotting is now done
    # inside the combined runtime subplot using pooled statistics.
    return

def print_init_stats(
    agg_init: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None,
) -> None:
    if agg_init is None:
        return

    agg_workers, env_mean_agg, env_std_agg, farm_mean_agg, farm_std_agg = agg_init

    print("Pooled Env Init and Farm Init statistics (seconds) across all experiments:")

    for w, em, es, fm, fs in zip(
        agg_workers, env_mean_agg, env_std_agg, farm_mean_agg, farm_std_agg
    ):

        if np.isnan(em) and np.isnan(fm):
            continue

        print(
            "  workers={:d}: Env mean={:.3f}, std={:.3f}; Farm mean={:.3f}, std={:.3f}".format(
                int(w),
                float(em),
                float(es),
                float(fm),
                float(fs),
            )
        )


def infer_label_from_experiment(exp_name: str) -> str:
    m = re.search(r"_(\d+)t_(\d+)s", exp_name)
    if m:
        tasks, secs = m.groups()
        return f"{tasks}/{secs}"
    return exp_name


def main() -> None:
    args = parse_args()

    # Derive logs_farm root from the project layout: this script lives in
    # <project_root>/plots/, so logs are expected in <project_root>/logs_farm/.
    logs_root = PROJECT_ROOT / "logs_farm"

    cfg = get_config()
    last_n_raw = str(cfg.get("number_of_most_recent_logs", "None"))
    last_n: int | None
    if last_n_raw != "None" and last_n_raw.isdigit():
        last_n = int(last_n_raw)
    else:
        last_n = None

    labels: List[str]
    if args.labels is not None and len(args.labels) == len(args.experiments):
        labels = list(args.labels)
    else:
        labels = [infer_label_from_experiment(name) for name in args.experiments]

    series: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    # Aggregate Env/Farm init samples across all experiments per worker
    agg_env: Dict[int, List[float]] = {}
    agg_farm: Dict[int, List[float]] = {}

    for exp_name, label in zip(args.experiments, labels):
        log_dir = logs_root / exp_name
        per_runtime, per_env, per_farm = collect_metrics_for_experiment(log_dir, last_n)
        workers, runtimes, speedup = compute_runtime_and_speedup(per_runtime)
        # Average Env Init Time and Farm Init Time per worker for line plots
        env_means = np.full_like(workers, np.nan, dtype=float)
        farm_means = np.full_like(workers, np.nan, dtype=float)
        for i, w in enumerate(workers):
            w_int = int(w)
            env_vals = per_env.get(w_int, [])
            farm_vals = per_farm.get(w_int, [])
            if env_vals:
                env_means[i] = float(np.mean(env_vals))
            if farm_vals:
                farm_means[i] = float(np.mean(farm_vals))
        series.append((label, workers, runtimes, speedup, env_means, farm_means))

        # Accumulate raw samples for pooled statistics
        for w_int, vals in per_env.items():
            agg_env.setdefault(w_int, []).extend(vals)
        for w_int, vals in per_farm.items():
            agg_farm.setdefault(w_int, []).extend(vals)

    # Build pooled mean/std across all experiments per worker
    if agg_env or agg_farm:
        all_worker_keys = sorted(set(agg_env.keys()) | set(agg_farm.keys()))
        agg_workers = np.array(all_worker_keys, dtype=float)
        env_mean_agg = np.full_like(agg_workers, np.nan, dtype=float)
        env_std_agg = np.full_like(agg_workers, np.nan, dtype=float)
        farm_mean_agg = np.full_like(agg_workers, np.nan, dtype=float)
        farm_std_agg = np.full_like(agg_workers, np.nan, dtype=float)
        for i, w_int in enumerate(all_worker_keys):
            env_vals = agg_env.get(w_int, [])
            farm_vals = agg_farm.get(w_int, [])
            if env_vals:
                env_mean_agg[i] = float(np.mean(env_vals))
                env_std_agg[i] = (
                    float(np.std(env_vals, ddof=1)) if len(env_vals) > 1 else 0.0
                )
            if farm_vals:
                farm_mean_agg[i] = float(np.mean(farm_vals))
                farm_std_agg[i] = (
                    float(np.std(farm_vals, ddof=1)) if len(farm_vals) > 1 else 0.0
                )
        agg_init = (agg_workers, env_mean_agg, env_std_agg, farm_mean_agg, farm_std_agg)
    else:
        agg_init = None

    render_comparison_plots(series, args.output_dir, agg_init)
    print_init_stats(agg_init)


if __name__ == "__main__":  # pragma: no cover
    main()
