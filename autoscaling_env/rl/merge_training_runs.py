import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
# The project root is the directory that contains the 'autoscaling_env' package.
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from autoscaling_env.rl.utils import prepare_output_directory, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--prefix", type=str, default="sarsa_run")
    parser.add_argument("--output-prefix", type=str, default="sarsa_merged")
    parser.add_argument(
        "--title",
        type=str,
        default="SARSA Training (QoS, Workers, Scaling)",
    )
    parser.add_argument("runs", nargs="*", type=Path)
    return parser.parse_args()


def _load_episodes(path: Path) -> List[Dict[str, Any]]:
    data = path.read_text(encoding="utf-8")
    payload = json.loads(data)
    if isinstance(payload, dict):
        episodes = payload.get("episodes")
        if episodes is None:
            raise ValueError(f"File {path} does not contain 'episodes'")
    elif isinstance(payload, list):
        episodes = payload
    else:
        raise ValueError(f"Unsupported metrics structure in {path}")
    if not isinstance(episodes, list):
        raise ValueError(f"Episodes entry in {path} is not a list")
    return episodes


def _load_evaluations(path: Path) -> List[Dict[str, Any]]:
    data = path.read_text(encoding="utf-8")
    payload = json.loads(data)
    if isinstance(payload, dict):
        entries = payload.get("checkpoints") or payload.get("episodes") or []
    elif isinstance(payload, list):
        entries = payload
    else:
        entries = []
    if not isinstance(entries, list):
        return []
    return entries


def _parse_episode_summary_line(line: str) -> Dict[str, Any] | None:
    marker = "Episode "
    if (
        marker not in line
        or " | " not in line
        or "reward=" not in line
        or "mean_qos=" not in line
    ):
        return None
    try:
        after_marker = line.split(marker, 1)[1]
        episode_part, metrics_part = after_marker.split(" | ", 1)
    except ValueError:
        return None
    episode_text = episode_part.split("/", 1)[0].strip()
    try:
        episode_number = int(episode_text)
    except ValueError:
        return None
    metrics: Dict[str, Any] = {"episode": episode_number}
    for chunk in metrics_part.strip().split():
        if "=" not in chunk:
            continue
        key, value = chunk.split("=", 1)
        metrics[key] = value.rstrip("|,")
    if "reward" not in metrics or "mean_qos" not in metrics:
        return None
    return metrics


def _normalize_episode_from_summary(raw: Dict[str, Any]) -> Dict[str, Any]:
    episode_raw = raw.get("episode")
    if episode_raw is None:
        raise ValueError("Episode summary missing episode number")
    try:
        episode = int(episode_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid episode number: {episode_raw}") from exc
    result: Dict[str, Any] = {"episode": episode}

    def _assign_float(target_key: str, source_key: str) -> None:
        value = raw.get(source_key)
        if value is None:
            return
        try:
            result[target_key] = float(value)
        except (TypeError, ValueError):
            return

    _assign_float("total_reward", "reward")
    _assign_float("mean_qos", "mean_qos")
    if "final_qos" in raw:
        _assign_float("final_qos", "final_qos")
    elif "mean_qos" in result:
        result["final_qos"] = result["mean_qos"]

    _assign_float("steps", "steps")
    _assign_float("processed_tasks", "tasks")
    _assign_float("epsilon", "epsilon")
    _assign_float("scale_up_count", "scale_up")
    _assign_float("scale_down_count", "scale_down")
    _assign_float("noop_count", "noop")
    _assign_float("max_workers", "max_workers")
    _assign_float("mean_workers", "mean_workers")
    _assign_float("qos_violations", "qos_violations")
    _assign_float("unfinished_tasks", "unfinished")

    return result


def _build_episodes_from_log(log_path: Path) -> List[Dict[str, Any]]:
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")
    episodes: List[Dict[str, Any]] = []
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parsed = _parse_episode_summary_line(line)
            if parsed is None:
                continue
            try:
                normalized = _normalize_episode_from_summary(parsed)
            except ValueError:
                continue
            episodes.append(normalized)
    if not episodes:
        raise ValueError(f"Log file does not contain episode summaries: {log_path}")
    episodes.sort(key=lambda ep: int(ep.get("episode", 0)))
    return episodes


def _parse_eval_line(line: str) -> Dict[str, Any] | None:
    if "[EVAL]" not in line or "Episode " not in line or "|" not in line:
        return None
    try:
        after_eval = line.split("[EVAL]", 1)[1]
        after_episode = after_eval.split("Episode ", 1)[1]
        episode_part, metrics_part = after_episode.split("|", 1)
    except ValueError:
        return None
    episode_text = episode_part.strip().split()[0]
    try:
        episode = int(episode_text)
    except ValueError:
        return None
    values: Dict[str, Any] = {"episode": episode}
    for token in metrics_part.strip().split():
        if "=" not in token:
            continue
        key, raw_value = token.split("=", 1)
        raw_value = raw_value.rstrip(",")
        mean_str = raw_value
        std_str = None
        if "±" in raw_value:
            mean_str, std_str = raw_value.split("±", 1)

        def _strip_percent(text: str) -> str:
            return text[:-1] if text.endswith("%") else text

        mean_str = _strip_percent(mean_str)
        try:
            mean_val = float(mean_str)
        except (TypeError, ValueError):
            continue
        values[key] = mean_val
        if std_str is not None:
            std_str = _strip_percent(std_str)
            try:
                std_val = float(std_str)
            except (TypeError, ValueError):
                continue
            values[f"{key}_std"] = std_val
    return values


def _normalize_eval_from_values(values: Dict[str, Any]) -> Dict[str, Any] | None:
    episode_raw = values.get("episode")
    if episode_raw is None:
        return None
    try:
        episode = int(episode_raw)
    except (TypeError, ValueError):
        return None
    # Logs may use either "total_reward" or plain "reward".
    total_reward = values.get("total_reward", values.get("reward"))
    # Logs may use either "final_qos" (fraction or percent) or plain "qos" (percent).
    final_qos = values.get("final_qos", values.get("qos"))
    if total_reward is None or final_qos is None:
        return None
    try:
        mean_total_reward = float(total_reward)
    except (TypeError, ValueError):
        return None
    std_total_reward = 0.0
    if "total_reward_std" in values:
        try:
            std_total_reward = float(values["total_reward_std"])
        except (TypeError, ValueError):
            std_total_reward = 0.0
    try:
        final_qos_percent = float(final_qos)
    except (TypeError, ValueError):
        return None
    final_qos_std_percent = 0.0
    if "final_qos_std" in values:
        try:
            final_qos_std_percent = float(values["final_qos_std"])
        except (TypeError, ValueError):
            final_qos_std_percent = 0.0
    entry: Dict[str, Any] = {
        "episode": episode,
        "mean_total_reward": mean_total_reward,
        "std_total_reward": std_total_reward,
        "mean_final_qos": final_qos_percent / 100.0,
        "std_final_qos": final_qos_std_percent / 100.0,
    }
    scaling_mean = values.get("scaling")
    if scaling_mean is not None:
        try:
            entry["mean_scaling_actions"] = float(scaling_mean)
        except (TypeError, ValueError):
            pass
        scaling_std = values.get("scaling_std")
        if scaling_std is not None:
            try:
                entry["std_scaling_actions"] = float(scaling_std)
            except (TypeError, ValueError):
                pass
    # Max workers may be logged as either "max_workers" or "workers".
    max_workers_mean = values.get("max_workers", values.get("workers"))
    if max_workers_mean is not None:
        try:
            entry["mean_max_workers"] = float(max_workers_mean)
        except (TypeError, ValueError):
            pass
        max_workers_std = values.get("max_workers_std")
        if max_workers_std is not None:
            try:
                entry["std_max_workers"] = float(max_workers_std)
            except (TypeError, ValueError):
                pass
    mean_qos_percent = values.get("mean_qos")
    if mean_qos_percent is not None:
        try:
            entry["mean_mean_qos"] = float(mean_qos_percent) / 100.0
        except (TypeError, ValueError):
            pass
        mean_qos_std = values.get("mean_qos_std")
        if mean_qos_std is not None:
            try:
                entry["std_mean_qos"] = float(mean_qos_std) / 100.0
            except (TypeError, ValueError):
                pass
    return entry


def _build_evaluations_from_log(log_path: Path) -> List[Dict[str, Any]]:
    if not log_path.exists():
        return []
    evaluations: List[Dict[str, Any]] = []
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parsed = _parse_eval_line(line)
            if parsed is None:
                continue
            entry = _normalize_eval_from_values(parsed)
            if entry is None:
                continue
            evaluations.append(entry)
    evaluations.sort(key=lambda entry: int(entry.get("episode", 0)))
    return evaluations


def _ensure_run_metrics(
    run_dir: Path,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    metrics_path = run_dir / "training_metrics.json"
    eval_path = run_dir / "evaluation_metrics.json"
    log_path = run_dir / "logs" / "training.log"

    episodes: List[Dict[str, Any]] = []
    evaluations: List[Dict[str, Any]] = []

    if metrics_path.exists():
        episodes = _load_episodes(metrics_path)
        # If the existing metrics appear to be missing richer fields (e.g. mean_workers
        # for older runs), rebuild them from the log when possible.
        needs_upgrade = False
        if episodes:
            sample = episodes[0]
            # Keys that are useful for plotting QoS, workers, and scaling.
            important_keys = {
                "scale_up_count",
                "scale_down_count",
                "noop_count",
                "max_workers",
                "final_qos",
                "mean_workers",
            }
            if not important_keys.issubset(sample.keys()):
                needs_upgrade = True

        if needs_upgrade and log_path.exists():
            try:
                upgraded = _build_episodes_from_log(log_path)
            except Exception:  # noqa: BLE001
                # Fall back to the original metrics if log parsing fails.
                pass
            else:
                if upgraded:
                    episodes = upgraded
                    write_json({"episodes": episodes}, metrics_path)
    elif log_path.exists():
        episodes = _build_episodes_from_log(log_path)
        write_json({"episodes": episodes}, metrics_path)

    if episodes:
        if eval_path.exists():
            evaluations = _load_evaluations(eval_path)
            if not evaluations and log_path.exists():
                parsed_evals = _build_evaluations_from_log(log_path)
                if parsed_evals:
                    evaluations = parsed_evals
                    write_json({"checkpoints": evaluations}, eval_path)
        elif log_path.exists():
            evaluations = _build_evaluations_from_log(log_path)
            if evaluations:
                write_json({"checkpoints": evaluations}, eval_path)
            else:
                write_json({"checkpoints": []}, eval_path)

    return episodes, evaluations


def _smooth_series(values: np.ndarray, window: int) -> np.ndarray:
    if values.size == 0 or window <= 1:
        return values
    if not np.all(np.isfinite(values)):
        return values
    window = min(window, values.size)

    pad = window // 2
    padded = np.pad(values, pad, mode="edge")
    kernel = np.ones(window, dtype=np.float64) / window
    smoothed = np.convolve(padded, kernel, mode="valid")
    if smoothed.size > values.size:
        start = (smoothed.size - values.size) // 2
        smoothed = smoothed[start : start + values.size]
    return smoothed


def _render_qos_workers_scaling(
    episodes: List[Dict[str, Any]],
    evaluation_entries: List[Dict[str, Any]] | None,
    output_path: Path,
    title: str = "SARSA Training (QoS, Workers, Scaling)",
) -> Path:
    if not episodes:
        raise ValueError("No episode data provided for plotting")

    evaluation_entries = evaluation_entries or []

    episode_numbers = np.array(
        [float(ep.get("episode", 0.0)) for ep in episodes], dtype=float
    )

    # Detect boundaries between different source runs (phases), so we can
    # indicate where one run ends and the next begins in the merged plot.
    phase_boundaries: List[float] = []
    first_episode = episodes[0]
    prev_run = first_episode.get("source_run")
    if prev_run is not None:
        for idx in range(1, len(episodes)):
            current_run = episodes[idx].get("source_run")
            if current_run != prev_run:
                # Place the boundary at the last episode number of the
                # previous run (e.g., 50 when merging 50-episode and
                # 30-episode runs).
                boundary_raw = episodes[idx - 1].get("episode", idx)
                try:
                    boundary = float(boundary_raw)
                except (TypeError, ValueError):
                    boundary = float(idx)
                phase_boundaries.append(boundary)
                prev_run = current_run

    final_qos = np.array(
        [float(ep.get("final_qos", ep.get("mean_qos", np.nan))) for ep in episodes],
        dtype=float,
    )
    max_workers = np.array(
        [float(ep.get("max_workers", np.nan)) for ep in episodes], dtype=float
    )
    scale_up_counts = np.array(
        [float(ep.get("scale_up_count", np.nan)) for ep in episodes], dtype=float
    )
    scale_down_counts = np.array(
        [float(ep.get("scale_down_count", np.nan)) for ep in episodes], dtype=float
    )
    scaling_events = scale_up_counts + scale_down_counts

    smooth_window = 0
    if len(episodes) >= 3:
        smooth_window = min(15, max(3, len(episodes) // 10 or 3))

    qos_smooth = (
        _smooth_series(final_qos, smooth_window) if smooth_window else final_qos
    )
    workers_smooth = (
        _smooth_series(max_workers, smooth_window) if smooth_window else max_workers
    )
    scaling_smooth = (
        _smooth_series(scaling_events, smooth_window)
        if smooth_window
        else scaling_events
    )

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

    fig, (ax_qos, ax_workers, ax_scaling) = plt.subplots(
        3,
        1,
        sharex=True,
        figsize=(10, 8),
    )

    # Final QoS subplot (top)
    ax_qos.set_ylabel("Final QoS")
    qos_mask = np.isfinite(final_qos)
    if np.any(qos_mask):
        ax_qos.plot(
            episode_numbers[qos_mask],
            final_qos[qos_mask],
            color="tab:green",
            alpha=0.35,
            linewidth=1,
            label="Final QoS (raw)",
        )
        ax_qos.plot(
            episode_numbers[qos_mask],
            qos_smooth[qos_mask],
            color="tab:green",
            linewidth=2,
            label="Final QoS (moving avg)",
        )

    eval_qos = None
    eval_qos_std = None
    eval_episodes = None
    if evaluation_entries:
        eval_episodes = np.array(
            [entry["episode"] for entry in evaluation_entries], dtype=float
        )
        eval_qos = np.array(
            [entry.get("mean_final_qos", np.nan) for entry in evaluation_entries],
            dtype=float,
        )
        eval_qos_std = np.array(
            [entry.get("std_final_qos", 0.0) for entry in evaluation_entries],
            dtype=float,
        )
        if np.isfinite(eval_qos).any():
            ax_qos.plot(
                eval_episodes,
                eval_qos,
                color="tab:orange",
                marker="s",
                linewidth=1.5,
                label="Evaluation QoS (mean)",
            )
            if np.any(eval_qos_std > 0):
                ax_qos.fill_between(
                    eval_episodes,
                    eval_qos - eval_qos_std,
                    eval_qos + eval_qos_std,
                    color="tab:orange",
                    alpha=0.15,
                )

    ax_qos.grid(True, alpha=0.3)
    if ax_qos.get_legend_handles_labels()[0]:
        ax_qos.legend(loc="best", frameon=False)

    # Workers subplot (middle)
    ax_workers.set_ylabel("Workers")
    workers_mask = np.isfinite(max_workers)
    if np.any(workers_mask):
        ax_workers.plot(
            episode_numbers[workers_mask],
            max_workers[workers_mask],
            color="tab:olive",
            alpha=0.35,
            linewidth=1,
            label="Max Workers (raw)",
        )
        ax_workers.plot(
            episode_numbers[workers_mask],
            workers_smooth[workers_mask],
            color="tab:olive",
            linewidth=2,
            label="Max Workers (moving avg)",
        )

    eval_workers = None
    eval_workers_std = None
    if evaluation_entries and eval_episodes is not None:
        eval_workers = np.array(
            [entry.get("mean_max_workers", np.nan) for entry in evaluation_entries],
            dtype=float,
        )
        eval_workers_std = np.array(
            [entry.get("std_max_workers", 0.0) for entry in evaluation_entries],
            dtype=float,
        )
        if np.isfinite(eval_workers).any():
            ax_workers.plot(
                eval_episodes,
                eval_workers,
                color="tab:olive",
                marker="^",
                linewidth=1.5,
                label="Evaluation Max Workers (mean)",
            )
            if np.any(eval_workers_std > 0):
                ax_workers.fill_between(
                    eval_episodes,
                    eval_workers - eval_workers_std,
                    eval_workers + eval_workers_std,
                    color="tab:olive",
                    alpha=0.15,
                )

    ax_workers.grid(True, alpha=0.3)
    if ax_workers.get_legend_handles_labels()[0]:
        ax_workers.legend(loc="best", frameon=False)

    # Scaling actions subplot (bottom)
    ax_scaling.set_xlabel("Episode")
    ax_scaling.set_ylabel("Scaling Actions")
    scaling_mask = np.isfinite(scaling_events)
    if np.any(scaling_mask):
        ax_scaling.plot(
            episode_numbers[scaling_mask],
            scaling_events[scaling_mask],
            color="tab:red",
            alpha=0.35,
            linewidth=1.5,
            label="Scaling Actions (raw)",
        )
        ax_scaling.plot(
            episode_numbers[scaling_mask],
            scaling_smooth[scaling_mask],
            color="tab:red",
            linewidth=2.5,
            label="Scaling Actions (moving avg)",
        )

    eval_scaling = None
    eval_scaling_std = None
    if evaluation_entries and eval_episodes is not None:
        eval_scaling = np.array(
            [entry.get("mean_scaling_actions", np.nan) for entry in evaluation_entries],
            dtype=float,
        )
        eval_scaling_std = np.array(
            [entry.get("std_scaling_actions", 0.0) for entry in evaluation_entries],
            dtype=float,
        )
        if np.isfinite(eval_scaling).any():
            ax_scaling.plot(
                eval_episodes,
                eval_scaling,
                color="tab:red",
                marker="d",
                linewidth=1.5,
                label="Evaluation Scaling (mean)",
            )
            if np.any(eval_scaling_std > 0):
                ax_scaling.fill_between(
                    eval_episodes,
                    eval_scaling - eval_scaling_std,
                    eval_scaling + eval_scaling_std,
                    color="tab:red",
                    alpha=0.15,
                )

    ax_scaling.grid(True, alpha=0.3)
    if ax_scaling.get_legend_handles_labels()[0]:
        ax_scaling.legend(loc="best", frameon=False)

    # Add vertical dotted lines at phase boundaries (e.g., between the first
    # 50 episodes and the last 30 episodes when merging two DQN runs).
    if phase_boundaries:
        for boundary in phase_boundaries:
            for axis in (ax_qos, ax_workers, ax_scaling):
                axis.axvline(
                    boundary,
                    color="gray",
                    linestyle="--",
                    linewidth=1.2,
                    alpha=0.7,
                )

    fig.suptitle(title, fontsize=20)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()

    runs_dir = args.runs_dir.resolve()
    if not runs_dir.exists() or not runs_dir.is_dir():
        raise SystemExit(f"Runs directory not found: {runs_dir}")

    if args.runs:
        run_dirs = [p.resolve() for p in args.runs]
    else:
        run_dirs = [
            p
            for p in runs_dir.iterdir()
            if p.is_dir() and p.name.startswith(args.prefix)
        ]
        run_dirs.sort()

    merged_dir = prepare_output_directory(runs_dir, prefix=args.output_prefix)

    merged_episodes: List[Dict[str, Any]] = []
    merged_eval: List[Dict[str, Any]] = []
    episode_offset = 0
    used_run_dirs: List[Path] = []

    for run_dir in run_dirs:
        episodes, evaluations = _ensure_run_metrics(run_dir)
        if not episodes:
            continue
        used_run_dirs.append(run_dir)
        run_start = episode_offset + 1

        for index, episode in enumerate(episodes, start=1):
            episode_copy: Dict[str, Any] = dict(episode)
            local_episode_raw = episode_copy.get("episode", index)
            try:
                local_episode = int(local_episode_raw)
            except (TypeError, ValueError):
                local_episode = index
            episode_copy["episode"] = local_episode + episode_offset
            if "source_run" not in episode_copy:
                episode_copy["source_run"] = run_dir.name
            merged_episodes.append(episode_copy)

        episode_offset = merged_episodes[-1]["episode"]

        if evaluations:
            sorted_evaluations = sorted(
                evaluations, key=lambda entry: int(entry.get("episode", 0))
            )
            for entry in sorted_evaluations:
                entry_copy: Dict[str, Any] = dict(entry)
                local_episode_eval = entry_copy.get("episode")
                try:
                    local_episode_eval_int = int(local_episode_eval)
                except (TypeError, ValueError):
                    continue
                entry_copy["episode"] = local_episode_eval_int + run_start - 1
                if "source_run" not in entry_copy:
                    entry_copy["source_run"] = run_dir.name
                merged_eval.append(entry_copy)

    if not merged_episodes:
        raise SystemExit("No episodes found in any runs (check logs and metrics files)")

    metrics_output = merged_dir / "training_metrics.json"
    write_json({"episodes": merged_episodes}, metrics_output)

    eval_output: Path | None = None
    if merged_eval:
        eval_output = merged_dir / "evaluation_metrics.json"
        write_json({"checkpoints": merged_eval}, eval_output)

    plots_dir = merged_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    curves_path = plots_dir / "training_curves.png"
    _render_qos_workers_scaling(merged_episodes, merged_eval, curves_path, title=args.title)

    print(f"Merged {len(used_run_dirs)} runs into {merged_dir}")
    print(f"Total episodes: {len(merged_episodes)}")
    print(f"Training metrics: {metrics_output}")
    if eval_output is not None:
        print(f"Evaluation metrics: {eval_output}")
    print(f"Metrics plot (QoS, workers, scaling): {curves_path}")


if __name__ == "__main__":
    main()
