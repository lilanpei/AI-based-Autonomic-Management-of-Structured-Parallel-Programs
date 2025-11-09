#!/usr/bin/env python3
"""Plot SARSA training metrics stored in training_metrics.json or training.log."""

import argparse
from pathlib import Path

import json
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
try:
    from scipy.ndimage import uniform_filter1d
except ImportError:  # pragma: no cover - fallback when SciPy missing
    uniform_filter1d = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "metrics",
        type=Path,
        help="Path to training_metrics.json produced by train_sarsa.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path for the generated plot (defaults to sibling plots/training_curves.png)",
    )
    parser.add_argument(
        "--evaluation",
        type=Path,
        default=None,
        help="Optional evaluation_metrics.json path to overlay checkpoint evaluation curves",
    )
    return parser.parse_args()


def _normalize_episode(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        episode_number = int(payload["episode"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Episode entry is missing a valid 'episode' number") from exc

    reward_raw = None
    for key in ("total_reward", "reward"):
        if key in payload and payload[key] is not None:
            reward_raw = payload[key]
            break
    if reward_raw is None:
        raise ValueError(f"Episode {episode_number} is missing 'total_reward'/'reward'")

    try:
        reward = float(reward_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Episode {episode_number} has invalid reward value: {reward_raw}") from exc

    mean_qos_raw = payload.get("mean_qos")
    if mean_qos_raw is None:
        raise ValueError(f"Episode {episode_number} is missing 'mean_qos'")
    try:
        mean_qos = float(mean_qos_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Episode {episode_number} has invalid mean_qos value: {mean_qos_raw}") from exc

    epsilon_raw = payload.get("epsilon")
    epsilon: float | None
    if epsilon_raw is None:
        epsilon = None
    else:
        try:
            epsilon = float(epsilon_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Episode {episode_number} has invalid epsilon value: {epsilon_raw}") from exc

    optional_float_keys = (
        "scale_up_count",
        "scale_down_count",
        "max_workers",
        "noop_count",
        "steps",
        "processed_tasks",
        "qos_violations",
        "unfinished_tasks",
        "final_qos",
    )

    result: dict[str, Any] = {
        "episode": episode_number,
        "total_reward": reward,
        "mean_qos": mean_qos,
        "epsilon": epsilon,
    }

    for key in optional_float_keys:
        if key not in payload or payload[key] is None:
            continue
        try:
            result[key] = float(payload[key])
        except (TypeError, ValueError):
            continue

    return result


def _load_from_json(path: Path) -> list[dict[str, Any]]:
    data = path.read_text(encoding="utf-8")
    payload = json.loads(data)

    if isinstance(payload, dict):
        episodes_raw = payload.get("episodes")
        if episodes_raw is None:
            raise ValueError("JSON metrics file must contain an 'episodes' array")
    elif isinstance(payload, list):
        episodes_raw = payload
    else:
        raise ValueError("JSON metrics file must be either an object with 'episodes' or a list of episodes")

    return [_normalize_episode(ep) for ep in episodes_raw]


def _parse_episode_line(line: str) -> dict[str, Any] | None:
    marker = "Episode "
    if marker not in line:
        return None

    try:
        after_marker = line.split(marker, 1)[1]
        episode_part, metrics_part = after_marker.split(" | ", 1)
    except ValueError:
        return None

    try:
        episode_number = int(episode_part.split("/", 1)[0])
    except ValueError:
        return None

    metrics: dict[str, Any] = {"episode": episode_number}
    for chunk in metrics_part.strip().split():
        if "=" not in chunk:
            continue
        key, value = chunk.split("=", 1)
        metrics[key] = value.rstrip("|,")

    required_keys = {"reward", "mean_qos"}
    if not required_keys.issubset(metrics.keys()):
        return None

    return metrics


def _load_from_log(path: Path) -> list[dict[str, Any]]:
    episodes: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parsed = _parse_episode_line(line)
            if parsed is None:
                continue
            episodes.append(_normalize_episode(parsed))

    if not episodes:
        raise ValueError("Log file does not contain parsable episode summaries")

    return episodes


def load_episodes(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return _load_from_json(path)

    if suffix in {".log", ".txt"}:
        return _load_from_log(path)

    try:
        return _load_from_json(path)
    except Exception:
        return _load_from_log(path)


def _normalize_evaluation_entry(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        episode = int(payload["episode"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Evaluation entry missing valid 'episode'") from exc

    def _require_float(key: str) -> float:
        value = payload.get(key)
        if value is None:
            raise ValueError(f"Evaluation entry for episode {episode} missing '{key}'")
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Evaluation entry for episode {episode} has invalid '{key}' value: {value}"
            ) from exc

    entry = {
        "episode": episode,
        "mean_total_reward": _require_float("mean_total_reward"),
        "mean_final_qos": _require_float("mean_final_qos"),
    }

    for key in ("std_total_reward", "std_final_qos"):
        raw = payload.get(key)
        entry[key] = float(raw) if raw is not None else 0.0

    for optional_key in (
        "mean_scaling_actions",
        "std_scaling_actions",
        "mean_max_workers",
        "std_max_workers",
    ):
        raw = payload.get(optional_key)
        if raw is not None:
            try:
                entry[optional_key] = float(raw)
            except (TypeError, ValueError):
                continue

    return entry


def load_evaluation(path: Path) -> list[dict[str, Any]]:
    data = path.read_text(encoding="utf-8")
    payload = json.loads(data)

    if isinstance(payload, dict):
        entries = payload.get("checkpoints") or payload.get("episodes")
        if entries is None:
            raise ValueError("Evaluation metrics must contain 'checkpoints' or 'episodes'")
    elif isinstance(payload, list):
        entries = payload
    else:
        raise ValueError("Evaluation metrics must be a list or object with 'checkpoints'")

    return [_normalize_evaluation_entry(entry) for entry in entries]


def render_training_curves(
    episodes: list[dict[str, Any]],
    evaluation_entries: list[dict[str, Any]] | None = None,
    output_path: Path | None = None,
    title: str = "SARSA Training Curves",
) -> Path:
    if not episodes:
        raise ValueError("No episode data provided for plotting")

    evaluation_entries = evaluation_entries or []

    episode_numbers = np.array([ep["episode"] for ep in episodes], dtype=float)
    total_rewards = np.array([ep["total_reward"] for ep in episodes], dtype=float)
    final_qos = np.array(
        [
            float(ep["final_qos"]) if "final_qos" in ep and ep["final_qos"] is not None else ep["mean_qos"]
            for ep in episodes
        ],
        dtype=float,
    )

    def _smooth(values: np.ndarray, window: int) -> np.ndarray:
        if values.size == 0 or window <= 1:
            return values
        if not np.all(np.isfinite(values)):
            return values
        window = min(window, values.size)

        if uniform_filter1d is not None:
            return uniform_filter1d(values, size=window, mode="nearest")

        # Fallback: pad with edge values before convolution to mimic "nearest" handling.
        pad = window // 2
        padded = np.pad(values, pad, mode="edge")
        kernel = np.ones(window, dtype=np.float64) / window
        smoothed = np.convolve(padded, kernel, mode="valid")
        # For even window sizes, "valid" leaves one extra sample; trim to original length.
        if smoothed.size > values.size:
            start = (smoothed.size - values.size) // 2
            smoothed = smoothed[start:start + values.size]
        return smoothed

    def _auto_limits(
        *series: np.ndarray,
        margin_ratio: float = 0.1,
        low_quantile: float = 0.02,
        high_quantile: float = 0.98,
        floor_zero: bool = False,
        integer: bool = False,
        min_span: float = 0.0,
    ) -> tuple[float, float] | None:
        data_segments = []
        for arr in series:
            if arr is None:
                continue
            flat = np.asarray(arr, dtype=float).ravel()
            if flat.size:
                data_segments.append(flat[np.isfinite(flat)])
        if not data_segments:
            return None
        filtered_segments = [seg for seg in data_segments if seg.size]
        if not filtered_segments:
            return None
        data = np.concatenate(filtered_segments)
        if data.size == 0:
            return None
        lo = np.quantile(data, low_quantile)
        hi = np.quantile(data, high_quantile)
        if np.isclose(lo, hi):
            span = max(min_span, max(1.0, abs(lo)) * margin_ratio * 2)
        else:
            span = hi - lo
            if span < min_span:
                span = min_span
        margin = span * margin_ratio
        lo_adj = lo - margin
        hi_adj = hi + margin

        if floor_zero:
            lo_adj = max(0.0, lo_adj)

        if integer:
            lo_adj = float(np.floor(lo_adj))
            hi_adj = float(np.ceil(hi_adj))
            if lo_adj == hi_adj:
                hi_adj = lo_adj + 1.0

        return lo_adj, hi_adj

    smooth_window = 0
    if len(episodes) >= 3:
        smooth_window = min(15, max(3, len(episodes) // 10 or 3))

    rewards_smooth = _smooth(total_rewards, smooth_window) if smooth_window else total_rewards
    qos_smooth = _smooth(final_qos, smooth_window) if smooth_window else final_qos

    max_workers = np.array([float(ep.get("max_workers", np.nan)) for ep in episodes], dtype=float)
    scale_up_counts = np.array([float(ep.get("scale_up_count", np.nan)) for ep in episodes], dtype=float)
    scale_down_counts = np.array([float(ep.get("scale_down_count", np.nan)) for ep in episodes], dtype=float)
    scaling_events = scale_up_counts + scale_down_counts
    scaling_smooth = _smooth(scaling_events, smooth_window) if smooth_window else scaling_events
    workers_smooth = _smooth(max_workers, smooth_window) if smooth_window else max_workers

    fig, (ax_reward, ax_qos, ax_workers, ax_scaling) = plt.subplots(
        4,
        1,
        sharex=True,
        figsize=(12, 14),
        constrained_layout=False,
    )
    fig.suptitle(title)

    # Reward subplot
    ax_reward.set_ylabel("Total Reward", color="tab:blue")
    ax_reward.plot(
        episode_numbers,
        total_rewards,
        color="tab:blue",
        alpha=0.35,
        linewidth=1,
        label="Total Reward (raw)",
    )
    ax_reward.plot(
        episode_numbers,
        rewards_smooth,
        color="tab:blue",
        linewidth=2,
        label="Total Reward (smoothed)",
    )
    ax_reward.tick_params(axis="y", labelcolor="tab:blue")
    ax_reward.grid(True, alpha=0.3)

    # QoS subplot
    ax_qos.set_ylabel("Final QoS", color="tab:green")
    ax_qos.plot(
        episode_numbers,
        final_qos,
        color="tab:green",
        linestyle="--",
        alpha=0.35,
        linewidth=1,
        label="Final QoS (raw)",
    )
    ax_qos.plot(
        episode_numbers,
        qos_smooth,
        color="tab:green",
        linestyle="--",
        linewidth=2,
        label="Final QoS (smoothed)",
    )
    ax_qos.tick_params(axis="y", labelcolor="tab:green")
    ax_qos.grid(True, alpha=0.3)

    # Max workers subplot
    ax_workers.set_ylabel("Max Workers")
    workers_mask = np.isfinite(max_workers)
    if np.any(workers_mask):
        ax_workers.plot(
            episode_numbers[workers_mask],
            max_workers[workers_mask],
            color="tab:olive",
            linewidth=1.5,
            alpha=0.35,
            label="Max Workers (raw)",
        )
        if smooth_window:
            smooth_workers_mask = np.isfinite(workers_smooth)
            ax_workers.plot(
                episode_numbers[smooth_workers_mask],
                workers_smooth[smooth_workers_mask],
                color="tab:olive",
                linewidth=2.5,
                label=f"Max Workers (moving avg {smooth_window})",
            )
    ax_workers.grid(True, alpha=0.3)

    # Scaling actions subplot
    ax_scaling.set_xlabel("Episode")
    ax_scaling.set_ylabel("Scaling Actions")
    total_mask = np.isfinite(scaling_events)

    if np.any(total_mask):
        ax_scaling.plot(
            episode_numbers[total_mask],
            scaling_events[total_mask],
            color="tab:red",
            linewidth=1.5,
            alpha=0.35,
            label="Scaling Actions (raw)",
        )
        if smooth_window:
            smooth_mask = np.isfinite(scaling_smooth)
            ax_scaling.plot(
                episode_numbers[smooth_mask],
                scaling_smooth[smooth_mask],
                color="tab:red",
                linewidth=2.5,
                label=f"Scaling Actions (moving avg {smooth_window})",
            )
    ax_scaling.grid(True, alpha=0.3)

    axes = [ax_reward, ax_qos, ax_workers, ax_scaling]

    eval_rewards = None
    eval_qos = None
    eval_scaling = None
    eval_max_workers = None
    if evaluation_entries:
        eval_episodes = np.array([entry["episode"] for entry in evaluation_entries], dtype=float)
        eval_rewards = np.array([entry["mean_total_reward"] for entry in evaluation_entries], dtype=float)
        eval_rewards_std = np.array([entry.get("std_total_reward", 0.0) for entry in evaluation_entries], dtype=float)
        eval_qos = np.array([entry["mean_final_qos"] for entry in evaluation_entries], dtype=float)
        eval_qos_std = np.array([entry.get("std_final_qos", 0.0) for entry in evaluation_entries], dtype=float)
        eval_scaling = np.array(
            [entry.get("mean_scaling_actions", np.nan) for entry in evaluation_entries], dtype=float
        )
        eval_scaling_std = np.array(
            [entry.get("std_scaling_actions", 0.0) for entry in evaluation_entries], dtype=float
        )
        eval_max_workers = np.array(
            [entry.get("mean_max_workers", np.nan) for entry in evaluation_entries], dtype=float
        )
        eval_max_workers_std = np.array(
            [entry.get("std_max_workers", 0.0) for entry in evaluation_entries], dtype=float
        )

        ax_reward.plot(
            eval_episodes,
            eval_rewards,
            color="tab:purple",
            marker="o",
            linewidth=1.5,
            label="Evaluation Reward (mean)",
        )
        if np.any(eval_rewards_std > 0):
            ax_reward.fill_between(
                eval_episodes,
                eval_rewards - eval_rewards_std,
                eval_rewards + eval_rewards_std,
                color="tab:purple",
                alpha=0.15,
            )

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

        if np.isfinite(eval_max_workers).any():
            ax_workers.plot(
                eval_episodes,
                eval_max_workers,
                color="tab:olive",
                marker="^",
                linewidth=1.5,
                label="Evaluation Max Workers (mean)",
            )
            if np.any(eval_max_workers_std > 0):
                ax_workers.fill_between(
                    eval_episodes,
                    eval_max_workers - eval_max_workers_std,
                    eval_max_workers + eval_max_workers_std,
                    color="tab:olive",
                    alpha=0.15,
                )

        if np.isfinite(eval_scaling).any():
            ax_scaling.plot(
                eval_episodes,
                eval_scaling,
                color="tab:red",
                marker="d",
                linewidth=1.5,
                label="Evaluation Scaling Actions (mean)",
            )
            if np.any(eval_scaling_std > 0):
                ax_scaling.fill_between(
                    eval_episodes,
                    eval_scaling - eval_scaling_std,
                    eval_scaling + eval_scaling_std,
                    color="tab:red",
                    alpha=0.15,
                )

    if ax_reward.get_legend_handles_labels()[0]:
        ax_reward.legend(loc="best", frameon=False)
    if ax_qos.get_legend_handles_labels()[0]:
        ax_qos.legend(loc="best", frameon=False)
    if ax_workers.get_legend_handles_labels()[0]:
        ax_workers.legend(loc="best", frameon=False)
    if ax_scaling.get_legend_handles_labels()[0]:
        ax_scaling.legend(loc="best", frameon=False)

    ax_reward.set_ylim(-200.0, 200.0)

    qos_limits = _auto_limits(
        final_qos,
        qos_smooth,
        eval_qos,
        margin_ratio=0.05,
        low_quantile=0.02,
        high_quantile=0.98,
        min_span=0.05,
    )
    if qos_limits is not None:
        lower = max(0.6, qos_limits[0])
        upper = max(1.1, qos_limits[1])
        ax_qos.set_ylim(lower, upper)
    ax_qos.yaxis.set_major_locator(MaxNLocator(nbins=6, prune="both"))

    ax_workers.set_ylim(12.0, 32.0)
    ax_workers.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))

    ax_scaling.set_ylim(12.0, 32.0)
    ax_scaling.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))

    fig.tight_layout(rect=(0, 0, 1, 0.97))

    if output_path is None:
        raise ValueError("output_path must be provided when rendering plots programmatically")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()

    if not args.metrics.exists():
        raise FileNotFoundError(f"Metrics file not found: {args.metrics}")

    episodes = load_episodes(args.metrics)

    if not episodes:
        raise ValueError("Metrics file does not contain any episode data")

    eval_path = args.evaluation
    if eval_path is None:
        candidate = args.metrics.parent / "evaluation_metrics.json"
        if candidate.exists():
            eval_path = candidate

    evaluation_entries: list[dict[str, Any]] = []
    if eval_path is not None:
        if not eval_path.exists():
            print(f"[WARN] Evaluation metrics file not found: {eval_path}")
        else:
            try:
                evaluation_entries = load_evaluation(eval_path)
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] Unable to load evaluation metrics from {eval_path}: {exc}")

    if args.output:
        output_path = args.output
    else:
        output_path = args.metrics.parent / "plots" / "training_curves.png"
        output_path.parent.mkdir(exist_ok=True)

    render_training_curves(episodes, evaluation_entries, output_path)
    print(f"Plot saved to {output_path}")


if __name__ == "__main__":
    main()
