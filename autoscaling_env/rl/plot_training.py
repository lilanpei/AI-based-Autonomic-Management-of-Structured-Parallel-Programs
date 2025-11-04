#!/usr/bin/env python3
"""Plot SARSA training metrics stored in training_metrics.json or training.log."""

import argparse
from pathlib import Path

import json
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


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

    return {
        "episode": episode_number,
        "total_reward": reward,
        "mean_qos": mean_qos,
        "epsilon": epsilon,
    }


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

    episode_numbers = np.array([ep["episode"] for ep in episodes])
    total_rewards = np.array([ep["total_reward"] for ep in episodes])
    mean_qos = np.array([ep["mean_qos"] for ep in episodes])
    epsilon_values = [ep.get("epsilon") for ep in episodes]
    has_epsilon = all(value is not None for value in epsilon_values)
    epsilons = np.array(epsilon_values, dtype=float) if has_epsilon else None

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

    if evaluation_entries:
        eval_episodes = np.array([entry["episode"] for entry in evaluation_entries], dtype=float)
        eval_rewards = np.array([entry["mean_total_reward"] for entry in evaluation_entries], dtype=float)
        eval_rewards_std = np.array([entry.get("std_total_reward", 0.0) for entry in evaluation_entries], dtype=float)
        eval_qos = np.array([entry["mean_final_qos"] for entry in evaluation_entries], dtype=float)
        eval_qos_std = np.array([entry.get("std_final_qos", 0.0) for entry in evaluation_entries], dtype=float)

        ax1.plot(
            eval_episodes,
            eval_rewards,
            color="tab:purple",
            marker="o",
            linewidth=1.5,
            label="Evaluation Reward (mean)",
        )
        if np.any(eval_rewards_std > 0):
            ax1.fill_between(
                eval_episodes,
                eval_rewards - eval_rewards_std,
                eval_rewards + eval_rewards_std,
                color="tab:purple",
                alpha=0.15,
            )

        ax2.plot(
            eval_episodes,
            eval_qos,
            color="tab:orange",
            marker="s",
            linewidth=1.5,
            label="Evaluation QoS (mean)",
        )
        if np.any(eval_qos_std > 0):
            ax2.fill_between(
                eval_episodes,
                eval_qos - eval_qos_std,
                eval_qos + eval_qos_std,
                color="tab:orange",
                alpha=0.15,
            )

    axes = [ax1, ax2]
    if epsilons is not None:
        ax3 = ax1.twinx()
        ax3.spines["right"].set_position(("outward", 60))
        ax3.set_ylabel("Epsilon", color="tab:red")
        ax3.plot(episode_numbers, epsilons, color="tab:red", linestyle=":", label="Epsilon")
        ax3.tick_params(axis="y", labelcolor="tab:red")
        axes.append(ax3)

    handles, labels = [], []
    for axis in axes:
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
