#!/usr/bin/env python3
"""Compare reduced vs full quadratic timing models on existing calibration data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=Path(__file__).resolve().parent / "calibration_results.json",
        help="Path to calibration_results.json produced by calibrate_direct.py",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=None,
        help="Optional path to save comparison plot (defaults next to results file)",
    )
    return parser.parse_args()


def load_calibration_data(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Calibration results not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    episodes = payload.get("results")
    if not episodes:
        raise ValueError("Calibration file missing 'results' entries")

    sizes = np.array([entry["image_size"] for entry in episodes], dtype=float)
    times = np.array([entry["mean"] for entry in episodes], dtype=float)
    return sizes, times


def fit_models(sizes: np.ndarray, times: np.ndarray) -> Dict[str, Dict[str, float]]:
    def model_reduced(x: np.ndarray, a: float, b: float) -> np.ndarray:
        return a * x**2 + b

    def model_full(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
        return a * x**2 + b * x + c

    params_reduced, _ = curve_fit(model_reduced, sizes, times)
    params_full, _ = curve_fit(model_full, sizes, times)

    preds_reduced = model_reduced(sizes, *params_reduced)
    preds_full = model_full(sizes, *params_full)

    ss_tot = np.sum((times - np.mean(times)) ** 2)
    ss_res_reduced = np.sum((times - preds_reduced) ** 2)
    ss_res_full = np.sum((times - preds_full) ** 2)

    r2_reduced = 1 - ss_res_reduced / ss_tot
    r2_full = 1 - ss_res_full / ss_tot

    return {
        "reduced": {
            "params": params_reduced,
            "preds": preds_reduced,
            "rss": ss_res_reduced,
            "r2": r2_reduced,
        },
        "full": {
            "params": params_full,
            "preds": preds_full,
            "rss": ss_res_full,
            "r2": r2_full,
        },
    }


def print_summary(sizes: np.ndarray, times: np.ndarray, metrics: Dict[str, Dict[str, float]]) -> None:
    reduced = metrics["reduced"]
    full = metrics["full"]

    a_r, b_r = reduced["params"]
    a_f, b_f, c_f = full["params"]

    print("\n=== Quadratic Model Comparison ===")
    print("Reduced model: time = a * size^2 + b")
    print(f"  a = {a_r:.6e}, b = {b_r:.6e}")
    print(f"  R^2 = {reduced['r2']:.6f}, RSS = {reduced['rss']:.6e}")

    print("\nFull model: time = a * size^2 + b * size + c")
    print(f"  a = {a_f:.6e}, b = {b_f:.6e}, c = {c_f:.6e}")
    print(f"  R^2 = {full['r2']:.6f}, RSS = {full['rss']:.6e}")

    delta_rss = reduced["rss"] - full["rss"]
    print(f"\nΔRSS (reduced - full): {delta_rss:.6e}")
    print("If ΔRSS is near zero and R^2 values match, dropping the linear term is justified.")

    print("\nPer-size absolute % error (reduced model):")
    for size, actual, pred in zip(sizes, times, reduced["preds"]):
        err_pct = abs(pred - actual) / actual * 100.0
        print(f"  {int(size)} px: actual={actual*1000:.3f} ms, predicted={pred*1000:.3f} ms, error={err_pct:.2f}%")


def plot_comparison(
    sizes: np.ndarray,
    times: np.ndarray,
    metrics: Dict[str, Dict[str, float]],
    output_path: Path,
) -> None:
    x_vals = np.linspace(sizes.min(), sizes.max(), 300)

    def reduced_curve(x: np.ndarray) -> np.ndarray:
        a, b = metrics["reduced"]["params"]
        return a * x**2 + b

    def full_curve(x: np.ndarray) -> np.ndarray:
        a, b, c = metrics["full"]["params"]
        return a * x**2 + b * x + c

    # Convert to milliseconds for readability
    times_ms = times * 1000.0
    reduced_curve_ms = reduced_curve(x_vals) * 1000.0
    full_curve_ms = full_curve(x_vals) * 1000.0
    reduced_preds_ms = metrics["reduced"]["preds"] * 1000.0
    full_preds_ms = metrics["full"]["preds"] * 1000.0

    fig, (ax_curve, ax_resid) = plt.subplots(
        2,
        1,
        figsize=(11, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
    )

    ax_curve.scatter(
        sizes,
        times_ms,
        color="black",
        label="Measured",
        s=60,
        edgecolor="white",
        linewidth=0.6,
        zorder=3,
    )
    ax_curve.plot(x_vals, reduced_curve_ms, color="tab:blue", label="Reduced Fit", linewidth=2.0)
    ax_curve.plot(
        x_vals,
        full_curve_ms,
        color="tab:orange",
        linestyle="--",
        linewidth=2.0,
        label="Full Fit",
    )
    ax_curve.set_title("Calibration Timing Models: Reduced vs Full Quadratic", fontsize=14)
    ax_curve.set_ylabel("Processing Time (ms)")
    ax_curve.grid(True, alpha=0.3)
    ax_curve.legend(loc="best")

    residual_reduced_ms = reduced_preds_ms - times_ms
    residual_full_ms = full_preds_ms - times_ms

    ax_resid.axhline(0.0, color="gray", linestyle=":", linewidth=1.0)
    ax_resid.plot(
        sizes,
        residual_reduced_ms,
        marker="o",
        color="tab:blue",
        label="Reduced",
        linewidth=1.2,
        markersize=6,
    )
    ax_resid.plot(
        sizes,
        residual_full_ms,
        marker="s",
        color="tab:orange",
        label="Full",
        linewidth=1.2,
        markersize=6,
    )
    ax_resid.set_xlabel("Image Size (pixels)")
    ax_resid.set_ylabel("Residual (ms)")
    ax_resid.grid(True, alpha=0.25)
    ax_resid.legend(loc="lower right")

    reduced = metrics["reduced"]
    full = metrics["full"]
    delta_rss = reduced["rss"] - full["rss"]
    summary_lines = [
        "Fit quality:",
        "  Reduced → R² = {:.6f}, RSS = {:.3e}".format(reduced["r2"], reduced["rss"]),
        "  Full     → R² = {:.6f}, RSS = {:.3e}".format(full["r2"], full["rss"]),
        "  ΔRSS (reduced − full) = {:.3e}".format(delta_rss),
    ]
    ax_curve.annotate(
        "\n".join(summary_lines),
        xy=(0.6, 0.28),
        xycoords="axes fraction",
        fontsize=10,
        ha="left",
        va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffffff", alpha=0.92, edgecolor="#cccccc"),
    )

    coeff_lines = [
        "Coefficients:",
        "  Reduced → a = {:.3e}, b = {:.3e}".format(reduced["params"][0], reduced["params"][1]),
        "  Full     → a = {:.3e}, b = {:.3e}, c = {:.3e}".format(
            full["params"][0], full["params"][1], full["params"][2]
        ),
    ]
    ax_curve.annotate(
        "\n".join(coeff_lines),
        xy=(0.05, 0.55),
        xycoords="axes fraction",
        fontsize=10,
        ha="left",
        va="center",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffffff", alpha=0.92, edgecolor="#cccccc"),
    )

    fig.align_labels()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved to {output_path}")


def main() -> None:
    args = parse_args()
    sizes, times = load_calibration_data(args.results)
    metrics = fit_models(sizes, times)
    print_summary(sizes, times, metrics)

    if args.plot is None:
        args.plot = args.results.parent / "calibration_model_comparison.png"
    plot_comparison(sizes, times, metrics, args.plot)


if __name__ == "__main__":
    main()
