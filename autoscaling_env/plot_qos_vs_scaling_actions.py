import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, default=None)
    parser.add_argument("--max-runs", type=int, default=10)
    parser.add_argument("--agents", type=str, default=None)
    parser.add_argument("--save", type=str, default="qos_vs_scaling_scatter.png")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    # Default to the latest comparison run that includes DQN/SARSA/baselines.
    default_json = script_dir / "runs/comparison/compare_20251119_112155/aggregated_results.json"
    json_path = Path(args.json) if args.json else default_json

    with open(json_path, "r") as f:
        data = json.load(f)

    present_agents = list(data.get("agents", {}).keys())

    preferred = ["DQN", "SARSA", "ReactiveAverage", "ReactiveMaximum"]
    if args.agents:
        selected = [a.strip() for a in args.agents.split(",") if a.strip()]
    else:
        selected = [a for a in preferred if a in present_agents]
        if not selected:
            selected = present_agents

    colors = {
        "DQN": "#d62728",
        "SARSA": "#1f77b4",
        "ReactiveAverage": "#ff7f0e",
        "ReactiveMaximum": "#2ca02c",
    }
    markers = {"DQN": "D", "SARSA": "o", "ReactiveAverage": "s", "ReactiveMaximum": "^"}

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

    plt.figure(figsize=(7, 5), dpi=140)

    plotted = 0
    for agent in selected:
        agent_data = data.get("agents", {}).get(agent)
        if not agent_data:
            continue

        records = agent_data.get("records", [])[: max(0, args.max_runs)]
        xs, ys = [], []
        for r in records:
            s = r.get("summary", {})
            if "scaling_actions" in s and "final_qos_total" in s:
                xs.append(s["scaling_actions"])
                ys.append(s["final_qos_total"])
        if xs and ys:
            plt.scatter(
                xs,
                ys,
                s=70,
                label=f"{agent} (n={len(xs)})",
                c=colors.get(agent, "#444"),
                marker=markers.get(agent, "o"),
                alpha=0.9,
                edgecolors="white",
                linewidths=0.7,
                zorder=3,
            )
            x_mean = sum(xs) / len(xs)
            y_mean = sum(ys) / len(ys)
            plt.scatter(
                [x_mean],
                [y_mean],
                s=140,
                label=f"{agent} mean",
                c=colors.get(agent, "#444"),
                marker="X",
                alpha=0.95,
                edgecolors="black",
                linewidths=1.2,
                zorder=4,
            )
            plotted += len(xs)

    plt.xlabel("Scaling Actions per Episode")
    plt.ylabel("Final QoS")
    plt.ylim(0.0, 1.1)
    plt.title(f"Final QoS vs Scaling Actions ({args.max_runs} runs)")
    plt.grid(True, linestyle="--", alpha=0.4, zorder=0)
    if plotted:
        plt.legend(frameon=True, loc="lower right", ncol=2)
    plt.tight_layout()

    if args.save:
        out = Path(args.save)
        if not out.suffix:
            out = out.with_suffix(".png")
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, bbox_inches="tight", dpi=180)
        print(f"Saved to {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
