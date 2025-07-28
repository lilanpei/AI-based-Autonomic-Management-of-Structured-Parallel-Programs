import os
import re
import numpy as np
import matplotlib.pyplot as plt
import sys

# Add the parent directory of 'orchestrator' to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from orchestrator.utilities import get_config

configuration = get_config()
last_n = str(configuration.get("number_of_most_recent_logs"))  # Number of last runs to consider
log_dir = configuration.get("log_dir")
plot_output_dir = configuration.get("plot_output_dir")

# Create directory if it doesn't exist
os.makedirs(plot_output_dir, exist_ok=True)

pattern = re.compile(r'logs_controller_w(\d+)_(\d+)\.txt')
timer_env_end = re.compile(r"\[TIMER\] Environment initialization completed at \[(.*?)\] seconds.")
timer_farm_end = re.compile(r"\[TIMER\] Farm initialization completed at \[(.*?)\] seconds.")
timer_task_start = re.compile(r"\[TIMER\] Task generation started at \[(.*?)\] seconds.")
timer_program_end = re.compile(r"\[TIMER\] Program/Monitoring completed at \[(.*?)\] seconds.")

# Step 1: Gather and sort log files
log_groups = {}
for fname in os.listdir(log_dir):
    match = pattern.match(fname)
    if match:
        w = int(match.group(1))
        t = int(match.group(2))
        log_groups.setdefault(w, []).append((t, fname))

# Keep only the last n (by timestamp) for each worker count
selected_logs = []
for w, logs in log_groups.items():
    logs.sort()  # sort by time
    if last_n != "None" and last_n.isdigit():
        last_n = int(last_n)
        if len(logs) > last_n:
            logs = logs[-last_n:]
    # Extend selected logs with the last n logs for this worker count
    selected_logs.extend([(w, fname) for _, fname in logs])

# Step 2: Parse selected logs
timing_data = {
    "Env Init Time": {},
    "Farm Init Time": {},
    "Total Init Time": {},
    "Task Run Time": {},
    "Program Total Time": {}
}

for w, fname in selected_logs:
    with open(os.path.join(log_dir, fname)) as f:
        env_end = farm_end = task_start = prog_end = None
        for line in f:
            if "Environment initialization completed" in line:
                env_end = float(timer_env_end.search(line).group(1))
                timing_data["Env Init Time"].setdefault(w, []).append(env_end)
            if "Farm initialization completed" in line:
                farm_end = float(timer_farm_end.search(line).group(1))
                if env_end is not None:
                    timing_data["Farm Init Time"].setdefault(w, []).append(farm_end - env_end)
                timing_data["Total Init Time"].setdefault(w, []).append(farm_end)
            if "Task generation started" in line:
                task_start = float(timer_task_start.search(line).group(1))
            if "Program/Monitoring completed" in line:
                prog_end = float(timer_program_end.search(line).group(1))
                if task_start is not None:
                    timing_data["Task Run Time"].setdefault(w, []).append(prog_end - task_start)
                timing_data["Program Total Time"].setdefault(w, []).append(prog_end)

# Step 3: Prepare data
workers = sorted(set(k for d in timing_data.values() for k in d))

# ------------------ FIGURE 1: SCATTER PLOT ------------------
fig1, ax1 = plt.subplots(figsize=(10, 6))
colors = plt.cm.tab10.colors
for idx, (label, data) in enumerate(timing_data.items()):
    x, y = [], []
    for w in sorted(data.keys()):
        x += [w] * len(data[w])
        y += data[w]
    ax1.scatter(x, y, label=label, color=colors[idx % len(colors)])
ax1.set_xscale("log", base=2)
ax1.set_xticks(workers)
ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax1.set_xlabel("Number of Workers")
ax1.set_ylabel("Time (s)")
ax1.set_title(f"Scatter Plot: Raw Timings vs. Number of Workers")
ax1.grid(True)
ax1.legend()
plt.tight_layout()
plt.savefig(f"{plot_output_dir}/scatter_metrics.png", dpi=600)

# ------------------ FIGURE 2: ERROR BARS ------------------
fig2, ax2 = plt.subplots(figsize=(10, 6))
for idx, (label, data) in enumerate(timing_data.items()):
    means = [np.mean(data[w]) for w in workers]
    stds = [np.std(data[w]) for w in workers]
    ax2.errorbar(workers, means, yerr=stds, label=label, fmt='o-', capsize=3)
ax2.set_xscale("log", base=2)
ax2.set_xticks(workers)
ax2.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax2.set_xlabel("Number of Workers")
ax2.set_ylabel("Time (s)")
ax2.set_title(f"Error Bar Plot: Avg ± Std Timings vs. Number of Workers")
ax2.grid(True)
ax2.legend()
plt.tight_layout()
plt.savefig(f"{plot_output_dir}/errorbar_metrics.png", dpi=600)

# ------------------ FIGURE 3: SCALING METRICS ------------------
# Time metrics
avg_run_times = [np.mean(timing_data["Task Run Time"][w]) for w in workers]
avg_env_init_times = [np.mean(timing_data["Env Init Time"][w]) for w in workers]
avg_farm_init_times = [np.mean(timing_data["Farm Init Time"][w]) for w in workers]
avg_init_times = [np.mean(timing_data["Total Init Time"][w]) for w in workers]
avg_run_times = [np.mean(timing_data["Task Run Time"][w]) for w in workers]
avg_tot_times = [np.mean(timing_data["Program Total Time"][w]) for w in workers]
baseline = avg_run_times[0]
speedups = [baseline / t for t in avg_run_times]
efficiencies = [s / w for s, w in zip(speedups, workers)]

fig3, axes = plt.subplots(1, 3, figsize=(15, 4))

# Runtime
axes[0].plot(workers, avg_run_times, 'o-', label='Runtime')
axes[0].set_title(f"Runtime vs. Number of Workers")
axes[0].set_ylabel("Time (s)")

# Speedup
axes[1].plot(workers, speedups, 'o-', label='Speedup')
axes[1].plot(workers, workers, '--', label='Ideal')
axes[1].set_title(f"Speedup vs. Number of Workers")

# Efficiency
axes[2].plot(workers, efficiencies, 'o-', label='Efficiency')
axes[2].plot(workers, [1 for _ in workers], '--', label='Ideal')
axes[2].set_ylim(0, 1.1)
axes[2].set_title(f"Efficiency vs. Number of Workers")

for ax in axes:
    ax.set_xscale("log", base=2)
    ax.set_xticks(workers)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("Number of Workers")
    ax.grid(True)
    ax.legend()

plt.tight_layout()
plt.savefig(f"{plot_output_dir}/scaling_metrics.png", dpi=600)

# ------------------ FIGURE 4: SCATTER TIME METRICS PLOT ------------------
fig4, ax4 = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
colors = plt.cm.tab10.colors
axes = ax4.flatten()

# Define the keys you want to plot
plot_keys = ["Env Init Time", "Farm Init Time", "Task Run Time", "Program Total Time"]

for idx, label in enumerate(plot_keys):
    data = timing_data[label]
    ax = axes[idx]
    x, y = [], []
    for w in sorted(data.keys()):
        x += [w] * len(data[w])
        y += data[w]
    ax.scatter(x, y, label=label, color=colors[idx % len(colors)])
    ax.set_xscale("log", base=2)
    ax.set_xticks(workers)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("Number of Workers")
    ax.set_ylabel("Time (s)")
    ax.set_title(f"Scatter Plot: {label}")
    ax.grid(True)
    ax.legend()

plt.tight_layout()
plt.savefig(f"{plot_output_dir}/scatter_per_time_metrics.png", dpi=600)

# ------------------ FIGURE 5: ERROR BAR TIME METRICS PLOT ------------------
fig5, ax5 = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
axes = ax5.flatten()
colors = plt.cm.tab10.colors

# Keys to plot
plot_keys = ["Env Init Time", "Farm Init Time", "Task Run Time", "Program Total Time"]

for idx, label in enumerate(plot_keys):
    data = timing_data[label]
    means = [np.mean(data[w]) for w in workers]
    stds = [np.std(data[w]) for w in workers]

    ax = axes[idx]
    ax.errorbar(workers, means, yerr=stds, label=label, fmt='o-', color=colors[idx % len(colors)], capsize=3)
    ax.set_xscale("log", base=2)
    ax.set_xticks(workers)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("Number of Workers")
    ax.set_ylabel("Time (s)")
    ax.set_title(f"{label} (Avg ± Std) vs. Number of Workers")
    ax.grid(True)
    ax.legend()

plt.tight_layout()
plt.savefig(f"{plot_output_dir}/errorbar_per_time_metrics.png", dpi=600)