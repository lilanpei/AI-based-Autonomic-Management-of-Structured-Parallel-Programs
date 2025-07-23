import os
import re
import numpy as np
import matplotlib.pyplot as plt

log_dir = "/home/lanpei/AI-based-Autonomic-Management-of-Structured-Parallel-Programs/orchestrator/logs_farm/0723"
pattern = re.compile(r'logs_controller_w(\d+)_(\d+)\.txt')
timer_start_line = re.compile(r"\[TIMER\] Task generation started at \[(.*?)\] seconds.")
timer_end_line = re.compile(r"\[TIMER\] Program/Monitoring completed at \[(.*?)\] seconds.")

timings = {}

# Step 1–2: Parse each log file
for fname in os.listdir(log_dir):
    match = pattern.match(fname)
    if not match:
        continue
    worker_count = int(match.group(1))
    with open(os.path.join(log_dir, fname)) as f:
        for line in f:
            if "[TIMER] Task generation started at" in line:
                start_time_val = float(timer_start_line.search(line).group(1))
            if "[TIMER] Program/Monitoring completed at" in line:
                end_time_val = float(timer_end_line.search(line).group(1))
                timings.setdefault(worker_count, []).append(end_time_val - start_time_val)

# Step 3: Compute metrics
workers = sorted(timings.keys())
# runtime = T(w)
avg_times = [np.mean(timings[w]) for w in workers]
baseline = avg_times[0]  # w1 time
# speedup(w) = T(1) / T(w)
speedups = [baseline / t for t in avg_times]
# efficiency(w) = speedup(w) / w
efficiency = [s / w for s, w in zip(speedups, workers)]

print(f"Workers: {workers}")
print(f"Runtime: {avg_times}")
print(f"Speedups: {speedups}")
print(f"Efficiencies: {efficiency}")

# Step 4: Plot all metrics
plt.figure(figsize=(12, 4))

# Runtime
plt.subplot(1, 3, 1)
plt.plot(workers, avg_times, 'o-', label='Runtime')
plt.xlabel("Number of Workers")
plt.ylabel('Time (s)')
plt.grid(True)
plt.xscale('log', base=2)
plt.yscale('log', base=2)
plt.xticks(workers, labels=workers)
plt.title("Runtime vs. Number of Workers")

# Speedup
plt.subplot(1, 3, 2)
plt.plot(workers, speedups, 'o-', label='Speedup')
plt.plot(workers, workers, linestyle='--', label='Ideal Linear Speedup')
plt.xlabel("Number of Workers")
plt.ylabel('Speedup')
plt.grid(True)
plt.xscale('log', base=2)
plt.yscale('log', base=2)
plt.xticks(workers, labels=workers)
plt.title("Speedup vs. Number of Workers")
plt.legend()

# Efficiency
plt.subplot(1, 3, 3)
plt.plot(workers, efficiency, 'o-', label='Efficiency')
plt.xlabel("Number of Workers")
plt.ylabel('Efficiency')
plt.grid(True)
plt.xscale('log', base=2)
plt.yscale('log', base=2)
plt.ylim(0, 1.1)
plt.xticks(workers, labels=workers)
plt.title("Efficiency vs. Number of Workers")

plt.tight_layout()
plt.savefig("scaling_metrics.png")
plt.show()