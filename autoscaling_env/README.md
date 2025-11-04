# Autoscaling Environment

This directory contains the **OpenFaaS Autoscaling Gym Environment** for reinforcement learning-based autoscaling. It provides a Gym-like interface for training and evaluating autoscaling policies.

---

## 📁 Directory Structure

```
autoscaling_env/
├── baselines/                     # Reactive baseline policies
│   ├── reactive_policies.py       # ReactiveAverage, ReactiveMaximum
│   ├── test_reactive_baselines.py # Test baseline policies (supports --horizon)
│   └── README.md                  # Baseline documentation
├── rl/                            # RL agents and training
│   ├── sarsa_agent.py    # SARSA
│   ├── train_sarsa.py    # Training script
│   ├── test_sarsa.py     # Evaluation script (per-episode plots + logging)
│   ├── plot_training.py  # Training plotter
│   ├── utils.py          # Utility functions
│   └── README.md
├── compare_policies.py            # Single-episode SARSA vs reactive comparison
└── openfaas_autoscaling_env.py    # Main Gym environment (9D observation)
```

---

## 🎯 Environment Overview

### **OpenFaaSAutoscalingEnv**

A Gym-compatible environment for autoscaling OpenFaaS workers.

**Observation Space** (9-dimensional):
- `input_queue`: Tasks in input queue
- `worker_queue`: Tasks in worker queue
- `result_queue`: Tasks in result queue
- `output_queue`: Completed tasks in output queue
- `workers`: Current number of worker instances
- `avg_processing_time`: Average task processing time (seconds)
- `max_processing_time`: Maximum task processing time (seconds)
- `arrival_rate`: Task arrival rate (tasks/second)
- `qos_rate`: QoS success rate (0-1, deadline compliance)

**Action Space** (3 discrete actions):
- `0`: Scale down by 1 worker
- `1`: No change
- `2`: Scale up by 1 worker

**Reward Function** (configurable):
```
reward =
  w_qos * (qos_rate - target_qos)
+ penalty_scale * (
- w_queue * max(0, worker_q / queue_target - 1)
- w_backlog * max(0, target_qos - qos_rate) * (1 + worker_q / queue_target)
- w_workers * (max(0, workers - min_workers) / (max_workers - min_workers))
- w_scaling * max(0, |delta| - scaling_tolerance)
- [w_idle * (max(0, workers - min_workers) / (max_workers - min_workers)) if worker_q ≤ idle_queue_threshold else 0]
+ success_bonus
)
+ [backlog_relief_weight * max(0, target_qos - qos_rate) * max(0, worker_q / queue_target - 1) * max(delta, 0)]
+ [qos_recovery_weight * max(0, qos_rate - qos_rate_prev) if qos_rate < target_qos else 0]
+ [queue_relief_weight * max(0, worker_q / queue_target - 1) * max(delta, 0)]
+ [scale_up_credit_scale * max(0, delta_prev) while scale_up_credit_steps remain]
```

Where:
- `success_bonus = w_qos * (success_bonus_bias + success_bonus_scale * max(0, qos_rate - target_qos))` when QoS meets/exceeds the target and backlog ≤ 50% of `queue_target`, providing a stronger positive signal as QoS improves.
- `penalty_scale`, `queue_relief_weight`, `scale_up_credit_steps`, `scale_up_credit_scale`, `w_*` weights, `target_qos`, `queue_target`, `idle_queue_threshold`, `scaling_tolerance`, `success_bonus_bias`, `success_bonus_scale`, `backlog_relief_weight`, and `qos_recovery_weight` are set in `utilities/configuration.yml` under the `reward` block.
- `w_qos * (qos_rate - target_qos)`: Positive incentive when QoS meets the goal, negative when it falls short.
- `penalty_scale` keeps the large negative components bounded so they do not swamp exploratory steps.
- `- w_queue * max(0, worker_q / queue_target - 1)` penalizes worker queue backlog beyond the desired target length.
- `- w_backlog * max(0, target_qos - qos_rate) * (1 + worker_q / queue_target)` amplifies backlog penalties when QoS drops below target.
- `- w_workers * (max(0, workers - min_workers) / (max_workers - min_workers))` discourages holding excess workers beyond the configured minimum.
- `- w_scaling * max(0, |delta| - scaling_tolerance)` ignores small adjustments within the tolerance window before penalising larger swings.
- `- w_idle * (max(0, workers - min_workers) / (max_workers - min_workers))` applies when the worker queue is at/below `idle_queue_threshold`, scaling the penalty by excess capacity.
- `backlog_relief_weight * max(0, target_qos - qos_rate) * max(0, worker_q / queue_target - 1) * max(delta, 0)` rewards scale-up actions only when backlog exceeds the target **and** QoS is below target, keeping the signal tied to urgent recovery steps.
- `qos_recovery_weight * max(0, qos_rate - qos_rate_prev)` adds a bonus when QoS improves between steps while still below target, reinforcing recovery efforts during backlog.
- `queue_relief_weight * max(0, worker_q / queue_target - 1) * max(delta, 0)` gives an extra boost to scale-up moves taken while the queue is still over target, further reinforcing corrective scaling.
- `scale_up_credit_scale * max(0, delta_prev)` is applied for `scale_up_credit_steps` timesteps after a scale-up, keeping a positive shaping signal active while the added workers drain the backlog.
- Terminal shaping now applies `penalty_scale × unfinished_penalty × min(pending_tasks, queue_target)` at the horizon, capping the negative spike, or adds `completion_bonus` when all tasks finish early.
- `delta` is the **applied** scaling change after clipping to `[min_workers, max_workers]`. The environment waits the remaining step duration, refreshes QoS/queue metrics, and then evaluates the reward, so SARSA receives `(s, a, r, s')` tuples where `r` reflects the action's observed effect.

---

## 🚀 Quick Start

### **1. Test Baseline Policies**

```bash
cd autoscaling_env
python test_reactive_baselines.py --agent both --steps 30 --step-duration 8 --horizon 8
```

**Output**: Comparison plots and statistics for ReactiveAverage and ReactiveMaximum

### **2. Train SARSA Agent**

```bash
cd rl
python train_sarsa.py --episodes 100 --max-steps 30 --step-duration 8 --initial-workers 12 --eval-episodes 3 --phase-shuffle --phase-shuffle-seed 42
```

**Output**: Model saved to `models/sarsa/sarsa_final.pkl` plus
`evaluation_metrics.json` that captures checkpoint mean ± std reward/QoS during
training.

### **3. Evaluate Trained Model**

```bash
python test_sarsa.py --model models/sarsa/sarsa_final.pkl --initial-workers 12
```

**Output**: Timestamped run directory with logs, plots, and metrics under `runs/`

### **4. Compare SARSA with Reactive Baselines on a Single Episode**

```bash
python compare_policies.py \
  --model rl/runs/sarsa_run_<timestamp>/models/sarsa_final.pkl \
  --initial-workers 12 \
  --max-steps 30 \
  --agents agent reactiveaverage reactivemaximum
```

**Output**: Per-step logs for selected policies, aggregated summaries, and a
shared comparison plot under `autoscaling_env/runs/comparison/compare_<timestamp>/`.
Rebuild plots from an existing run with `--plot-only --input-dir <run_dir>
[--agents ...]` to focus on specific policies.

---

## 🎓 Key Features

### **1. Task-Driven Episode Termination**

Each episode:
- Resets workers to initial count (default: 1)
- Generates tasks based on phase configurations
- **Terminates when**: All tasks completed OR max_steps reached
- Automatically calculates `phase_duration` to cover episode
- Waits for all tasks to complete before ending so QoS reflects all finished tasks

**Key improvements**:
- Episodes end naturally when work is done instead of arbitrary step limits
- Terminal reward shaping penalises unfinished tasks at the max-step boundary
  and rewards early completion (see `utilities/configuration.yml`)

### **2. Task Generation**

Phase-based generation with configurable patterns:
- **Constant**: Steady task rate (e.g., 300 tasks/min)
- **Burst**: Sudden spike in task rate
- **Sine wave**: Periodic variation

### **3. QoS Tracking**

Tasks have deadlines:
- QoS success: task completes before deadline
- QoS failure: task exceeds deadline
- QoS rate: percentage of successful tasks

### **4. Observation Window with Accurate Timestamping**

Metrics calculated over sliding window (default: 10s):
- Average processing time from recent tasks
- Maximum processing time from recent tasks
- Arrival rate from task completion rate
- QoS rate from recent task outcomes

**Key improvement**: Uses actual task completion timestamps (not collection time) for accurate metrics!

### **5. Scaling Constraints**

- Min workers: 1
- Max workers: 32
- Actions clipped to valid range

---

## 📊 Evaluation Metrics

Agents are evaluated on:

1. **Total Reward**: Cumulative reward over episode
2. **Mean QoS Rate**: Average QoS across all steps
3. **Mean Workers**: Average worker count (resource efficiency)
4. **Scaling Actions**: Number of scaling operations (stability)
5. **QoS Violations**: Number of steps with QoS < 80%

---

## 🔧 Configuration

Key parameters in environment initialization:

- `max_workers`: Maximum workers (default: 32)
- `min_workers`: Minimum workers (default: 1)
- `initial_workers`: Starting workers (default: 1)
- `observation_window`: Metric window in seconds (default: 10)
- `step_duration`: **Total** step time including scaling (default: 20s)
- `max_steps`: Maximum steps per episode (default: 15)

**Important**: `step_duration` is the total time per step (scaling + observation), not just wait time!

---

## 🎯 Baseline Policies

See `baselines/README.md` for details.

### **ReactiveAverage**

- Horizon-based planning with average processing time (default horizon = step duration)
- Safety factor configurable; suitable for stable workloads

### **ReactiveMaximum**

- Horizon-based planning with maximum processing time and safety margin
- Faster to react to bursts while respecting QoS constraints

---

## 🤖 RL Agents

See `rl/README.md` for details.

### **SARSA (State-Action-Reward-State-Action)**

- **Algorithm**: On-policy TD control
- **State Discretization**: Tile coding
- **Exploration**: Epsilon-greedy

---

## 📚 Documentation

- **Baselines**: `baselines/README.md`
- **RL Agents**: `rl/README.md`

---

## 🔗 Integration

### **With OpenFaaS Functions**

Environment interacts with functions via:
- Kubernetes API (scaling workers)
- Redis queues (task generation, monitoring)
- OpenFaaS gateway (function invocation)

### **With Orchestrator**

Environment uses utilities from:
- `utilities/utilities.py`: Deployment, scaling, monitoring
- `utilities/configuration.yml`: Configuration parameters

---

## 📝 Summary

The autoscaling environment provides:

- ✅ **Gym-compatible interface** for RL training
- ✅ **Realistic workload simulation** with phase-based generation
- ✅ **QoS-aware rewards** for deadline compliance
- ✅ **Baseline policies** for comparison
- ✅ **Comprehensive evaluation** metrics

**Train RL agents to learn optimal autoscaling policies!** 🚀
