# Autoscaling Environment

This directory contains the **OpenFaaS Autoscaling Gym Environment** for reinforcement learning-based autoscaling. It provides a Gym-like interface for training and evaluating autoscaling policies.

---

## 📁 Directory Structure

```
autoscaling_env/
├── openfaas_autoscaling_env.py   # Main Gym environment (9D observation)
├── test_reactive_baselines.py    # Test baseline policies
├── baselines/                     # Reactive baseline policies
│   ├── reactive_baselines.py      # ReactiveAverage, ReactiveMaximum
│   └── README.md                  # Baseline documentation
├── rl/                            # RL agents and training
│   ├── sarsa_agent.py             # SARSA with tile coding
│   ├── train_sarsa.py             # Training script
│   ├── test_sarsa.py              # Evaluation script
│   └── README.md                  # RL documentation
├── models/                        # Saved RL models
│   └── sarsa/                     # SARSA checkpoints
└── plots/                         # Performance plots
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

**Action Space** (5 discrete actions):
- `0`: Scale down by 2 workers
- `1`: Scale down by 1 worker
- `2`: No change
- `3`: Scale up by 1 worker
- `4`: Scale up by 2 workers

**Reward Function**:
```
reward = qos_reward + queue_penalty + worker_cost + scaling_penalty + efficiency_bonus + violation_penalty
```

Components:
- `qos_reward`: +10 × qos_rate (encourage high QoS)
- `queue_penalty`: -0.1 × queue_length (discourage buildup)
- `worker_cost`: -1.0 × workers (encourage efficiency)
- `scaling_penalty`: -2.0 if action ≠ 0 (discourage unnecessary scaling)
- `efficiency_bonus`: +5.0 if qos > 95% and workers < 10
- `violation_penalty`: -20.0 if qos < 80% (strong penalty)

---

## 🚀 Quick Start

### **1. Test Baseline Policies**

```bash
cd autoscaling_env
python test_reactive_baselines.py --agent both --max-steps 30 --step-duration 20
```

**Output**: Comparison plots and statistics for ReactiveAverage and ReactiveMaximum

### **2. Train SARSA Agent**

```bash
cd rl
python train_sarsa.py --episodes 50 --max-steps 30 --step-duration 20
```

**Output**: Model saved to `models/sarsa/sarsa_final.pkl`

### **3. Evaluate Trained Model**

```bash
python test_sarsa.py --model models/sarsa/sarsa_final.pkl --compare-baselines
```

**Output**: Comparison table and plots in `plots/`

---

## 🎓 Key Features

### **1. Task-Driven Episode Termination**

Each episode:
- Resets workers to initial count (default: 1)
- Generates tasks based on phase configurations
- **Terminates when**: All tasks completed OR max_steps reached
- Automatically calculates `phase_duration` to cover episode
- Waits for all tasks to complete before ending

**Key improvement**: Episodes end naturally when work is done, not at arbitrary step limits!

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

- Conservative scaling based on average processing time
- Good for stable workloads

### **ReactiveMaximum**

- Aggressive scaling based on maximum processing time
- Good for bursty workloads

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
