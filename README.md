# AI-based Autonomic Management of Structured Parallel Programs

**Reinforcement Learning-driven Autoscaling for OpenFaaS Serverless Workflows**

This project implements an **RL-based autoscaling system** for serverless parallel task processing using OpenFaaS, Kubernetes, and Redis. It provides a Gym-compatible environment for training and evaluating autoscaling policies.

---

## 🎯 Project Overview

### **What It Does**

- **Serverless Task Processing**: Producer-consumer workflow with OpenFaaS functions and calibrated image workloads
- **Dynamic Autoscaling**: RL agents learn optimal scaling policies (1-32 workers) with single-step adjustments informed by config-driven workload phases
- **QoS-Aware**: Tracks deadline compliance and optimizes for QoS targets
- **Baseline Comparison**: Reactive policies for benchmarking RL performance

### **Key Components**

1. **OpenFaaS Functions**: Emitter, Worker (scalable), Collector (gamma-sampled image workloads)
2. **Autoscaling Environment**: Gym-compatible RL environment
3. **RL Agents**: SARSA, lightweight DQN
4. **Baseline Policies**: ReactiveAverage, ReactiveMaximum
5. **Orchestrator**: Workflow management and monitoring

---

## 📁 Project Structure

```
AI-based-Autonomic-Management-of-Structured-Parallel-Programs/
├── autoscaling_env/                  # RL environment, baselines, comparison tooling
│   ├── compare_policies.py           # Single-episode SARSA vs reactive comparison script
│   ├── openfaas_autoscaling_env.py   # Gym-compatible OpenFaaS autoscaling environment
│   ├── baselines/                    # Reactive baseline policies & docs
│   │   ├── reactive_policies.py
│   │   ├── test_reactive_baselines.py
│   │   ├── logs/                     # Baseline evaluation logs (gitignored)
│   │   ├── plots/                    # Baseline plots (gitignored)
│   │   └── README.md
│   ├── rl/                           # RL agents (SARSA + lightweight DQN) & tooling
│   │   ├── sarsa_agent.py            # Tabular SARSA agent
│   │   ├── dqn_agent.py              # Compact neural DQN agent
│   │   ├── train_sarsa.py            # SARSA training CLI with per-step logs
│   │   ├── train_dqn.py              # DQN training CLI mirroring SARSA workflow
│   │   ├── test_sarsa.py             # SARSA evaluation & plotting
│   │   ├── test_dqn.py               # DQN evaluation & plotting
│   │   ├── plot_training.py          # Training curve renderer
│   │   ├── utils.py                  # Shared helpers (logging, directories)
│   │   └── README.md
│   └── runs/                         # Timestamped evaluation/comparison outputs (gitignored)
│       └── comparison/
├── image_processing/                 # Workload calibration & task generation utilities
│   ├── task_generator.py             # Phase-based arrival generator
│   ├── calibrate_direct.py           # Image processing calibration pipeline
│   ├── calibration_results.json
│   ├── calibration_plot.png
│   ├── task_arrival_rates.png
│   └── README.md
├── openfaas-functions/               # Deployed OpenFaaS functions
│   ├── emitter/
│   ├── worker/
│   ├── collector/
│   ├── template/                     # Custom python3-http-skeleton template
│   └── stack.yaml
├── openfaas/                         # OpenFaaS deployment manifests
├── utilities/                        # Shared orchestration helpers
│   ├── configuration.yml             # System/experiment configuration
│   └── utilities.py
├── plots/                            # Repository-level figures
├── kind/                             # KIND cluster helpers
├── cleanup.sh                        # Utility script for tearing down resources
└── README.md                         # You are here
```

---

## 🏗️ Architecture

### **Workflow Pattern**

```
  input_queue → EMITTER → worker_queue → WORKER (Nx) → result_queue → COLLECTOR → output_queue
                                       ↑            ↓
                                       └─ RL Agent ─┘
                                        (Scale 1-32)

RL Agent observes: [input_q, worker_q, result_q, output_q, workers,
                    avg_time, max_time, arrival_rate, qos_rate]
```

### **RL Training Loop**

```
1. Observe: 9D state [input_q, worker_q, result_q, output_q, workers,
                      avg_time, max_time, arrival_rate, qos_rate]
2. Decide: RL agent (tabular SARSA or lightweight DQN) selects action {-1, 0, +1}
           via epsilon-greedy exploration
3. Act: Scale workers up/down via Kubernetes API
4. Reward: Based on QoS, queue length, worker cost, scaling penalty
5. Learn:
   - **SARSA**: On-policy TD update of the discretised Q-table (optionally with eligibility traces)
   - **DQN**: Store transition, sample mini-batches from replay, and apply Double-DQN updates to the policy network
```

---

## 🚀 Quick Start

### **Prerequisites**

- Kubernetes cluster with OpenFaaS installed
- Redis deployed in cluster
- `faas-cli`, `kubectl`, Python 3.13+
- Port forwards: `8080` (OpenFaaS), `6379` (Redis)

### **1. Deploy OpenFaaS Functions**

```bash
cd openfaas-functions
faas-cli up -f stack.yaml
```

### **2. Test Baseline Policies**

```bash
cd autoscaling_env
python test_reactive_baselines.py --agent both --steps 50 --step-duration 10 --horizon 10
```

### **3. Train SARSA Agent (tabular)**

```bash
cd autoscaling_env/rl
python train_sarsa.py --episodes 100 --max-steps 30 --step-duration 10 --initial-workers 12 --eval-episodes 5 --phase-shuffle --phase-shuffle-seed 42
```

Each run automatically evaluates the latest checkpoint every `--checkpoint-every`
episodes (and at the end of training), logging mean ± std reward/QoS metrics to
`evaluation_metrics.json` alongside the usual `training_metrics.json`.

### **4. Train DQN Agent (lightweight)**

```bash
cd autoscaling_env/rl
python train_dqn.py --episodes 120 --max-steps 30 --step-duration 8 --initial-workers 12 --eval-episodes 3
```

The DQN trainer mirrors SARSA's reporting: timestamped `dqn_run_*` directories with
per-step logs (reward, QoS, **mean/max workers, processed tasks, QoS violations, unfinished tasks**),
checkpoints (`.pt`), `training_metrics.json`, optional `evaluation_metrics.json`, and plots.

> **Tip:** Tune the Polyak target smoothing with `--target-tau` (default `0.01`). Set `1.0` to fall back to the previous hard-copy target network.

### **5. Evaluate Trained Model**

```bash
# Evaluate SARSA
python test_sarsa.py --model runs/sarsa_run_<timestamp>/models/sarsa_final.pkl --initial-workers 12

# Evaluate DQN
python test_dqn.py --model runs/dqn_run_<timestamp>/models/dqn_final.pt --initial-workers 12
```

Both evaluation scripts now emit SARSA-style per-step tables, per-episode plots,
and JSON summaries with the same aggregate metrics as the comparison tooling
(`total_reward`, `mean_reward`, `mean_qos`, `final_qos`, `mean/max workers`, scaling/no-op counts).

---

### **6. Plot Single-Episode Comparison (SARSA, DQN & Baselines)**

```bash
cd autoscaling_env/rl
python compare_policies.py \
  --model rl/runs/sarsa_run_<timestamp>/models/sarsa_final.pkl \
  --dqn-model rl/runs/dqn_run_<timestamp>/models/dqn_final.pt \
  --initial-workers 12 \
  --max-steps 40 \
  --agents sarsa dqn reactiveaverage reactivemaximum
```

Generates a shared-step comparison plot, aggregated statistics, and a log under
`autoscaling_env/runs/comparison/compare_<timestamp>/`. Re-run the plot without
new simulations using `--plot-only --input-dir <existing_run> [--agents ...]`.

> **Re-use baselines:** pass `--baseline-cache <previous_compare_dir>` to reuse
> cached ReactiveAverage / ReactiveMaximum results while re-evaluating SARSA or DQN.

---

## 🎯 Key Features

### **1. Custom OpenFaaS Template**

**Problem**: Standard templates are for short-lived functions  
**Solution**: Custom `python3-http-skeleton` template for long-running functions

**Features**:
- Indefinite execution with `while True` loop and gamma-based image-size sampling
- Flask + Waitress HTTP server
- 12-hour timeouts
- Shared utilities integration

### **2. RL Environment**

**Gym-compatible interface** for training autoscaling policies:

- **Observation**: 9D vector [input_q, worker_q, result_q, output_q, workers, avg_time, max_time, arrival_rate, qos_rate]
- **Action**: 3 discrete actions (-1, 0, +1 workers)
- **Reward**: Configurable blend of QoS delta, normalized queue penalty, worker/idle cost, and scaling penalty (see `utilities/configuration.yml`)
- **Workload**: Task generator draws processing times from a gamma distribution (mean 1.5 s, shape 4.0 by default) and derives image sizes via the calibrated quadratic model
- **Episode Termination**: Task-driven (ends when all tasks complete or max steps reached)

### **3. RL Agent Toolkit**

- **Algorithms**: Tabular SARSA with discretisation + eligibility traces; lightweight DQN with replay buffer, Double DQN target updates, and observation normalisation
- **Training Scripts**: `train_sarsa.py` / `train_dqn.py` share CLI ergonomics, per-step logging tables, checkpointing, evaluation hooks, and plotting outputs
- **Evaluation utilities**: `test_sarsa.py` and `test_dqn.py` manage run directories, emit the same step-by-step tables, and generate per-episode plots. `compare_policies.py` captures single-episode trajectories for SARSA and both reactive baselines on shared axes.

### **4. Baseline Policies**

- **ReactiveAverage**: Conservative, horizon-based planning with average processing time
- **ReactiveMaximum**: Safety-focused, horizon-based planning with max processing time and safety factor

### **5. QoS Tracking**

- Task deadlines based on processing time
- QoS success/failure tracking
- Deadline compliance metrics

### **6. Graceful Scaling**

- **Scale up**: Deploy new workers, invoke immediately
- **Scale down**: SYN/ACK protocol, finish current task
- **Workload shaping**: Emitter reads `phase_definitions` from `utilities/configuration.yml` to produce configurable multi-phase arrivals

---

## 📊 Performance Comparison

| Method          | QoS Rate | Avg Workers | Training Time | Complexity |
|-----------------|----------|-------------|---------------|------------|
| **DQN**         |          |             |               | Medium     |
| **SARSA**       |          |             |               | Medium     |
| ReactiveAverage |          |             | 0 (no train)  | Low        |
| ReactiveMaximum |          |             | 0 (no train)  | Low        |

**Expected:SARSA achieves better QoS than baselines through learning!**

---

## 🎓 Design Principles

### **1. Separation of Concerns**

- **Emitter**: Task generation
- **Worker**: Task processing (scalable)
- **Collector**: Result aggregation

### **2. Loose Coupling**

- Redis queues for communication
- No direct HTTP calls between functions
- Easy to add/remove workers

### **3. Horizontal Scalability**

- Only workers scale (1-32 instances)
- Emitter and collector remain at 1 instance

### **4. Graceful Degradation**

- Workers poll with timeout (no busy-waiting)
- SYN/ACK protocol for graceful shutdown
- Workers finish current task before exiting

### **5. Observability**

- Real-time queue monitoring
- QoS tracking (deadline compliance)
- Performance metrics (work time, queue lengths)

---

## 🔧 Configuration

Key parameters in `utilities/configuration.yml`:

- **Workload**: `base_rate` (300 tasks/min), `phase_definitions` (list of per-phase multipliers, durations, oscillation controls) consumed by `image_processing/task_generator.py`
- **Scaling**: `min_workers` (1), `max_workers` (32), `initial_workers` (12)
- **Observation window**: `observation_window` (10s), `step_duration` (10s by default across scaling + observation), optional reactive `horizon`
- **Processing time sampling**: `target_mean_processing_time` (default 1.5 s) and `processing_time_shape` (default 4.0) control gamma draws for image sizes in the function utilities
- **Reward**: `reward` block with QoS target, queue target, idle threshold, and per-term weights
- **QoS**: Task deadlines based on calibrated processing time model (DEADLINE_COEFFICIENT = 2.0)
- **SARSA**: `learning_rate` (0.1), `gamma` (0.99), `epsilon_decay` (0.995)
- **DQN**: Replay capacity, batch size, epsilon schedule, target update cadence are configurable via `train_dqn.py`

---

## 📚 Documentation

### **Component READMEs**

- **OpenFaaS Functions**: `openfaas-functions/README.md`
- **Autoscaling Environment**: `autoscaling_env/README.md`
- **Baseline Policies**: `autoscaling_env/baselines/README.md`
- **RL Agents**: `autoscaling_env/rl/README.md`

---

## 🐛 Troubleshooting

### **Port Forward Issues**

```bash
# Kill existing port forwards
pkill -f "port-forward"

# Restart port forwards
kubectl port-forward -n openfaas svc/gateway 8080:8080 &
kubectl port-forward -n redis svc/redis-master 6379:6379 &
```

### **Function Not Responding**

```bash
# Check function status
faas-cli list
kubectl get pods -n openfaas-fn

# Check logs
kubectl logs -n openfaas-fn -l faas_function=worker-1 --tail=50
```

### **Redis Connection Issues**

```bash
# Test Redis connection
redis-cli ping

# Check queue lengths
redis-cli LLEN worker_queue
```

### **Training Issues**

- **QoS stays low**: Increase observation window, adjust reward weights
- **Continuous scale-up**: Revisit reward queue weight and reactive horizon settings
- **Training too slow**: Reduce steps/episode, reduce step duration

---

## 🚀 Next Steps

### **1. Experiment with Workloads**

Modify `configuration.yml` to test different patterns:
- Constant load
- Burst patterns
- Sine wave variations

### **2. Tune Hyperparameters**

Adjust RL agent parameters:
- Learning rate
- Discount factor
- Observation window
- Reward weights

### **3. Extend Functionality**

Add new features:
- Multi-objective optimization
- Cost-aware scaling
- Predictive scaling
- Custom reward functions

### **4. Deploy to Production**

Use trained models for real workloads:
- Load model from checkpoint
- Integrate with monitoring system
- Set up alerting for QoS violations

---

## 📝 Summary

This project provides:

- ✅ **Complete RL-based autoscaling system** for OpenFaaS
- ✅ **Gym-compatible environment** for training
- ✅ **Baseline policies** for comparison
- ✅ **QoS-aware rewards** for deadline compliance
- ✅ **Graceful scaling** with SYN/ACK protocol
- ✅ **Comprehensive documentation** and guides

**Train RL agents to learn optimal autoscaling policies for serverless workflows!** 🎉

---
