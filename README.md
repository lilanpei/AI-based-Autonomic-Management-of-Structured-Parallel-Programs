# AI-based Autonomic Management of Structured Parallel Programs

**Reinforcement Learning-driven Autoscaling for OpenFaaS Serverless Workflows**

This project implements an **RL-based autoscaling system** for serverless parallel task processing using OpenFaaS, Kubernetes, and Redis. It provides a Gym-compatible environment for training and evaluating autoscaling policies.

---

## 🎯 Project Overview

### **What It Does**

- **Serverless Task Processing**: Producer-consumer workflow with OpenFaaS functions
- **Dynamic Autoscaling**: RL agents learn optimal scaling policies (1-32 workers) with single-step adjustments
- **QoS-Aware**: Tracks deadline compliance and optimizes for QoS targets
- **Baseline Comparison**: Reactive policies for benchmarking RL performance

### **Key Components**

1. **OpenFaaS Functions**: Emitter, Worker (scalable), Collector
2. **Autoscaling Environment**: Gym-compatible RL environment
3. **RL Agents**: SARSA
4. **Baseline Policies**: ReactiveAverage, ReactiveMaximum
5. **Orchestrator**: Workflow management and monitoring

---

## 📁 Project Structure

```
AI-based-Autonomic-Management-of-Structured-Parallel-Programs/
├── openfaas-functions/       # OpenFaaS function implementations
│   ├── emitter/              # Task generator (1 instance)
│   ├── worker/               # Task processor (1-32 instances)
│   ├── collector/            # Result aggregator (1 instance)
│   ├── template/             # Custom python3-http-skeleton template
│   └── stack.yaml            # Deployment configuration
│
├── autoscaling_env/          # RL environment and agents
│   ├── baselines/                     # Reactive baseline policies
│   │   ├── reactive_policies.py       # ReactiveAverage, ReactiveMaximum
│   │   ├── test_reactive_baselines.py # Baseline testing (with optional horizon tuning)
│   │   ├── plots/                     # Performance plots
│   │   ├── logs/                      # Log files
│   │   └── README.md                  # Baseline documentation
│   ├── rl/                            # RL agents and training
│   └── openfaas_autoscaling_env.py    # Gym environment
│
└── utilities/                # Shared utilities
    ├── utilities.py          # Deployment, scaling, monitoring
    └── configuration.yml     # System configuration
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
2. Decide: SARSA agent selects action {-1, 0, +1}
3. Act: Scale workers up/down via Kubernetes API
4. Reward: Based on QoS, queue length, worker cost, scaling penalty
5. Learn: Update Q-table using SARSA (on-policy TD control)
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

### **3. Train SARSA Agent**

```bash
cd autoscaling_env/rl
python train_sarsa.py --episodes 50 --max-steps 30 --step-duration 20
```

### **4. Evaluate Trained Model**

```bash
python test_sarsa.py --model models/sarsa/sarsa_final.pkl --compare-baselines
```

---

## 🎯 Key Features

### **1. Custom OpenFaaS Template**

**Problem**: Standard templates are for short-lived functions  
**Solution**: Custom `python3-http-skeleton` template for long-running functions

**Features**:
- Indefinite execution with `while True` loop
- Flask + Waitress HTTP server
- 12-hour timeouts
- Shared utilities integration

### **2. RL Environment**

**Gym-compatible interface** for training autoscaling policies:

- **Observation**: 9D vector [input_q, worker_q, result_q, output_q, workers, avg_time, max_time, arrival_rate, qos_rate]
- **Action**: 3 discrete actions (-1, 0, +1 workers)
- **Reward**: Configurable blend of QoS delta, normalized queue penalty, worker/idle cost, and scaling penalty (see `utilities/configuration.yml`)
- **Episode Termination**: Task-driven (ends when all tasks complete or max steps reached)

### **3. SARSA**

- **Algorithm**: On-policy TD control
- **State Discretization**: Tile coding
- **Exploration**: Epsilon-greedy

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

---

## 📊 Performance Comparison

| Method          | QoS Rate | Avg Workers | Training Time | Complexity |
|-----------------|----------|-------------|---------------|------------|
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

- **Workload**: `base_rate` (300 tasks/min), `phase_duration`, `number_of_phases`
- **Scaling**: `min_workers` (1), `max_workers` (32), `initial_workers` (3)
- **Environment**: `observation_window` (10s), `step_duration` (10s by default across scaling + observation), optional reactive `horizon`
- **Reward**: `reward` block with QoS target, queue target, idle threshold, and per-term weights
- **QoS**: Task deadlines based on calibrated processing time model (DEADLINE_COEFFICIENT = 2.0)
- **SARSA**: `learning_rate` (0.1), `gamma` (0.99), `epsilon_decay` (0.995)

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
