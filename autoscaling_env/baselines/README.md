# Baseline Autoscaling Policies

This directory contains **reactive baseline policies** for autoscaling comparison. These policies serve as benchmarks to evaluate the performance of RL-based autoscaling agents.

---

## 📁 Files

- **`reactive_baselines.py`**: Policy implementations (ReactiveAverage, ReactiveMaximum)
- **`README.md`**: This documentation

---

## 🎯 Baseline Policies

### **1. ReactiveAverage**

**Strategy**: Horizon-based scaling using average processing time

**Formula**:
```
tasks_per_worker = max(horizon_seconds / avg_time, 1)
workers_for_queue = worker_queue / tasks_per_worker
workers_for_arrivals = arrival_rate * avg_time
optimal_workers = (workers_for_queue + workers_for_arrivals) * safety_factor
```

**Defaults**:
- `horizon_seconds` = environment `step_duration` (override via `--horizon`/`set_horizon`)
- `safety_factor` = 1.0

**Behavior**:
- Conservative scaling tuned to current control window
- Smooth adjustments informed by backlog and expected arrivals
- Good for stable workloads

---

### **2. ReactiveMaximum**

**Strategy**: Horizon-based scaling using maximum processing time (worst case)

**Formula**:
```
service_time = max(max_time, avg_time, 0.1)
tasks_per_worker = max(horizon_seconds / service_time, 1)
workers_for_queue = worker_queue / tasks_per_worker
workers_for_arrivals = arrival_rate * service_time
optimal_workers = (workers_for_queue + workers_for_arrivals) * safety_factor
```

**Defaults**:
- `horizon_seconds` = environment `step_duration`
- `safety_factor` = 1.0

**Behavior**:
- Protective against QoS violations
- Responds faster to bursts while remaining bounded
- Better for variable or unpredictable workloads

---

## 🔧 Key Features

### **Queue-Based Scaling**

Both policies calculate optimal workers based on:
- Current worker queue length
- Processing time (average or maximum)
- Simple formula: `workers = (queue / time) + (arrival_rate × time)`

### **Idle Handling**

When queue is empty or processing time is zero:
- Return no-op action (action = 2)
- Prevents unnecessary scaling

### **Action Mapping**

Policies map the worker delta to the 3-action space used by the environment:
- `delta ≥ 0.5`: Scale up by 1 (`action = 2`)
- `delta ≤ -0.5`: Scale down by 1 (`action = 0`)
- Otherwise: No change (`action = 1`)

---

## 🚀 Usage

### **Test Single Policy**

```bash
cd autoscaling_env
python test_reactive_baselines.py --agent average --steps 15
```

### **Test Both Policies**

```bash
python test_reactive_baselines.py --agent both --steps 50 --step-duration 10 --horizon 10
```

### **Compare with RL Agent**

```bash
cd rl
python train_ppo.py --mode compare --model models/ppo_final.pt --episodes 3
```

---

## 📊 Evaluation Metrics

Policies are evaluated on:

1. **Total Reward**: Cumulative reward over episode
2. **QoS Rate**: Percentage of tasks meeting deadline
3. **Average Workers**: Mean worker count (resource efficiency)
4. **Scaling Actions**: Number of scaling operations (stability)

---

## 🎓 Design Principles

### **Reactive (Not Predictive)**

Policies react to current metrics, don't predict future load:
- Simple and interpretable
- No training required
- Fast decision-making

---

## 📈 Expected Performance

### **ReactiveAverage**

- **Strengths**: Stable, resource-efficient
- **Weaknesses**: Slow to respond to bursts
- **Best for**: Constant or gradually changing workloads

### **ReactiveMaximum**

- **Strengths**: Fast response to bursts
- **Weaknesses**: May over-provision, higher cost
- **Best for**: Bursty or unpredictable workloads

### **RL Agents**

- **Strengths**: Adaptive, learns optimal policy
- **Weaknesses**: Requires training, more complex
- **Best for**: Complex workload patterns

---

## 🔗 Integration

Baseline policies use the same interface as RL agents:

- **Observation**: `[queue_length, workers, avg_time, arrival_rate, qos_rate]`
- **Action**: `{0: -2, 1: -1, 2: 0, 3: +1, 4: +2}`
- **Environment**: `OpenFaaSAutoscalingEnv`

This allows fair comparison between baselines and RL agents.

---

## 📚 Related Documentation

- **Main Environment**: `../README.md`
- **RL Agents**: `../rl/README.md`

---

## 📝 Summary

Baseline policies provide:

- ✅ **Simple, interpretable** autoscaling rules
- ✅ **No training required** - ready to use
- ✅ **Fair comparison** for RL agents
- ✅ **Two strategies**: conservative (average) and aggressive (maximum)

**Use these as benchmarks to validate RL agent performance!** 🎯
