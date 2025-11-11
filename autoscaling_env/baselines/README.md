# Baseline Autoscaling Policies

This directory contains **reactive baseline policies** for autoscaling comparison. These policies serve as benchmarks to evaluate the performance of RL-based autoscaling agents.

---

## 📁 Files

- **`reactive_policies.py`**: Policy implementations (ReactiveAverage, ReactiveMaximum)
- **`README.md`**: This documentation

---

## 🎯 Baseline Policies

### **1. ReactiveAverage** – FARM-inspired heuristic with average processing time

**Strategy**: Balance backlog clearance and anticipated arrivals during the control horizon using the analytical FARM skeleton.

**Formula**:
```
service_time         = max(avg_time, 0.1)
horizon              = max(horizon_seconds, service_time, 1.0)
workers_for_arrival  = arrival_rate * service_time
workers_for_queue    = (worker_queue / horizon) * service_time
active_tasks         = mirror_counter - worker_queue - result_queue - output_queue
active_tasks_workers = min(workers, max(0, active_tasks - collector_adjustment))
busy_workers         = (active_tasks_workers / horizon) * service_time * adjustment_factor
optimal_workers      = (workers_for_queue + workers_for_arrival) * safety_factor
delta                = optimal_workers - (workers - busy_workers)
```

**Defaults**:
- `horizon_seconds` = 8.0 s (override via `--horizon`/`set_horizon`)
- `safety_factor` = 1.0
- `adjustment_factor` = 0.5 (down-weights mirrored busy capacity to leave buffer)

**Behavior**:
- Conservative scaling matched to average service time
- Smooth adjustments informed by mirrored active-task telemetry
- Good for stable workloads with gradual changes

---

### **2. ReactiveMaximum**

**Strategy**: Same analytical FARM structure as ReactiveAverage but substitutes the worst-case service time for extra protection.

**Formula**:
```
service_time         = max(max_time, avg_time, 0.1)
horizon              = max(horizon_seconds, service_time, 1.0)
workers_for_arrival  = arrival_rate * service_time
workers_for_queue    = (worker_queue / horizon) * service_time
active_tasks         = mirror_counter - worker_queue - result_queue - output_queue
active_tasks_workers = min(workers, max(0, active_tasks - collector_adjustment))
busy_workers         = (active_tasks_workers / horizon) * service_time * adjustment_factor
optimal_workers      = (workers_for_queue + workers_for_arrival) * safety_factor
delta                = optimal_workers - (workers - busy_workers)
```

**Defaults**:
- `horizon_seconds` = 8.0 s
- `safety_factor` = 1.0
- `adjustment_factor` = 1.0 (uses full mirrored busy capacity to react quickly)

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
- Mirrored enqueue counter (`worker_queue:enqueued_total`) to infer active tasks
- Simple formula: `workers = (queue / time) + (arrival_rate × time)` adjusted by busy-worker estimates

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
python test_reactive_baselines.py --agent average --steps 30
```

### **Test Both Policies**

```bash
python test_reactive_baselines.py --agent both --steps 30 --step-duration 8 --horizon 8
```

---

## 📊 Evaluation Metrics

Policies are evaluated on the same episode metrics emitted by SARSA/DQN comparisons:

1. **Total Reward**: Cumulative reward over the episode
2. **Mean QoS Rate / Final QoS**: Percentage of tasks meeting deadline
3. **Mean / Max Workers**: Resource usage
4. **Scaling vs No-op Actions**: Measures policy aggressiveness and stability

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

Baseline policies expose lightweight lifecycle hooks so they can share mirrored telemetry with the environment:

- `reset()` clears mirrored counters at episode boundaries.
- `update_from_info(info)` ingests `info['enqueued_total']` each step to keep busy-worker estimates in sync.

They otherwise follow the same observation/action conventions as RL agents, enabling fair comparisons.

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
