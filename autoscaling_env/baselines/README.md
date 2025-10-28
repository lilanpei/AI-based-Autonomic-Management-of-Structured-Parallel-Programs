# Baseline Autoscaling Policies

This directory contains **reactive baseline policies** for autoscaling comparison. These policies serve as benchmarks to evaluate the performance of RL-based autoscaling agents.

---

## 📁 Files

- **`reactive_baselines.py`**: Policy implementations (ReactiveAverage, ReactiveMaximum)
- **`README.md`**: This documentation

---

## 🎯 Baseline Policies

### **1. ReactiveAverage**

**Strategy**: Scale based on average processing time

**Formula**:
```
optimal_workers = (worker_queue / avg_time) + (arrival_rate × avg_time)
```

**Components**:
- **First term**: Workers needed to clear current queue
- **Second term**: Workers needed to handle incoming tasks

**Behavior**:
- Conservative scaling
- Smooth adjustments based on average load
- Good for stable workloads

---

### **2. ReactiveMaximum**

**Strategy**: Scale based on maximum processing time

**Formula**:
```
optimal_workers = (worker_queue / max_time) + (arrival_rate × max_time)
```

**Components**:
- **First term**: Workers needed to clear current queue
- **Second term**: Workers needed to handle incoming tasks

**Behavior**:
- Aggressive scaling
- Responds to worst-case scenarios
- Better for bursty workloads

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

Policies map calculated deltas to discrete actions:
- `delta ≥ 2.5`: Scale up by 2
- `delta ≥ 1.0`: Scale up by 1
- `delta ≤ -2.5`: Scale down by 2
- `delta ≤ -1.0`: Scale down by 1
- Otherwise: No change

---

## 🚀 Usage

### **Test Single Policy**

```bash
cd autoscaling_env
python test_reactive_baselines.py --agent average --steps 15
```

### **Test Both Policies**

```bash
python test_reactive_baselines.py --agent both --steps 15
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
