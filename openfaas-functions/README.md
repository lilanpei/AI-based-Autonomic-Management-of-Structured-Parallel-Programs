# OpenFaaS Functions - Skeleton Architecture

This directory contains the **OpenFaaS function implementations** for AI-based autoscaling. The functions implement a **producer-consumer workflow** for parallel task processing with RL-driven dynamic scaling.

---

## 📁 Directory Structure

```
openfaas-functions/
├── emitter/              # Task generator (1 instance)
├── worker/               # Task processor (1-32 instances, scalable)
├── collector/            # Result aggregator (1 instance)
├── template/
│   └── python3-http-skeleton/  # Custom long-running template
│       ├── Dockerfile            # Container definition
│       ├── index.py              # Flask + Waitress HTTP server
│       └── utilities/            # Shared utilities (symlinked)
└── stack.yaml            # Deployment configuration
```

---

## 🏗️ Architecture Overview

### **Workflow Pattern**

```
  input_queue         worker_queue        result_queue      output_queue
      │                    │                    │                 │
      ▼                    ▼                    ▼                 ▼
┌──────────┐         ┌──────────┐          ┌──────────┐      ┌---------┐
│ EMITTER  │────────▶│ WORKER-1 │─────────▶│COLLECTOR │─────▶│ RESULTS │
│  (1x)    │         ├──────────┤          │  (1x)    │      │         │
└──────────┘         │ WORKER-2 │          └──────────┘      └---------┘
                     ├──────────┤
                     │ WORKER-N │
                     └──────────┘
                          ▲
                          │
                    RL Agent Scales
```

### **Queue Flow**

- **input_queue**: Phase configurations (constant, burst, sine wave patterns)
- **worker_queue**: Tasks ready for processing
- **result_queue**: Completed task results
- **output_queue**: Final results with QoS metrics

### **Control Queues**

- **worker_control_syn_queue**: Scale-down signals (SYN)
- **worker_control_ack_queue**: Worker acknowledgments (ACK)
- **emitter/collector_control_syn_queue**: Termination signals

---

## 🎯 Function Roles

### **1. Emitter (Producer)**

**Purpose**: Generate and distribute tasks to workers

**Responsibilities**:
- Read phase configurations from `input_queue`
- Generate tasks based on workload patterns (constant, burst, sine wave)
- Push tasks to `worker_queue` for processing
- Support graceful termination

**Task Generation**:
- Phase-based generation with configurable rates (e.g., 300 tasks/min) pulled from `utilities/configuration.yml` `phase_definitions`
- Loads model and workload settings from `utilities/configuration.yml`
- Draws processing times via gamma sampling (`target_mean_processing_time`, `processing_time_shape`) and inverts the calibrated model to choose image sizes
- Sets task deadline: `2.0 × expected_duration`
- Streams multi-phase workloads defined in `phase_definitions`
- Each task includes: unique ID, simulated processing time, deadline, timestamps
- Uses calibrated model for realistic processing time simulation

**Control**: Responds to `TERMINATE` messages for graceful shutdown

---

### **2. Worker (Consumer)**

**Purpose**: Process tasks in parallel

**Responsibilities**:
- Poll `worker_queue` for tasks
- Simulate image processing workload
- Calculate QoS compliance (deadline met/missed)
- Push results to `result_queue`
- Support graceful scale-down

**Processing**:
- Simulates image processing with configurable processing times
- Tracks QoS: compares actual processing time vs. deadline
- Each result includes: task_id, work_time, QoS status, timestamps

**Control Messages**:
- `SCALE_DOWN` (SYN): Gracefully stop worker, requires ACK
- `TERMINATE`: Force stop worker

**Scalability**:
- **Horizontal scaling**: RL agent dynamically adds/removes workers (1-32)
- **Independent processing**: Each worker polls queue independently
- **Graceful shutdown**: Finish current task → Send ACK → Exit

---

### **3. Collector (Aggregator)**

**Purpose**: Collect and aggregate results

**Responsibilities**:
- Poll `result_queue` for completed tasks
- Calculate QoS metrics (deadline compliance)
- Push final results to `output_queue`

**QoS Tracking**:
- Compares task work time vs. deadline
- Marks task as QoS success/failure
- Aggregates metrics for RL agent observation

**Control**: Responds to `TERMINATE` messages for graceful shutdown

---

## 🔧 Custom Template: `python3-http-skeleton`

### **Why Custom Template?**

Standard OpenFaaS templates are for **short-lived, stateless functions**. Our workflow needs **long-running, stateful functions** that continuously poll Redis queues.

### **Key Design Decisions**

#### **1. Long-Running Execution**

- **Standard**: Function exits after one request
- **Our design**: Function runs indefinitely with `while True` loop
- **Benefit**: Continuous queue polling without cold starts

#### **2. HTTP Server: Flask + Waitress**

- **Flask**: Lightweight HTTP framework for request handling
- **Waitress**: Production WSGI server (no GIL issues)
- **OpenFaaS watchdog**: Proxies HTTP requests to Flask (port 5000)

#### **3. Extended Timeouts**

All timeouts set to **12 hours** in `stack.yaml`:
- `read_timeout`, `write_timeout`, `exec_timeout`
- Functions run until explicitly terminated via control messages
- No timeout-based interruptions

#### **4. Shared Utilities**

Utilities from `../utilities/` are copied into container:
- Redis client management
- Configuration loading
- Task generation logic
- QoS calculation
- Control message handling

#### **5. Resource Limits**

Low resource limits (10m CPU, 20Mi memory):
- Functions are I/O-bound (Redis polling)
- Minimal CPU/memory usage
- High pod density on cluster

---

## 🚀 Deployment Workflow

### **Build, Push, Deploy**

Use `faas-cli up` to build, push, and deploy all functions:

```bash
cd openfaas-functions
faas-cli up -f stack.yaml
```

This builds Docker images, pushes to registry (`lilanpei/*:latest`), and deploys to OpenFaaS.

### **Orchestrator Invocation**

Functions are invoked by `orchestrator/workflow_controller.py`:

```bash
cd orchestrator
python workflow_controller.py --tasks 100 --workers 3
```

**This does**:
1. Deploys emitter-1, collector-1, worker-1/2/3
2. Scales OpenFaaS queue-worker deployment
3. Invokes functions with payload (Redis config, timeouts, etc.)
4. Monitors queue progress in real-time

---

## 🎛️ Dynamic Scaling

### **Scale Up** (Add Workers)

Triggered by RL agent or `worker_scaler.py`:

1. Deploy new worker functions (e.g., worker-4, worker-5)
2. Scale OpenFaaS queue-worker deployment
3. Invoke new workers with payload
4. Workers start polling `worker_queue`

**Result**: More parallel processing capacity

### **Scale Down** (Remove Workers)

Graceful shutdown with SYN/ACK protocol:

1. Send `SCALE_DOWN` (SYN) to `worker_control_syn_queue`
2. Workers finish current task
3. Workers send ACK to `worker_control_ack_queue`
4. Orchestrator deletes worker functions

**Result**: Workers exit gracefully without dropping tasks

---

## 📊 Control Message Protocol

### **SYN/ACK Pattern**

Ensures graceful shutdown without dropping tasks:

```
Orchestrator → Worker:  SCALE_DOWN (SYN)
Worker:                 Finish current task
Worker → Orchestrator:  SCALE_DOWN (ACK)
Orchestrator:           Delete function
Worker:                 Pod terminates
```

### **Message Types**

- **SCALE_DOWN**: Graceful worker shutdown (requires ACK)
- **TERMINATE**: Force stop (no ACK required)

Messages include: type, action (SYN/ACK), timestamp, pod_name

---

## 🔍 Monitoring & Debugging

### **Function Status**

- `faas-cli list` - List deployed functions
- `kubectl get pods -n openfaas-fn` - Check pod status
- `kubectl logs -n openfaas-fn -l faas_function=worker-1` - View logs

### **Redis Queues**

- `redis-cli LLEN worker_queue` - Check queue length
- `redis-cli LRANGE worker_queue 0 5` - Peek at tasks

### **Workflow Monitoring**

`workflow_controller.py` provides real-time monitoring:
- Queue lengths (input, worker, result, output)
- Task completion rate (e.g., 23/100 tasks completed)
- QoS metrics (deadline compliance %)
- Worker replica count

---

## 🎓 Design Principles

### **1. Separation of Concerns**

Each function has a single responsibility:
- **Emitter**: Task generation
- **Worker**: Task processing  
- **Collector**: Result aggregation

### **2. Horizontal Scalability**

Only workers scale (1-32 instances):
- Emitter: 1 instance (single producer)
- Worker: N instances (parallel consumers)
- Collector: 1 instance (single aggregator)

### **3. Loose Coupling**

Redis queues enable decoupled communication:
- No service discovery needed
- No direct HTTP calls between functions
- Easy to add/remove workers dynamically

### **4. Graceful Degradation**

- Workers poll with timeout (no busy-waiting)
- SYN/ACK protocol for graceful shutdown
- Workers finish current task before exiting

### **5. Observability**

- Detailed logging with timestamps
- QoS tracking (deadline compliance %)
- Performance metrics (work time, queue lengths)

---

## 🔗 Integration with Orchestrator

### **Workflow Controller** (`orchestrator/workflow_controller.py`)

**Responsibilities**:
- Initialize OpenFaaS environment
- Deploy functions (emitter, collector, N workers)
- Generate task configurations (phase-based)
- Monitor queue progress in real-time
- Analyze results (QoS, completion rate)

**Usage**: `python workflow_controller.py --tasks 100 --workers 3`

### **Worker Scaler** (`orchestrator/worker_scaler.py`)

**Responsibilities**:
- Scale workers up (deploy new instances)
- Scale workers down (graceful shutdown with SYN/ACK)
- Handle control message protocol

**Usage**: `python worker_scaler.py +2` (scale up by 2)

---

## 📚 Related Documentation

- **Configuration**: `../utilities/configuration.yml`
- **Utilities**: `../utilities/utilities.py`

---

## 🎯 Key Takeaways

1. **Custom template** enables long-running, stateful functions
2. **Producer-consumer pattern** with Redis queues for loose coupling
3. **Horizontal scaling** of workers for parallel processing
4. **Graceful shutdown** via SYN/ACK control protocol
5. **QoS tracking** for deadline-aware task processing
6. **RL-driven scaling** for dynamic workload management

---

## 🚀 Quick Start

```bash
# 1. Build and deploy functions
cd openfaas-functions
faas-cli up -f stack.yaml

# 2. Deploy workflow with 3 workers
cd ../orchestrator
python workflow_controller.py --tasks 100 --workers 3

# 3. Monitor in separate terminals
watch -n 2 'kubectl get pods -n openfaas-fn'
watch -n 2 'redis-cli LLEN worker_queue'

# 4. Scale workers (manual or via RL agent)
python worker_scaler.py +2  # Add 2 workers
```

---

## 📝 Summary

The OpenFaaS functions implement a **scalable, fault-tolerant workflow** for parallel task processing:

- ✅ **Emitter**: Generates tasks from phase configurations
- ✅ **Worker**: Processes tasks in parallel (1-32 instances)
- ✅ **Collector**: Aggregates results and tracks QoS
- ✅ **Custom template**: Supports long-running, stateful functions
- ✅ **Control protocol**: Graceful scaling via SYN/ACK messages
- ✅ **Redis queues**: Loose coupling and high throughput
- ✅ **RL integration**: Dynamic autoscaling based on workload

**This architecture enables RL-based autoscaling for dynamic workload management!** 🎉
