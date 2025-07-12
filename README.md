# AI-based Autonomic Management of Structured Parallel Programs

## 🎯 Objective

This project demonstrates an **AI-enabled autonomic workflow orchestration system** for structured parallel programs, leveraging **OpenFaaS** for serverless task execution and **Redis** for message passing. It supports different parallel execution patterns (e.g., **pipeline** and **farm**) with dynamic feedback and scalability.

Key features:
- Autonomous task generation, processing, and collection
- Workflow modes: Pipeline and Farm
- Manual and automatic worker scaling
- Quality-of-Service (QoS) monitoring

---

## 🗂️ File Structure

```bash
AI-based-Autonomic-Management-of-Structured-Parallel-Programs/
├── openfaas-functions/
│   ├── emitter/
│   │   ├── handler.py           # Emits tasks from input queue to worker queue
│   │   └── requirements.txt    # Dependencies
│   ├── worker/
│   │   ├── handler.py           # Performs matrix multiplication and push results into result queue
│   │   └── requirements.txt    # Dependencies
│   ├── collector/
│   │   ├── handler.py           # Collects results, sort result by emit timestamp into output queue, and evaluates QoS, sends feedback to input queue if enabled
│   │   └── requirements.txt     # Dependencies
│   └── stack.yaml               # OpenFaaS stack definition
│
└── orchestrator/
    ├── env_init.py              # Initializes Redis queues and default config
    ├── emitter_init.py          # Triggers emitter function with configuration
    ├── worker_init.py           # Scales worker and invokes OpenFaaS worker function
    ├── collector_init.py        # Triggers collector function with or without feedback
    ├── worker_scaler.py         # Manually scales worker pods up/down (+1, -1, etc.)
    ├── task_generator.py        # Generates synthetic matrix multiplication tasks
    ├── workflow_controller.py   # Automates full pipeline/farm workflow
    ├── run_workflow.py          # Testing and debugging
    └── utilities.py             # Shared utility functions for config and K8s client
```

---

## 🚀 How to Run

### ✅ Prerequisites

- Kubernetes with **OpenFaaS** installed
- `faas-cli`, `kubectl`, `Python 3.13.5`
- Redis serves both OpenFaaS functions and local scripts
- Install Python dependencies:
  ```bash
  pip install -r orchestrator/requirements.txt

---

## 🔧 Step-by-Step Instructions

### 1. Build and Deploy OpenFaaS Functions

```bash
cd openfaas-functions
faas-cli up
```
### 2. Start Workflow
Choose between **Pipeline** and **Farm** execution modes:

### ▶️ Pipeline Mode

- Starts with `env_init.py` to sets up the Redis queues and default configuration.
- One worker only
- Optional feedback to generate new tasks
- Example with feedback enabled:

```bash
cd orchestrator
python workflow_controller.py --mode pipeline --workers 1 --feedback --tasks 1000
```

### 🌾 Farm Mode

- Starts with `env_init.py`
- N workers (set via --workers flag)
- Optional feedback
- Example with 4 workers and feedback disabled:

```bash
cd orchestrator
python workflow_controller.py --mode farm --workers 4 --feedback --tasks 1000
```
## 🔁 Manual Worker Scaling
While the workflow is running, use `worker_scaler.py` to manually adjust worker replicas:

```bash
# Scale up by 2 workers
python orchestrator/worker_scaler.py +2

# Scale down by 1 worker
python orchestrator/worker_scaler.py -1
```
> Note: The scale range is limited to -2, -1, 0, +1, +2

## 📊 Monitoring & Output

- **Collector** summarizes total tasks processed and deadline success rate (QoS %)
- **Redis output** queue contains final structured results
- When `collector_feedback_flag=True`, additional tasks are auto-generated based on prior results

```bash
# Testing and Debugging
python run_workflow.py python workflow_controller.py --mode farm --workers 4 --tasks 10
```
## ⚙️ Customization

- Update `orchestrator/configuration.yml` to customize:
  - Redis host/port
  - Input/output/result queue names
- Adjust default task deadlines in `task_generator.py` as needed
