# Autoscaling Environment

The `autoscaling_env` package houses the **OpenFaaS Autoscaling Gym Environment** together with baseline policies, SARSA training scripts, and comparison tooling. Use it to prototype, train, and benchmark autoscaling strategies under realistic workload traces.

---

## At a Glance

| Path | Purpose |
|------|---------|
| `openfaas_autoscaling_env.py` | Gym environment (9-D observations, 3 discrete actions) with mirrored enqueue telemetry |
| `baselines/` | Reactive policies and helpers |
| `rl/` | SARSA & lightweight DQN agents plus training/eval utilities |
| `compare_policies.py` | One-shot evaluation across policies |

---

## Environment Essentials

### Observation Vector (9)

`[input_queue, worker_queue, result_queue, output_queue, workers, avg_processing_time, max_processing_time, arrival_rate, qos_rate]`

> **Arrival rate** is now derived from a mirrored Redis enqueue counter (`<queue>:enqueued_total`), eliminating the previous lag between submissions and completions.

### Action Space

`[-1, 0, +1]` → scale down, hold, or scale up a single worker. Actions are clipped to `[min_workers, max_workers]`.

### Reward Summary

The per-step reward blends four core terms with two conditional bonuses (`utilities/configuration.yml`):

1. **QoS delta** – `w_qos * (qos_rate - target_qos)` rewards hitting the QoS target and penalises shortfalls.
2. **Backlog pressure** – `- w_backlog * (max(0, worker_q / queue_target - 1))²` scales quadratically with queue overflow.
3. **Scaling friction** – `- w_scaling * |delta|` discourages oscillation; delta is the applied replica change.
4. **Efficiency drag** – `- w_eff * max(0, workers - balanced_workers)` when queues are idle (`worker_q ≤ idle_queue_threshold`) and QoS is healthy.

Conditional bonuses provide directional guidance:

- **Overload scale-up** – `+ w_scale_up * |delta|` when backlog/QoS indicate stress and the action adds workers.
- **Calm scale-down** – `+ w_scale_down * |delta|` when the system is healthy and the action removes workers.
- **Stability** – `+ w_qos * 0.1` for maintaining QoS ≥ target with queues under 50% of `queue_target`.

Tuning knobs: `target_qos`, `queue_target`, `balanced_workers`, `idle_queue_threshold`, and the weight set `{qos, backlog, scaling, efficiency, scale.up, scale.down}`.

Terminal shaping draws from `reward.unfinished_penalty_scale` (negative when work remains at max steps) and `reward.completion_bonus` (positive for draining queues early).

---

## Quick Start

```bash
# 1. Explore reactive baselines
cd autoscaling_env/baselines
python test_reactive_baselines.py --agent both --steps 30 --step-duration 8 --horizon 8

# 2. Train SARSA (100 episodes × 30 steps)
cd rl
python train_sarsa.py --episodes 100 --max-steps 30 --step-duration 8 \
    --initial-workers 12 --eval-episodes 3 --phase-shuffle --phase-shuffle-seed 42

# 3. Train lightweight DQN (120 episodes × 30 steps)
python train_dqn.py --episodes 120 --max-steps 30 --step-duration 8 \
    --initial-workers 12 --eval-episodes 3 \
    --target-tau 0.01

# 4. Evaluate trained models
python test_sarsa.py --model runs/sarsa_run_<timestamp>/models/sarsa_final.pkl --initial-workers 12
python test_dqn.py --model runs/dqn_run_<timestamp>/models/dqn_final.pt --initial-workers 12

# 5. Compare SARSA vs reactive policies
python compare_policies.py --model rl/runs/sarsa_run_<timestamp>/models/sarsa_final.pkl \
    --initial-workers 12 --max-steps 30 --agents sarsa reactiveaverage reactivemaximum
```

Each trainer creates timestamped run directories (e.g., `sarsa_run_*`, `dqn_run_*`) with per-step logs,
metrics (reward, QoS, **mean/max workers, processed tasks, QoS violations, unfinished tasks**), checkpoints,
and plots. Evaluation scripts mirror the SARSA-style tabular reporting and per-episode figures.

- RL training runs are written under `autoscaling_env/rl/runs/` when invoked from the `rl/` directory.
- Policy comparison runs are written under `autoscaling_env/runs/comparison/` when using `compare_policies.py`.

---

## Key Features

- **Task-aware episode termination** – Finish early when all queues drain; terminal penalties/bonuses configured in `utilities/configuration.yml`.
- **Phase-based workload generator** – Constant, burst, or oscillating task rates controlled by YAML config.
- **Mirrored enqueue telemetry** – Task submissions increment `worker_queue:enqueued_total`, allowing step-level arrival-rate estimates without waiting for completion samples.
- **Accurate QoS tracking** – Metrics computed from completion timestamps (not sampling instants).
- **Sliding observation window** – Rolling averages of processing times, arrivals, and QoS over a configurable horizon.
- **Strict scaling constraints** – Worker counts clipped to `min_workers` … `max_workers` each step.

---

## Evaluation Metrics

We track the following per episode:

- Total reward
- Mean QoS rate and final QoS (task-level)
- Mean/max workers
- Scaling / no-op counts
- QoS violation count (tasks exceeding the QoS deadline)
- Unfinished task count (tasks remaining when episode ends)

These metrics are emitted to JSON, plots, and comparison logs for downstream analysis.

---

## Configuration Highlights

Environment constructor knobs:

- `max_workers`, `min_workers`, `initial_workers`
- `observation_window` (seconds)
- `step_duration` (full step time including scaling latency)
- `max_steps`

Reward knobs live under `reward` in `utilities/configuration.yml` (targets, thresholds, and weights listed above).

---

## Baselines & RL Agents

- **ReactiveAverage / ReactiveMaximum** – Horizon-based heuristics that track mirrored enqueue totals to estimate busy workers (see `baselines/README.md`).
- **SARSA** – Tabular agent with discretised observations, eligibility traces (λ) and epsilon-greedy exploration (details in `rl/README.md`).
- **Lightweight DQN** – Two-layer neural network with replay buffer, Double DQN target updates, and observation normalisation for fast inference (details in `rl/README.md`).

Training artefacts include `training_metrics.json`, QoS/reward plots, `training_state_visits.json` for SARSA state coverage, and per-run checkpoints (`.pkl`/`.pt`).

---

## Comparisons & Plots

`compare_policies.py` evaluates SARSA, optionally a DQN checkpoint, and the reactive baselines, saving:

- Step-by-step logs per agent
- Aggregated summaries (`aggregated_results.json`) with mean reward/QoS/worker metrics and scaling/no-op counts
- Comparison plots (`comparison_plots.png`)

Use `--plot-only --input-dir <run_dir>` to regenerate visualisations from saved results. To skip recomputing
reactive baselines on subsequent runs, pass `--baseline-cache <existing_compare_dir>`.

---

## Integration Notes

- **OpenFaaS Skeleton (Farm)** – Managed through the Kubernetes API and OpenFaaS gateway.
- **Task pipeline** – Redis queues buffer tasks between emitter, workers, and collector functions.
- **Shared utilities** – `utilities/utilities.py` orchestrates deployments; configuration is centralised in `utilities/configuration.yml`.

---

## Summary

The autoscaling environment offers a complete playground for experimenting with scaling policies:

- ✅ Gym-compatible RL interface
- ✅ Realistic workload simulations
- ✅ Tunable QoS-aware rewards
- ✅ Baseline policy benchmarks
- ✅ Rich diagnostics for evaluation and comparison

Train, tweak, and compare autoscaling strategies—all within one reproducible toolkit. 🚀
