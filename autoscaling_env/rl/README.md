# Reinforcement Learning (RL) Toolkit

This package hosts the reinforcement-learning stack used to train tabular SARSA
policies against the `OpenFaaSAutoscalingEnv`. It complements the reactive
baselines by providing a configuration-driven workflow for training, evaluating,
plotting, and persisting RL agents that understand the multi-phase workload
produced by `utilities/configuration.yml`.

## Contents

| File | Purpose |
| --- | --- |
| `sarsa_agent.py` | Tabular SARSA agent with observation discretisation, epsilon decay, and persistence helpers. |
| `train_sarsa.py` | Command-line training loop with logging, checkpointing, and metric export. |
| `test_sarsa.py` | Deterministic policy roll-out with run-directory logging and per-episode plotting. |
| `plot_training.py` | Utility to regenerate training plots from stored metrics. |
| `utils.py` | Shared helpers (discretisation factory, logging, run directory management, JSON export). |
| `../compare_policies.py` | Compare trained SARSA against reactive baselines on a shared single-episode plot. |
| `runs/` | Default output directory for experiments (ignored by git). |

## Prerequisites

- Python 3.10+
- Project dependencies installed (see repository root for instructions).
- `gym` (or `gymnasium`) with NumPy 1.x compatibility. The scripts emit a warning
  if an older Gym build is detected; migrating to Gymnasium is recommended but
  not yet mandatory.
- Access to a Kubernetes/OpenFaaS cluster configured by `utilities/configuration.yml`.
  Training invokes the real `OpenFaaSAutoscalingEnv`, so ensure you can deploy
  functions and connect to Redis from the machine running the script.

> **Tip:** Run all commands from the project root and prefer the module form
> (`python -m autoscaling_env.rl.<script>`) so package imports resolve without
> modifying `PYTHONPATH`.

## Training a SARSA Agent

```bash
python -m autoscaling_env.rl.train_sarsa \
  --episodes 100 \
  --max-steps 30 \
  --step-duration 8 \
  --observation-window 8 \
  --initial-workers 12 \
  --bins 4 30 4 40 16 18 18 15 20 \
  --eval-episodes 3 \
  --phase-shuffle \
  --phase-shuffle-seed 42
```

Key arguments:

- `--episodes`: number of training episodes.
- `--max-steps`: cap on steps per episode; episodes end earlier if the workload
  drains.
- `--step-duration`, `--observation-window`, `--initial-workers`: forwarded to
  `OpenFaaSAutoscalingEnv`.
- `--bins`: nine integers defining discretisation granularity for each element
  of the observation vector (queues, workers, processing times, arrival rate,
  QoS). Higher values increase state resolution but enlarge the Q-table.
- `--alpha`, `--gamma`, `--epsilon`, `--epsilon-min`, `--epsilon-decay`: SARSA
  hyperparameters.
- `--checkpoint-every`: save intermediate models every _N_ episodes; when
  `--eval-episodes > 0`, each checkpoint runs greedy evaluation rollouts.
- `--eval-episodes`: number of evaluation episodes per checkpoint (and after
  training). Aggregated metrics (reward, final QoS across all tasks, scaling
  actions, max workers) are written to `evaluation_metrics.json`.
- `--output-dir`: base directory for experiment artefacts (default `runs/`). Each run
  gets a timestamped subdirectory containing logs, plots, metrics, and models.
- `--phase-shuffle`: randomizes phase order each training episode (for training only).
- `--phase-shuffle-seed`: seed for the random phase shuffling.

### Produced Artefacts

Each run directory includes:

```
<runs>/sarsa_run_<timestamp>/
  logs/training.log          # Structured log output
  models/sarsa_final.pkl     # Final SARSA agent (pickle)
  models/sarsa_epXXXX.pkl    # Optional checkpoints
  training_metrics.json      # Episode-level training metrics
  evaluation_metrics.json    # Checkpoint evaluation summaries (if enabled)
  plots/training_curves.png  # Training curves (with evaluation overlay when present)
```

Interrupting training with `Ctrl+C` triggers a graceful shutdown that persists
`models/sarsa_interrupt.pkl` and notes how many episodes completed.

## Evaluating a Saved Model

```bash
python -m autoscaling_env.rl.test_sarsa \
  --model autoscaling_env/rl/runs/sarsa_run_<timestamp>/models/sarsa_final.pkl \
  --episodes 10 \
  --max-steps 30 \
  --step-duration 8 \
  --observation-window 8 \
  --initial-workers 12 \
```

The evaluation script executes a greedy policy (always selecting the
highest-valued action) and prints per-episode summaries. A JSON report is written
via `--output` (default `runs/eval_results.json`).

### Compare Against Reactive Baselines on a Single Episode

```bash
python -m autoscaling_env.compare_policies \
  --model autoscaling_env/rl/runs/sarsa_run_<timestamp>/models/sarsa_final.pkl \
  --max-steps 30 \
  --initial-workers 12 \
  --agents agent reactiveaverage reactivemaximum
```

Creates a timestamped directory under `autoscaling_env/runs/comparison/` with
per-step logs, aggregated mean/std summaries, and a shared plot for the selected
policies. Once `aggregated_results.json` exists for a run you can reproduce
plots or focus on a subset of policies without rerunning simulations:

```bash
python -m autoscaling_env.compare_policies \
  --plot-only \
  --input-dir autoscaling_env/runs/comparison/compare_<timestamp> \
  --agents reactiveaverage
```

## Re-plotting Training Metrics

If you adjust plotting styles or regenerate charts after cleaning the `plots/`
folder, use:

```bash
python -m autoscaling_env.rl.plot_training \
  autoscaling_env/rl/runs/sarsa_run_<timestamp>/training_metrics.json \
  --output autoscaling_env/rl/runs/sarsa_run_<timestamp>/plots/training_curves.png
```

The plotter accepts either `training_metrics.json` or `logs/training.log` and
overlays checkpoint evaluation means ± 1σ whenever `evaluation_metrics.json` is
present (or supplied via `--evaluation`). If `--output` is omitted, the figure
is saved next to the metrics file inside a `plots/` subdirectory.

## Configuration Tips

- **Discretisation:** Start with coarser bins, then increase select dimensions
  (e.g., worker count, QoS) once training stabilises. Remember that the total
  number of table entries scales with the product of all bin counts.
- **Workflow spin-up:** Initialising the OpenFaaS workflow can take up to a
  minute; the training script logs each phase (queue reset, worker scaling,
  task generation) so you can monitor progress.
- **Gym warning:** If you see the deprecation notice for Gym, migrate to
  Gymnasium by replacing `import gym` with `import gymnasium as gym` in
  `openfaas_autoscaling_env.py` and ensuring the dependency is installed.

## Troubleshooting

- `TypeError: 'float' object cannot be interpreted as an integer`: ensure all
  `--bins` values are integers.
- Observation fallback errors (e.g., missing phase config) were hardened to rely
  on `phase_definitions`. Verify `utilities/configuration.yml` contains the
  phases you expect if arrival-rate defaults look off.

---

With the RL toolkit wired in, you can iterate on policy learning while keeping
full traceability of logs, metrics, and trained agents. Happy experimenting! 🚀
