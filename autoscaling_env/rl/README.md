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
| `test_sarsa.py` | Deterministic policy roll-out for evaluation of a saved SARSA model. |
| `plot_training.py` | Utility to regenerate training plots from stored metrics. |
| `utils.py` | Shared helpers (discretisation factory, logging, run directory management, JSON export). |
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
  --max-steps 50 \
  --step-duration 8 \
  --observation-window 8 \
  --initial-workers 12 \
  --bins 10 10 10 10 16 10 10 10 10 \
  --initialize-workflow
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
- `--checkpoint-every`: save intermediate models every _N_ episodes.
- `--output-dir`: base directory for experiment artefacts (default `runs/`). Each run
  gets a timestamped subdirectory containing logs, plots, metrics, and models.
- `--initialize-workflow`: ensures the OpenFaaS workflow is deployed before the
  first episode. Omit if the cluster is already primed.

### Produced Artefacts

Each run directory includes:

```
<runs>/sarsa_run_<timestamp>/
  logs/training.log          # Structured log output
  models/sarsa_final.pkl     # Final SARSA agent (pickle)
  models/sarsa_epXXXX.pkl    # Optional checkpoints
  training_metrics.json      # Episode-level metrics
  plots/training_curves.png  # Reward/QoS/epsilon curves (auto-generated)
```

Interrupting training with `Ctrl+C` triggers a graceful shutdown that persists
`models/sarsa_interrupt.pkl` and notes how many episodes completed.

## Evaluating a Saved Model

```bash
python -m autoscaling_env.rl.test_sarsa \
  --model autoscaling_env/rl/runs/sarsa_run_<timestamp>/models/sarsa_final.pkl \
  --episodes 10 \
  --max-steps 50 \
  --step-duration 8 \
  --observation-window 8 \
  --initial-workers 12 \
  --initialize-workflow
```

The evaluation script executes a greedy policy (always selecting the
highest-valued action) and prints per-episode summaries. A JSON report is written
via `--output` (default `runs/eval_results.json`).

## Re-plotting Training Metrics

If you adjust plotting styles or regenerate charts after cleaning the `plots/`
folder, use:

```bash
python -m autoscaling_env.rl.plot_training \
  autoscaling_env/rl/runs/sarsa_run_<timestamp>/training_metrics.json \
  --output autoscaling_env/rl/runs/sarsa_run_<timestamp>/plots/training_curves.png
```

The script rebuilds the combined Reward/QoS/Epsilon plot from the stored JSON.
If `--output` is omitted, the plot is saved alongside the metrics file under a
`plots/` subdirectory.

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
