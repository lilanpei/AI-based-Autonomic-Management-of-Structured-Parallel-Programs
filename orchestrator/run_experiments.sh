#!/bin/bash
# run_experiments.sh

# Arguments
MODE="farm"
TASKS=1000
CYCLES=1
REPEATS=5
WINDOW_DURATION_LIST=(30 100 200 300)
WORKERS_LIST=(32 16 8 4 2 1)

for WINDOW in "${WINDOW_DURATION_LIST[@]}"; do
  for WORKERS in "${WORKERS_LIST[@]}"; do
    for ((i=1; i<=REPEATS; i++)); do
      echo "[INFO] Running experiment: workers=${WORKERS}, run ${i}/${REPEATS}"
      python run_workflow.py python workflow_controller.py \
        --mode "$MODE" \
        --workers "$WORKERS" \
        --tasks "$TASKS" \
        --cycles "$CYCLES" \
        --window "$WINDOW"

      if [ $? -ne 0 ]; then
        echo "[WARNING] Run failed for workers=${WORKERS} (run $i), skipping to next..."
      else
        echo "[SUCCESS] Completed run $i for workers=${WORKERS}"
      fi
      sleep 20 # Sleep to avoid overwhelming the system
    done
  done
done