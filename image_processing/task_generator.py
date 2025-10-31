#!/usr/bin/env python3
"""
Four-Phase Task Generator

Generates image processing tasks with four distinct arrival patterns:
- Phase 1: Steady Low Load (30% of base rate)
- Phase 2: Steady High Load (150% of base rate)
- Phase 3: Slow Oscillation (50-150% of base rate, 1 cycle)
- Phase 4: Fast Oscillation (30-170% of base rate, 4 cycles)

Task Type:
- Single task type: image_processing (complete pipeline)
- Pipeline: Thumbnail → Compression → Metadata → Conversion
- Based on calibrated timing model
"""

import os
import sys
import time
import json
import uuid
import redis
import random
import numpy as np
from datetime import datetime

# Add the project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from utilities.utilities import get_config, init_redis_client, get_utc_now

# Single task type: image_processing (complete pipeline)
TASK_TYPES = {
    "image_processing": {
        "name": "image_processing",
        "description": "Complete image processing pipeline (all 4 stages)",
        # Expected durations from calibration (complete pipeline)
        "duration_512": 45,      # ms for 512x512
        "duration_1024": 175,    # ms for 1024x1024
        "duration_2048": 708,    # ms for 2048x2048
        "duration_4096": 2790,   # ms for 4096x4096
        # Image size distribution
        "image_sizes": [512, 1024, 2048, 4096],
        "size_weights": [0.1, 0.3, 0.5, 0.1]  # Favor 2048 (typical size)
    }
}

# QoS Model: Deadline-Based
# Deadline = Coefficient × Expected Duration × User Priority
DEADLINE_COEFFICIENT = 2.0  # Can be 2 or 3 for different strictness levels
USER_PRIORITY = 1.0          # Simplified: always 1.0 for now


def calculate_deadline(expected_duration, coefficient=None, user_priority=None):
    """
    Calculate deadline using QoS model:
    Deadline = Coefficient × Expected Duration × User Priority

    Args:
        expected_duration: Expected processing time (seconds)
        coefficient: Deadline coefficient (default: DEADLINE_COEFFICIENT)
        user_priority: User priority multiplier (default: USER_PRIORITY)

    Returns:
        deadline: Deadline in seconds
    """
    if coefficient is None:
        coefficient = DEADLINE_COEFFICIENT
    if user_priority is None:
        user_priority = USER_PRIORITY

    # QoS Model: Deadline = Coefficient × Expected Duration × User Priority
    deadline = coefficient * expected_duration * user_priority

    # Apply bounds
    min_deadline = 0.1   # 100ms minimum
    max_deadline = 30.0  # 30s maximum

    return max(min_deadline, min(deadline, max_deadline))


def generate_task(program_start_time):
    """
    Generate an image processing task (complete pipeline)

    Returns:
        dict: Task payload
    """
    task_config = TASK_TYPES["image_processing"]
    priority = "normal"

    # Select image size based on distribution
    image_size = random.choices(
        task_config["image_sizes"],
        weights=task_config["size_weights"]
    )[0]

    # Get expected duration from calibration data
    duration_map = {
        512: task_config["duration_512"],
        1024: task_config["duration_1024"],
        2048: task_config["duration_2048"],
        4096: task_config["duration_4096"]
    }
    expected_duration = duration_map[image_size] / 1000  # Convert to seconds

    # Calculate deadline using QoS model
    deadline = calculate_deadline(expected_duration)

    return {
        "task_id": str(uuid.uuid4()),
        "task_gen_timestamp": (get_utc_now() - program_start_time).total_seconds(),
        "task_application": task_config["name"],
        "task_priority": priority,
        "task_expected_duration": expected_duration,
        "task_data_size": image_size,
        "task_deadline": deadline
    }


def push_task_to_queue(redis_client, queue_name, task_json, task_index):
    """
    Push a task to Redis queue with error handling

    Returns:
        bool: True if successful
    """
    try:
        redis_client.lpush(queue_name, task_json)
        return True
    except redis.exceptions.ConnectionError as e:
        print(f"[ERROR] Redis connection failed: {e}. Retrying in 5s...")
        time.sleep(5)
        try:
            new_client = init_redis_client()
            new_client.lpush(queue_name, task_json)
            print(f"[INFO] Retry succeeded for task {task_index}.")
            return True
        except Exception as retry_e:
            print(f"[ERROR] Retry failed for task {task_index}: {retry_e}", file=sys.stderr)
    except Exception as e:
        print(f"[ERROR] Failed to push task {task_index}: {e}", file=sys.stderr)

    return False


def get_phase_rate(phase, base_rate, time_in_phase, phase_duration):
    """
    Calculate task arrival rate for current phase and time

    Args:
        phase: Phase number (1-4)
        base_rate: Base arrival rate (tasks/minute)
        time_in_phase: Time elapsed in current phase (seconds)
        phase_duration: Total duration of phase (seconds)

    Returns:
        float: Tasks per minute for current time
    """
    if phase == 1:
        # Phase 1: Steady Low Load (30% of base rate)
        return base_rate * 0.3

    elif phase == 2:
        # Phase 2: Steady High Load (150% of base rate)
        return base_rate * 1.5

    elif phase == 3:
        # Phase 3: Slow Oscillation (period = phase_duration)
        # Oscillates between 50% and 150% of base rate
        progress = time_in_phase / phase_duration  # 0 to 1
        oscillation = 0.5 * np.sin(2 * np.pi * progress) + 1.0  # 0.5 to 1.5
        return base_rate * oscillation

    elif phase == 4:
        # Phase 4: Fast Oscillation (4 cycles per phase)
        # Oscillates between 30% and 170% of base rate
        progress = time_in_phase / phase_duration  # 0 to 1
        oscillation = 0.7 * np.sin(8 * np.pi * progress) + 1.0  # 0.3 to 1.7
        return base_rate * oscillation

    else:
        return base_rate


def get_target_tasks_for_phase(phase, base_rate, phase_duration):
    """
    Calculate exact target number of tasks for a phase
    
    Args:
        phase: Phase number (1-4)
        base_rate: Base arrival rate (tasks/minute)
        phase_duration: Total duration of phase (seconds)
    
    Returns:
        int: Exact target number of tasks
    """
    if phase == 1:
        # Phase 1: Steady 30%
        avg_rate = base_rate * 0.3
    elif phase == 2:
        # Phase 2: Steady 150%
        avg_rate = base_rate * 1.5
    elif phase == 3:
        # Phase 3: Oscillation 50-150%, average = 100%
        avg_rate = base_rate * 1.0
    elif phase == 4:
        # Phase 4: Oscillation 30-170%, average = 100%
        avg_rate = base_rate * 1.0
    else:
        avg_rate = base_rate
    
    return int(avg_rate * (phase_duration / 60))


def generate_tasks_for_phase(phase, phase_name, base_rate, phase_duration, window_duration,
                             redis_client, input_queue, program_start_time):
    """
    Generate tasks for one phase with specific arrival pattern
    Uses controlled Poisson to hit exact target while maintaining variance

    Args:
        phase: Phase number (1-4)
        phase_name: Phase description
        base_rate: Base arrival rate (tasks/minute)
        phase_duration: Total duration of this phase (seconds)
        window_duration: Duration of each generation window (seconds)
        redis_client: Redis client
        input_queue: Queue name
        program_start_time: Program start timestamp

    Returns:
        int: Total tasks generated in this phase
    """
    # Calculate exact target for this phase
    target_tasks = get_target_tasks_for_phase(phase, base_rate, phase_duration)
    
    print(f"\n{'='*70}")
    print(f"PHASE {phase}: {phase_name}")
    print(f"{'='*70}")
    print(f"Duration: {phase_duration}s ({phase_duration/60:.1f} min)")
    print(f"Window size: {window_duration}s")
    print(f"Target tasks: {target_tasks}")
    print(f"{'='*70}\n")

    num_windows = int(phase_duration / window_duration)
    phase_start_time = time.time()
    total_tasks_generated = 0

    for window in range(num_windows):
        window_start = time.time()
        time_in_phase = (window_start - phase_start_time)

        # Calculate arrival rate for this window
        rate = get_phase_rate(phase, base_rate, time_in_phase, phase_duration)
        expected_tasks = rate * (window_duration / 60)
        
        # Controlled Poisson: adjust to hit exact target
        remaining_tasks = target_tasks - total_tasks_generated
        remaining_windows = num_windows - window
        
        if remaining_windows > 1:
            # Use Poisson with adjustment to stay on track
            adjusted_expected = remaining_tasks / remaining_windows
            num_tasks = np.random.poisson(adjusted_expected)
            # Clamp to remaining tasks
            num_tasks = min(num_tasks, remaining_tasks)
        else:
            # Last window: generate exactly remaining tasks
            num_tasks = remaining_tasks

        print(f"\n[Window {window+1}/{num_windows}] Time: {time_in_phase:.1f}s, Rate: {rate:.2f} tasks/min, Tasks: {num_tasks}")

        if num_tasks <= 0:
            print("  No tasks in this window")
            # No sleep - move to next window immediately to avoid timing overhead
            continue

        # Generate inter-arrival times (exponential distribution for Poisson process)
        inter_arrival_times = np.random.exponential(window_duration / num_tasks, num_tasks)

        tasks_in_window = 0
        task_start_time = time.time()
        cumulative_delay = 0
        
        for i, delay in enumerate(inter_arrival_times):
            # Calculate target time for this task (absolute time)
            cumulative_delay += delay
            target_time = task_start_time + cumulative_delay
            
            # Generate task
            task_payload = generate_task(program_start_time)
            task_json = json.dumps(task_payload)

            # Push to queue
            success = push_task_to_queue(redis_client, input_queue, task_json, i + 1)

            if success:
                tasks_in_window += 1
                total_tasks_generated += 1

            # Sleep until target time (compensates for processing time)
            current_time = time.time()
            sleep_time = target_time - current_time
            if sleep_time > 0:
                time.sleep(sleep_time)

        print(f"  Generated {tasks_in_window} tasks (total: {total_tasks_generated}/{target_tasks})")

        # Note: No window enforcement sleep - timing is controlled by cumulative_delay

    print(f"\n[PHASE {phase} COMPLETE] Total tasks generated: {total_tasks_generated}")
    return total_tasks_generated


def generate_phase_configs():
    """Generate phase configuration objects using configuration.yml definitions."""
    config = get_config()
    configured_phases = config.get("phase_definitions")
    base_rate = config.get("base_rate")
    number_of_phases = config.get("number_of_phases")

    if not configured_phases:
        raise ValueError("phase_definitions missing from configuration.yml")

    phases = []
    for _ , phase_cfg in enumerate(configured_phases):
        multiplier = phase_cfg.get("phase_multiplier")
        duration = int(phase_cfg.get("phase_duration"))
        window = int(phase_cfg.get("window_duration"))

        target_tasks = int(base_rate * multiplier * (duration / 60))

        phase = {
            "type": "phase_config",
            "phase_number": phase_cfg.get("phase_number"),
            "phase_name": phase_cfg.get("phase_name"),
            "base_rate": base_rate,
            "phase_multiplier": multiplier,
            "phase_duration": duration,
            "window_duration": window,
            "target_tasks": target_tasks,
            "phase_pattern": phase_cfg.get("phase_pattern"),
        }

        if phase["phase_pattern"].startswith("oscillation"):
            phase["oscillation_min"] = phase_cfg.get("oscillation_min")
            phase["oscillation_max"] = phase_cfg.get("oscillation_max")
            phase["oscillation_cycles"] = phase_cfg.get("oscillation_cycles")

        phases.append(phase)

        print(f"\n{'='*70}")
        print(f"PHASE CONFIG GENERATOR")
        print(f"{'='*70}")
        print(f"Number of phases: {number_of_phases}")
        print(f"Phase {phase['phase_number']}:")
        print(f"Base rate: {phase['base_rate']} tasks/min")
        print(f"Multiplier: {phase['phase_multiplier']}")
        print(f"Duration: {phase['phase_duration']}s")
        print(f"Window duration: {phase['window_duration']}s")
        print(f"Target tasks: {phase['target_tasks']}")
        print(f"Pattern: {phase['phase_pattern']}")
        if phase['phase_pattern'].startswith("oscillation"):
            print(f"Oscillation min: {phase['oscillation_min']}")
            print(f"Oscillation max: {phase['oscillation_max']}")
            print(f"Oscillation cycles: {phase['oscillation_cycles']}")
        print(f"{'='*70}\n")

    return phases


def main():
    """Main: Generate phase configs and push to input_queue"""

    # Get program start time
    program_start_time_str = os.getenv("START_TIMESTAMP")
    if program_start_time_str:
        program_start_time = datetime.fromisoformat(program_start_time_str)
    else:
        print("[ERROR] START_TIMESTAMP environment variable not set.", file=sys.stderr)
        sys.exit(1)

    task_generation_start_time = (get_utc_now() - program_start_time).total_seconds()
    print(f"[TIMER] Phase config generation started at [{task_generation_start_time:.4f}] seconds.")

    # Connect to Redis
    redis_host = os.getenv("REDIS_HOSTNAME", "localhost")
    redis_port = int(os.getenv("REDIS_PORT", 6379))
    input_queue = os.getenv("INPUT_QUEUE_NAME", "input_queue")
    
    try:
        redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        redis_client.ping()
        print(f"[INFO] Connected to Redis at {redis_host}:{redis_port}")
    except redis.exceptions.ConnectionError as e:
        print(f"[ERROR] Failed to connect to Redis: {e}", file=sys.stderr)
        sys.exit(1)

    # Generate phase configurations
    phase_configs = generate_phase_configs()

    print(f"[INFO] Generated {len(phase_configs)} phase configurations")
    print(f"[INFO] Pushing phase configs to '{input_queue}'...\n")

    # Push phase configs to input_queue
    for phase_config in phase_configs:
        phase_json = json.dumps(phase_config)
        redis_client.lpush(input_queue, phase_json)
        
        print(f"[INFO] Pushed Phase {phase_config['phase_number']}: {phase_config['phase_name']}")
        print(f"       Pattern: {phase_config['phase_pattern']}")
        print(f"       Target tasks: {phase_config['target_tasks']}")
        print(f"       Duration: {phase_config['phase_duration']}s")
        print()
    
    task_generation_end_time = (get_utc_now() - program_start_time).total_seconds()
    total_duration_actual = task_generation_end_time - task_generation_start_time
    
    total_tasks = sum(p['target_tasks'] for p in phase_configs)

    print(f"\n{'='*70}")
    print(f"TASK GENERATION COMPLETE")
    print(f"{'='*70}")
    print(f"[TIMER] Started at: {task_generation_start_time:.4f}s")
    print(f"[TIMER] Ended at: {task_generation_end_time:.4f}s")
    print(f"[TIMER] Total duration: {total_duration_actual:.4f}s ({total_duration_actual/60:.2f} min)")
    print(f"[STATS] Total tasks generated: {total_tasks}")
    print(f"[STATS] Average rate: {total_tasks / (total_duration_actual/60):.2f} tasks/min")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
