import os
import json
import redis
import sys
import time
import uuid
import random
import math
import numpy as np
import logging
from datetime import datetime, timezone

# Configure logging
log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
numeric_log_level = getattr(logging, log_level, logging.INFO)

root_logger = logging.getLogger()
if not root_logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
    root_logger.addHandler(handler)

root_logger.setLevel(numeric_log_level)

logger = logging.getLogger(__name__)
logger.setLevel(numeric_log_level)

backoff_factor = 2
retries = 10
_redisClient = None  # private global inside module

def get_utc_now():
    """
    Returns the current UTC time as a timezone-aware datetime object.
    """
    return datetime.now(timezone.utc)

def get_redis_client():
    """Get the global Redis client, initializing if needed."""
    global _redisClient
    if _redisClient is None:
        try:
            _redisClient = init_redis_client()
        except redis.exceptions.ConnectionError as e:
            logger.error(f"Redis connection failed: {e}")
            return {"statusCode": 500, "body": f"Redis connection failed: {e}"}
    return _redisClient

def init_redis_client():
    """
    Initializes a Redis client with fallback values.
    Works for both local and in-cluster use.
    """
    host = os.getenv("REDIS_HOSTNAME", "redis-master.redis.svc.cluster.local")
    port = int(os.getenv("REDIS_PORT", 6379))

    try:
        return redis.Redis(host=host, port=port, decode_responses=True)
    except redis.exceptions.RedisError as e:
        logger.error(f"Redis initialization failed: {e}")
        sys.exit(1)

def reinit_redis_client():
    """Reinitialize and replace the global Redis client."""
    global _redis_client
    logger.info("Reinitializing Redis client...")
    _redis_client = init_redis_client()

def parse_request_body(event):
    try:
        body = json.loads(event.body)
        if not body:
            raise ValueError("Empty request body")
        return body
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        raise ValueError(f"ERROR: {e} - Raw Body: {str(event.body)[:512]}")

def safe_redis_call(func):
    try:
        return func()
    except redis.exceptions.ConnectionError as e:
        logger.error(f"Redis connection error: {e}. Retrying...")
        time.sleep(1)
        reinit_redis_client()
        return func()

def fetch_control_message(redis_client, control_queue):
    try:
        control_raw = safe_redis_call(lambda: redis_client.rpop(control_queue))
        return json.loads(control_raw) if control_raw else None
    except Exception as e:
        raise ValueError(f"[ERROR] Failed to parse control message: {e}")

def send_start_signal(redis_client, start_queue, pod_name, start_timestamp):
    """
    Sends a start signal to start queue.
    """
    start_signal = {
        "type": "START",
        "action": "SYNC",
        "start_timestamp": str(start_timestamp),
        "pod_name": pod_name,
        "message": f"{pod_name} is ready to start processing tasks."
    }

    safe_redis_call(lambda: redis_client.lpush(start_queue, json.dumps(start_signal)))
    logger.info(f"{pod_name} sent START signal: {start_signal}")


def extract_result(raw_result, program_start_time):
    try:
        result = json.loads(raw_result)
        if not isinstance(result, dict):
            raise ValueError("Result is not a JSON object")

        required_keys = ['task_id', 'task_gen_timestamp', 'task_deadline', 'task_work_timestamp']
        if not all(k in result for k in required_keys):
            raise ValueError("Result missing required keys")

        # Extract timestamps
        deadline = result.get("task_deadline")
        work_ts = result.get("task_work_timestamp")
        gen_ts = result.get("task_gen_timestamp")
        priority = result.get("task_priority", "normal")
        task_id = result.get("task_id")
        logger.info(f"Extracted result for task {task_id}")
        now = (get_utc_now() - program_start_time).total_seconds()
        deadline_met = (now - gen_ts) <= deadline

        # Calculate current time
        now = (datetime.now(timezone.utc) - program_start_time).total_seconds()

        # Calculate actual completion time (from generation to now)
        actual_time = now - gen_ts

        # Calculate boolean QoS
        deadline_met = deadline >= actual_time

        return {
            "task_id": task_id,
            "task_application": result.get("task_application"),
            "task_gen_timestamp": gen_ts,
            "task_result_data": result.get("task_result_data"),
            "task_emit_time": result.get("task_emit_time"),
            "task_emit_timestamp" : result.get("task_emit_timestamp"),
            "task_deadline": deadline,
            "task_collect_time": now - work_ts,
            "task_collect_timestamp": now,
            "task_output_size": result.get("task_output_size"),
            "task_work_time": result.get("task_work_time"),
            "task_work_timestamp": work_ts,
            "task_completion_time": actual_time,
            "task_QoS": deadline_met
        }
    except (json.JSONDecodeError, ValueError) as e:
        raise ValueError(f"ERROR: Malformed result: {e} - Raw: {raw_result[:256]}")

def simulate_processing_time(image_size, calibrated_model_a, calibrated_model_b, calibrated_model_seed, calibrated_model_r_squared):
    """
    Simulate processing time using calibrated model

    Args:
        image_size: Image dimension in pixels

    Returns:
        float: Simulated processing time in seconds
    """

    # Log model info
    logger.debug(f"Model: time = {calibrated_model_a:.2e} × size² + {calibrated_model_b:.6f}")
    logger.debug(f"R² = {calibrated_model_r_squared}")

    a = calibrated_model_a
    b = calibrated_model_b

    # Calculate expected time using calibrated model
    expected_time = a * image_size**2 + b

    # Add realistic variance (±10%)
    actual_time = expected_time * random.uniform(0.9, 1.1)

    # Simulate processing
    time.sleep(actual_time)

    return actual_time

def process_image_processing(task):
    """
    Simulate complete image processing pipeline
    Pipeline: Thumbnail → Compression → Metadata → Conversion
    Uses calibrated model for complete pipeline timing
    """
    image_size = task.get("task_data_size")
    processing_time = task.get("task_processing_time_simulated")
    logger.debug(f"Simulating processing time: {processing_time:.2f} seconds")

    # Simulate processing
    time.sleep(processing_time)

    # Simple result: just indicate success
    result = {
        "status": "completed",
        "image_size": image_size
    }

    return result, processing_time

def process_image_task(task, program_start_time):
    """
    Process image_processing task with simulated timing

    Uses calibrated model for realistic processing simulation:
    - Model: time = a × size² + b
    - Complete pipeline: Thumbnail → Compression → Metadata → Conversion

    Args:
        task: Task payload dict (must have task_application="image_processing")
        program_start_time: Program start timestamp

    Returns:
        dict: Processing result with timing
    """

    # Process using calibrated simulation
    output_data, processing_time = process_image_processing(task)

    # Simplified result
    return {
        "status": output_data.get("status", "completed"),
        "image_size": output_data.get("image_size"),
        "processing_time": processing_time
    }

def extract_generated_task(raw_task, program_start_time, calibrated_model_a, calibrated_model_b, calibrated_model_seed, calibrated_model_r_squared):
    try:
        task = json.loads(raw_task)
        if not isinstance(task, dict):
            raise ValueError("Task is not a JSON object")

        required_keys = ['task_application', 'task_data_size', 'task_deadline']
        if not all(k in task for k in required_keys):
            raise ValueError("Task missing required keys")

        task_id = task.get('task_id')
        logger.info(f"Extracted task ID: {task_id}")
        now = (get_utc_now() - program_start_time).total_seconds()

        processing_time_simulated = simulate_processing_time(task.get('task_data_size'), calibrated_model_a, calibrated_model_b, calibrated_model_seed, calibrated_model_r_squared)

        return {
            "task_id": task_id,
            "task_application": task.get('task_application'),
            "task_data": task.get('task_data'),
            "task_data_size": task.get('task_data_size'),
            "task_deadline": task.get('task_deadline'),
            "task_gen_timestamp": task.get('task_gen_timestamp'),
            "task_emit_timestamp" : now,
            "task_emit_time":  now - task.get('task_gen_timestamp'),
            "task_processing_time_simulated": processing_time_simulated,
            "task_processing_time_expected": task.get('task_expected_duration')
        }
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Malformed task: {e} - Raw: {raw_task[:256]}")
        raise ValueError(f"ERROR: Malformed task: {e}")

# TASK GENERATION FUNCTIONS (for emitter generate mode)

# Single task type: image_processing (complete pipeline)
TASK_TYPES = {
    "image_processing": {
        "name": "image_processing",
        "description": "Complete image processing pipeline (all 4 stages)",
        # Image size sampling parameters (derived from calibrated model)
        "min_image_size": 512,
        "max_image_size": 4096,
        "target_mean_processing_time": 1.5,  # seconds
        "processing_time_shape": 4.0  # Gamma shape for sampling variability
    }
}

# QoS Model: Deadline-Based
DEADLINE_COEFFICIENT = 2.0
USER_PRIORITY = 1.0


def calculate_task_deadline(expected_duration, coefficient=None, user_priority=None):
    """Calculate deadline using QoS model"""
    if coefficient is None:
        coefficient = DEADLINE_COEFFICIENT
    if user_priority is None:
        user_priority = USER_PRIORITY

    deadline = coefficient * expected_duration * user_priority

    # Apply bounds
    min_deadline = 0.1   # 100ms minimum
    max_deadline = 30.0  # 30s maximum

    return max(min_deadline, min(deadline, max_deadline))


def generate_task_for_generation(program_start_time, calibrated_model_a=None, calibrated_model_b=None, 
                                 calibrated_model_seed=None, calibrated_model_r_squared=None):
    """
    Generate an image processing task for task generation mode
    Includes simulated processing time calculation
    """
    task_config = TASK_TYPES["image_processing"]
    priority = "normal"

    if calibrated_model_a is None or calibrated_model_b is None:
        logger.error("Calibrated model not provided")
        raise ValueError("Calibrated model not provided")

    # Sample processing time from a gamma distribution centered on the target mean
    target_mean = task_config.get("target_mean_processing_time", 1.5)
    shape = task_config.get("processing_time_shape", 4.0)
    scale = target_mean / shape if shape > 0 else target_mean

    min_size = task_config.get("min_image_size", 512)
    max_size = task_config.get("max_image_size", 4096)
    min_time = calibrated_model_a * (min_size ** 2) + calibrated_model_b
    max_time = calibrated_model_a * (max_size ** 2) + calibrated_model_b

    processing_time_sampled = 0.0
    for _ in range(5):
        candidate = random.gammavariate(shape, scale)
        if candidate > calibrated_model_b:
            processing_time_sampled = candidate
            break
    if processing_time_sampled <= calibrated_model_b:
        processing_time_sampled = target_mean

    # Clip sampled processing time to calibrated range
    processing_time_sampled = max(min_time, min(processing_time_sampled, max_time))

    # Invert calibrated model to derive image size from processing time
    denom = max(calibrated_model_a, 1e-12)
    size_continuous = math.sqrt(max((processing_time_sampled - calibrated_model_b) / denom, 0.0))
    image_size = int(max(min_size, min(size_continuous, max_size)))

    # Recompute processing time based on rounded image size
    processing_time_simulated = calibrated_model_a * (image_size ** 2) + calibrated_model_b
    processing_time_simulated = max(0.001, processing_time_simulated)

    # Get current UTC time
    task_gen_timestamp = (get_utc_now() - program_start_time).total_seconds()
    logger.debug(
        "Calibrated model sample: size=%d, sampled_time=%.3fs, recomputed_time=%.3fs",
        image_size,
        processing_time_sampled,
        processing_time_simulated,
    )

    # Calculate deadline using QoS model (based on calibrated or expected duration)
    deadline = calculate_task_deadline(processing_time_simulated)

    # Final validation: ensure processing time is positive
    processing_time_simulated = max(0.001, processing_time_simulated)

    expected_duration = processing_time_simulated

    return {
        "task_id": str(uuid.uuid4()),
        "task_gen_timestamp": task_gen_timestamp,
        "task_application": task_config["name"],
        "task_priority": priority,
        "task_expected_duration": expected_duration,
        "task_data_size": image_size,
        "task_deadline": deadline,
        "task_processing_time_simulated": processing_time_simulated,
        # Add emit timestamps (same as gen since pushed directly to worker_queue)
        "task_emit_timestamp": task_gen_timestamp,
        "task_emit_time": 0.0  # No emit delay since generated directly
    }


def get_phase_rate(phase_config_data, time_in_phase):
    """Calculate task arrival rate for current phase and time"""
    phase_pattern = phase_config_data.get("phase_pattern")
    base_rate = phase_config_data.get("base_rate")
    phase_duration = phase_config_data.get("phase_duration")
    phase_multiplier = phase_config_data.get("phase_multiplier")

    if phase_pattern.startswith("oscillation"):
        oscillation_min = phase_config_data.get("oscillation_min")
        oscillation_max = phase_config_data.get("oscillation_max")
        oscillation_cycles = phase_config_data.get("oscillation_cycles")

    if phase_pattern == "steady":
        # Phase: Steady Low/High Load (phase_multiplier of base rate)
        return base_rate * phase_multiplier

    elif phase_pattern == "oscillation":
        # Phase: Slow(1 cycle per phase)/Fast(4 cycles per phase) Oscillation
        # Oscillates between oscillation_min and oscillation_max of base rate
        progress = time_in_phase / phase_duration  # 0 to 1
        amplitude = (oscillation_max - oscillation_min) / 2.0
        midpoint = (oscillation_max + oscillation_min) / 2.0
        oscillation = amplitude * np.sin(2 * np.pi * oscillation_cycles * progress) + midpoint
        return base_rate * oscillation

    else:
        return base_rate


def push_task_to_worker_queue(redis_client, queue_name, task_json):
    """Push a task to Redis queue with error handling"""
    try:
        redis_client.lpush(queue_name, task_json)
        return True
    except Exception as e:
        logger.error(f"Failed to push task: {e}")
        return False


def generate_tasks_for_phase(phase_config_data,
                             redis_client, worker_queue, program_start_time,
                             calibrated_model_a=None, calibrated_model_b=None,
                             calibrated_model_seed=None, calibrated_model_r_squared=None):
    """
    Generate tasks for one phase with specific arrival pattern
    Uses controlled Poisson to hit exact target while maintaining variance
    """
    # Calculate exact target for this phase
    phase = phase_config_data.get("phase_number")
    phase_name = phase_config_data.get("phase_name")
    base_rate = phase_config_data.get("base_rate")
    phase_duration = phase_config_data.get("phase_duration")
    window_duration = phase_config_data.get("window_duration")
    target_tasks = phase_config_data.get("target_tasks")

    logger.info(f"\n{'='*70}")
    logger.info(f"PHASE {phase}: {phase_name}")
    logger.info(f"{'='*70}")
    logger.info(f"Duration: {phase_duration}s ({phase_duration/60:.1f} min)")
    logger.info(f"Window size: {window_duration}s")
    logger.info(f"Target tasks: {target_tasks}")
    logger.info(f"{'='*70}\n")

    num_windows = int(phase_duration / window_duration)
    phase_start_time = time.time()
    total_tasks_generated = 0

    for window in range(num_windows):
        window_start = time.time()
        time_in_phase = (window_start - phase_start_time)

        # Calculate arrival rate for this window
        rate = get_phase_rate(phase_config_data, time_in_phase)
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

        logger.info(f"\n[Window {window+1}/{num_windows}] Time: {time_in_phase:.1f}s, Rate: {rate:.2f} tasks/min, Tasks: {num_tasks}")

        if num_tasks <= 0:
            logger.debug("  No tasks in this window")
            # No sleep - move to next window immediately
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

            # Generate task with calibrated model parameters
            task_payload = generate_task_for_generation(
                program_start_time,
                calibrated_model_a=calibrated_model_a,
                calibrated_model_b=calibrated_model_b,
                calibrated_model_seed=calibrated_model_seed,
                calibrated_model_r_squared=calibrated_model_r_squared
            )
            task_json = json.dumps(task_payload)

            # Push directly to worker_queue (not input_queue!)
            success = push_task_to_worker_queue(redis_client, worker_queue, task_json)

            if success:
                tasks_in_window += 1
                total_tasks_generated += 1

            # Sleep until target time (compensates for processing time)
            current_time = time.time()
            sleep_time = target_time - current_time
            if sleep_time > 0:
                time.sleep(sleep_time)

        logger.info(f"  Generated {tasks_in_window} tasks (total: {total_tasks_generated}/{target_tasks})")

    logger.info(f"\n[PHASE {phase} COMPLETE] Total tasks generated: {total_tasks_generated}")
    return total_tasks_generated
