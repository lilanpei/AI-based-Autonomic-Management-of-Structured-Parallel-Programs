import os
import sys
import json
import time
import numpy as np
from datetime import datetime
from utils import (
    get_redis_client,
    parse_request_body,
    safe_redis_call,
    fetch_control_message,
    send_start_signal,
    get_utc_now,
    prepare_matrices
)

def load_calibrated_model():
    """
    Load calibrated timing model from configuration.yml

    Workflow:
    1. Run: python3 calibrate_direct.py
    2. Copy values from calibration_results.json to configuration.yml
    3. Worker loads from configuration.yml

    Returns:
        dict: Calibrated model parameters
    """
    try:
        config = get_config()
        if "calibrated_model" in config:
            model = config["calibrated_model"]
            return {
                "a": float(model.get("a", 1.95e-08)),
                "b": float(model.get("b", 0.001200)),
                "seed": int(model.get("seed", 42)),
                "r_squared": float(model.get("r_squared", 1.0)),
                "source": "configuration.yml"
            }
    except Exception as e:
        print(f"Warning: Could not load from configuration: {e}")

    # Fallback to default values if config not found
    print("Warning: Using default calibration values. Please run calibration and update configuration.yml")
    return {
        "a": 1.95e-08,
        "b": 0.001200,
        "seed": 42,
        "r_squared": 1.0,
        "source": "default (fallback)"
    }

def simulate_processing_time(image_size):
    """
    Simulate processing time using calibrated model

    Args:
        image_size: Image dimension in pixels

    Returns:
        float: Simulated processing time in seconds
    """

    # Load calibrated model
    CALIBRATED_MODEL = load_calibrated_model()

    # Log model info
    print(f"Calibrated model loaded from: {CALIBRATED_MODEL['source']}")
    print(f"Model: time = {CALIBRATED_MODEL['a']:.2e} × size² + {CALIBRATED_MODEL['b']:.6f}")
    print(f"R² = {CALIBRATED_MODEL.get('r_squared', 'N/A')}")

    a = CALIBRATED_MODEL["a"]
    b = CALIBRATED_MODEL["b"]

    # Calculate expected time using calibrated model
    expected_time = a * image_size**2 + b

    # Add realistic variance (±10%)
    actual_time = expected_time * random.uniform(0.9, 1.1)

    # Simulate processing
    time.sleep(actual_time)

    return actual_time

def process_image_processing(image_size):
    """
    Simulate complete image processing pipeline
    Pipeline: Thumbnail → Compression → Metadata → Conversion
    Uses calibrated model for complete pipeline timing
    """
    processing_time = simulate_processing_time(image_size)

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
    task_data = task.get("task_data", {})
    image_size = task_data.get("image_size", 2048)

    # Start timing
    start_time = time.perf_counter()

    # Process using calibrated simulation
    output_data, processing_time = process_image_processing(image_size)

    # Calculate actual elapsed time
    actual_time = time.perf_counter() - start_time

    # Simplified result
    return {
        "status": output_data.get("status", "completed"),
        "image_size": image_size,
        "processing_time": processing_time
    }

def handle(event, context):
    pod_name = os.environ.get("HOSTNAME")
    tasks_processed = 0
    redis_client = get_redis_client()
    body = parse_request_body(event)
    if not body:
        return {"statusCode": 400, "body": "Invalid JSON in request body."}

    worker_q, result_q = body.get('worker_queue_name'), body.get('result_queue_name')
    control_syn_q, control_ack_q = body.get('worker_control_syn_queue_name'), body.get('worker_control_ack_queue_name')
    start_q, processing_delay = body.get('worker_start_queue_name'), body.get('processing_delay')
    wait_time, program_start_time_str = body.get("wait_time"), body.get("program_start_time")

    if program_start_time_str:
        program_start_time = datetime.fromisoformat(program_start_time_str)
    else:
        print("[ERROR] START_TIMESTAMP environment variable not set.", file=sys.stderr)
        sys.exit(1)

    if not all([worker_q, result_q, control_syn_q, control_ack_q, start_q, processing_delay, wait_time, program_start_time]):
        return {"statusCode": 400, "body": "Missing required fields in request body."}

    print(f"\n[TIMER] Invoked at {(get_utc_now() - program_start_time).total_seconds():.4f} on pod {pod_name}")
    print(f"[INFO] Worker received body: {body}")

    # send start signal to start queue
    send_start_signal(redis_client, start_q, pod_name, (get_utc_now() - program_start_time).total_seconds())

    previous_iteration_start = None

    while True:
        iteration_start = (get_utc_now() - program_start_time).total_seconds()
        print(f"[TIMER]-------------Iteration start at {iteration_start:.4f}-----------------")
        if previous_iteration_start:
            print(f"[TIMER] Iteration time: {(iteration_start - previous_iteration_start):.4f} sec")
        previous_iteration_start = iteration_start

        try:
            control_msg = fetch_control_message(redis_client, control_syn_q)
            if control_msg and control_msg.get("action") == "SYN":
                if control_msg.get("type") == "SCALE_DOWN":
                    print(f"[INFO] Received SCALE_DOWN control message: {control_msg}")
                    # Acknowledge the scale down request
                    ack_msg = {
                        "type": "SCALE_DOWN",
                        "action": "ACK",
                        "ack_timestamp": (get_utc_now() - program_start_time).total_seconds(),
                        "task_id": control_msg.get("task_id"),
                        "pod_name": pod_name,
                        "message": "Worker pod is exiting as instructed."
                    }

                    safe_redis_call(lambda: redis_client.lpush(control_ack_q, json.dumps(ack_msg)))
                    print(f"[INFO] Sent ACK for control message: {ack_msg}")
                    return {
                        "statusCode": 200,
                        "body": f"Worker pod {pod_name} acknowledged scale down."
                    }
                elif control_msg.get("type") == "TERMINATE":
                    print(f"[INFO] Received TERMINATE control message: {control_msg}")
                    return {
                        "statusCode": 200,
                        "body": f"Worker pod {pod_name} acknowledged termination."
                    }
                else:
                    print(f"[WARNING] Unknown control message type: {control_msg.get('type')}")

            # Pop task from queue
            raw_task = safe_redis_call(lambda: redis_client.rpop(worker_q))

            if not raw_task:
                # No tasks available, wait for a while
                time.sleep(float(wait_time))
                continue
            else:
                # Extract task
                print(f"\n[TIMER] got task at {(get_utc_now() - program_start_time).total_seconds():.4f} on pod {pod_name}")
                _, task_json = raw_task
                task = json.loads(task_json)
                tasks_processed += 1

                task_id = task.get("task_id")
                task_type = task.get("task_application")
                task_priority = task.get("task_priority", "normal")
                print(f"[INFO] Processing task ID: {task_id}")

                # Validate task type (only image_processing supported)
                if task_type != "image_processing":
                    raise ValueError("Unsupported task application")

                # Process the task
                result_data = process_image_task(task, program_start_time)

                # Add any additional processing delay (if configured)
                if processing_delay > 0:
                    time.sleep(float(processing_delay))

                # Calculate timestamps
                now = (get_utc_now() - program_start_time).total_seconds()
                emit_ts = task.get("task_emit_timestamp")

                # Build simplified result
                result = {
                    "task_id": task_id,
                    "task_application": task.get("task_application"),
                    "task_priority": task_priority,
                    "task_gen_timestamp": task.get("task_gen_timestamp"),
                    "task_deadline": task.get("task_deadline"),
                    "task_emit_time": task.get("task_emit_time"),
                    "task_emit_timestamp" : emit_ts,
                    "task_work_time": now - emit_ts,
                    "task_work_timestamp": now,
                    "processing_time": result_data["processing_time"],
                    "image_size": result_data["image_size"],
                    "status": result_data["status"]
                }

                # Push result to queue
                safe_redis_call(lambda: redis_client.lpush(result_q, json.dumps(result)))
                print(f"[TIMER] Processed {tasks_processed} tasks at {now} from {worker_q} and pushed result to '{result_q}' on pod {pod_name}")

        except Exception as e:
            print(f"[ERROR] Failed to process task: {e}", file=sys.stderr)
            break

    print(f"[INFO] Worker processed {tasks_processed} tasks.")

    return {
        "statusCode": 200,
        "body": f"Worker processed task from '{worker_q}' to '{result_q}'"
    }
