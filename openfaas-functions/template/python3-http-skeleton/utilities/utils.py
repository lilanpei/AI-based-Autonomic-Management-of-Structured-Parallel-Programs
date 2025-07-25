import os
import json
import redis
import sys
import time
import uuid
import numpy as np
from datetime import datetime, timezone

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
            print(f"[ERROR] Redis connection failed: {e}", file=sys.stderr)
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
        print(f"[ERROR] Redis initialization failed: {e}", file=sys.stderr)
        sys.exit(1)

def reinit_redis_client():
    """Reinitialize and replace the global Redis client."""
    global _redis_client
    print("[INFO] Reinitializing Redis client...")
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
        print(f"[ERROR] Redis connection error: {e}. Retrying...")
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
    print(f"[INFO] {pod_name} sent START signal: {start_signal}")

def deadline_for_matrix(n: int, coeff: float, cap: int, floor: int) -> int:
    """
    Analytic deadline (seconds) for an nxn matrix multiplication.

    deadline = min( max(floor, coeff · n³), cap )   # seconds

    • n       - size of the matrix (n x n).
    • coeff   - tunes how many seconds per floating-point operation.
    • cap     - hard upper bound so the system never waits 'forever'.
    • floor   - hard lower bound so the system never waits less than 1 second.
    """
    return int(min(max(floor, coeff * n**3), cap))

def extract_result(raw_result, program_start_time):
    try:
        result = json.loads(raw_result)
        if not isinstance(result, dict):
            raise ValueError("Result is not a JSON object")

        required_keys = ['task_id', 'task_gen_timestamp', 'task_deadline', 'task_work_timestamp']
        if not all(k in result for k in required_keys):
            raise ValueError("Result missing required keys")

        deadline = result.get("task_deadline")
        work_ts = result.get("task_work_timestamp")
        gen_ts = result.get("task_gen_timestamp")
        task_id = result.get("task_id")
        print(f"[INFO] Extracted result for task {task_id}")
        now = (get_utc_now() - program_start_time).total_seconds()
        deadline_met = (now - gen_ts) <= deadline

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
            "task_QoS": deadline_met
        }
    except (json.JSONDecodeError, ValueError) as e:
        raise ValueError(f"ERROR: Malformed result: {e} - Raw: {raw_result[:256]}")

def feedback_task_generation(result, redis_client, input_q, program_start_time, deadline_coeff, deadline_cap, deadline_floor):

    feedback_task = {
        "task_id": str(uuid.uuid4()),
        "task_gen_timestamp": (get_utc_now() - program_start_time).total_seconds(),
        "task_application": result["task_application"],
        "task_data": None,  # Actual data will be generated by workers
        "task_data_size": result["task_output_size"][0],
        "task_deadline": deadline_for_matrix(result["task_output_size"][0], float(deadline_coeff), int(deadline_cap), int(deadline_floor))  # seconds
    }
    safe_redis_call(lambda: redis_client.lpush(input_q, json.dumps(feedback_task)))
    print(f"[INFO] Feedback task generated: {feedback_task['task_id']} for result {result['task_id']}")

def extract_task(raw_task, program_start_time):
    try:
        task = json.loads(raw_task)
        if not isinstance(task, dict):
            raise ValueError("Task is not a JSON object")

        required_keys = ['task_application', 'task_data', 'task_data_size', 'task_deadline']
        if not all(k in task for k in required_keys):
            raise ValueError("Task missing required keys")

        task_id = task.get('task_id')
        print(f"[INFO] Extracted task ID: {task_id}")
        now = (get_utc_now() - program_start_time).total_seconds()

        return {
            "task_id": task_id,
            "task_application": task.get('task_application'),
            "task_data": task.get('task_data'),
            "task_data_size": task.get('task_data_size'),
            "task_deadline": task.get('task_deadline'),
            "task_gen_timestamp": task.get('task_gen_timestamp'),
            "task_emit_timestamp" : now,
            "task_emit_time":  now - task.get('task_gen_timestamp'),
        }
    except (json.JSONDecodeError, ValueError) as e:
        raise ValueError(f"ERROR: Malformed task: {e} - Raw: {raw_task[:256]}", file=sys.stderr)

def prepare_matrices(task):
    task_data, task_size = task.get("task_data"), task.get("task_data_size")
    if task_data:
        matrix_a = np.array(task_data.get("matrix_A"))
        matrix_b = np.array(task_data.get("matrix_B"))
    else:
        size = task_size or 10
        rows, cols = (size, size) if isinstance(size, int) else size
        matrix_a = np.random.rand(rows, cols)
        matrix_b = np.random.rand(cols, rows)
    if matrix_a.shape[1] != matrix_b.shape[0]:
        raise ValueError("Incompatible matrix dimensions")
    return matrix_a, matrix_b
