import os
import sys
import json
import time
import redis
from datetime import datetime
from zoneinfo import ZoneInfo

redisClient = None
backoff_factor = 2
retries = 10

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

def parse_request_body(event):
    try:
        body = json.loads(event.body)
        if not body:
            raise ValueError("Empty request body")
        return body
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        raise ValueError(f"ERROR: {e} - Raw Body: {str(event.body)[:512]}", file=sys.stderr)

def safe_redis_call(func):
    try:
        return func()
    except redis.exceptions.ConnectionError as e:
        print(f"[ERROR] Redis connection error: {e}. Retrying...")
        time.sleep(1)
        global redisClient
        redisClient = init_redis_client()
        return func()

def fetch_control_message(redis_client, control_queue):
    try:
        control_raw = safe_redis_call(lambda: redis_client.rpop(control_queue))
        return json.loads(control_raw) if control_raw else None
    except Exception as e:
        raise ValueError(f"[ERROR] Failed to parse control message: {e}")

def extract_task(raw_task):
    try:
        task = json.loads(raw_task)
        if not isinstance(task, dict):
            raise ValueError("Task is not a JSON object")

        required_keys = ['task_application', 'task_data', 'task_data_size', 'task_deadline']
        if not all(k in task for k in required_keys):
            raise ValueError("Task missing required keys")

        task_id = task.get('task_id')
        print(f"[DEBUG] Extracted task ID: {task_id}")
        now = time.time()

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

def send_start_signal(redis_client, start_queue, pod_name, start_timestamp):
    """
    Sends a start signal to start queue.
    """
    start_signal = {
        "type": "START",
        "action": "SYNC",
        "start_timestamp": start_timestamp.strftime('%Y-%m-%d %H:%M:%S %Z'),
        "pod_name": pod_name,
        "message": f"{pod_name} is ready to start processing tasks."
    }

    safe_redis_call(lambda: redisClient.lpush(start_queue, json.dumps(start_signal)))
    print(f"[INFO] {pod_name} sent START signal: {start_signal}")

def handle(event, context):
    now = datetime.now(ZoneInfo("Europe/Rome"))
    pod_name = os.environ.get("HOSTNAME")
    print(f"\n[Emitter] Invoked at {now.strftime('%Y-%m-%d %H:%M:%S %Z')} on pod {pod_name}")
    tasks_generated = 0
    global redisClient
    if redisClient is None:
        try:
            redisClient = init_redis_client()
        except redis.exceptions.ConnectionError as e:
            print(f"[CRITICAL] Redis connection failed: {e}", file=sys.stderr)
            return {"statusCode": 500, "body": f"Redis connection failed: {e}"}

    body = parse_request_body(event)
    if not body:
        return {"statusCode": 400, "body": "Invalid JSON in request body."}

    input_q, worker_q = body.get('input_queue_name'), body.get('worker_queue_name')
    control_syn_q, start_q = body.get('emitter_control_syn_queue_name'), body.get('emitter_start_queue_name')
    print(f"[DEBUG] Emitter received body: {body}")

    if not all([input_q, worker_q, start_q]):
        return {"statusCode": 400, "body": "Missing required fields in request body."}

    # send start signal to start queue
    send_start_signal(redisClient, start_q, pod_name, now)

    previous_iteration_start = None
    attempt = 0

    while True:
        print("------------------------------")
        iteration_start = time.time()
        if previous_iteration_start:
            print(f"[INFO] Iteration time: {iteration_start - previous_iteration_start:.2f} sec")
        previous_iteration_start = iteration_start

        try:
            control_msg = fetch_control_message(redisClient, control_syn_q)
            if control_msg and control_msg.get("action") == "SYN" and control_msg.get("type") == "TEMINATE":
                print(f"[INFO] Received TERMINATE control message: {control_msg}")
                return {
                    "statusCode": 200,
                    "body": f"Emitter pod {pod_name} acknowledged termination."
                }

            raw_task = safe_redis_call(lambda: redisClient.rpop(input_q))

            if not raw_task:
                attempt += 1
                sleep_time = backoff_factor ** attempt
                print(f"[INFO] No task found in '{input_q}' for {attempt} tries, waiting {sleep_time} seconds...")
                time.sleep(sleep_time)
                if attempt >= retries:
                    print("[WARNING] No tasks found for a long time, exiting emitter.")
                    return {
                        "statusCode": 200,
                        "body": f"Emitter pod {pod_name} exited due to inactivity."
                    }
                continue
            else:
                attempt = 0
                now = datetime.now(ZoneInfo("Europe/Rome"))
                print(f"\n[Emitter] got task at {now.strftime('%Y-%m-%d %H:%M:%S %Z')} on pod {pod_name}")
                task = extract_task(raw_task)
                tasks_generated += 1

                safe_redis_call(lambda: redisClient.lpush(worker_q, json.dumps(task)))
                print(f"[INFO] Pushed {tasks_generated} tasks from '{input_q}' to '{worker_q}'")

        except Exception as e:
            print(f"ERROR: Unexpected error: {e}", file=sys.stderr)
            return {
                "statusCode": 500,
                "body": f"Unexpected error: {e}"
            }
    print(f"[INFO] Emitter processed {tasks_generated} tasks.")

    return {
        "statusCode": 200,
        "body": f"Emitter processed task from '{input_q}' to '{worker_q}'"
    }
