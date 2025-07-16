import os
import sys
import json
import time
import redis
import requests
from threading import Thread
from datetime import datetime
from zoneinfo import ZoneInfo

redisClient = None

def init_redis_client():
    return redis.Redis(
        host=os.getenv('redis_hostname', 'redis-master.redis.svc.cluster.local'),
        port=os.getenv('redis_port'),
        decode_responses=True
    )

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
        # time.sleep(5)
        global redisClient
        redisClient = init_redis_client()
        return func()

def async_function(func, *args, **kwargs):
    """
    Runs any function asynchronously with given arguments.

    Args:
        func (callable): The function to execute.
        *args: Positional arguments for the function.
        **kwargs: Keyword arguments for the function.
    """
    thread = Thread(target=func, args=args, kwargs=kwargs)
    thread.daemon = True  # Optional: thread dies with the main program
    thread.start()

def invoke_function_sync(function_name, payload, gateway_url="http://gateway.openfaas.svc.cluster.local:8080"):
    """Asynchronously invoke OpenFaaS function."""
    url = f"{gateway_url}/sync-function/{function_name}"
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            print(f"[WARM-UP] Successfully invoked '{function_name}'")
        else:
            print(f"[WARM-UP] Failed to invoke '{function_name}' - Status {response.status_code}")
    except Exception as e:
        print(f"[WARM-UP] Error invoking '{function_name}': {e}")

# def invoke_function_async(function_name, payload, gateway_url="http://gateway.openfaas.svc.cluster.local:8080"):
#     """Invoke OpenFaaS function asynchronously."""
#     async_function((invoke_function_sync), function_name, payload, gateway_url)

def invoke_function_async(function_name, payload, gateway_url="http://gateway.openfaas.svc.cluster.local:8080"):
    """Asynchronously invoke OpenFaaS function."""
    url = f"{gateway_url}/async-function/{function_name}"
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        print(f"[INFO] Async invoked '{function_name}'")
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Async invocation failed: {e}", file=sys.stderr)

def extract_task(raw_task):
    try:
        task = json.loads(raw_task)
        if not isinstance(task, dict):
            raise ValueError("Task is not a JSON object")

        required_keys = ['task_application', 'task_data', 'task_data_size', 'task_deadline']
        if not all(k in task for k in required_keys):
            raise ValueError("Task missing required keys")

        now = time.time()

        return {
            "task_id": task.get('task_id'),
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

def handle(event, context):
    now = datetime.now(ZoneInfo("Europe/Rome"))
    print(f"\n[Emitter] Invoked at {now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
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
    print(f"[DEBUG] Emitter received body: {body}")

    if not all([input_q, worker_q]):
        return {"statusCode": 400, "body": "Missing required fields in request body."}

    tasks_generated = 0
    iteration_end = None

    while True:
        print("------------------------------")
        iteration_start = time.time()
        if iteration_end:
            print(f"[INFO] Time since last iteration: {iteration_start - iteration_end:.2f} sec")

        try:
            raw_task = safe_redis_call(lambda: redisClient.rpop(input_q))

            if not raw_task:
                print(f"[INFO] No task in '{input_q}', waiting for tasks...")
                # time.sleep(5)
                # invoke_function_async("emitter", body)
                continue # break
            else:
                task = extract_task(raw_task)

                safe_redis_call(lambda: redisClient.lpush(worker_q, json.dumps(task)))
                # invoke_function_async("worker", body)
                print(f"[INFO] Pushed task {task['task_id']} to '{worker_q}'")

                tasks_generated += 1
                if tasks_generated % 100 == 0:
                    print(f"[INFO] Generated {tasks_generated} tasks")

            iteration_end = time.time()
            print(f"[INFO] Iteration completed in {iteration_end - iteration_start:.2f}s")

        except Exception as e:
            print(f"ERROR: Unexpected error: {e}", file=sys.stderr)
            # break
            return {
                "statusCode": 500,
                "body": f"Unexpected error: {e}"
            }

    return {
        "statusCode": 200,
        # "body": f"Emitter processed {tasks_generated} tasks from '{input_q}' to '{worker_q}'"
        "body": f"Emitter processed task from '{input_q}' to '{worker_q}'"
    }
