import os
import sys
import json
import time
import redis
import requests
import numpy as np
from threading import Thread

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
        print(f"ERROR: {e} - Raw Body: {str(event.body)[:512]}", file=sys.stderr)
        return None

def async_function(func, *args, **kwargs):
    """
    Runs any function asynchronously with given arguments.

    Args:
        func (callable): The function to execute.
        *args: Positional arguments for the function.
        **kwargs: Keyword arguments for the function.
    """
    thread = Thread(target=func, args=args, kwargs=kwargs)
    # thread.daemon = True  # Optional: thread dies with the main program
    thread.start()

def invoke_function_sync(function_name, payload, gateway_url="http://127.0.0.1:8080"):
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

# def reinvoke_self(payload):
#     async_function((invoke_function_sync), "worker", payload, "http://127.0.0.1:8080")

def reinvoke_self(payload):
    try:
        response = requests.post(
            "http://gateway.openfaas.svc.cluster.local:8080/async-function/worker",
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"}
        )
        print(f"[INFO] Reinvoked worker - Status: {response.status_code}, Body: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Self-reinvoke failed: {e}", file=sys.stderr)

def safe_redis_call(func):
    try:
        return func()
    except redis.exceptions.ConnectionError as e:
        print(f"[ERROR] Redis connection error: {e}. Retrying...")
        time.sleep(5)
        global redisClient
        redisClient = init_redis_client()
        return func()

def fetch_control_message(redis_client, control_queue):
    try:
        control_raw = safe_redis_call(lambda: redis_client.rpop(control_queue))
        return json.loads(control_raw) if control_raw else None
    except Exception as e:
        print(f"[ERROR] Failed to parse control message: {e}")
        return None

def prepare_matrices(task):
    task_data, task_size = task.get("data"), task.get("size")
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

def handle(event, context):
    print(f"[INFO] Worker invoked at {time.strftime('%Y-%m-%d %H:%M:%S')} on pod {os.environ.get('HOSTNAME')}")
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

    worker_q, result_q = body.get('worker_queue_name'), body.get('result_queue_name')
    control_syn_q, control_ack_q = body.get('control_syn_queue_name'), body.get('control_ack_queue_name')

    if not all([worker_q, result_q, control_syn_q, control_ack_q]):
        return {"statusCode": 400, "body": "Missing required fields in request body."}

    tasks_processed = 0
    iteration_end = None

    while True:
        print("------------------------------")
        iteration_start = time.time()
        if iteration_end:
            print(f"[INFO] Time since last iteration: {iteration_start - iteration_end:.2f} sec")

        try:
            control_msg = fetch_control_message(redisClient, control_syn_q)
            if control_msg and control_msg.get("type") == "SCALE_DOWN" and control_msg.get("action") == "SYN":
                ack_msg = {
                    "type": "SCALE_DOWN",
                    "action": "ACK",
                    "ack_timestamp": time.time(),
                    "task_id": control_msg.get("task_id"),
                    "pod_name": os.environ.get("HOSTNAME"),
                    "message": "Worker pod is exiting as instructed."
                }

                safe_redis_call(lambda: redisClient.lpush(control_ack_q, json.dumps(ack_msg)))
                print(f"[INFO] Sent ACK for control message: {ack_msg}")
                break

            raw_task = safe_redis_call(lambda: redisClient.rpop(worker_q))

            if not raw_task:
                print(f"[INFO] No task found in '{worker_q}', reinvoking and exiting...")
                # time.sleep(10)
                # reinvoke_self(body)
                break

            task = json.loads(raw_task)
            if task.get("application") != "matrix_multiplication":
                raise ValueError("Unsupported task application")

            matrix_a, matrix_b = prepare_matrices(task)
            result_matrix = np.dot(matrix_a, matrix_b)
            now = time.time()
            result = {
                "task_id": task.get("id"),
                "result_data": None,
                "task_application": task.get("application"),
                "task_emit_timestamp": task.get("timestamp"),
                "task_deadline": task.get("deadline"),
                "output_size": result_matrix.shape,
                "emit_time": now - task.get("timestamp", now),
                "complete_time": 2,
                "complete_timestamp": now
            }
            safe_redis_call(lambda: redisClient.lpush(result_q, json.dumps(result)))
            print(f"[INFO] Processed task {task.get('id')} and pushed result to '{result_q}'")
            tasks_processed += 1

            if tasks_processed % 50 == 0:
                print(f"[INFO] Processed {tasks_processed} tasks")

        except Exception as e:
            print(f"[ERROR] Failed to process task: {e}", file=sys.stderr)

        iteration_end = time.time()

    return {
        "statusCode": 200,
        "body": f"Worker processed {tasks_processed} tasks from '{worker_q}' to '{result_q}'"
    }
