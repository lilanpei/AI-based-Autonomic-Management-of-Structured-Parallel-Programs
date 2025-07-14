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

    # tasks_processed = 0
    # iteration_end = None

    # while True:
    print("------------------------------")
    iteration_start = time.time()
    # if iteration_end:
    #     print(f"[INFO] Time since last iteration: {iteration_start - iteration_end:.2f} sec")

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
            return {
                "statusCode": 200,
                "body": f"Worker pod {os.environ.get('HOSTNAME')} acknowledged scale down."
            }

        raw_task = safe_redis_call(lambda: redisClient.rpop(worker_q))

        if not raw_task:
            print(f"[INFO] No task found in '{worker_q}', waiting for tasks...")
            # time.sleep(5)
            # invoke_function_async("worker", body)
            # continue # break
        else:
            task = json.loads(raw_task)
            if task.get("task_application") != "matrix_multiplication":
                raise ValueError("Unsupported task application")

            task_id = task.get("task_id")
            matrix_a, matrix_b = prepare_matrices(task)
            result_matrix = np.dot(matrix_a, matrix_b)
            time.sleep(2)  # Simulate processing delay
            now = time.time()
            emit_ts = task.get("task_emit_timestamp")
            result = {
                "task_id": task_id,
                "task_result_data": None,
                "task_application": task.get("task_application"),
                "task_gen_timestamp": task.get("task_gen_timestamp"),
                "task_deadline": task.get("task_deadline"),
                "task_output_size": result_matrix.shape,
                "task_emit_time": task.get("task_emit_time"),
                "task_emit_timestamp" : emit_ts,
                "task_work_time": now - emit_ts,
                "task_work_timestamp": now
            }
            safe_redis_call(lambda: redisClient.lpush(result_q, json.dumps(result)))
            invoke_function_async("collector", body)
            print(f"[INFO] Processed task {task_id} and pushed result to '{result_q}'")

        # tasks_processed += 1
        # if tasks_processed % 50 == 0:
        #     print(f"[INFO] Processed {tasks_processed} tasks")

        iteration_end = time.time()
        print(f"[INFO] Iteration completed in {iteration_end - iteration_start:.2f}s")

    except Exception as e:
        print(f"[ERROR] Failed to process task: {e}", file=sys.stderr)
        # break
        return {
            "statusCode": 500,
            "body": f"Failed to process task: {e}"
        }

    return {
        "statusCode": 200,
        # "body": f"Worker processed {tasks_processed} tasks from '{worker_q}' to '{result_q}'"
        "body": f"Worker processed task from '{worker_q}' to '{result_q}'"
    }
