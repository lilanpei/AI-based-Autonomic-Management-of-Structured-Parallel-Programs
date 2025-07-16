import os
import sys
import json
import time
import uuid
import redis
import requests
import numpy as np
from threading import Thread
from datetime import datetime
from zoneinfo import ZoneInfo

redisClient = None

def init_redis_client():
    return redis.Redis(
        host=os.getenv("redis_hostname", "redis-master.redis.svc.cluster.local"),
        port=os.getenv("redis_port"),
        decode_responses=True
    )

def deadline_for_matrix(n: int, coeff: float = 2.5e-7, cap: int = 30, floor: int = 1) -> int:
    """
    Analytic deadline (seconds) for an nxn matrix multiplication.

    deadline = min( max(floor, coeff · n³), cap )   # seconds

    • n       - size of the matrix (n x n).
    • coeff   - tunes how many seconds per floating-point operation.
    • cap     - hard upper bound so the system never waits 'forever'.
    • floor   - hard lower bound so the system never waits less than 1 second.
    """
    return int(min(max(floor, coeff * n**3), cap))

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

def extract_result(raw_result):
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
        now = time.time()
        deadline_met = (now - gen_ts) <= deadline

        return {
            "task_id": result.get('task_id'),
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
        raise ValueError(f"ERROR: Malformed result: {e} - Raw: {raw_result[:256]}", file=sys.stderr)

def feedback_task_generation(result, redisClient, input_q, body):
    feedback_task = {
        "task_id": str(uuid.uuid4()),
        "task_gen_timestamp": time.time(),
        "task_application": result["task_application"],
        "task_data": None,  # Actual data will be generated by workers
        "task_data_size": result["task_output_size"][0],
        "task_deadline": deadline_for_matrix(result["task_output_size"][0])  # seconds
    }
    safe_redis_call(lambda: redisClient.lpush(input_q, json.dumps(feedback_task)))
    # invoke_function_async("emitter", body)
    print(f"[INFO] Feedback task generated: {feedback_task['task_id']} for result {result['task_id']}")

def handle(event, context):
    now = datetime.now(ZoneInfo("Europe/Rome"))
    print(f"\n[Collector] Invoked at {now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
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

    result_q, output_q = body.get('result_queue_name'), body.get("output_queue_name")
    input_q, feedback_flag = body.get('input_queue_name'), body.get("collector_feedback_flag")
    print(f"[DEBUG] Collector received body: {body}")

    if not all([input_q, result_q, output_q, feedback_flag is not None]):
        return {"statusCode": 400, "body": "Missing required fields in request body."}

    num_results = 0
    iteration_end = None

    while True:
        print("------------------------------")
        iteration_start = time.time()
        if iteration_end:
            print(f"[INFO] Time since last iteration: {iteration_start - iteration_end:.2f} sec")
        try:
            raw_result = safe_redis_call(lambda: redisClient.rpop(result_q))

            if not raw_result:
                print(f"[INFO] No result in '{result_q}', waiting for results...")
                # time.sleep(5)
                # invoke_function_async("emitter", body)
                continue # break
            else:
                # Fetch and parse results
                result = extract_result(raw_result)
                safe_redis_call(lambda: redisClient.lpush(output_q, json.dumps(result)))

                # Feedback task generation
                print(f"[DEBUG] Feedback flag: {feedback_flag}, Input queue: {input_q}")
                if feedback_flag and input_q:
                    print(f"[INFO] Feedback : Generating task from result {result['task_id']}")
                    feedback_task_generation(result, redisClient, input_q, body)

                print(f"[INFO] Processed result {result['task_id']} from '{result_q}' and pushed to '{output_q}'")

            iteration_end = time.time()
            print(f"[INFO] Iteration completed in {iteration_end - iteration_start:.2f} seconds.")

        except Exception as e:
            print(f"ERROR: Unexpected error: {e}", file=sys.stderr)
            # break
            return {
                "statusCode": 500,
                "body": f"Unexpected error: {e}"
            }

    return {
        "statusCode": 200,
        "body": f"Processed result from '{result_q}' and pushed to '{output_q}'."
    }
