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

def handle(event, context):
    print(f"\n[Collector] Invoked at {time.strftime('%Y-%m-%d %H:%M:%S')}")
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
    input_q, feedback_flag = body.get('input_queue_name'), body.get("collector_feedback_flag", False)

    if not all([input_q, result_q, output_q, feedback_flag is not None]):
        return {"statusCode": 400, "body": "Missing required fields in request body."}

    num_results = 0
    # iteration_end = None

    # while True:
    print("------------------------------")
    iteration_start = time.time()
    # if iteration_end:
    #     print(f"[INFO] Time since last iteration: {iteration_start - iteration_end:.2f} sec")
    try:
        queue_length = safe_redis_call(lambda: redisClient.llen(result_q))
        if queue_length == 0:
            print(f"[INFO] Queue '{result_q}' empty. Waiting for results...")
            # time.sleep(5)
            # invoke_function_async("collector", body)
            # continue # break
        else:
            # Fetch and parse results
            raw_results = []
            for _ in range(queue_length):
                raw = safe_redis_call(lambda: redisClient.rpop(result_q))
                if raw:
                    try:
                        raw_results.append(json.loads(raw))
                    except json.JSONDecodeError:
                        print(f"[WARN] Skipped invalid JSON result: {raw[:128]}")

            # Process results
            num_results = len(raw_results)
            if num_results > 0:
                print(f"[INFO] Processing {num_results} results from '{result_q}'")

                for result in sorted(raw_results, key=lambda r: r.get("task_gen_timestamp")):
                    task_id = result.get("task_id")
                    gen_ts = result.get("task_gen_timestamp")
                    deadline = result.get("task_deadline")
                    work_ts = result.get("task_work_timestamp")

                    now = time.time()
                    deadline_met = (now - gen_ts) <= deadline

                    structured = {
                        "task_id": task_id,
                        "collect_iteration_time": iteration_start - now,
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

                    safe_redis_call(lambda: redisClient.rpush(output_q, json.dumps(structured)))
                    print(f"[INFO] Pushed result of task {task_id} to '{output_q}'")

                    # Feedback task generation
                    if feedback_flag and input_q:
                        print(f"[INFO] Feedback: Generating task from result {task_id}")
                        feedback_task = {
                            "task_id": str(uuid.uuid4()),
                            "task_gen_timestamp": time.time(),
                            "task_application": structured["task_application"],
                            "task_data": None,  # Actual data will be generated by workers
                            "task_data_size": structured["output_size"][0],
                            "task_deadline": deadline_for_matrix(structured["output_size"][0])  # seconds
                        }
                        safe_redis_call(lambda: redisClient.lpush(input_q, json.dumps(feedback_task)))
                        invoke_function_async("emitter", body)

                print(f"[INFO] Processed {num_results} results from '{result_q}' and pushed to '{output_q}'.")

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
        "body": f"Processed {num_results} results from '{result_q}' and pushed to '{output_q}'."
    }
