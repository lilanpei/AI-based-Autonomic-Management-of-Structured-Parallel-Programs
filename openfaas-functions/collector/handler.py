import os
import sys
import json
import time
import redis
import requests
import numpy as np

redisClient = None

def init_redis_client():
    return redis.Redis(
        host=os.getenv("redis_hostname", "redis-master.redis.svc.cluster.local"),
        port=os.getenv("redis_port"),
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

def reinvoke_self(payload):
    try:
        response = requests.post(
            "http://gateway.openfaas.svc.cluster.local:8080/async-function/collector",
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"}
        )
        print(f"[INFO] Reinvoked collector - Status: {response.status_code}, Body: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Self-reinvoke failed: {e}", file=sys.stderr)

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

    tasks_processed = 0
    iteration_end = None

    while True:
        print("------------------------------")
        iteration_start = time.time()
        if iteration_end:
            print(f"[INFO] Time since last iteration: {iteration_start - iteration_end:.2f} sec")
        try:
            queue_length = safe_redis_call(lambda: redisClient.llen(result_q))
            if queue_length == 0:
                print(f"[INFO] Queue '{result_q}' empty. Reinvoking...")
                time.sleep(10)
                reinvoke_self(body)
                break

            # Fetch and parse results
            raw_results = []
            for _ in range(queue_length):
                raw = safe_redis_call(lambda: redisClient.rpop(result_q))
                if not raw:
                    continue
                try:
                    raw_results.append(json.loads(raw))
                except json.JSONDecodeError:
                    print(f"[WARN] Skipped invalid JSON result: {raw[:128]}")

            # Process results
            for result in sorted(raw_results, key=lambda r: r.get("task_emit_timestamp")):
                task_id = result.get("task_id")
                emit_ts = result.get("task_emit_timestamp")
                deadline = result.get("task_deadline")
                complete_ts = result.get("complete_timestamp")

                collect_ts = time.time()
                deadline_met = (collect_ts - emit_ts) <= deadline

                structured = {
                    "task_id": task_id,
                    "task_application": result.get("task_application"),
                    "result_data": result.get("result_data"),
                    "emit_time": result.get("emit_time"),
                    "task_emit_timestamp": emit_ts,
                    "task_deadline": deadline,
                    "collect_time": collect_ts - complete_ts,
                    "task_collect_timestamp": collect_ts,
                    "output_size": result.get("output_size"),
                    "complete_time": result.get("complete_time"),
                    "complete_timestamp": complete_ts,
                    "QoS": deadline_met
                }

                safe_redis_call(lambda: redisClient.rpush(output_q, json.dumps(structured)))
                print(f"[INFO] Pushed result of task {task_id} to '{output_q}'")

                # Feedback task generation
                if feedback_flag and input_q:
                    print(f"[INFO] Feedback: Generating task from result {task_id}")
                    feedback_task = {
                        "task_application": structured["task_application"],
                        "task_data": None,
                        "task_data_size": structured["output_size"],
                        "task_deadline": 1
                    }
                    safe_redis_call(lambda: redisClient.lpush(input_q, json.dumps(feedback_task)))

            iteration_end = time.time()
            tasks_processed = len(raw_results)
            print(f"[INFO] Processed {tasks_processed} results from '{result_q}' and pushed to '{output_q}'.")
            print(f"[INFO] Iteration completed in {iteration_end - iteration_start:.2f} seconds.")

        except Exception as e:
            print(f"ERROR: Unexpected error: {e}", file=sys.stderr)
            break

    return {
        "statusCode": 200,
        "body": f"Processed {tasks_processed} results from '{result_q}' and pushed to '{output_q}'."
    }
