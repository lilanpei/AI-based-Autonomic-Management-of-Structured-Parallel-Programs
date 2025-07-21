import os
import sys
import json
import time
import redis
from datetime import datetime
from zoneinfo import ZoneInfo
from utils import (
    backoff_factor,
    retries,
    get_redis_client,
    parse_request_body,
    safe_redis_call,
    fetch_control_message,
    send_start_signal,
    extract_task
)

def handle(event, context):
    now = datetime.now(ZoneInfo("Europe/Rome"))
    pod_name = os.environ.get("HOSTNAME")
    print(f"\n[Emitter] Invoked at {now.strftime('%Y-%m-%d %H:%M:%S %Z')} on pod {pod_name}")
    tasks_generated = 0
    redis_client = get_redis_client()
    body = parse_request_body(event)
    if not body:
        return {"statusCode": 400, "body": "Invalid JSON in request body."}

    input_q, worker_q = body.get('input_queue_name'), body.get('worker_queue_name')
    control_syn_q, start_q = body.get('emitter_control_syn_queue_name'), body.get('emitter_start_queue_name')
    print(f"[DEBUG] Emitter received body: {body}")

    if not all([input_q, worker_q, start_q]):
        return {"statusCode": 400, "body": "Missing required fields in request body."}

    # send start signal to start queue
    send_start_signal(redis_client, start_q, pod_name, now)

    previous_iteration_start = None
    attempts = 0

    while True:
        print("------------------------------")
        iteration_start = time.time()
        if previous_iteration_start:
            print(f"[INFO] Iteration time: {iteration_start - previous_iteration_start:.2f} sec")
        previous_iteration_start = iteration_start

        try:
            control_msg = fetch_control_message(redis_client, control_syn_q)
            if control_msg and control_msg.get("action") == "SYN" and control_msg.get("type") == "TEMINATE":
                print(f"[INFO] Received TERMINATE control message: {control_msg}")
                return {
                    "statusCode": 200,
                    "body": f"Emitter pod {pod_name} acknowledged termination."
                }

            raw_task = safe_redis_call(lambda: redis_client.rpop(input_q))

            if not raw_task:
                attempts += 1
                sleep_time = backoff_factor ** attempts
                print(f"[INFO] No task found in '{input_q}' for {attempts} tries, waiting {sleep_time} seconds...")
                time.sleep(sleep_time)
                if attempts >= retries:
                    print("[WARNING] No tasks found for a long time, exiting emitter.")
                    return {
                        "statusCode": 200,
                        "body": f"Emitter pod {pod_name} exited due to inactivity."
                    }
                continue
            else:
                attempts = 0
                now = datetime.now(ZoneInfo("Europe/Rome"))
                print(f"\n[Emitter] got task at {now.strftime('%Y-%m-%d %H:%M:%S %Z')} on pod {pod_name}")
                task = extract_task(raw_task)
                tasks_generated += 1

                safe_redis_call(lambda: redis_client.lpush(worker_q, json.dumps(task)))
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
