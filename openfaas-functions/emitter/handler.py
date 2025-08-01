import os
import sys
import json
import time
from datetime import datetime
from utils import (
    get_redis_client,
    parse_request_body,
    safe_redis_call,
    fetch_control_message,
    send_start_signal,
    get_utc_now,
    extract_task
)

def handle(event, context):
    pod_name = os.environ.get("HOSTNAME")
    tasks_generated = 0
    redis_client = get_redis_client()
    body = parse_request_body(event)
    if not body:
        return {"statusCode": 400, "body": "Invalid JSON in request body."}

    input_q, worker_q = body.get('input_queue_name'), body.get('worker_queue_name')
    control_syn_q, start_q = body.get('emitter_control_syn_queue_name'), body.get('emitter_start_queue_name')
    wait_time, program_start_time_str = body.get("wait_time"), body.get("program_start_time")

    if program_start_time_str:
        program_start_time = datetime.fromisoformat(program_start_time_str)
    else:
        print("[ERROR] START_TIMESTAMP environment variable not set.", file=sys.stderr)
        sys.exit(1)

    if not all([input_q, worker_q, start_q, wait_time, program_start_time]):
        return {"statusCode": 400, "body": "Missing required fields in request body."}

    print(f"[INFO] Emitter received body: {body}")
    print(f"\n[TIMER] Invoked at {(get_utc_now() - program_start_time).total_seconds():.4f} on pod {pod_name}")

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
            if control_msg and control_msg.get("action") == "SYN" and control_msg.get("type") == "TERMINATE":
                print(f"[INFO] Received TERMINATE control message: {control_msg}")
                return {
                    "statusCode": 200,
                    "body": f"Emitter pod {pod_name} acknowledged termination."
                }

            raw_task = safe_redis_call(lambda: redis_client.rpop(input_q))

            if not raw_task:
                print(f"[INFO] No task found in '{input_q}', waiting {wait_time} seconds...")
                time.sleep(float(wait_time))
                continue
            else:
                print(f"\n[TIMER] got task at {(get_utc_now() - program_start_time).total_seconds():.4f} on pod {pod_name}")
                task = extract_task(raw_task, program_start_time)
                tasks_generated += 1

                safe_redis_call(lambda: redis_client.lpush(worker_q, json.dumps(task)))
                print(f"[TIMER] Pushed {tasks_generated} tasks at {(get_utc_now() - program_start_time).total_seconds():.4f} from '{input_q}' to '{worker_q}' on pod {pod_name}")

        except Exception as e:
            print(f"ERROR: Unexpected error: {e}", file=sys.stderr)
            break

    print(f"[INFO] Emitter processed {tasks_generated} tasks.")

    return {
        "statusCode": 200,
        "body": f"Emitter processed task from '{input_q}' to '{worker_q}'"
    }
