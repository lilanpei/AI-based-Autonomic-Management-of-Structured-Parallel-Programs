import os
import sys
import json
import time
import uuid
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
    deadline_for_matrix,
    extract_result,
    feedback_task_generation
)

def handle(event, context):
    now = datetime.now(ZoneInfo("Europe/Rome"))
    pod_name = os.environ.get("HOSTNAME")
    print(f"\n[Collector] Invoked at {now.strftime('%Y-%m-%d %H:%M:%S %Z')} on pod {pod_name}")
    num_results = 0
    redis_client = get_redis_client()
    body = parse_request_body(event)
    if not body:
        return {"statusCode": 400, "body": "Invalid JSON in request body."}

    result_q, output_q = body.get('result_queue_name'), body.get("output_queue_name")
    input_q, feedback_flag = body.get('input_queue_name'), body.get("collector_feedback_flag")
    control_syn_q, start_q = body.get('collector_control_syn_queue_name'), body.get('collector_start_queue_name')
    print(f"[DEBUG] Collector received body: {body}")

    if not all([input_q, result_q, output_q, start_q, feedback_flag is not None]):
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
                    "body": f"Collector pod {pod_name} acknowledged termination."
                }

            raw_result = safe_redis_call(lambda: redis_client.rpop(result_q))

            if not raw_result:
                attempts += 1
                sleep_time = backoff_factor ** attempts
                print(f"[INFO] No result found in '{result_q}' for {attempts} tries, waiting {sleep_time} seconds...")
                time.sleep(sleep_time)
                if attempts >= retries:
                    print("[WARNING] No results found for a long time, exiting collector.")
                    return {
                        "statusCode": 200,
                        "body": f"Collector pod {pod_name} exited due to inactivity."
                    }
                continue
            else:
                attempts = 0
                now = datetime.now(ZoneInfo("Europe/Rome"))
                print(f"\n[Collector] got result at {now.strftime('%Y-%m-%d %H:%M:%S %Z')} on pod {pod_name}")
                # Fetch and parse results
                result = extract_result(raw_result)
                num_results += 1
                safe_redis_call(lambda: redis_client.lpush(output_q, json.dumps(result)))

                # Feedback task generation
                if feedback_flag and input_q:
                    print(f"[INFO] Feedback : Generating task from result {result['task_id']}")
                    feedback_task_generation(result, redis_client, input_q, body)

                print(f"[Collector] Processed {num_results} results from '{result_q}' and pushed to '{output_q}'")

        except Exception as e:
            print(f"ERROR: Unexpected error: {e}", file=sys.stderr)
            return {
                "statusCode": 500,
                "body": f"Unexpected error: {e}"
            }
    print(f"[Collector] Collector processed {num_results} results.")

    return {
        "statusCode": 200,
        "body": f"Processed result from '{result_q}' and pushed to '{output_q}'."
    }
