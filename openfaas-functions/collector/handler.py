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
    extract_result,
    feedback_task_generation
)

def handle(event, context):
    pod_name = os.environ.get("HOSTNAME")
    num_results = 0
    redis_client = get_redis_client()
    body = parse_request_body(event)
    if not body:
        return {"statusCode": 400, "body": "Invalid JSON in request body."}

    result_q, output_q = body.get('result_queue_name'), body.get("output_queue_name")
    input_q, feedback_flag = body.get('input_queue_name'), body.get("collector_feedback_flag")
    control_syn_q, start_q = body.get('collector_control_syn_queue_name'), body.get('collector_start_queue_name')
    wait_time, program_start_time_str = body.get("wait_time"), body.get("program_start_time")
    deadline_coeff, deadline_cap, deadline_floor = body.get("deadline_coeff"), body.get("deadline_cap"), body.get("deadline_floor")

    if program_start_time_str:
        program_start_time = datetime.fromisoformat(program_start_time_str)
    else:
        print("[ERROR] START_TIMESTAMP environment variable not set.", file=sys.stderr)
        sys.exit(1)

    if not all([input_q, result_q, output_q, start_q, wait_time, program_start_time, feedback_flag is not None]):
        return {"statusCode": 400, "body": "Missing required fields in request body."}

    print(f"\n[TIMER] Invoked at {(get_utc_now() - program_start_time).total_seconds():.4f} on pod {pod_name}")
    print(f"[INFO] Collector received body: {body}")

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
                    "body": f"Collector pod {pod_name} acknowledged termination."
                }

            raw_result = safe_redis_call(lambda: redis_client.rpop(result_q))

            if not raw_result:
                print(f"[INFO] No result found in '{result_q}', waiting {wait_time} seconds...")
                time.sleep(float(wait_time))
                continue
            else:
                print(f"\n[TIMER] got result at {(get_utc_now() - program_start_time).total_seconds():.4f} on pod {pod_name}")
                # Fetch and parse results
                result = extract_result(raw_result, program_start_time)
                num_results += 1
                safe_redis_call(lambda: redis_client.lpush(output_q, json.dumps(result)))

                # Feedback task generation
                if feedback_flag and input_q:
                    print(f"[INFO] Feedback : Generating task from result {result['task_id']}")
                    feedback_task_generation(result, redis_client, input_q, program_start_time, deadline_coeff, deadline_cap, deadline_floor)

                print(f"[TIMER] Processed {num_results} results at {(get_utc_now() - program_start_time).total_seconds():.4f} from '{result_q}' to '{output_q}' on pod {pod_name}")

        except Exception as e:
            print(f"ERROR: Unexpected error: {e}", file=sys.stderr)
            break

    print(f"[Collector] Collector processed {num_results} results.")

    return {
        "statusCode": 200,
        "body": f"Processed result from '{result_q}' and pushed to '{output_q}'."
    }
