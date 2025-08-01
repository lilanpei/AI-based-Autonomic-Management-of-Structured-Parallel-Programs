import os
import sys
import json
import time
import numpy as np
from datetime import datetime
from utils import (
    get_redis_client,
    parse_request_body,
    safe_redis_call,
    fetch_control_message,
    send_start_signal,
    get_utc_now,
    prepare_matrices
)

def handle(event, context):
    pod_name = os.environ.get("HOSTNAME")
    tasks_processed = 0
    redis_client = get_redis_client()
    body = parse_request_body(event)
    if not body:
        return {"statusCode": 400, "body": "Invalid JSON in request body."}

    worker_q, result_q = body.get('worker_queue_name'), body.get('result_queue_name')
    control_syn_q, control_ack_q = body.get('worker_control_syn_queue_name'), body.get('worker_control_ack_queue_name')
    start_q, processing_delay = body.get('worker_start_queue_name'), body.get('processing_delay')
    wait_time, program_start_time_str = body.get("wait_time"), body.get("program_start_time")

    if program_start_time_str:
        program_start_time = datetime.fromisoformat(program_start_time_str)
    else:
        print("[ERROR] START_TIMESTAMP environment variable not set.", file=sys.stderr)
        sys.exit(1)

    if not all([worker_q, result_q, control_syn_q, control_ack_q, start_q, processing_delay, wait_time, program_start_time]):
        return {"statusCode": 400, "body": "Missing required fields in request body."}

    print(f"\n[TIMER] Invoked at {(get_utc_now() - program_start_time).total_seconds():.4f} on pod {pod_name}")
    print(f"[INFO] Worker received body: {body}")

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
            if control_msg and control_msg.get("action") == "SYN":
                if control_msg.get("type") == "SCALE_DOWN":
                    print(f"[INFO] Received SCALE_DOWN control message: {control_msg}")
                    # Acknowledge the scale down request
                    ack_msg = {
                        "type": "SCALE_DOWN",
                        "action": "ACK",
                        "ack_timestamp": (get_utc_now() - program_start_time).total_seconds(),
                        "task_id": control_msg.get("task_id"),
                        "pod_name": pod_name,
                        "message": "Worker pod is exiting as instructed."
                    }

                    safe_redis_call(lambda: redis_client.lpush(control_ack_q, json.dumps(ack_msg)))
                    print(f"[INFO] Sent ACK for control message: {ack_msg}")
                    return {
                        "statusCode": 200,
                        "body": f"Worker pod {pod_name} acknowledged scale down."
                    }
                elif control_msg.get("type") == "TERMINATE":
                    print(f"[INFO] Received TERMINATE control message: {control_msg}")
                    return {
                        "statusCode": 200,
                        "body": f"Worker pod {pod_name} acknowledged termination."
                    }
                else:
                    print(f"[WARNING] Unknown control message type: {control_msg.get('type')}")

            raw_task = safe_redis_call(lambda: redis_client.rpop(worker_q))

            if not raw_task:
                time.sleep(float(wait_time))
                continue
            else:
                print(f"\n[TIMER] got task at {(get_utc_now() - program_start_time).total_seconds():.4f} on pod {pod_name}")
                task = json.loads(raw_task)
                tasks_processed += 1
                if task.get("task_application") != "matrix_multiplication":
                    raise ValueError("Unsupported task application")

                task_id = task.get("task_id")
                print(f"[INFO] Processing task ID: {task_id}")
                matrix_a, matrix_b = prepare_matrices(task)
                result_matrix = np.dot(matrix_a, matrix_b)
                time.sleep(float(processing_delay))  # Simulate processing delay
                now = (get_utc_now() - program_start_time).total_seconds()
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
                safe_redis_call(lambda: redis_client.lpush(result_q, json.dumps(result)))
                print(f"[TIMER] Processed {tasks_processed} tasks at {now} from {worker_q} and pushed result to '{result_q}' on pod {pod_name}")

        except Exception as e:
            print(f"[ERROR] Failed to process task: {e}", file=sys.stderr)
            break

    print(f"[INFO] Worker processed {tasks_processed} tasks.")

    return {
        "statusCode": 200,
        "body": f"Worker processed task from '{worker_q}' to '{result_q}'"
    }
