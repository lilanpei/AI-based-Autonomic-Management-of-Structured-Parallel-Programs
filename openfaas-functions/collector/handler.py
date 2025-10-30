import os
import sys
import json
import time
import logging
from datetime import datetime
from utils import (
    get_redis_client,
    parse_request_body,
    safe_redis_call,
    fetch_control_message,
    send_start_signal,
    get_utc_now,
    extract_result,
)

# Logger configured in utilities.utils
logger = logging.getLogger(__name__)

def handle(event, context):
    pod_name = os.environ.get("HOSTNAME")
    num_results = 0
    redis_client = get_redis_client()
    body = parse_request_body(event)
    if not body:
        return {"statusCode": 400, "body": "Invalid JSON in request body."}

    input_q, result_q, output_q = body.get('input_queue_name'), body.get('result_queue_name'), body.get("output_queue_name")
    control_syn_q, start_q = body.get('collector_control_syn_queue_name'), body.get('collector_start_queue_name')
    wait_time, program_start_time_str = body.get("wait_time"), body.get("program_start_time")

    if program_start_time_str:
        program_start_time = datetime.fromisoformat(program_start_time_str)
    else:
        logger.error("START_TIMESTAMP environment variable not set.")
        sys.exit(1)

    if not all([input_q, result_q, output_q, start_q, wait_time, program_start_time]):
        return {"statusCode": 400, "body": "Missing required fields in request body."}

    logger.info(f"\n[TIMER] Invoked at {(get_utc_now() - program_start_time).total_seconds():.4f} on pod {pod_name}")
    logger.debug(f"Collector received body: {body}")

    # send start signal to start queue
    send_start_signal(redis_client, start_q, pod_name, (get_utc_now() - program_start_time).total_seconds())

    previous_iteration_start = None

    while True:
        iteration_start = (get_utc_now() - program_start_time).total_seconds()
        logger.debug(f"[TIMER]-------------Iteration start at {iteration_start:.4f}-----------------")
        if previous_iteration_start:
            logger.debug(f"[TIMER] Iteration time: {(iteration_start - previous_iteration_start):.4f} sec")
        previous_iteration_start = iteration_start

        try:
            control_msg = fetch_control_message(redis_client, control_syn_q)
            if control_msg and control_msg.get("action") == "SYN" and control_msg.get("type") == "TERMINATE":
                logger.info(f"Received TERMINATE control message: {control_msg}")
                return {
                    "statusCode": 200,
                    "body": f"Collector pod {pod_name} acknowledged termination."
                }

            raw_result = safe_redis_call(lambda: redis_client.rpop(result_q))

            if not raw_result:
                logger.debug(f"No result found in '{result_q}', waiting {wait_time} seconds...")
                time.sleep(float(wait_time))
                continue
            else:
                logger.info(f"\n[TIMER] got result at {(get_utc_now() - program_start_time).total_seconds():.4f} on pod {pod_name}")
                # Fetch and parse results with QoS calculation
                result = extract_result(raw_result, program_start_time)
                num_results += 1

                # Log QoS information
                logger.info(f"Task {result['task_id']}: {result['task_QoS']}")

                safe_redis_call(lambda: redis_client.lpush(output_q, json.dumps(result)))

                logger.info(f"[TIMER] Processed {num_results} results at {(get_utc_now() - program_start_time).total_seconds():.4f} from '{result_q}' to '{output_q}' on pod {pod_name}")

        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            break

    logger.info(f"Collector processed {num_results} results.")

    return {
        "statusCode": 200,
        "body": f"Processed result from '{result_q}' and pushed to '{output_q}'."
    }
