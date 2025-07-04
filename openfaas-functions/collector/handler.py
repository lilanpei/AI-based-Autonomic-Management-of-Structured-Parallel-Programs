import os
import sys
import json
import time
import redis
import numpy as np

redisClient = None

def init_redis_client():
    redis_hostname = os.getenv('redis_hostname', default='redis-master.redis.svc.cluster.local')
    redis_port = os.getenv('redis_port')
    return redis.Redis(
        host=redis_hostname,
        port=redis_port,
        decode_responses=True
    )

def handle(event, context):
    print("!!!!!!!!!!!!! Collector function invoked !!!!!!!!!!!!!")
    global redisClient

    if redisClient is None:
        try:
            redisClient = init_redis_client()
        except redis.exceptions.ConnectionError as e:
            print(f"Redis init error: {str(e)}. Retrying...")
            time.sleep(5)
            try:
                redisClient = init_redis_client()
            except Exception as e:
                print(f"CRITICAL ERROR: Redis reinit failed: {e}", file=sys.stderr)
                return {"statusCode": 500, "body": f"Redis failure: {e}"}

    try:
        request_body = json.loads(event.body)
        collector_feedback_flag = request_body.get("collector_feedback_flag", False)
        result_queue_name = request_body.get("result_queue_name")
        output_queue_name = request_body.get("output_queue_name")
        input_queue_name = request_body.get("input_queue_name")
    except (json.JSONDecodeError, TypeError) as e:
        return {"statusCode": 400, "body": f"Invalid or missing JSON in request body. {str(e)}"}

    if not result_queue_name or not output_queue_name:
        return {"statusCode": 400, "body": "Missing 'result_queue_name' or 'output_queue_name'."}

    total_tasks = 0
    tasks_met_deadline = 0

    while True:
        try:
            raw_result = redisClient.lpop(result_queue_name)
            if raw_result is None:
                print(f"[INFO] Result queue '{result_queue_name}' is empty. Waiting for new results...")
                time.sleep(10)
                continue  # Wait before checking again

            result = json.loads(raw_result)
            task_id = result.get("task_id")
            task_application = result.get("task_application")
            task_emit_timestamp = result.get("task_emit_timestamp")
            task_deadline = result.get("task_deadline")
            complete_timestamp = result.get("complete_timestamp")
            result_data = result.get("result_data")
            output_size = result.get("output_size")
            complete_time = result.get("complete_time")

            task_collect_timestamp = time.time()
            deadline_exceeded = (task_collect_timestamp - task_emit_timestamp) > task_deadline
            qos = not deadline_exceeded

            structured_result = {
                "task_id": task_id,
                "result_data": result_data,
                "task_application": task_application,
                "task_emit_timestamp": task_emit_timestamp,
                "task_deadline": task_deadline,
                "task_collect_timestamp": task_collect_timestamp,
                "output_size": output_size,
                "complete_time": complete_time,
                "complete_timestamp": complete_timestamp,
                "QoS": qos
            }

            redisClient.lpush(output_queue_name, json.dumps(structured_result))
            print(f"[INFO] Result for task {task_id} pushed to output queue with QoS={qos}.")

            total_tasks += 1
            if qos:
                tasks_met_deadline += 1

            print(f"[INFO] Collector feedback flag: {collector_feedback_flag}, Input queue name: {input_queue_name}")
            if collector_feedback_flag and input_queue_name:
                print(f"[INFO] Collector feedback enabled. Processing task {task_id} for feedback.")
                matrix = np.array(result_data.get("result_matrix"))
                new_task = {
                    "task_application": task_application,
                    "task_data": {
                        "matrix_A": matrix.tolist(),
                        "matrix_B": matrix.tolist()
                    },
                    "task_data_size": matrix.size,
                    "task_deadline": 1
                }
                redisClient.lpush(input_queue_name, json.dumps(new_task))
                print(f"[INFO] Feedback task based on task {task_id} pushed to input queue.")

        except json.JSONDecodeError as e:
            print(f"[ERROR] JSON decoding failed: {str(e)}")
            continue
        except redis.exceptions.ConnectionError as e:
            print(f"[ERROR] Redis error: {str(e)}. Retrying...")
            time.sleep(5)
            redisClient = init_redis_client()
            continue
        except Exception as e:
            print(f"[ERROR] Unexpected error: {str(e)}", file=sys.stderr)
            time.sleep(5)
            continue

    percentage = (tasks_met_deadline / total_tasks * 100) if total_tasks else 0
    print(f"[SUMMARY] Total Tasks: {total_tasks}, Met Deadline: {tasks_met_deadline}, QoS Success Rate: {percentage:.2f}%")

    return {
        "statusCode": 200,
        "body": f"Collector processed {total_tasks} tasks from '{result_queue_name}' with {percentage:.2f}% meeting deadline."
    }
