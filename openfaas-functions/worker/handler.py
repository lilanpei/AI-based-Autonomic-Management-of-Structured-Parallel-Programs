import os
import sys
import json
import time
import redis
import requests
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
    print("!!!!!!!!!!!!! Worker function invoked !!!!!!!!!!!!!")
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

    # Parse request body
    try:
        request_body = json.loads(event.body)
        start_flag = request_body.get('start_flag')
        worker_q_name = request_body.get('worker_queue_name')
        result_q_name = request_body.get('result_queue_name')
        print(f"[INFO] Received request with start_flag={start_flag}, worker_queue_name='{worker_q_name}', result_queue_name='{result_q_name}'")
    except (json.JSONDecodeError, TypeError):
        return {"statusCode": 400, "body": "Invalid or missing JSON in request body."}

    if not worker_q_name or not result_q_name or start_flag is None:
        return {"statusCode": 400, "body": "Missing 'start_flag', 'worker_queue_name', or 'result_queue_name'."}

    if not start_flag:
        print("[INFO] start_flag is False. Worker will not run.")
        return {"statusCode": 200, "body": "Worker exited as instructed (start_flag is False)."}

    tasks_processed = 0

    while True:
        time.sleep(1)
        try:
            try:
                pop_start = time.time()
                raw_task = redisClient.lpop(worker_q_name)
                pop_time = time.time() - pop_start
                print(f"[INFO] lpop from '{worker_q_name}' took {pop_time} seconds.")
            except redis.exceptions.ConnectionError as e:
                print(f"Redis lpop error: {str(e)}. Retrying...")
                time.sleep(5)
                try:
                    redisClient = init_redis_client()
                    retry_pop_start = time.time()
                    raw_task = redisClient.lpop(worker_q_name)
                    retry_pop_time = time.time() - retry_pop_start
                    print(f"[INFO] Recovered: lpop from '{worker_q_name}' took {retry_pop_time} seconds.")
                except Exception as e:
                    return {"statusCode": 500, "body": f"Redis failure: {e}"}

            if raw_task is None:
                print(f"[INFO] No tasks in queue '{worker_q_name}'. Reinvoking self...")
                time.sleep(10)

                payload = {
                    "start_flag": True,
                    "worker_queue_name": worker_q_name,
                    "result_queue_name": result_q_name
                }

                try:
                    response = requests.post(
                        "http://gateway.openfaas.svc.cluster.local:8080/async-function/worker",
                        data=json.dumps(payload),
                        headers={"Content-Type": "application/json"}
                    )
                    print(f"[INFO] Reinvoked worker - Status: {response.status_code}")
                    print("Response Body:", response.text)
                except requests.exceptions.RequestException as e:
                    print(f"ERROR: Self-reinvoke failed: {e}", file=sys.stderr)

                break  # Exit current loop after reinvoking

            try:
                task = json.loads(raw_task)
                task_id = task.get("id")
                task_data = task.get("data")
                task_application = task.get("application")
                task_deadline = task.get("deadline")
                task_emit_timestamp = task.get("timestamp")
                print(f"[INFO] Processing task ID: {task_id}, Application: {task_application}")

                if task_application != "matrix_multiplication":
                    raise ValueError(f"Unsupported task_application: {task_application}")

                if not task_data or not isinstance(task_data, dict):
                    raise ValueError("Invalid task_data format.")

                matrix_a = np.array(task_data.get("matrix_A"))
                matrix_b = np.array(task_data.get("matrix_B"))

                if matrix_a.shape[1] != matrix_b.shape[0]:
                    raise ValueError("Matrix dimensions incompatible for multiplication.")

            except Exception as e:
                print(f"Task parsing or validation failed: {e}", file=sys.stderr)
                continue

            start_time = time.time()
            result_matrix = np.dot(matrix_a, matrix_b)
            end_time = time.time()
            print(f"[INFO] Task ID: {task_id} completed in {end_time - start_time} seconds.")

            structured_result = {
                "task_id": task_id,
                "result_data": {"result_matrix":result_matrix.tolist()},  # Convert numpy array to list for JSON serialization
                "task_application": task_application,
                "task_emit_timestamp": task_emit_timestamp,
                "task_deadline": task_deadline,
                "output_size": result_matrix.size,
                "complete_time": end_time - start_time,
                "complete_timestamp": end_time
            }

            try:
                push_start = time.time()
                redisClient.lpush(result_q_name, json.dumps(structured_result))
                push_time = time.time() - push_start
                print(f"[INFO] lpush to '{result_q_name}' took {push_time} seconds.")
            except redis.exceptions.ConnectionError as e:
                print(f"Redis lpush error: {str(e)}. Retrying...")
                time.sleep(5)
                try:
                    redisClient = init_redis_client()
                    retry_push_start = time.time()
                    redisClient.lpush(result_q_name, json.dumps(structured_result))
                    retry_push_time = time.time() - retry_push_start
                    print(f"[INFO] Recovered: lpush to '{result_q_name}' took {retry_push_time} seconds.")
                except Exception as e:
                    return {"statusCode": 500, "body": f"Redis failure: {e}"}

            tasks_processed += 1
            if tasks_processed % 50 == 0:
                print(f"[INFO] Processed {tasks_processed} tasks so far...")

        except Exception as e:
            print(f"ERROR: Unexpected error: {e}", file=sys.stderr)
            time.sleep(5)
            continue

    return {
        "statusCode": 200,
        "body": f"Worker processed {tasks_processed} tasks from '{worker_q_name}' and pushed results to '{result_q_name}'."
    }
