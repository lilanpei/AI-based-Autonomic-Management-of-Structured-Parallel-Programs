import os
import sys
import json
import time
import uuid
import redis
import requests

redisClient = None

def init_redis_client():
    redisHostname = os.getenv('redis_hostname', default='redis-master.redis.svc.cluster.local')
    redisPort = os.getenv('redis_port')

    return redis.Redis(
        host=redisHostname,
        port=redisPort,
        decode_responses=True
    )


def handle(event, context):
    print(f"!!!!!!!!!!!!! Emitter function invoked at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))} !!!!!!!!!!!!!")
    global redisClient
    tasks_generated_count = 0
    iteration_end = None

    while True:
        print("------------------------------")
        iteration_start = time.time()
        print(f"[INFO] Emitter loop iteration started at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(iteration_start))}")

        if iteration_end is not None:
            gap_time = iteration_start - iteration_end
            print(f"[INFO] Time gap since last iteration: {gap_time:.2f} seconds")

        try:
            if redisClient is None:
                try:
                    redisClient = init_redis_client()
                except redis.exceptions.ConnectionError as e:
                    print(f"Redis connection error: {str(e)}. Attempting to reinitialize.")
                    time.sleep(5)
                    try:
                        redisClient = init_redis_client() # Reinitialize blocking client
                    except Exception as init_e:
                        print(f"CRITICAL ERROR: Redis reinit failed: {init_e}", file=sys.stderr)
                        return {"statusCode": 500, "body": f"Redis failure: {init_e}"}

            # Parse request body
            try:
                request_body = json.loads(event.body)
                input_q_name = request_body.get('input_queue_name')
                worker_q_name = request_body.get('worker_queue_name')
            except (json.JSONDecodeError, TypeError):
                print(f"ERROR: Invalid or non-JSON request body (truncated): '{str(event.body)[:512]}'", file=sys.stderr)
                return {"statusCode": 400, "body": "Invalid or missing JSON in request body."}
            except Exception as e:
                print(f"ERROR: Unexpected error parsing request body: {e}", file=sys.stderr)
                return {"statusCode": 500, "body": f"Internal error parsing request: {e}"}

            if not input_q_name or not worker_q_name:
                print("ERROR: 'input_queue_name' or 'worker_queue_name' not provided in request body.", file=sys.stderr)
                return {"statusCode": 400, "body": "Missing 'input_queue_name' or 'worker_queue_name'."}

            try:
                pop_start = time.time()
                raw_input_task = redisClient.rpop(input_q_name)
                pop_time = time.time() - pop_start
                print(f"[INFO] rpop from '{input_q_name}' took {pop_time} seconds.")
            except redis.exceptions.ConnectionError as e:
                print(f"Redis rpop connection error: {str(e)}. Attempting to reinitialize.")
                time.sleep(5)
                try:
                    redisClient = init_redis_client() # Reinitialize blocking client
                    retry_pop_start = time.time()
                    raw_input_task = redisClient.rpop(input_q_name)
                    retry_pop_time = time.time() - retry_pop_start
                    print(f"[INFO] Recovered: rpop from '{input_q_name}' took {retry_pop_time} seconds.")
                except Exception as init_e:
                    print(f"CRITICAL ERROR: Redis reinit and rpop failed: {init_e}", file=sys.stderr)
                    return {"statusCode": 500, "body": f"Redis failure: {init_e}"}
                
            if raw_input_task is None:
                print(f"[INFO] No more tasks in queue '{input_q_name}'. Waiting for new tasks...")
                time.sleep(10)  # No tasks in queue, wait before checking again
                # continue
                # Reinvoke self to check for new tasks
                print(f"[INFO] Reinvoking self to check for new tasks in '{input_q_name}'...")
                # Prepare payload for self-reinvoke
                payload = {
                    "start_flag": True,
                    "input_queue_name": input_q_name,
                    "worker_queue_name": worker_q_name,
                }

                try:
                    response = requests.post(
                        "http://gateway.openfaas.svc.cluster.local:8080/async-function/emitter",
                        data=json.dumps(payload),
                        headers={"Content-Type": "application/json"}
                    )
                    print(f"[INFO] Reinvoked emitter - Status: {response.status_code}")
                    print("Response Body:", response.text)
                except requests.exceptions.RequestException as e:
                    print(f"ERROR: Self-reinvoke failed: {e}", file=sys.stderr)

                break  # Exit current loop after reinvoking

            try:
                input_task = json.loads(raw_input_task)
                if not isinstance(input_task, dict):
                    raise ValueError("Expected task to be a JSON object.")
                
                required_keys = ['task_application', 'task_data', 'task_data_size', 'task_deadline']
                if not all(k in input_task for k in required_keys):
                    raise ValueError(f"Missing one of required keys: {required_keys}")

                original_task_application = input_task.get("task_application")
                original_task_data = input_task.get("task_data")
                original_task_size = input_task.get("task_data_size")
                original_task_deadline = input_task.get("task_deadline")

            except json.JSONDecodeError as e:
                print(f"ERROR: Could not parse or validate task: '{raw_input_task[:256]}...' - {str(e)}", file=sys.stderr)
                continue # break

            new_task_id = str(uuid.uuid4())
            print(f"[INFO] Generated new task ID: {new_task_id}")
            new_task_timestamp = time.time()

            structured_task = {
                "id": new_task_id,
                "application": original_task_application,
                "data": original_task_data,
                "size": original_task_size,
                "deadline": original_task_deadline,
                "timestamp": new_task_timestamp,
            }

            try:
                push_start = time.time()
                redisClient.lpush(worker_q_name, json.dumps(structured_task))
                push_time = time.time() - push_start
                print(f"[INFO] lpush to '{worker_q_name}' took {push_time} seconds.")
            except redis.exceptions.ConnectionError as e:
                print(f"Redis lpush connection error: {str(e)}. Attempting to reinitialize.")
                time.sleep(5)
                try:
                    redisClient = init_redis_client() # Reinitialize blocking client
                    retry_push_start = time.time()
                    redisClient.lpush(worker_q_name, json.dumps(structured_task))
                    retry_push_time = time.time() - retry_push_start
                    print(f"[INFO] Recovered: lpush to '{worker_q_name}' took {retry_push_time} seconds.")
                except Exception as init_e:
                    print(f"CRITICAL ERROR: Redis reinit and lpush failed: {init_e}", file=sys.stderr)
                    return {"statusCode": 500, "body": f"Redis failure: {init_e}"}

            tasks_generated_count += 1

            if tasks_generated_count % 100 == 0:
                print(f"[INFO] Processed {tasks_generated_count} tasks so far...")

            iteration_end = time.time()
            print(f"[INFO] Emitter loop iteration completed in {iteration_end - iteration_start:.2f} seconds.")

        except Exception as e:
            print(f"ERROR: Unexpected error during processing: {str(e)}", file=sys.stderr)
            break

    return {
        "statusCode": 200,
        "body": f"Emitter processed {tasks_generated_count} tasks from '{input_q_name}' and pushed to '{worker_q_name}'."
    }
