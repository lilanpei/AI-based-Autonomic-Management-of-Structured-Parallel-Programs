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
    print(f"!!!!!!!!!!!!! Worker function invoked at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))} !!!!!!!!!!!!!")
    pod_name = os.environ.get("HOSTNAME")
    print(f"This function is running in pod: {pod_name}")
    global redisClient
    tasks_processed = 0
    iteration_end = None

    while True:
        print("------------------------------")
        iteration_start = time.time()
        print(f"[INFO] Worker loop iteration started at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(iteration_start))}")

        if iteration_end is not None:
            gap_time = iteration_start - iteration_end
            print(f"[INFO] Time gap since last iteration: {gap_time:.2f} seconds")

        try:
            worker_start_time = time.time()
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
                if not request_body:
                    return {"statusCode": 400, "body": "Empty request body."}
                start_flag = request_body.get('start_flag')
                worker_q_name = request_body.get('worker_queue_name')
                result_q_name = request_body.get('result_queue_name')
                control_syn_q_name = request_body.get('control_syn_queue_name')
                control_ack_q_name = request_body.get('control_ack_queue_name')
                print(f"[INFO] Received request with start_flag={start_flag}, control_syn_q_name='{control_syn_q_name}', control_ack_q_name='{control_ack_q_name}'")
            except (json.JSONDecodeError, TypeError):
                print(f"ERROR: Invalid or non-JSON request body (truncated): '{str(event.body)[:512]}'", file=sys.stderr)
                return {"statusCode": 400, "body": "Invalid or missing JSON in request body."}

            print(f"[DEBUG] Worker queue name: {worker_q_name}, Result queue name: {result_q_name}")
            if not worker_q_name or not result_q_name or start_flag is None:
                return {"statusCode": 400, "body": "Missing 'start_flag', 'worker_queue_name', or 'result_queue_name'."}
            print(f"[DEBUG] Worker pod: {pod_name} is ready to process tasks from '{worker_q_name}' and push results to '{result_q_name}'.")
            # if not start_flag:
            #     print(f"[INFO] start_flag is False. Worker pod: {pod_name} will be deleted.")
            #     control_message_timestamp = time.time()
            #     structured_control_message = {
            #         "task_id": task_id,
            #         "pod_name": pod_name,
            #         "control_message_timestamp": control_message_timestamp,
            #         "message": "Worker pod is exiting as instructed."
            #     }

            #     return {"statusCode": 200, "body": structured_control_message}

            try:
                print(f"[DEBUG] Waiting for control message from '{control_syn_q_name}' queue...")
                control_raw = redisClient.rpop(control_syn_q_name)
                if control_raw:
                    control_task = json.loads(control_raw)
                    print(f"[DEBUG] Received control message: {control_task}")

                    # Handle SCALE_DOWN-SYN control message
                    control_type = control_task.get("type")
                    control_action = control_task.get("action")
                    task_id = control_task.get("task_id")

                    if control_type == "SCALE_DOWN" and control_action == "SYN":
                        print(f"[INFO] SCALE_DOWN-SYN received. Pod '{pod_name}' will acknowledge and exit.")

                        structured_ack = {
                            "type": "SCALE_DOWN",
                            "action": "ACK",
                            "ack_timestamp": time.time(),
                            "task_id": task_id,
                            "pod_name": pod_name,
                            "message": "Worker pod is exiting as instructed."
                        }

                        try:
                            print(f"[INFO] Pushing ACK message to '{control_ack_q_name}' queue.")
                            redisClient.lpush(control_ack_q_name, json.dumps(structured_ack))
                            break  # Exit loop after ACK

                        except redis.exceptions.ConnectionError as push_err:
                            print(f"[ERROR] Redis connection error on LPUSH: {push_err}. Retrying...")
                            time.sleep(5)
                            try:
                                redisClient = init_redis_client()
                                redisClient.lpush(control_ack_q_name, json.dumps(structured_ack))
                                break  # Successfully pushed after reconnect
                            except Exception as push_fail:
                                return {"statusCode": 500, "body": f"Redis failure on ACK push: {push_fail}"}

            except redis.exceptions.ConnectionError as conn_err:
                print(f"[ERROR] Redis connection error on RPOP: {conn_err}. Retrying after delay...")
                time.sleep(5)
                try:
                    redisClient = init_redis_client()
                    continue  # Retry on reconnect
                except Exception as reinit_err:
                    return {"statusCode": 500, "body": f"Redis reinit failed: {reinit_err}"}

            except Exception as parse_err:
                print(f"[ERROR] Failed to parse control message: {parse_err}")

            try:
                pop_start = time.time()
                raw_task = redisClient.rpop(worker_q_name)
                pop_time = time.time() - pop_start
                print(f"[INFO] rpop from '{worker_q_name}' took {pop_time} seconds.")
            except redis.exceptions.ConnectionError as e:
                print(f"Redis rpop error: {str(e)}. Retrying...")
                time.sleep(5)
                try:
                    redisClient = init_redis_client()
                    retry_pop_start = time.time()
                    raw_task = redisClient.rpop(worker_q_name)
                    retry_pop_time = time.time() - retry_pop_start
                    print(f"[INFO] Recovered: rpop from '{worker_q_name}' took {retry_pop_time} seconds.")
                except Exception as e:
                    return {"statusCode": 500, "body": f"Redis failure: {e}"}

            if raw_task is None:
                print(f"[INFO] No tasks in queue '{worker_q_name}'. Reinvoking self...")
                time.sleep(10)
                # continue  # No tasks to process, wait before checking again
                # Reinvoke self to check for new tasks
                print(f"[INFO] Reinvoking self to check for new tasks in '{worker_q_name}'...")
                # Prepare payload for self-reinvoke
                payload = {
                    "start_flag": True,
                    "worker_queue_name": worker_q_name,
                    "result_queue_name": result_q_name,
                    "control_syn_queue_name": control_syn_q_name,
                    "control_ack_queue_name": control_ack_q_name
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
                task_size = task.get("size")
                task_application = task.get("application")
                task_deadline = task.get("deadline")
                task_emit_timestamp = task.get("timestamp")
                print(f"[INFO] Processing task ID: {task_id}, Application: {task_application}")

                if task_application != "matrix_multiplication":
                    raise ValueError(f"Unsupported task_application: {task_application}")

                if not task_data or not isinstance(task_data, dict):
                    # Generate the two matrices for multiplication
                    # Handle different task_size cases
                    print(f"[INFO] Generating matrices for task ID: {task_id} with size: {task_size}")
                    if isinstance(task_size, int):
                        if task_size == 0:
                            rows = cols = 10 # Default size if task_size is 0
                        else:
                            rows = cols = task_size
                    else:
                        rows = task_size[0]
                        cols = task_size[1]

                    # Generate matrices
                    matrix_a = np.random.rand(rows, cols)
                    matrix_b = np.random.rand(cols, rows)
                else:
                    matrix_a = np.array(task_data.get("matrix_A"))
                    matrix_b = np.array(task_data.get("matrix_B"))

                if matrix_a.shape[1] != matrix_b.shape[0]:
                    raise ValueError("Matrix dimensions incompatible for multiplication.")

            except Exception as e:
                print(f"Task parsing or validation failed: {e}", file=sys.stderr)
                continue

            result_matrix = np.dot(matrix_a, matrix_b)
            time.sleep(2)  # Simulate processing time
            end_time = time.time()
            print(f"[INFO] Task ID: {task_id} completed in {end_time - worker_start_time} seconds.")

            structured_result = {
                "task_id": task_id,
                # "result_data": {"result_matrix":result_matrix.tolist()},  # Convert numpy array to list for JSON serialization
                "result_data": None,  # Placeholder for result matrix
                "task_application": task_application,
                "task_emit_timestamp": task_emit_timestamp,
                "task_deadline": task_deadline,
                "output_size": result_matrix.shape,
                "emit_time": worker_start_time - task_emit_timestamp,
                "complete_time": end_time - worker_start_time,
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

            iteration_end = time.time()
            print(f"[INFO] Worker loop iteration completed in {iteration_end - iteration_start:.2f} seconds.")

        except Exception as e:
            print(f"ERROR: Unexpected error: {e}", file=sys.stderr)
            break

    return {
        "statusCode": 200,
        "body": f"Worker processed {tasks_processed} tasks from '{worker_q_name}' and pushed results to '{result_q_name}'."
    }
