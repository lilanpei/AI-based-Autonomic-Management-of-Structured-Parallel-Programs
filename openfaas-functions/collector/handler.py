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
    total_tasks = 0
    tasks_met_deadline = 0

    while True:
        print("------------------------------")
        print("[INFO] Starting new collector loop iteration...")
        try:
            if redisClient is None:
                try:
                    redisClient = init_redis_client()
                except redis.exceptions.ConnectionError as e:
                    print(f"[ERROR] Redis init failed: {e}. Retrying in 5 seconds...")
                    time.sleep(5)
                    redisClient = init_redis_client()

            try:
                request_body = json.loads(event.body)
                collector_feedback_flag = request_body.get("collector_feedback_flag", False)
                result_queue_name = request_body.get("result_queue_name")
                output_queue_name = request_body.get("output_queue_name")
                input_queue_name = request_body.get("input_queue_name")
            except (json.JSONDecodeError, TypeError) as e:
                return {"statusCode": 400, "body": f"Invalid request body: {str(e)}"}

            if not result_queue_name or not output_queue_name:
                return {"statusCode": 400, "body": "Missing 'result_queue_name' or 'output_queue_name'."}

            # Fetch all results from result_queue
            queue_length = redisClient.llen(result_queue_name)
            if queue_length == 0:
                print(f"[INFO] Result queue '{result_queue_name}' is empty. Waiting...")
                time.sleep(10)
                continue

            raw_results = []
            for _ in range(queue_length):
                raw = redisClient.lpop(result_queue_name)
                if raw:
                    try:
                        result = json.loads(raw)
                        raw_results.append(result)
                    except json.JSONDecodeError:
                        print(f"[WARN] Skipping invalid JSON result: {raw}")
                        continue

            # Sort results by emit timestamp
            sorted_results = sorted(raw_results, key=lambda r: r.get("task_emit_timestamp", 0))

            # Process and push sorted results
            for result in sorted_results:
                task_id = result.get("task_id")
                task_application = result.get("task_application")
                task_emit_timestamp = result.get("task_emit_timestamp")
                task_deadline = result.get("task_deadline")
                complete_timestamp = result.get("complete_timestamp")
                result_data = result.get("result_data")
                output_size = result.get("output_size")
                emit_time = result.get("emit_time")
                complete_time = result.get("complete_time")

                task_collect_timestamp = time.time()
                deadline_exceeded = (task_collect_timestamp - task_emit_timestamp) > task_deadline
                qos = not deadline_exceeded

                structured_result = {
                    "task_id": task_id,
                    "result_data": result_data,
                    "task_application": task_application,
                    "emit_time": emit_time,
                    "task_emit_timestamp": task_emit_timestamp,
                    "task_deadline": task_deadline,
                    "collect_time": task_collect_timestamp - complete_timestamp,
                    "task_collect_timestamp": task_collect_timestamp,
                    "output_size": output_size,
                    "complete_time": complete_time,
                    "complete_timestamp": complete_timestamp,
                    "QoS": qos
                }

                try:
                    redisClient.rpush(output_queue_name, json.dumps(structured_result))
                    print(f"[INFO] RPUSHED task {task_id} to output queue.")
                except redis.exceptions.ConnectionError as e:
                    print(f"[ERROR] Redis push failed: {e}. Retrying...")
                    time.sleep(5)
                    redisClient = init_redis_client()
                    redisClient.rpush(output_queue_name, json.dumps(structured_result))

                total_tasks += 1
                if qos:
                    tasks_met_deadline += 1

                # Feedback logic
                if collector_feedback_flag and input_queue_name:
                    print(f"[INFO] Feedback enabled. Generating new task from {task_id}.")
                    if result_data is not None:
                        matrix = np.array(result_data.get("result_matrix"))

                    new_task = {
                        "task_application": task_application,
                        "task_data": None,  # Placeholder if needed
                        "task_data_size": output_size,
                        "task_deadline": 1
                    }

                    try:
                        redisClient.lpush(input_queue_name, json.dumps(new_task))
                        print(f"[INFO] Feedback task for {task_id} pushed to input queue.")
                    except redis.exceptions.ConnectionError as e:
                        print(f"[ERROR] Redis feedback push failed: {e}. Retrying...")
                        time.sleep(5)
                        redisClient = init_redis_client()
                        redisClient.lpush(input_queue_name, json.dumps(new_task))

        except Exception as e:
            print(f"[ERROR] Unexpected error: {e}", file=sys.stderr)
            break

    qos_rate = (tasks_met_deadline / total_tasks * 100) if total_tasks else 0
    print(f"[SUMMARY] Total Tasks: {total_tasks}, Met Deadline: {tasks_met_deadline}, QoS Success Rate: {qos_rate:.2f}%")

    return {
        "statusCode": 200,
        "body": f"Collector processed {total_tasks} tasks from '{result_queue_name}' with {qos_rate:.2f}% meeting deadline."
    }
