import os
import json
import time
import redis
import random
import numpy as np

redisClient = None

def initRedis():
    redisHostname = os.getenv('redis_hostname', default='redis-master.redis.svc.cluster.local')
    redisPort = os.getenv('redis_port')

    return redis.Redis(
        host=redisHostname,
        port=redisPort,
        decode_responses=True
    )

# basic numpy matrix multiplication
def matmul(n):
    A = np.random.rand(n, n)
    B = np.random.rand(n, n)

    start = time.time()
    C = np.matmul(A, B)
    latency = time.time() - start
    return latency

def handle(event, context):
    print(f"!!!!!!!!!!!!! Worker function invoked !!!!!!!!!!!!!")
    global redisClient

    if redisClient == None:
        redisClient = initRedis()

    task_queue_key = os.getenv('TASK_QUEUE_NAME', 'task_queue')
    results_queue_key = os.getenv('RESULTS_QUEUE_NAME', 'results_queue')

    while True:
        task = None
        try:
            # BLPOP to block until a task is available from the task queue (blocking call)
            queue_name, task_json = redisClient.blpop(task_queue_key, timeout=0)

            if not task_json:
                time.sleep(0.1)
                continue

            task = json.loads(task_json)

            task_id = int(task['id'])
            task_size = task['size']

            print(f"Worker processing task: {task_id} : {task_size}")

            result_latency = matmul(task_size)
            completion_time = time.time()

            print(f"Task {task_id} completed with result (latency): {result_latency:.4f}s")

            result_data = {
                "task_id": task_id,
                "original_task_size": task_size,
                "latency_seconds": result_latency,
                "completion_timestamp": completion_time,
                "worker_pod": os.getenv("HOSTNAME", "unknown")
            }

            # --- Push the result to the results queue (blocking call) ---
            redisClient.lpush(results_queue_key, json.dumps(result_data))
            print(f"Pushed result for task {task_id} to '{results_queue_key}'.")

        except json.JSONDecodeError as e:
            print(f"Error decoding task JSON from queue: {task_json} - {str(e)}")
        except redis.exceptions.ConnectionError as e:
            print(f"Redis connection error: {str(e)}. Attempting to reinitialize.")
            redisClient = initRedis() # Reinitialize blocking client
        except Exception as e:
            print(f"An unhandled error occurred in worker for task {task.get('id', 'N/A')}: {str(e)}")
            time.sleep(1)

    return {
        "statusCode": 200, 
        "body": "Worker function is continuously processing tasks."
    }
