import os
import json
import redis
import time

redisClient = None

def initRedis():
    redisHostname = os.getenv('redis_hostname', default='redis-master.redis.svc.cluster.local')
    redisPort = os.getenv('redis_port')

    return redis.Redis(
        host=redisHostname,
        port=redisPort,
        decode_responses=True
    )

def handle(event, context):
    print(f"!!!!!!!!!!!!! Collector function invoked !!!!!!!!!!!!!")
    global redisClient

    if redisClient == None:
        redisClient = initRedis()

    results_queue_key = os.getenv('RESULTS_QUEUE_NAME', 'results_queue')
    completed_results_set_key = os.getenv('COMPLETED_RESULTS_SET_NAME', 'completed_tasks_results')

    while True:
        result_data_json = None
        try:
            # BLPOP to block until a result is available from the results queue (blocking call)
            queue_name, result_data_json = redisClient.blpop(results_queue_key, timeout=0)

            if not result_data_json:
                time.sleep(0.1)
                continue

            result_data = json.loads(result_data_json)

            task_id = int(result_data.get('task_id'))
            if not task_id:
                raise ValueError("Task ID (timestamp) missing or invalid in result_data.")

            print(f"Collector received result for task: {task_id}")

            result_data["collection_timestamp"] = time.time()

            # Store result in a Redis Sorted Set (blocking call)
            redisClient.zadd(completed_results_set_key, {json.dumps(result_data): task_id})

            print(f"Stored result for task {task_id} in '{completed_results_set_key}' Sorted Set.")

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON payload from queue: {result_data_json} - {str(e)}")
        except ValueError as e:
            print(f"Validation error for received result: {str(e)}")
        except redis.exceptions.ConnectionError as e:
            print(f"Redis connection error: {str(e)}. Attempting to reinitialize.")
            redisClient = initRedis()
        except Exception as e:
            print(f"An unexpected error occurred in collector: {str(e)}")
            time.sleep(1)

    return {
        "statusCode": 200,
        "body": "Collector function is continuously processing results."
    }
