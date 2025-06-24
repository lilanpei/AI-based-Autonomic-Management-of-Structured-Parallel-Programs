import os
import json
import random
import time
import redis

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
    print(f"!!!!!!!!!!!!! Task-generator function invoked !!!!!!!!!!!!!")
    global redisClient

    if redisClient == None:
        redisClient = initRedis()

    # Add task to queue
    # Generate a timestamp-based ID
    task_id = int(time.time() * 1000000) # Current time in microseconds as an integer
    task_size = random.randint(10, 100)
    task = {"id": task_id, "size": task_size, "creation_time": time.time()} # Added creation_time for explicit tracking
    redisClient.lpush("task_queue", json.dumps(task))
    print(f"Received task: {task}")

    return {
        "statusCode": 200,
        "body": f"Added task {task['id']} : {task['size']} to queue"
    }
