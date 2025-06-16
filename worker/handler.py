import os
import json
import time
import uuid
import redis
import requests
import random
import numpy as np
from time import time, sleep

redisClient = None

def initRedis():
    redisHostname = os.getenv('redis_hostname', default='redis-master.redis.svc.cluster.local')
    redisPort = os.getenv('redis_port')

    with open('/var/openfaas/secrets/redis-password', 'r') as s:
        redisPassword = s.read()

    return redis.Redis(
        host=redisHostname,
        port=redisPort,
        password=redisPassword,
    )

# basic numpy matrix multiplication
def matmul(n):
    A = np.random.rand(n, n)
    B = np.random.rand(n, n)

    start = time()
    C = np.matmul(A, B)
    latency = time() - start
    return latency

# openfaas event handler function
def handle(event, context):
    print(f"!!!!!!!!!!!!! Event: {event} !!!!!!!!!!!!!")
    global redisClient

    if redisClient == None:
        redisClient = initRedis()

    while True:
        # Process tasks until queue is empty
        task_json = redisClient.rpop("task_queue")  # Non-blocking pop
    
        if not task_json:
            # If queue is empty, sleep for a while before checking again
            print("Queue is empty, waiting for tasks...")
            # sleep(1)  # Sleep for 1 second before checking again
            # requests.post('http://127.0.0.1:8080/function/woker')
            break  # Exit when queue is empty
            
        try:
            task = json.loads(task_json)
            # Process task here
            print(f"Processing task: {task['id']} : {task['size']}")
            size = task['size']
            # input = [10, 100, 1000]
            # n = random.randint(0, 2)
            # result = matmul(input[n])
            result = matmul(size)
            print(f"Task {task['id']} completed with result: {result}")
        
        # If there are remaining tasks, send a request to the matmul function itself to process the next task
        # Check remaining tasks in the queue
        # remainingWork = redisClient.llen("task_queue")
        # print(f"Remaining tasks in queue: {remainingWork}")
        # if remainingWork > 0:
        #     requests.post('http://127.0.0.1:8080/function/woker')

        except Exception as e:
            print(f"Error processing task: {str(e)}")

    # If we reach here, it means the break condition was met
    # After processing all tasks, check if the queue is empty
    print("All tasks processed. Queue is empty.")
    # Return a response indicating completion
    return {
        "status": "Completed",
        "message": "All tasks processed. Queue is empty."
    }