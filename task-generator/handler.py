import os
import json
import random
import time
import uuid
import redis
import requests

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

def handle(event, context):
    global redisClient

    if redisClient == None:
        redisClient = initRedis()
        
    # Add task to queue
    TaskSize = random.randint(10, 100)
    task = {"id": str(uuid.uuid4()), "size": TaskSize}
    redisClient.lpush("task_queue", json.dumps(task))
    print(f"Received task: {task}")
    # requests.post('http://127.0.0.1:8080/function/woker')

    return {
        "statusCode": 200,
        "body": f"Added task {task['id']} : {task['size']} to queue"
    }
