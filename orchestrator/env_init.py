import sys
import time
import redis
from utilities import get_config, init_redis_client, clear_queues, scale_worker_deployment, restart_function
if __name__ == "__main__":
    config = get_config()

    try:
        redis_client = init_redis_client()
    except redis.exceptions.ConnectionError as e:
        print(f"Redis connection error: {str(e)}. Attempting to reinitialize.")
        time.sleep(5)
        try:
            redis_client = init_redis_client() # Reinitialize blocking client
        except Exception as init_e:
            print(f"CRITICAL ERROR: Redis reinit failed: {init_e}", file=sys.stderr)
            raise

    clear_queues(redis_client, None)  # Clear all queues before generating tasks

    # Restart all functions
    for fn in ["emitter", "worker", "collector"]:
        restart_function(fn)
    time.sleep(5)  # Wait for the deployments to stabilize

    # Scale down to 1 replica before generating tasks
    scale_worker_deployment(1, "emitter", "openfaas-fn") # Ensure the emitter deployment is scaled down to 1 replica
    scale_worker_deployment(1, "worker", "openfaas-fn")  # Ensure the worker deployment is scaled down to 1 replica
    scale_worker_deployment(1, "collector", "openfaas-fn")  # Ensure the worker deployment is scaled down to 1 replica
    time.sleep(5)  # Wait for the deployments to stabilize
