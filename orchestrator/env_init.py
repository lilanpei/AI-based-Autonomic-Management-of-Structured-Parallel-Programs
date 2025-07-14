import sys
import time
import redis
from utilities import (
    get_config, init_redis_client,
    clear_queues,
    scale_function_deployment,
    restart_function
)


def init_redis_connection():
    """
    Initializes the Redis connection with retry logic.

    Returns:
        redis.Redis: A Redis client instance.

    Raises:
        Exception: If Redis cannot be connected after retries.
    """
    try:
        return init_redis_client()
    except redis.exceptions.ConnectionError as e:
        print(f"[ERROR] Redis connection error: {str(e)}. Attempting to reinitialize...")
        time.sleep(5)
        try:
            return init_redis_client()  # Attempt to reinitialize the Redis client
        except Exception as init_e:
            print(f"[CRITICAL ERROR] Redis reinit failed: {init_e}", file=sys.stderr)
            raise


def restart_all_functions():
    """
    Restarts all the functions (emitter, worker, collector).
    """
    for fn in ["emitter", "worker", "collector"]:
        restart_function(fn)
    print("[INFO] Waiting for function deployments to stabilize...")
    time.sleep(5)  # Allow time for function deployments to stabilize


def scale_deployments_to_single_replica():
    """
    Scales down the emitter, worker, and collector functions to 1 replica each.
    """
    for fn in ["emitter", "worker", "collector"]:
        scale_function_deployment(1, fn, "openfaas-fn")
        print(f"[INFO] Scaled down {fn} function deployment to 1 replica.")


def clear_all_queues(redis_client):
    """
    Clears all Redis queues before generating tasks.

    Args:
        redis_client (redis.Redis): Redis client instance.
    """
    clear_queues(redis_client, None)
    print("[INFO] Cleared all Redis queues.")


def main():
    # Load configuration
    config = get_config()

    # Initialize Redis connection
    redis_client = init_redis_connection()

    # Restart functions
    restart_all_functions()

    # Scale down deployments to 1 replica
    scale_deployments_to_single_replica()

    # Clear Redis queues
    clear_all_queues(redis_client)

    # Wait for deployments to stabilize before task generation
    time.sleep(5)
    print("[INFO] Environment initialization complete.")


if __name__ == "__main__":
    main()
