import sys
import time
import json
import uuid
import redis
import random
import numpy as np
from threading import Thread
from utilities import get_config, init_redis_client, generate_matrix

def generate_task_payload(matrix_size):
    """
    Construct a task payload for matrix multiplication.

    Args:
        matrix_size (int): The size of the matrix (NxN).

    Returns:
        dict: Task payload dictionary.
    """
    return {
        "task_id": str(uuid.uuid4()),
        "task_gen_timestamp": time.time(),
        "task_application": "matrix_multiplication",
        "task_data": None,  # Actual data will be generated by workers
        "task_data_size": matrix_size,
        "task_deadline": deadline_for_matrix(matrix_size)  # seconds
    }

def deadline_for_matrix(n: int, coeff: float = 2.5e-7, cap: int = 30, floor: int = 1) -> int:
    """
    Analytic deadline (seconds) for an nxn matrix multiplication.

    deadline = min( max(floor, coeff · n³), cap )   # seconds

    • n       - size of the matrix (n x n).
    • coeff   - tunes how many seconds per floating-point operation.
    • cap     - hard upper bound so the system never waits 'forever'.
    • floor   - hard lower bound so the system never waits less than 1 second.
    """
    return int(min(max(floor, coeff * n**3), cap))

def push_task_to_queue(redis_client, queue_name, task_json, task_index):
    """
    Push a task to Redis queue with error handling and retry logic.

    Args:
        redis_client (Redis): Redis client instance.
        queue_name (str): Redis queue name.
        task_json (str): Task payload in JSON string format.
        task_index (int): Index of the task (for logging).

    Returns:
        bool: True if push succeeded, False otherwise.
    """
    try:
        start_time = time.time()
        redis_client.lpush(queue_name, task_json)
        duration = time.time() - start_time
        print(f"[INFO] Task {task_index} pushed to '{queue_name}' in {duration:.4f}s.")
        return True
    except redis.exceptions.ConnectionError as e:
        print(f"[ERROR] Redis connection failed: {e}. Retrying in 5s...")
        time.sleep(5)
        try:
            new_client = init_redis_client()
            new_client.lpush(queue_name, task_json)
            print(f"[INFO] Retry succeeded for task {task_index}.")
            return True
        except Exception as retry_e:
            print(f"[CRITICAL] Retry failed for task {task_index}: {retry_e}", file=sys.stderr)
    except Exception as e:
        print(f"[ERROR] Failed to push task {task_index}: {e}", file=sys.stderr)

    return False

def generate_and_push_tasks(num_tasks, redis_client, input_queue, payload):
    """
    Generate and push matrix multiplication tasks to Redis queue.

    Args:
        num_tasks (int): Number of tasks to generate.
        redis_client (Redis): Redis client instance.
        input_queue (str): Name of the Redis queue.
    """
    print(f"[INFO] Starting task generation: {num_tasks} tasks to '{input_queue}'")

    for i in range(1, num_tasks + 1):
        matrix_size = random.randint(100, 1000)
        task_payload = generate_task_payload(matrix_size)
        task_json = json.dumps(task_payload)
        json_size = len(task_json.encode('utf-8'))

        success = push_task_to_queue(redis_client, input_queue, task_json, i)

        if success:
            print(f"[INFO] Task {i}/{num_tasks}: size={json_size} bytes, matrix={matrix_size}x{matrix_size}")
            if i % 100 == 0 or i == num_tasks:
                print(f"[INFO] Batch update: {i} tasks pushed.")
        else:
            print(f"[ERROR] Skipped task {i} due to Redis failure.")

    print(f"[INFO] Task generation complete: {num_tasks} tasks pushed to '{input_queue}'.")

def task_producer(requests_in_window, redis_client, input_queue, payload, window_duration=30):
    """
    Produces tasks over a time window with Poisson-distributed inter-arrival times.

    Args:
        requests_in_window (int): Number of tasks in this time window.
        redis_client (redis.Redis): Redis client instance.
        input_queue (str): Redis input queue name.
        payload (dict): Payload for invoking functions.
        window_duration (int): Duration of the window in seconds.
    """
    if requests_in_window <= 0:
        print("[INFO] No tasks to generate in this interval.")
        time.sleep(window_duration)
        return

    avg_interval = window_duration / requests_in_window
    inter_arrival_times = np.random.poisson(lam=avg_interval, size=requests_in_window)

    total_elapsed = 0
    for i, delay in enumerate(inter_arrival_times):
        before = time.time()
        Thread(target=generate_and_push_tasks, args=(1, redis_client, input_queue, payload)).start()
        print(f"[DEBUG] Scheduled task {i + 1}/{requests_in_window}, sleeping {delay}s...")
        after = time.time()
        adjusted_delay = max(0, delay - (after - before))  # Ensure non-negative sleep time
        if adjusted_delay > 0:
            time.sleep(adjusted_delay)
        # total_elapsed += delay

    # remaining_time = window_duration - total_elapsed
    # if remaining_time > 0:
    #     print(f"[DEBUG] Sleeping additional {remaining_time}s to finish the interval.")
    #     time.sleep(remaining_time)

def main():
    task_genration_start_time = time.time()
    if len(sys.argv) != 4:
        print("Usage: python task_generator.py <number_of_tasks_to_generate> <num_cycles> <feedback_enabled>")
        print("Example: python task_generator.py 100 10 False")
        sys.exit(1)

    try:
        num_tasks = int(sys.argv[1])
        num_cycles = int(sys.argv[2])
        feedback_flag = sys.argv[3].lower() == "true"
        if num_cycles <= 0:
            raise ValueError("Number of cycles must be a positive integer.")
        if num_tasks <= 0:
            raise ValueError("Number of tasks must be a positive integer.")
    except ValueError as ve:
        print(f"[ERROR] Invalid task number: {ve}", file=sys.stderr)
        sys.exit(1)
    print(f"[INFO] Generating {num_tasks} tasks across {num_cycles} cycles with feedback={feedback_flag}")
    config = get_config()
    payload = {
        "input_queue_name": config["input_queue_name"],
        "worker_queue_name": config["worker_queue_name"],
        "result_queue_name": config["result_queue_name"],
        "output_queue_name": config["output_queue_name"],
        "emitter_control_syn_queue_name": config["emitter_control_syn_queue_name"],
        "worker_control_syn_queue_name": config["worker_control_syn_queue_name"],
        "worker_control_ack_queue_name": config["worker_control_ack_queue_name"],
        "collector_control_syn_queue_name": config["collector_control_syn_queue_name"],
        "emitter_start_queue_name": config["emitter_start_queue_name"],
        "worker_start_queue_name": config["worker_start_queue_name"],
        "collector_start_queue_name": config["collector_start_queue_name"],
        "collector_feedback_flag": feedback_flag,
    }

    try:
        redis_client = init_redis_client()
    except redis.exceptions.ConnectionError as e:
        print(f"[ERROR] Initial Redis connection failed: {e}. Retrying...")
        time.sleep(5)
        try:
            redis_client = init_redis_client()
        except Exception as init_e:
            print(f"[CRITICAL] Redis reinitialization failed: {init_e}", file=sys.stderr)
            sys.exit(1)

    np.random.seed(29)  # for reproducibility

    # simulate multiple runs (num_cycles cycles of 30 seconds each)
    print(f"[INFO] Starting task generation across {num_tasks} tasks over {num_cycles} cycles of 30 seconds each.")
    for cycle in range(num_cycles):
        print(f"\n[CYCLE {cycle+1}] Generating {num_tasks} tasks over 30s...")
        task_producer(num_tasks, redis_client, config['input_queue_name'], payload, window_duration=30)

    task_generation_end_time = time.time()
    print(f"[INFO] Task generation completed across all cycles in {task_generation_end_time - task_genration_start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
