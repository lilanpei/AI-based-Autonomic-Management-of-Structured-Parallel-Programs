import subprocess
import argparse
import sys
import time
import redis
import json
from utilities import (
    get_config,
    init_redis_client,
    init_pipeline,
    init_farm,
    get_current_worker_replicas,
    invoke_function_async
)


def run_script(script_name, args=[]):
    """
    Asynchronously runs the specified script with the provided arguments.

    Args:
        script_name (str): The script to run.
        args (list): List of arguments to pass to the script.
    """
    cmd = ["python", script_name] + [str(arg) for arg in args]
    print(f"[ASYNC RUNNING] {' '.join(cmd)}")
    subprocess.Popen(cmd)


def init_redis_client_with_retry():
    """
    Initializes the Redis client with retry logic.

    Returns:
        redis.Redis: A Redis client instance.

    Raises:
        Exception: If Redis cannot be connected after retries.
    """
    try:
        return init_redis_client()
    except redis.exceptions.ConnectionError as e:
        print(f"[ERROR] Redis connection error: {str(e)}. Retrying...")
        time.sleep(5)
        return init_redis_client()


def monitor_worker_replicas():
    """
    Monitors the current number of worker replicas.
    """
    try:
        replicas = get_current_worker_replicas()
        print(f"\n[WORKER STATUS]")
        print(f"  Current worker replicas: {replicas}")
    except Exception as e:
        print(f"[WARNING] Could not retrieve worker replicas: {e}")
    return replicas


def analyze_output_queue(redis_client, total_tasks):
    """
    Analyzes the output queue in Redis, calculating statistics such as task completion times and QoS performance.

    Args:
        redis_client (redis.Redis): The Redis client.
        total_tasks (int): The total number of tasks expected.

    Returns:
        total_results (int): The total number of completed tasks in the output queue.
    """
    results = redis_client.lrange("output_queue", 0, -1)
    total_results = len(results)

    print(f"\n[PROGRESS] {total_results}/{total_tasks} tasks completed ({(total_results / total_tasks):.2%})")

    if total_results == 0:
        print("\n[OUTPUT QUEUE ANALYSIS] No results to analyze.")
        return total_results

    qos_exceed_count = 0
    total_compute_time = 0.0
    total_comm_time = 0.0
    total_emit_time = 0.0
    total_collect_time = 0.0

    for index, raw in enumerate(results):
        try:
            result = json.loads(raw)
            task_id = result.get("task_id")
            emit_time = result.get("task_emit_time", 0.0)
            collect_time = result.get("task_collect_time", 0.0)
            work_time = result.get("task_work_time", 0.0)

            comm_time = emit_time + collect_time

            # print(f"[INFO] Result {index} (task_id: {task_id}):")
            # print(f"  Emit Time    : {emit_time}")
            # print(f"  Collect Time : {collect_time}")
            # print(f"  Compute Time : {work_time}")
            # print(f"[DEBUG] result: {result}")
            if not result.get("task_QoS", False):
                qos_exceed_count += 1
            total_compute_time += work_time
            total_comm_time += comm_time
            total_emit_time += emit_time
            total_collect_time += collect_time
        except Exception as e:
            print(f"[WARNING] Failed to parse result: {e}")

    avg_compute_time = total_compute_time / total_results
    avg_comm_time = total_comm_time / total_results
    avg_emit_time = total_emit_time / total_results
    avg_collect_time = total_collect_time / total_results
    qos_percentage = (qos_exceed_count / total_results) * 100

    print("\n[OUTPUT QUEUE ANALYSIS]")
    print(f"  Total Results       : {total_results}")
    print(f"  QoS Exceed Count    : {qos_exceed_count} ({qos_percentage:.2f}%)")
    print(f"  Avg Compute Time    : {avg_compute_time:.4f} sec")
    print(f"  Avg Emit Time       : {avg_emit_time:.4f} sec")
    print(f"  Avg Collect Time    : {avg_collect_time:.4f} sec")
    print(f"  Avg Communication   : {avg_comm_time:.4f} sec")

    return total_results


def monitor_queues(interval=5, total_tasks=1000, start_time=None, feedback_enabled=False):
    """
    Monitors Redis queues and provides periodic status updates on task progress.

    Args:
        interval (int): The interval in seconds to wait between checks.
        total_tasks (int): The total number of tasks to be completed.
        start_time (float): The start time of task execution.
        feedback_enabled (bool): Flag indicating whether feedback is enabled.
    """
    config = get_config()
    QUEUE_NAMES = [
        config["input_queue_name"],
        config["worker_queue_name"],
        config["result_queue_name"],
        config["output_queue_name"],
        config["control_syn_queue_name"],
        config["control_ack_queue_name"]
    ]
    payload = {
        "input_queue_name": config["input_queue_name"],
        "worker_queue_name": config["worker_queue_name"],
        "result_queue_name": config["result_queue_name"],
        "output_queue_name": config["output_queue_name"],
        "control_syn_queue_name": config["control_syn_queue_name"],
        "control_ack_queue_name": config["control_ack_queue_name"],
        "collector_feedback_flag": False if not feedback_enabled else True,
    }

    redis_client = init_redis_client_with_retry()

    print("\n[INFO] Starting Redis queue monitoring...")

    end_time = None
    warm_up_enabled = False #True
    warm_up_interval = 5  # seconds
    last_warm_up_time = time.time()

    while True:
        now = time.time()
        print(f"---------------{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now))}---------------")
        print("\n[QUEUE STATUS]")
        for queue in QUEUE_NAMES:
            length = redis_client.llen(queue)
            print(f"  {queue}: {length} items")

        replicas = monitor_worker_replicas()

        # Periodic warm-up every warm_up_interval seconds
        # Warm-up conditionally based on queue contents
        if warm_up_enabled and now - last_warm_up_time >= warm_up_interval:
            try:
                input_queue_lenght = redis_client.llen(config["input_queue_name"])
                worker_queue_length = redis_client.llen(config["worker_queue_name"])
                result_queue_length = redis_client.llen(config["result_queue_name"])

                if input_queue_lenght > 0:
                    print("[WARM-UP] Invoking emitter (input queue not empty)...")
                    invoke_function_async("emitter", payload)

                if worker_queue_length > 0:
                    print("[WARM-UP] Invoking worker(s) (worker queue not empty)...")
                    for _ in range(replicas):
                        invoke_function_async("worker", payload)

                if result_queue_length > 0:
                    print("[WARM-UP] Invoking collector (result queue not empty)...")
                    invoke_function_async("collector", payload)
            except Exception as e:
                print(f"[WARM-UP] Error: {e}")
            last_warm_up_time = now
            warm_up_enabled = False  # Disable warm-up after first run

        total_results = analyze_output_queue(redis_client, total_tasks)

        if total_results == total_tasks and start_time and end_time is None:
            end_time = time.time()
            duration = end_time - start_time
            print(f"\n[TIMER] All {total_tasks} tasks completed.")
            print(f"[TIMER] End time recorded at {time.ctime(end_time)}")
            print(f"[TIMER] Total processing time: {duration:.2f} seconds")

            if not feedback_enabled:
                print("[INFO] Feedback disabled. Terminating program.")
                sys.exit(0)

        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description="Workflow Controller for AI Task Processing")
    parser.add_argument('--mode', choices=['pipeline', 'farm'], required=True, help='Select execution mode')
    parser.add_argument('--feedback', action=argparse.BooleanOptionalAction, help='Enable collector feedback')
    parser.add_argument('--tasks', type=int, default=100, help='Number of tasks to generate')
    parser.add_argument('--cycles', type=int, default=2, help='Number of cycles to generate tasks')
    parser.add_argument('--workers', type=int, default=1, help='Number of workers (used only in farm mode)')

    args = parser.parse_args()

    # Record start time
    program_start_time = time.time()
    print(f"[TIMER] Program started at {time.ctime(program_start_time)}")

    # Step 1: Initialize environment
    run_script("env_init.py")

    # Step 2: Give time for env to settle
    time.sleep(10)

    # Step 3: Start task generator asynchronously
    run_script("task_generator.py", [str(args.tasks), str(args.cycles), "True" if args.feedback else "False"])

    # Step 4: Start workers and collector based on mode
    if args.mode == "pipeline":
        init_pipeline("True" if args.feedback else "False")
    elif args.mode == "farm":
        init_farm(int(args.workers), "True" if args.feedback else "False")

    # Step 5: Begin monitoring queues
    total_tasks = int(args.tasks) * int(args.cycles)  # Assuming args.cycles cycles of tasks
    print(f"[INFO] Monitoring queues for {total_tasks} tasks across {int(args.cycles)} cycles.")
    monitor_queues(interval=3, total_tasks=total_tasks, start_time=program_start_time, feedback_enabled=args.feedback)


if __name__ == "__main__":
    main()
