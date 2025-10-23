import os
import argparse
import sys
import time
import json
from threading import Thread
from kubernetes import client, config

# Add the project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from utilities.utilities import (
    get_config,
    init_farm,
    get_redis_client_with_retry,
    initialize_environment,
    run_script,
    monitor_deployment_replicas,
    async_function,
    send_control_messages,
    get_utc_now,
    clear_queues
)


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

    print(f"\n[INFO] {total_results}/{total_tasks} tasks completed ({(total_results / total_tasks):.2%})")

    if total_results == 0:
        print("\n[INFO] [OUTPUT QUEUE ANALYSIS] No results to analyze.")
        return total_results

    qos_exceed_count = 0
    total_work_time = 0.0
    total_emit_time = 0.0
    total_collect_time = 0.0

    for _, raw in enumerate(results):
        try:
            result = json.loads(raw)
            emit_time = result.get("task_emit_time")
            work_time = result.get("task_work_time")
            collect_time = result.get("task_collect_time")

            if not result.get("task_QoS", False):
                qos_exceed_count += 1
            total_emit_time += emit_time
            total_work_time += work_time
            total_collect_time += collect_time
        except Exception as e:
            print(f"[WARNING] Failed to parse result: {e}")

    avg_emit_time = total_emit_time / total_results
    avg_work_time = total_work_time / total_results
    avg_collect_time = total_collect_time / total_results
    qos_percentage = (qos_exceed_count / total_results) * 100

    print("\n[INFO][OUTPUT QUEUE ANALYSIS]")
    print(f"[INFO]  Total Results         : {total_results}")
    print(f"[INFO]  Total Emit Time       : {total_emit_time:.4f} sec")
    print(f"[INFO]  Total Work Time       : {total_work_time:.4f} sec")
    print(f"[INFO]  Total Collect Time    : {total_collect_time:.4f} sec")
    print(f"[INFO]  QoS Exceed Count      : {qos_exceed_count} ({qos_percentage:.2f}%)")
    print(f"[INFO]  Avg Emit Time         : {avg_emit_time:.4f} sec")
    print(f"[INFO]  Avg Work Time         : {avg_work_time:.4f} sec")
    print(f"[INFO]  Avg Collect Time      : {avg_collect_time:.4f} sec")


def monitor_queues(program_start_time, interval=3, total_tasks=1000, configuration=None, redis_client=None):
    """
    Monitors Redis queues and provides periodic status updates on task progress.

    Args:
        program_start_time (float): The start time of program execution.
        interval (int): The interval in seconds to wait between checks.
        total_tasks (int): The total number of tasks to be completed.
    """
    if not configuration:
        configuration = get_config()
    if not redis_client:
        redis_client = get_redis_client_with_retry()
    try:
        config.load_incluster_config()
    except:
        try:
            config.load_kube_config()
        except Exception as e:
            print(f"[ERROR] Kubernetes config load failed: {e}", file=sys.stderr)
            sys.exit(1)
    apps_v1_api = client.AppsV1Api()

    QUEUE_NAMES = [
            configuration.get("input_queue_name"),
            configuration.get("worker_queue_name"),
            configuration.get("result_queue_name"),
            configuration.get("output_queue_name"),
            configuration.get("emitter_control_syn_queue_name"),
            configuration.get("worker_control_syn_queue_name"),
            configuration.get("worker_control_ack_queue_name"),
            configuration.get("collector_control_syn_queue_name"),
            configuration.get("emitter_start_queue_name"),
            configuration.get("worker_start_queue_name"),
            configuration.get("collector_start_queue_name")
        ]

    print("\n[INFO] Starting Redis queue monitoring...")

    end_time = None
    monitoring_start_time = (get_utc_now() - program_start_time).total_seconds()
    print(f"[TIMER] Monitoring started at [{monitoring_start_time:.4f}] seconds.")

    while True:
        now = (get_utc_now() - program_start_time).total_seconds()
        print(f"[INFO]------------- Monitoring window at [{now:.4f}] seconds -------------")
        print("\n[INFO] [QUEUE STATUS]")
        for queue in QUEUE_NAMES:
            length = redis_client.llen(queue)
            print(f"[INFO]  {queue}: {length} items")

        replicas = monitor_deployment_replicas(apps_v1_api=apps_v1_api, namespace="openfaas-fn", name_or_prefix="worker-", exact_match=False)
        analyze_output_queue(redis_client, total_tasks)
        results = redis_client.lrange("output_queue", 0, -1)
        total_results = len(results)

        if total_results == total_tasks and program_start_time and end_time is None:

            end_time = (get_utc_now() - program_start_time).total_seconds()
            print(f"\n[TIMER] Program/Monitoring completed at [{end_time:.4f}] seconds.")

            # Reset workflow init flag
            clear_queues(redis_client, [configuration.get("workflow_init_syn_queue_name")])

            for index, raw in enumerate(results):
                result = json.loads(raw)
                print(f"[DEBUG] Result {index} (task_id: {result.get("task_id")}):")
                print(f"[DEBUG]  Emit Time    : {result.get("task_emit_time"):.4f} sec")
                print(f"[DEBUG]  Collect Time : {result.get("task_collect_time"):.4f} sec")
                print(f"[DEBUG]  Compute Time : {result.get("task_work_time"):.4f} sec")
                print(f"[DEBUG] result: {result}")

            print(f"\n[INFO] All {total_tasks} tasks completed at {end_time:.4f} seconds.")
            print(f"[TIMER] Total monitoring time: [{(end_time - monitoring_start_time):.4f}] seconds.")
            # Send terminate control messages
            message = {
                "type": "TERMINATE",
                "action": "SYN",
                "message": "Terminate function from orchestrator",
                "SYN_timestamp": end_time
            }

            send_control_messages(message, redis_client, configuration.get("emitter_control_syn_queue_name"), 1)
            send_control_messages(message, redis_client, configuration.get("worker_control_syn_queue_name"), replicas)
            send_control_messages(message, redis_client, configuration.get("collector_control_syn_queue_name"), 1)
            print("[INFO] Termination control messages sent to all components.")

            print("[INFO] Terminating program.")
            sys.exit(0)

        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description="Workflow Controller for AI Task Processing")
    parser.add_argument('--tasks', type=int, default=100, help='Number of tasks to generate')
    parser.add_argument('--workers', type=int, default=1, help='Number of workers (used only in farm mode)')

    args = parser.parse_args()
    total_tasks = int(args.tasks)

    # Record start time
    program_start_time = get_utc_now()
    os.environ["START_TIMESTAMP"] = str(program_start_time)  # Set env var
    print(f"[TIMER] Program started at [{(get_utc_now() - program_start_time).total_seconds():.4f}] seconds")

    # Step 1: Initialize environment
    configuration = initialize_environment(program_start_time)
    payload = {
        "input_queue_name": configuration.get("input_queue_name"),
        "worker_queue_name": configuration.get("worker_queue_name"),
        "result_queue_name": configuration.get("result_queue_name"),
        "output_queue_name": configuration.get("output_queue_name"),
        "emitter_control_syn_queue_name": configuration.get("emitter_control_syn_queue_name"),
        "worker_control_syn_queue_name": configuration.get("worker_control_syn_queue_name"),
        "worker_control_ack_queue_name": configuration.get("worker_control_ack_queue_name"),
        "collector_control_syn_queue_name": configuration.get("collector_control_syn_queue_name"),
        "emitter_start_queue_name": configuration.get("emitter_start_queue_name"),
        "worker_start_queue_name": configuration.get("worker_start_queue_name"),
        "collector_start_queue_name": configuration.get("collector_start_queue_name"),
        "processing_delay": configuration.get("processing_delay"),
        "wait_time": configuration.get("wait_time"),
        "program_start_time": str(program_start_time),
        "calibrated_model_a": configuration.get("calibrated_model")["a"],
        "calibrated_model_b": configuration.get("calibrated_model")["b"],
        "calibrated_model_seed": configuration.get("calibrated_model")["seed"],
        "calibrated_model_r_squared": configuration.get("calibrated_model")["r_squared"],
        "base_rate": configuration.get("base_rate"),  # tasks/min - high enough to trigger autoscaling
        "phase_duration": configuration.get("phase_duration"),  # seconds per phase
        "window_duration": configuration.get("window_duration"),  # seconds per window
    }

    # Step 2: Workflow Controller Initialization
    redis_client = init_farm(program_start_time, configuration, int(args.workers), payload)

    # Step 3: Begin monitoring queues
    print(f"[INFO] Monitoring queues for {total_tasks} tasks.")
    async_function((monitor_queues), program_start_time, interval=3, total_tasks=total_tasks, configuration=configuration, redis_client=redis_client)

    # Step 3: Task generation
    # NEW DESIGN: Two-stage generation
    # 1. Task generator pushes phase configs to input_queue
    # 2. Emitter reads phase configs and generates actual tasks to worker_queue
    print(f"[INFO] Starting task generator (pushes phase configs to input_queue)")
    print(f"[INFO] Emitter will read phase configs and generate tasks to worker_queue")

    # Task generator pushes phase metadata (base_rate=300, 4 phases, 60s each)
    # Image sizes shifted to larger (40% 2048 + 40% 4096) for ~2s avg processing time
    run_script("../image_processing/task_generator.py", [300, 60, 1])

if __name__ == "__main__":
    main()
