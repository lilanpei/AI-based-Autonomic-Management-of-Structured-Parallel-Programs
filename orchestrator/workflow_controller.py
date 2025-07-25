import os
import argparse
import sys
import time
import json
from threading import Thread
from kubernetes import client, config
from utilities import (
    get_config,
    init_pipeline,
    init_farm,
    get_redis_client_with_retry,
    initialize_environment,
    run_script,
    monitor_worker_replicas,
    async_function,
    send_control_messages,
    get_utc_now
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

    print(f"\n[PROGRESS] {total_results}/{total_tasks} tasks completed ({(total_results / total_tasks):.2%})")

    if total_results == 0:
        print("\n[OUTPUT QUEUE ANALYSIS] No results to analyze.")
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

    print("\n[OUTPUT QUEUE ANALYSIS]")
    print(f"  Total Results         : {total_results}")
    print(f"  Total Emit Time       : {total_emit_time:.4f} sec")
    print(f"  Total Work Time       : {total_work_time:.4f} sec")
    print(f"  Total Collect Time    : {total_collect_time:.4f} sec")
    print(f"  QoS Exceed Count      : {qos_exceed_count} ({qos_percentage:.2f}%)")
    print(f"  Avg Emit Time         : {avg_emit_time:.4f} sec")
    print(f"  Avg Work Time         : {avg_work_time:.4f} sec")
    print(f"  Avg Collect Time      : {avg_collect_time:.4f} sec")


def monitor_queues(program_start_time, interval=3, total_tasks=1000, feedback_enabled=False, configuration=None, redis_client=None):
    """
    Monitors Redis queues and provides periodic status updates on task progress.

    Args:
        program_start_time (float): The start time of program execution.
        interval (int): The interval in seconds to wait between checks.
        total_tasks (int): The total number of tasks to be completed.
        feedback_enabled (bool): Flag indicating whether feedback is enabled.
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
        print("\n[QUEUE STATUS]")
        for queue in QUEUE_NAMES:
            length = redis_client.llen(queue)
            print(f"  {queue}: {length} items")

        replicas = monitor_worker_replicas(apps_v1_api=apps_v1_api)
        analyze_output_queue(redis_client, total_tasks)
        results = redis_client.lrange("output_queue", 0, -1)
        total_results = len(results)

        if total_results == total_tasks and program_start_time and end_time is None:

            end_time = (get_utc_now() - program_start_time).total_seconds()
            print(f"\n[TIMER] Program/Monitoring completed at [{end_time:.4f}] seconds.")
            for index, raw in enumerate(results):
                result = json.loads(raw)
                print(f"[INFO] Result {index} (task_id: {result.get("task_id")}):")
                print(f"[INFO]  Emit Time    : {result.get("task_emit_time"):.4f} sec")
                print(f"[INFO]  Collect Time : {result.get("task_collect_time"):.4f} sec")
                print(f"[INFO]  Compute Time : {result.get("task_work_time"):.4f} sec")
                print(f"[INFO] result: {result}")

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
    total_tasks = int(args.tasks) * int(args.cycles)  # Assuming args.cycles cycles of tasks
    feedback_flag = True if args.feedback else False
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
        "deadline_coeff": configuration.get("deadline_coeff"),
        "deadline_cap": configuration.get("deadline_cap"),
        "deadline_floor": configuration.get("deadline_floor"),
        "program_start_time": str(program_start_time),
        "collector_feedback_flag": feedback_flag
    }

    # Step 2: Workflow Controller Initialization
    if args.mode == "pipeline":
        redis_client = init_pipeline(program_start_time, configuration, payload)
    elif args.mode == "farm":
        redis_client = init_farm(program_start_time, configuration, int(args.workers), payload)

    # Step 3: Begin monitoring queues
    print(f"[INFO] Monitoring queues for {total_tasks} tasks across {int(args.cycles)} cycles.")
    async_function((monitor_queues), program_start_time, interval=3, total_tasks=total_tasks, feedback_enabled=feedback_flag, configuration=configuration, redis_client=redis_client)

    # Step 3: Start task generator script
    print(f"[INFO] Starting task generator with {args.tasks} tasks and {args.cycles} cycles with feedback={feedback_flag} at {(get_utc_now() - program_start_time).total_seconds():.4f} seconds")
    run_script("task_generator.py", [str(args.tasks), str(args.cycles), str(feedback_flag)])

if __name__ == "__main__":
    main()
