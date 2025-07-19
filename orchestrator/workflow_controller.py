import subprocess
import argparse
import sys
import time
import redis
import json
from threading import Thread
from kubernetes import client, config
from utilities import (
    get_config,
    init_redis_client,
    init_pipeline,
    init_farm,
    get_redis_client_with_retry,
    get_current_worker_replicas,
    invoke_function_async,
    initialize_environment,
    run_script,
    monitor_worker_replicas,
    async_function,
    send_control_messages
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
    total_compute_time = 0.0
    total_comm_time = 0.0
    total_emit_time = 0.0
    total_collect_time = 0.0

    for index, raw in enumerate(results):
        try:
            result = json.loads(raw)
            task_id = result.get("task_id")
            emit_time = result.get("task_emit_time")
            collect_time = result.get("task_collect_time")
            work_time = result.get("task_work_time")

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
    print(f"  Total Compute Time    : {total_compute_time:.4f} sec")
    print(f"  Total Emit Time       : {total_emit_time:.4f} sec")
    print(f"  Total Collect Time    : {total_collect_time:.4f} sec")
    print(f"  Total Communication   : {total_comm_time:.4f} sec")
    print(f"  QoS Exceed Count    : {qos_exceed_count} ({qos_percentage:.2f}%)")
    print(f"  Avg Compute Time    : {avg_compute_time:.4f} sec")
    print(f"  Avg Emit Time       : {avg_emit_time:.4f} sec")
    print(f"  Avg Collect Time    : {avg_collect_time:.4f} sec")
    print(f"  Avg Communication   : {avg_comm_time:.4f} sec")


def monitor_queues(interval=3, total_tasks=1000, program_start_time=None, task_generation_start_time=None, feedback_enabled=False, configuration=None, redis_client=None):
    """
    Monitors Redis queues and provides periodic status updates on task progress.

    Args:
        interval (int): The interval in seconds to wait between checks.
        total_tasks (int): The total number of tasks to be completed.
        program_start_time (float): The start time of program execution.
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
    monitoring_start_time = time.time()

    while True:
        now = time.time()
        print(f"---------------{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now))}---------------")
        print("\n[QUEUE STATUS]")
        for queue in QUEUE_NAMES:
            length = redis_client.llen(queue)
            print(f"  {queue}: {length} items")

        replicas = monitor_worker_replicas(apps_v1_api=apps_v1_api)
        analyze_output_queue(redis_client, total_tasks)
        results = redis_client.lrange("output_queue", 0, -1)
        total_results = len(results)

        if total_results == total_tasks and program_start_time and end_time is None:

            end_time = time.time()
            for index, raw in enumerate(results):
                result = json.loads(raw)
                print(f"[INFO] Result {index} (task_id: {result.get("task_id")}):")
                print(f"  Emit Time    : {result.get("task_emit_time")}")
                print(f"  Collect Time : {result.get("task_collect_time")}")
                print(f"  Compute Time : {result.get("task_work_time")}")
                print(f"[DEBUG] result: {result}")

            print(f"\n[TIMER] All {total_tasks} tasks completed.")
            print(f"[TIMER] End time recorded at {time.ctime(end_time)}")
            print(f"[TIMER] Total processing time start from monitoring: {end_time - monitoring_start_time:.2f} seconds")
            print(f"[TIMER] Total processing time start from task generation: {end_time - task_generation_start_time:.2f} seconds")
            print(f"[TIMER] Total processing time start from program start: {end_time - program_start_time:.2f} seconds")
            # Send terminate control messages
            message = {
                "type": "TERMINATE",
                "action": "SYN",
                "message": "Terminate function from orchestrator",
                "SYN_timestamp": time.strftime('%Y-%m-%d %H:%M:%S %Z', time.localtime(end_time))
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
    program_start_time = time.time()
    print(f"[TIMER] Program started at {time.ctime(program_start_time)}")

    # Step 1: Initialize environment
    configuration, redis_client = initialize_environment()
    time.sleep(20) # Give time for env to settle
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
        "collector_feedback_flag": feedback_flag
    }

    # Step 2: Start workers and collector based on mode
    if args.mode == "pipeline":
        init_pipeline(redis_client, configuration, payload)
    elif args.mode == "farm":
        init_farm(redis_client, configuration, int(args.workers), payload)
    # time.sleep(10) # Give time for env to settle

    # Step 3: Start task generator asynchronously
    now = time.time()
    print(f"[INFO] Environment initialization time: {now - program_start_time:.2f} seconds")
    print(f"[INFO] Starting task generator with {args.tasks} tasks and {args.cycles} cycles with feedback={feedback_flag} at {time.ctime(now)}")
    run_script("task_generator.py", [str(args.tasks), str(args.cycles), str(feedback_flag)])

    # Step 4: Begin monitoring queues
    print(f"[INFO] Monitoring queues for {total_tasks} tasks across {int(args.cycles)} cycles.")
    # async_function((monitor_queues), interval=3, total_tasks=total_tasks, program_start_time=program_start_time, feedback_enabled=feedback_flag, configuration=configuration, redis_client=redis_client)
    monitor_queues(interval=3, total_tasks=total_tasks, program_start_time=program_start_time, task_generation_start_time=now, feedback_enabled=feedback_flag, configuration=configuration, redis_client=redis_client)

    print("[INFO] Queue monitoring completed.")
    print(f"[TIMER] Total program execution time: {time.time() - program_start_time:.2f} seconds")
    print("[INFO] Workflow Controller execution completed.")
    print("[INFO] Exiting program.")

if __name__ == "__main__":
    main()
