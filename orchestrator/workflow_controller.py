import subprocess
import argparse
import sys
import time
import redis
import json
from utilities import get_config, init_redis_client, get_current_worker_replicas


def run_script(script_name, args=[]):
    """
    Always runs the script asynchronously in the background.
    """
    cmd = ["python", script_name] + [str(arg) for arg in args]
    print(f"[ASYNC RUNNING] {' '.join(cmd)}")
    subprocess.Popen(cmd)


def monitor_queues(interval=5, total_tasks=1000, start_time=None, feedback_enabled=False):
    config = get_config()
    QUEUE_NAMES = [
        config["input_queue_name"],
        config["worker_queue_name"],
        config["result_queue_name"],
        config["output_queue_name"],
        config["control_syn_queue_name"],
        config["control_ack_queue_name"]
    ]

    end_time = None

    try:
        try:
            r = init_redis_client()
        except redis.exceptions.ConnectionError as e:
            print(f"Redis connection error: {str(e)}. Attempting to reinitialize.")
            time.sleep(5)
            r = init_redis_client()

        print("\n[INFO] Starting Redis queue monitoring...")
        while True:
            print(f"---------------{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}---------------")
            print("\n[QUEUE STATUS]")
            for queue in QUEUE_NAMES:
                length = r.llen(queue)
                print(f"  {queue}: {length} items")

            # Monitoring worker replicas
            try:
                replicas = get_current_worker_replicas()
                print(f"\n[WORKER STATUS]")
                print(f"  Current worker replicas: {replicas}")
            except Exception as e:
                print(f"[WARNING] Could not retrieve worker replicas: {e}")

            # Analyze output queue
            results = r.lrange("output_queue", 0, -1)
            total_results = len(results)

            # Progress monitoring
            progress_ratio = total_results / total_tasks
            print(f"\n[PROGRESS] {total_results}/{total_tasks} tasks completed ({progress_ratio:.2%})")

            # Output result analysis
            qos_exceed_count = 0
            total_compute_time = 0.0
            total_comm_time = 0.0
            total_emit_time = 0.0
            total_collect_time = 0.0
            index = 0

            for raw in results:
                try:
                    result = json.loads(raw)
                    complete_time = result.get("complete_time", 0.0)
                    emit_time = result.get("emit_time", 0.0)
                    collect_time = result.get("collect_time", 0.0)
                    task_id = result.get("task_id")

                    print(f"[INFO] Result {index} (task_id: {task_id}):")
                    print(f"  Emit Time    : {emit_time}")
                    print(f"  Collect Time : {collect_time}")
                    print(f"  Compute Time : {complete_time}")

                    comm_time = emit_time + collect_time

                    if not result.get("QoS", False):
                        qos_exceed_count += 1
                    total_compute_time += complete_time
                    total_comm_time += comm_time
                    total_emit_time += emit_time
                    total_collect_time += collect_time
                    index += 1
                except Exception as e:
                    print(f"[WARNING] Failed to parse result: {e}")

            if total_results > 0:
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
            else:
                print("\n[OUTPUT QUEUE ANALYSIS] No results to analyze.")

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

    except KeyboardInterrupt:
        print("\n[INFO] Queue monitoring stopped by user.")


def main():
    parser = argparse.ArgumentParser(description="Workflow Controller for AI Task Processing")
    parser.add_argument('--mode', choices=['pipeline', 'farm'], required=True, help='Select execution mode')
    parser.add_argument('--feedback', action=argparse.BooleanOptionalAction, help='Enable collector feedback')
    parser.add_argument('--tasks', type=int, default=1000, help='Number of tasks to generate')
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
    run_script("task_generator.py", [str(args.tasks)])

    # Step 4: Start emitter
    run_script("emitter_init.py", ["True"])

    # Step 5: Start workers and collector based on mode
    if args.mode == "pipeline":
        run_script("worker_init.py", ["1", "True"])
        run_script("collector_init.py", ["True", "True" if args.feedback else "False"])
    elif args.mode == "farm":
        run_script("worker_init.py", [str(args.workers), "True"])
        run_script("collector_init.py", ["True", "True" if args.feedback else "False"])

    # Step 6: Begin monitoring queues
    monitor_queues(interval=5, total_tasks=args.tasks, start_time=program_start_time, feedback_enabled=args.feedback)


if __name__ == "__main__":
    main()
