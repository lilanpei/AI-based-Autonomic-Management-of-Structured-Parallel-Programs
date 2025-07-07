import subprocess
import argparse
import sys
import time
import redis
import json
from utilities import get_config, init_redis_client, get_current_worker_replicas

def run_script(script_name, args=[]):
    cmd = ["python", script_name] + [str(arg) for arg in args]
    print(f"[RUNNING] {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[ERROR] Script {script_name} failed.")
        sys.exit(1)

def monitor_queues(interval=5):
    config = get_config()
    QUEUE_NAMES = [
        config["input_queue_name"],
        config["worker_queue_name"],
        config["result_queue_name"],
        config["output_queue_name"]
    ]
    try:
        try:
            r = init_redis_client()
        except redis.exceptions.ConnectionError as e:
            print(f"Redis connection error: {str(e)}. Attempting to reinitialize.")
            time.sleep(5)
            try:
                r = init_redis_client()
            except Exception as init_e:
                print(f"CRITICAL ERROR: Redis reinit failed: {init_e}", file=sys.stderr)
                sys.exit(1)

        print("\n[INFO] Starting Redis queue monitoring...")
        while True:
            print("------------------------------")
            print("\n[QUEUE STATUS]")
            for queue in QUEUE_NAMES:
                length = r.llen(queue)
                print(f"  {queue}: {length} items")

            # Monitoring current worker replicas
            try:
                replicas = get_current_worker_replicas()
                print(f"\n[WORKER STATUS]")
                print(f"  Current worker replicas: {replicas}")
            except Exception as e:
                print(f"[WARNING] Could not retrieve worker replicas: {e}")

            # Analyze output queue
            results = r.lrange("output_queue", 0, -1)
            total_results = 0
            qos_exceed_count = 0
            total_compute_time = 0.0
            total_comm_time = 0.0
            total_emit_time = 0.0
            total_collect_time = 0.0

            for raw in results:
                try:
                    result = json.loads(raw)
                    total_results += 1

                    complete_time = result.get("complete_time", 0.0)
                    emit_time = result.get("emit_time", 0.0)
                    collect_time = result.get("collect_time", 0.0)
                    qos = result.get("QoS", False)

                    comm_time = emit_time + collect_time

                    if not qos:
                        qos_exceed_count += 1
                    total_compute_time += complete_time
                    total_comm_time += comm_time
                    total_emit_time += emit_time
                    total_collect_time += collect_time

                except Exception as e:
                    print(f"[WARNING] Failed to parse result: {e}")

            if total_results > 0:
                avg_compute_time = total_compute_time / total_results
                avg_comm_time = total_comm_time / total_results
                avg_emit_time = total_emit_time / total_results
                avg_collect_time = total_collect_time / total_results
                qos_percentage = (qos_exceed_count / total_results) * 100

                print("\n[OUTPUT QUEUE ANALYSIS]")
                print(f"  Total results       : {total_results}")
                print(f"  QoS exceed count    : {qos_exceed_count} ({qos_percentage:.2f}%)")
                print(f"  Avg compute time    : {avg_compute_time:.4f} sec")
                print(f"  Avg emit time       : {avg_emit_time:.4f} sec")
                print(f"  Avg collect time    : {avg_collect_time:.4f} sec")
                print(f"  Avg communication   : {avg_comm_time:.4f} sec")
            else:
                print("\n[OUTPUT QUEUE ANALYSIS] No results to analyze.")

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

    run_script("env_init.py")
    time.sleep(10)  # Allow time for environment initialization
    run_script("emitter_init.py", ["True"])  # Start emitter with start_flag=True

    if args.mode == "pipeline":
        run_script("worker_init.py", ["1", "True"])
        run_script("collector_init.py", ["True", "True" if args.feedback else "False"])
        run_script("task_generator.py", [str(args.tasks)])

    elif args.mode == "farm":
        run_script("worker_init.py", [str(args.workers), "True"])
        run_script("collector_init.py", ["True", "True" if args.feedback else "False"])
        run_script("task_generator.py", [str(args.tasks)])

    # Start Redis queue monitoring
    monitor_queues()

if __name__ == "__main__":
    main()
