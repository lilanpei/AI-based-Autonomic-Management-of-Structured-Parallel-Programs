import sys
import time
import json
import threading
import requests
from utilities import get_config, scale_worker_deployment


def invoke_worker_function_concurrently(replica_count):
    """
    Invokes the worker function via HTTP POST for each replica concurrently via the OpenFaaS gateway.
    """
    config_data = get_config()
    worker_q = config_data["worker_queue_name"]
    result_q = config_data["result_queue_name"]

    def invoke(i):
        payload = {
            "start_flag": True,
            "worker_queue_name": worker_q,
            "result_queue_name": result_q
        }

        try:
            response = requests.post(
                "http://127.0.0.1:8080/function/worker",  # Local gateway path
                data=json.dumps(payload),
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            print(f"[INFO] Invoked worker request {i} - Status: {response.status_code}")
            print("Response Body:", response.text)
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Failed to invoke worker request {i}: {e}", file=sys.stderr)

    threads = []
    for i in range(replica_count):
        t = threading.Thread(target=invoke, args=(i,))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()


def main():
    if len(sys.argv) != 2:
        print("Usage: python worker_init.py <replica_count>")
        sys.exit(1)

    try:
        replicas = int(sys.argv[1])
        if replicas < 0:
            raise ValueError("Replica count must be non-negative.")
    except ValueError:
        print("ERROR: Invalid replica count. Must be a non-negative integer.")
        sys.exit(1)

    scale_worker_deployment(replicas)
    invoke_worker_function_concurrently(replicas)


if __name__ == "__main__":
    main()
