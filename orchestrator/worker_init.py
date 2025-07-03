import sys
import time
import json
import requests
from utilities import get_config, init_redis_client, scale_worker_deployment

def invoke_worker_function(replica_count):
    """
    Invokes the worker function via HTTP POST for each replica.
    """
    config_data = get_config()
    worker_q = config_data["worker_queue_name"]
    output_q = config_data["output_queue_name"]

    for i in range(replica_count):
        payload = {
            "start_flag": True,
            "worker_queue_name": worker_q,
            "result_queue_name": output_q
        }

        try:
            response = requests.post(
                "http://127.0.0.1:8080/function/worker",
                data=json.dumps(payload),
                headers={"Content-Type": "application/json"}
            )
        except requests.exceptions.Timeout as retry_e:
            print(f"ERROR: Timeout while trying to invoke worker replica {i}: {retry_e}. Retrying...", file=sys.stderr)
            time.sleep(5)
            try:
                response = requests.post(
                    "http://127.0.0.1:8080/function/worker",
                    data=json.dumps(payload),
                    headers={"Content-Type": "application/json"}
                )
            except requests.exceptions.Timeout:
                print(f"Timeout occurred during retry for worker {i}. Exiting with error.", file=sys.stderr)
                return {"statusCode": 500, "body": f"Worker function retry failed: {retry_e}"}
        except requests.exceptions.RequestException as e:
            print(f"ERROR: Failed to invoke worker replica {i}: {e}", file=sys.stderr)
            return {"statusCode": 500, "body": f"Worker function invocation failed: {e}"}

        print(f"[INFO] Invoked worker replica {i} - Status: {response.status_code}")
        print("Response Body:", response.text)

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
    invoke_worker_function(replicas)

if __name__ == "__main__":
    main()
