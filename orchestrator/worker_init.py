import sys
from utilities import get_config, scale_worker_deployment, invoke_function_async

def main():
    if len(sys.argv) != 3:
        print("Usage: python worker_init.py <replica_count> <start_flag: True|False>")
        sys.exit(1)

    try:
        replicas = int(sys.argv[1])
        if replicas <= 0:
            raise ValueError("Replica count must be greater than zero.")
    except ValueError:
        print("ERROR: Invalid replica count. Must be a positive integer.")
        sys.exit(1)

    start_flag_input = sys.argv[2].lower()
    if start_flag_input not in ("true", "false"):
        print("ERROR: start_flag must be 'True' or 'False'.")
        sys.exit(1)
    start_flag = start_flag_input == "true"

    scale_worker_deployment(replicas)

    config = get_config()

    payload = {
        "start_flag": start_flag,
        "worker_queue_name": config["worker_queue_name"],
        "result_queue_name": config["result_queue_name"]
    }

    for i in range(replicas):
        invoke_function_async("worker", payload)

if __name__ == "__main__":
    main()
