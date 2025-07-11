import sys
from utilities import get_config, scale_function_deployment, invoke_function_async

def parse_replicas(arg: str) -> int:
    """Parse and validate the replica count."""
    try:
        replicas = int(arg)
        if replicas <= 0:
            raise ValueError()
        return replicas
    except ValueError:
        print("ERROR: <replica_count> must be a positive integer.")
        sys.exit(1)

def main():
    if len(sys.argv) != 2:
        print("Usage: python worker_init.py <replica_count>")
        sys.exit(1)

    replicas = parse_replicas(sys.argv[1])

    print(f"[INFO] Scaling worker function to {replicas} replicas...")
    scale_function_deployment(replicas)

    config = get_config()
    payload = {
        "worker_queue_name": config["worker_queue_name"],
        "result_queue_name": config["result_queue_name"],
        "control_syn_queue_name": config["control_syn_queue_name"],
        "control_ack_queue_name": config["control_ack_queue_name"],
    }

    print(f"[INFO] Invoking {replicas} worker function(s) with payload: {payload}")
    for i in range(replicas):
        invoke_function_async("worker", payload)

if __name__ == "__main__":
    main()
