import sys
import json
from utilities import get_config, scale_worker_deployment, invoke_worker_function_concurrently

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
