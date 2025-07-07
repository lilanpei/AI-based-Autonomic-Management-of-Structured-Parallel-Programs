import sys
import json
import time
from kubernetes import client, config
from utilities import get_config, scale_function_deployment, send_control_requests, get_current_worker_replicas

def parse_args():
    if len(sys.argv) != 2:
        print("Usage: python worker_scaler.py <scale_delta>")
        print("Example: python worker_scaler.py +1")
        sys.exit(1)

    delta = sys.argv[1]

    if delta not in ["+2", "+1", "0", "-1", "-2"]:
        print("ERROR: scale_delta must be one of +2, +1, 0, -1, -2")
        sys.exit(1)

    return int(delta)


def main():
    delta = parse_args()
    if delta == 0:
        print("[INFO] No scaling requested (delta = 0).")
        return

    current_replicas = get_current_worker_replicas()
    new_replicas = max(current_replicas + delta, 0)
    print(f"[INFO] Current replicas: {current_replicas}, New replicas: {new_replicas}")
    count = abs(delta)
    if delta > 0:
        print(f"[INFO] Trying to scale up by {count} replicas...")
        scale_function_deployment(new_replicas)
        time.sleep(5)  # Allow time for the deployment to scale up
        print(f"[INFO] Invoking {new_replicas} workers to start processing tasks...")
        send_control_requests(True, count=new_replicas)
    else:
        print(f"[INFO] Trying to scale down by {count} replicas...")
        if new_replicas > 0:
            print(f"[INFO] Sending {count} control requests to stop processing tasks...")
            send_control_requests(False, count=count)           
            time.sleep(40) # Allow time for the control requests to be processed
            scale_function_deployment(new_replicas)
        else:
            if current_replicas == 1:
                print("[INFO] Already at minimum replica count of 1, cannot scale down further.")
            else:
                print("[INFO] At least one replica remains, scaled down to 1 replica.")
                count = current_replicas - 1 # Ensure at least one replica remains
                send_control_requests(False, count=count)
                time.sleep(40) # Allow time for the control requests to be processed
                scale_function_deployment(1)

    print("[INFO] Waiting for the deployment to stabilize...")
    time.sleep(5)
    current_replicas = get_current_worker_replicas()
    print(f"[INFO] Current worker replicas after scaling: {current_replicas}")

if __name__ == "__main__":
    main()
