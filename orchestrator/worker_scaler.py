import sys
import json
import time
from kubernetes import client, config
from utilities import get_config, scale_worker_deployment, invoke_worker_function_concurrently

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

def get_current_worker_replicas(namespace="openfaas-fn", deployment_name="worker"):
    config.load_kube_config()
    apps_v1 = client.AppsV1Api()
    deployment = apps_v1.read_namespaced_deployment(deployment_name, namespace)
    return deployment.spec.replicas

def send_control_requests(start_flag, count):
    config_data = get_config()
    worker_q = config_data["worker_queue_name"]
    result_q = config_data["result_queue_name"]

    for _ in range(count):
        payload = {
            "start_flag": start_flag,
            "worker_queue_name": worker_q,
            "result_queue_name": result_q
        }
        # simulate OpenFaaS function trigger
        invoke_worker_function_concurrently(1)
        print(f"[INFO] Sent control request with start_flag={start_flag}")

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
        print(f"[INFO] Scaling up by {count} replicas...")
        scale_worker_deployment(new_replicas)
        time.sleep(5)  # Allow time for the deployment to scale up
        print(f"[INFO] Invoking {new_replicas} workers to start processing tasks...")
        send_control_requests(True, count=new_replicas)
    else:
        print(f"[INFO] Scaling down by {count} replicas...")
        if new_replicas > 1:
            print(f"[INFO] Sending {count} control requests to stop processing tasks...")
            send_control_requests(False, count=count)           
            time.sleep(10) # Allow time for the control requests to be processed
            scale_worker_deployment(new_replicas)
        else:
            print("[INFO] Already Scaled down to 1 replica, no further action needed.")
    
    print("[INFO] Waiting for the deployment to stabilize...")
    time.sleep(5)
    current_replicas = get_current_worker_replicas()
    print(f"[INFO] Successfully scaled to {current_replicas} replicas.")

if __name__ == "__main__":
    main()
