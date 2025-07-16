import sys
import json
import time
from utilities import (
    get_config,
    scale_function_deployment,
    get_current_worker_replicas,
    init_redis_client,
    send_control_messages,
    delete_pod_by_name,
    invoke_function_async,
    get_worker_pod_names,
    get_redis_client_with_retry,
    scale_queue_worker,
    get_queue_worker_replicas
)


VALID_DELTAS = {"+2", "+1", "0", "-1", "-2"}


def parse_args():
    """
    Parses and validates CLI arguments.
    """
    if len(sys.argv) != 3:
        print("Usage: python worker_scaler.py <scale_delta> <feedback_enabled>")
        print("Example: python worker_scaler.py +1 False")
        sys.exit(1)

    delta_str = sys.argv[1]
    feedback_flag = sys.argv[2].lower() == "true"

    if delta_str not in VALID_DELTAS:
        print("ERROR: scale_delta must be one of +2, +1, 0, -1, -2")
        sys.exit(1)

    return int(delta_str), feedback_flag


def scale_up(current, delta, payload):
    """
    Scales up worker pods and deploys new functions.
    """
    new_replicas = current + delta
    print(f"[INFO] Scaling up from {current} to {new_replicas} replicas...")
    scale_function_deployment(new_replicas)
    # time.sleep(5)

    for _ in range(delta):
        invoke_function_async("worker", payload)
    return new_replicas

def scale_down(current, delta, config, payload):
    """
    Scales down worker pods using control messages and ACKs.
    """
    redis_client = get_redis_client_with_retry()
    control_syn_q = config["control_syn_queue_name"]
    control_ack_q = config["control_ack_queue_name"]
    new_replicas = max(current + delta, 1)
    count = current - new_replicas

    print(f"[INFO] Preparing to scale down from {current} to {new_replicas} replicas...")
    print(f"[INFO] Sending {count} control requests...")

    # Send SCALE_DOWN control messages
    send_control_messages(redis_client, control_syn_q, count)

    # Wait for ACKs
    acked_pods = []
    while len(acked_pods) < count:
        print(f"[INFO] Waiting for ACKs... {len(acked_pods)}/{count}")
        msg_raw = redis_client.rpop(control_ack_q)
        if not msg_raw:
            time.sleep(5)
            continue

        try:
            msg = json.loads(msg_raw)
            # compensate for the original task invocation signal
            if msg.get("type") == "SCALE_DOWN" and msg.get("action") == "ACK":
                pod_name = msg.get("pod_name")
                acked_pods.append(pod_name)
                print(f"[INFO] ACK received from pod: {pod_name}")
        except Exception as e:
            print(f"[WARNING] Malformed ACK message: {e}")

    # Delete ACKed pods
    print("[INFO] Deleting ACKed pods...")
    for pod in acked_pods:
        delete_pod_by_name(pod)

    # Scale the function deployment down
    print(f"[INFO] Scaling deployment to {new_replicas} replicas...")
    scale_function_deployment(new_replicas)
    time.sleep(3)

    # Check pod consistency
    remaining_pods = set(get_worker_pod_names())
    for pod in acked_pods:
        if pod in remaining_pods:
            print(f"[ERROR] Pod '{pod}' was not deleted as expected!")
    for pod in remaining_pods:
        if pod not in acked_pods:
            print(f"[OK] Pod '{pod}' preserved.")
    return new_replicas

def main():
    delta, feedback_flag = parse_args()
    config = get_config()

    if delta == 0:
        print("[INFO] No scaling requested (delta = 0).")
        return

    payload = {
        "input_queue_name": config["input_queue_name"],
        "worker_queue_name": config["worker_queue_name"],
        "result_queue_name": config["result_queue_name"],
        "output_queue_name": config["output_queue_name"],
        "control_syn_queue_name": config["control_syn_queue_name"],
        "control_ack_queue_name": config["control_ack_queue_name"],
        "collector_feedback_flag": feedback_flag,
    }

    current_replicas = get_current_worker_replicas()
    print(f"[INFO] Current replicas: {current_replicas}")

    if delta > 0:
        new_replicas = scale_up(current_replicas, delta, payload)
    else:
        if current_replicas <= 1:
            print("[INFO] Only one replica present; cannot scale down further.")
        else:
            new_replicas = scale_down(current_replicas, delta, config, payload)

    scale_queue_worker(replicas=new_replicas)
    print("[INFO] Finalizing scaling...")
    time.sleep(5)
    current_worker_replicas = get_current_worker_replicas()
    current_queue_worker_replicas = get_queue_worker_replicas()
    print(f"[INFO] Current Worker Replicas: {current_worker_replicas}, Queue Worker Replicas: {current_queue_worker_replicas}")


if __name__ == "__main__":
    main()
