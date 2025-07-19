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
    safe_invoke_function_async,
    get_function_pod_names,
    get_redis_client_with_retry,
    scale_queue_worker,
    get_queue_worker_replicas
)


VALID_DELTAS = {"+2", "+1", "0", "-1", "-2"}


def parse_args():
    """
    Parses and validates CLI arguments.
    """
    if len(sys.argv) == 2:
        delta_str = sys.argv[1]
        feedback_flag = False
        print(f"[INFO] No feedback flag provided, defaulting to {feedback_flag}.")
    elif len(sys.argv) == 3:
        delta_str = sys.argv[1]
        feedback_flag = sys.argv[2].lower() == "true"
        print(f"[INFO] Feedback flag set to {feedback_flag}.")
    else:
        print("[ERROR] Invalid number of arguments. Expected 1 or 2 arguments.")
        print("Usage: python worker_scaler.py <scale_delta> [<feedback_enabled>]")
        print("Example: python worker_scaler.py +1 True")
        sys.exit(1)

    if delta_str not in VALID_DELTAS:
        print("ERROR: scale_delta must be one of +2, +1, 0, -1, -2")
        sys.exit(1)

    return int(delta_str), feedback_flag


def scale_up(current, delta, configuration, redis_client, payload, apps_v1_api=None):
    """
    Scales up worker pods and deploys new functions.
    """
    if not configuration:
        configuration = get_config()
    if not redis_client:
        redis_client = get_redis_client_with_retry()
    new_replicas = current + delta
    print(f"[INFO] Scaling up from {current} to {new_replicas} replicas...")
    scale_function_deployment(new_replicas, apps_v1_api=apps_v1_api, fn="worker", namespace="openfaas-fn")
    time.sleep(delta)
    scale_queue_worker(new_replicas+2, apps_v1_api=apps_v1_api, fn="worker", namespace="openfaas-fn")
    time.sleep(delta)
    safe_invoke_function_async("worker", payload, redis_client, configuration.get("worker_start_queue_name"), delta)

    return new_replicas

def scale_down(current, delta, configuration, redis_client, timeout=20, retries=2, backoff_factor=2, core_v1_api=None, apps_v1_api=None):
    """
    Scales down worker pods using control messages and ACKs.
    """
    if not configuration:
        configuration = get_config()
    if not redis_client:
        redis_client = get_redis_client_with_retry()

    control_syn_q = config.get("worker_control_syn_queue_name")
    control_ack_q = config.get("worker_control_ack_queue_name")
    new_replicas = max(current + delta, 1)
    count = current - new_replicas

    print(f"[INFO] Preparing to scale down from {current} to {new_replicas} replicas...")
    print(f"[INFO] Sending {count} control requests...")

    # Send SCALE_DOWN control messages
    message = {
        "type": "SCALE_DOWN",
        "action": "SYN",
        "message": "Scale down request from orchestrator",
        "SYN_timestamp": time.time()
    }
    send_control_messages(message, redis_client, control_syn_q, count, apps_v1_api=None)

    # Wait for ACKs
    acked_pods = []
    attempts = 0
    while len(acked_pods) < count:
        print(f"[INFO] Waiting for ACKs... {len(acked_pods)}/{count}")
        msg_raw = redis_client.rpop(control_ack_q)
        if not msg_raw:
            attempts += 1
            sleep_time = backoff_factor ** attempts
            print(f"[INFO] No ACK received for {attempts} tries, waiting {timeout+sleep_time} seconds...")
            time.sleep(timeout+sleep_time)
            if attempts >= retries:
                print("[WARNING] No ACKs received for a long time, exiting scale down.")
                print(f"[INFO] Current worker replicas: {get_current_worker_replicas(apps_v1_api=apps_v1_api)}")
                sys.exit(0)
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
        delete_pod_by_name(pod_name=pod, core_v1_api=core_v1_api)

    # Scale the function deployment down
    print(f"[INFO] Scaling deployment to {new_replicas} replicas...")
    scale_function_deployment(new_replicas, apps_v1_api=apps_v1_api, fn="worker", namespace="openfaas-fn")
    time.sleep(3)

    # Check pod consistency
    remaining_pods = set(get_function_pod_names("worker"), core_v1_api=core_v1_api)
    for pod in acked_pods:
        if pod in remaining_pods:
            print(f"[ERROR] Pod '{pod}' was not deleted as expected!")
    for pod in remaining_pods:
        if pod not in acked_pods:
            print(f"[OK] Pod '{pod}' preserved.")
    return new_replicas

def main():
    delta, feedback_flag = parse_args()
    redis_client = get_redis_client_with_retry()
    configuration = get_config()
    try:
        config.load_incluster_config()
    except:
        try:
            config.load_kube_config()
        except Exception as e:
            print(f"[ERROR] Kubernetes config load failed: {e}", file=sys.stderr)
            sys.exit(1)
    core_v1_api = client.CoreV1Api()
    apps_v1_api = client.AppsV1Api()

    if delta == 0:
        print("[INFO] No scaling requested (delta = 0).")
        return

    payload = {
        "input_queue_name": configuration.get("input_queue_name"),
        "worker_queue_name": configuration.get("worker_queue_name"),
        "result_queue_name": configuration.get("result_queue_name"),
        "output_queue_name": configuration.get("output_queue_name"),
        "emitter_control_syn_queue_name": configuration.get("emitter_control_syn_queue_name"),
        "worker_control_syn_queue_name": configuration.get("worker_control_syn_queue_name"),
        "worker_control_ack_queue_name": configuration.get("worker_control_ack_queue_name"),
        "collector_control_syn_queue_name": configuration.get("collector_control_syn_queue_name"),
        "emitter_start_queue_name": configuration.get("emitter_start_queue_name"),
        "worker_start_queue_name": configuration.get("worker_start_queue_name"),
        "collector_start_queue_name": configuration.get("collector_start_queue_name"),
        "collector_feedback_flag": feedback_flag,
    }

    current_replicas = get_current_worker_replicas(apps_v1_api=apps_v1_api)
    print(f"[INFO] Current replicas: {current_replicas}")

    if delta > 0:
        new_replicas = scale_up(current_replicas, delta, configuration, redis_client, payload, apps_v1_api=apps_v1_api)
    else:
        if current_replicas <= 1:
            print("[INFO] Only one replica present; cannot scale down further.")
        else:
            new_replicas = scale_down(current_replicas, delta, configuration, redis_client, core_v1_api=core_v1_api, apps_v1_api=apps_v1_api)
            scale_queue_worker(replicas=new_replicas+2, apps_v1_api=apps_v1_api, fn="worker", namespace="openfaas-fn")

    print("[INFO] Finalizing scaling...")
    time.sleep(2)
    current_worker_replicas = get_current_worker_replicas(apps_v1_api=apps_v1_api)
    current_queue_worker_replicas = get_queue_worker_replicas(apps_v1_api=apps_v1_api)
    print(f"[INFO] Current Worker Replicas: {current_worker_replicas}, Queue Worker Replicas: {current_queue_worker_replicas}")


if __name__ == "__main__":
    main()
