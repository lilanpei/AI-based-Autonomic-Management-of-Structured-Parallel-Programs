import os
import sys
import json
import time
import subprocess
from datetime import datetime
from kubernetes import client, config
from utilities import (
    get_config,
    scale_function_deployment,
    get_deployment_replicas,
    send_control_messages,
    safe_invoke_function_async,
    get_redis_client_with_retry,
    get_utc_now,
    deploy_function,
    run_faas_cli
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


def scale_up(program_start_time, current, delta, configuration, redis_client, payload, function_invoke_timeout, function_invoke_retries, apps_v1_api=None):
    """
    Scales up worker pods and deploys new function instances from index current+1 to current+delta.
    """

    if not configuration:
        configuration = get_config()
    if not redis_client:
        redis_client = get_redis_client_with_retry()

    new_replicas = current + delta
    print(f"[TIMER] Scaling up from {current} to {new_replicas} replicas at {(get_utc_now()-program_start_time).total_seconds():.4f} seconds")
    # Scaling up by deploying new function instances
    deployed_worker_names = deploy_function(function_name_prefix="worker", replicas=delta, max_retries=3, delay=1)
    print(f"[TIMER] Deployed {delta} new worker instances at {(get_utc_now()-program_start_time).total_seconds():.4f} seconds")
    # Scale the queue worker deployment to match the new number of replicas
    current_queue_worker_replicas = get_deployment_replicas(apps_v1_api=apps_v1_api, namespace="openfaas", name_or_prefix="queue-worker", exact_match=True)
    print(f"[INFO] Current Queue Worker Replicas: {current_queue_worker_replicas}")
    scale_function_deployment(current_queue_worker_replicas+delta, apps_v1_api=apps_v1_api, deployment_name="queue-worker", namespace="openfaas")
    print(f"[TIMER] Scaled queue worker deployment to {current_queue_worker_replicas+delta} replicas at {(get_utc_now()-program_start_time).total_seconds():.4f} seconds")

    safe_invoke_function_async(
        deployed_worker_names,
        payload,
        redis_client,
        configuration.get("worker_start_queue_name"),
        delta,
        timeout=function_invoke_timeout,
        retries=function_invoke_retries
    )
    print(f"[TIMER] Finished invoking {delta} worker functions at {(get_utc_now()-program_start_time).total_seconds():.4f} seconds")


def scale_down(program_start_time, current, delta, configuration, redis_client, timeout=1, retries=30, core_v1_api=None, apps_v1_api=None):
    """
    Scales down worker pods using control messages and ACKs.
    """
    if not configuration:
        configuration = get_config()
    if not redis_client:
        redis_client = get_redis_client_with_retry()

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

    control_syn_q = configuration.get("worker_control_syn_queue_name")
    control_ack_q = configuration.get("worker_control_ack_queue_name")
    new_replicas = max(current + delta, 1)
    count = current - new_replicas

    print(f"[TIMER] Scale down from {current} to {new_replicas} replicas at {(get_utc_now()-program_start_time).total_seconds():.4f} seconds")
    print(f"[TIMER] Sending {count} control requests at {(get_utc_now()-program_start_time).total_seconds():.4f} seconds...")

    # Send SCALE_DOWN control messages
    message = {
        "type": "SCALE_DOWN",
        "action": "SYN",
        "message": "Scale down request from orchestrator",
        "SYN_timestamp": (get_utc_now() - program_start_time).total_seconds(),
    }
    send_control_messages(message, redis_client, control_syn_q, count)
    print(f"[TIMER] Sent {count} control messages at {(get_utc_now()-program_start_time).total_seconds():.4f} seconds")

    # Wait for ACKs
    acked_pods = []
    attempts = 0
    while len(acked_pods) < count:
        print(f"[INFO] Waiting for ACKs... {len(acked_pods)}/{count}")
        msg_raw = redis_client.rpop(control_ack_q)
        if not msg_raw:
            attempts += 1
            print(f"[INFO] No ACK received for {attempts} tries, waiting {timeout} seconds...")
            time.sleep(timeout)
            if attempts >= retries:
                print("[WARNING] No ACKs received for a long time, exiting scale down.")
                print(f"[INFO] Current worker replicas: {get_deployment_replicas(apps_v1_api, namespace='openfaas-fn', name_or_prefix='worker-', exact_match=False)}")
                sys.exit(0)
            continue

        try:
            msg = json.loads(msg_raw)
            # Check if the message is an ACK for SCALE_DOWN
            if msg.get("type") == "SCALE_DOWN" and msg.get("action") == "ACK":
                pod_name = msg.get("pod_name")
                acked_pods.append(pod_name)
                print(f"[INFO] ACK received from pod: {pod_name}")
        except Exception as e:
            print(f"[WARNING] Malformed ACK message: {e}")
    print(f"[TIMER] Received ACKs for {len(acked_pods)} pods at {(get_utc_now()-program_start_time).total_seconds():.4f} seconds")

    # Delete functions based on ACKed pod names
    print("[INFO] Deleting functions for ACKed pods...")
    for pod in acked_pods:
        try:
            result = subprocess.run([
                "kubectl", "get", "pod", pod, "-n", "openfaas-fn",
                "-o", "jsonpath={.metadata.labels.faas_function}"
            ], capture_output=True, text=True, check=True)
            function_name = result.stdout.strip()
            if not function_name:
                print(f"[WARNING] No faas_function label found for pod {pod}, skipping deletion.")
                continue

            print(f"[INFO] Removing function: {function_name}")
            run_faas_cli(["remove", function_name])
            print(f"[TIMER] Finished removing function: {function_name} at {(get_utc_now()-program_start_time).total_seconds():.4f} seconds")

            # Wait for pod termination
            for attempt in range(retries):
                pod_list = core_v1_api.list_namespaced_pod(
                    namespace="openfaas-fn",
                    label_selector=f"faas_function={function_name}"
                ).items
                if not pod_list:
                    print(f"[TIMER] Function pod for '{function_name}' has terminated at {(get_utc_now()-program_start_time).total_seconds():.4f} seconds")
                    break
                print(f"[INFO] Waiting for pod of '{function_name}' to terminate (Attempt {attempt+1})...")
                time.sleep(timeout)
            else:
                print(f"[WARNING] Function pod for '{function_name}' still exists after timeout.")

        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Failed to fetch function name for pod {pod}: {e}")
        except Exception as e:
            print(f"[ERROR] Unexpected error while deleting function for pod {pod}: {e}")


def main():
    redis_client = get_redis_client_with_retry()
    queue = "workflow_init_syn_queue"
    init_len = redis_client.llen(queue)
    print(f"[INFO] Length of {queue}: {init_len}")
    if init_len == 0:
        print("[ERROR] Workflow init not completed. Exiting.")
        sys.exit(0)

    delta, feedback_flag = parse_args()
    configuration = get_config()
    init_msg = json.loads(redis_client.lindex(queue, -1))
    program_start_time = datetime.fromisoformat(init_msg["program_start_time"])
    print(f"[INFO] Workflow init completed at timestamp: {init_msg['SYN_timestamp']}")
    print(f"[INFO] Worker scaler start time: {(get_utc_now() - program_start_time).total_seconds():.4f} seconds")

    function_invoke_timeout = configuration.get("function_invoke_timeout")
    function_invoke_retries = configuration.get("function_invoke_retries")
    scale_down_timeout = configuration.get("scale_down_timeout")
    scale_down_retries = configuration.get("scale_down_retries")

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
        "processing_delay": configuration.get("processing_delay"),
        "wait_time": configuration.get("wait_time"),
        "deadline_coeff": configuration.get("deadline_coeff"),
        "deadline_cap": configuration.get("deadline_cap"),
        "deadline_floor": configuration.get("deadline_floor"),
        "program_start_time": str(program_start_time),
        "collector_feedback_flag": feedback_flag
    }

    current_replicas = get_deployment_replicas(apps_v1_api, namespace="openfaas-fn", name_or_prefix="worker-", exact_match=False)
    print(f"[INFO] Current replicas: {current_replicas}")

    if delta > 0:
        scale_up(program_start_time, current_replicas, delta, configuration, redis_client, payload, int(function_invoke_timeout), int(function_invoke_retries), apps_v1_api)
    else:
        if current_replicas <= 1:
            print("[INFO] Only one replica present; cannot scale down further.")
        else:
            scale_down(program_start_time, current_replicas, delta, configuration, redis_client, int(scale_down_timeout), int(scale_down_retries), core_v1_api, apps_v1_api)

    print(f"[TIMER] Finalizing scaling at {(get_utc_now() - program_start_time).total_seconds():.4f} seconds")
    time.sleep(2)
    current_worker_replicas = get_deployment_replicas(apps_v1_api, namespace="openfaas-fn", name_or_prefix="worker-", exact_match=False)
    current_queue_worker_replicas = get_deployment_replicas(apps_v1_api, namespace="openfaas", name_or_prefix="queue-worker", exact_match=True)
    print(f"[INFO] Current Worker Replicas: {current_worker_replicas}, Queue Worker Replicas: {current_queue_worker_replicas}")


if __name__ == "__main__":
    main()
