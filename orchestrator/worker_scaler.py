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
    get_worker_pod_names
)

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
    config = get_config()

    if delta == 0:
        print("[INFO] No scaling requested (delta = 0).")
        return

    payload = {
        "start_flag": "True",
        "worker_queue_name": config["worker_queue_name"],
        "result_queue_name": config["result_queue_name"],
        "control_syn_queue_name": config["control_syn_queue_name"],
        "control_ack_queue_name": config["control_ack_queue_name"]
    }
    current_replicas = get_current_worker_replicas()
    new_replicas = max(current_replicas + delta, 0)
    print(f"[INFO] Current replicas: {current_replicas}, New replicas: {new_replicas}")
    count = abs(delta)

    if delta > 0:
        print(f"[INFO] Scaling up by {count} replicas...")
        scale_function_deployment(new_replicas)
        time.sleep(5)

        for _ in range(new_replicas):
            invoke_function_async("worker", payload)

    else:
        print(f"[INFO] Scaling down by {count} replicas...")
        control_syn_q = config["control_syn_queue_name"]
        control_ack_q = config["control_ack_queue_name"]

        try:
            redis_client = init_redis_client()
        except Exception as e:
            print(f"[ERROR] Redis connection error: {e}")
            time.sleep(5)
            redis_client = init_redis_client()

        if current_replicas > 1:
            if new_replicas < 1:
                print("[INFO] Forcing at least 1 replica to remain.")
                count = current_replicas - 1
                new_replicas = 1
            print(f"[INFO] Sending {count} control requests to stop processing tasks...")

            for _ in range(current_replicas):
                invoke_function_async("worker", payload)

            time.sleep(5)
            send_control_messages(redis_client, control_syn_q, count)

            acked_pods = []
            while len(acked_pods) < count:
                print(f"[INFO] Waiting for ACKs... {len(acked_pods)}/{count}")
                msg_raw = redis_client.rpop(control_ack_q)
                if not msg_raw:
                    time.sleep(5)
                    continue

                msg = json.loads(msg_raw)
                if msg.get("type") == "SCALE_DOWN" and msg.get("action") == "ACK":
                    pod_name = msg.get("pod_name")
                    acked_pods.append(pod_name)
                    print(f"[INFO] ACK received from pod {pod_name}")

            all_pods_before = set(get_worker_pod_names())
            to_keep = list(all_pods_before - set(acked_pods))

            print("[INFO] Deleting ACKed pods...")
            for pod in acked_pods:
                delete_pod_by_name(pod)

            # time.sleep(5)

            print(f"[INFO] Scaling deployment down to {new_replicas}...")
            scale_function_deployment(new_replicas)

            time.sleep(3)
            all_pods_after = set(get_worker_pod_names())

            for pod in to_keep:
                if pod in all_pods_after:
                    print(f"[OK] Pod '{pod}' preserved.")
                else:
                    print(f"[ERROR] Pod '{pod}' was unintentionally lost!")

            for pod in acked_pods:
                if pod in all_pods_after:
                    print(f"[ERROR] Deleted pod '{pod}' still exists! Check scaling logic.")

        else:
            print("[INFO] Only one replica present; cannot scale down further.")

    print("[INFO] Finalizing scaling...")
    time.sleep(5)
    print(f"[INFO] Replicas now: {get_current_worker_replicas()}")

if __name__ == "__main__":
    main()
