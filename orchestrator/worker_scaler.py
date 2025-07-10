import sys
import json
import time
from utilities import get_config, scale_function_deployment, get_current_worker_replicas, init_redis_client, send_control_messages, delete_pod_by_name, invoke_function_async

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

    current_replicas = get_current_worker_replicas()
    new_replicas = max(current_replicas + delta, 0)
    print(f"[INFO] Current replicas: {current_replicas}, New replicas: {new_replicas}")
    count = abs(delta)
    if delta > 0:
        print(f"[INFO] Trying to scale up by {count} replicas...")
        scale_function_deployment(new_replicas)
        time.sleep(5)  # Allow time for the deployment to scale up
        print(f"[INFO] Invoking {new_replicas} workers to start processing tasks...")

        payload = {
            "start_flag": "True",
            "worker_queue_name": config["worker_queue_name"],
            "result_queue_name": config["result_queue_name"],
            "control_syn_queue_name": config["control_syn_queue_name"],
            "control_ack_queue_name": config["control_ack_queue_name"]
        }

        for _ in range(new_replicas):
            invoke_function_async("worker", payload)

    else:
        print(f"[INFO] Trying to scale down by {count} replicas...")
        control_syn_q = config["control_syn_queue_name"]
        control_ack_q = config["control_ack_queue_name"]
        try:
            redis_client = init_redis_client()
        except redis.exceptions.ConnectionError as e:
            print(f"[ERROR] Redis connection error: {str(e)}. Attempting to reinitialize.")
            time.sleep(5)
            redis_client = init_redis_client()

        if current_replicas != 1:
            if new_replicas < 1:
                print("[INFO] At least one replica remains, scaled down to 1 replica.")
                count = current_replicas - 1 # Ensure at least one replica remains
                new_replicas = 1
            print(f"[INFO] Sending {count} control requests to stop processing tasks...")
            send_control_messages(redis_client, control_syn_q, count=count)

            while True:
                print(f"[INFO] Waiting for ACKs for {count} control messages...")
                # Wait for ACKs from the control_ack_q
                msg_raw = redis_client.rpop(control_ack_q)
                if msg_raw is None:
                    print("[INFO] No ACK received yet, checking again...")
                    time.sleep(10)
                    continue

                msg = json.loads(msg_raw)
                control_type = msg.get("type")
                control_action = msg.get("action")
                task_id = msg.get("task_id")
                pod_name = msg.get("pod_name")
                ack_timestamp = msg.get("ack_timestamp")
                if control_type == "SCALE_DOWN" and control_action == "ACK":
                    print(f"[INFO] Received ACK for control message: {control_type}, task_id: {task_id}, pod_name: {pod_name}, ack_timestamp: {ack_timestamp}")
                    delete_pod_by_name(pod_name)
                    count -= 1
                    if count <= 0:
                        break
            time.sleep(5)
            print(f"[INFO] Successfully sent control messages and deleted corresponding pods.")
            print(f"[INFO] Scaling down to {new_replicas} replicas...")
            scale_function_deployment(new_replicas)
            print(f"[INFO] Successfully scaled down {abs(delta)} replicas.")
        else:
            print("[INFO] Already at minimum replica count of 1, cannot scale down further.")

    print("[INFO] Waiting for the deployment to stabilize...")
    time.sleep(5)
    current_replicas = get_current_worker_replicas()
    print(f"[INFO] Current worker replicas after scaling: {current_replicas}")

if __name__ == "__main__":
    main()
