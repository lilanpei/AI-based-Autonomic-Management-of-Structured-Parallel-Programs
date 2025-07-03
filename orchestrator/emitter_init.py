import sys
import json
import time
import requests
from threading import Thread
from utilities import get_config

def invoke_emitter_function(payload):
    """
    Sends an async POST request to the emitter function.
    Includes a retry on failure.
    """
    try:
        response = requests.post(
            "http://127.0.0.1:8080/function/emitter",
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"},
            timeout=125  # Slightly above typical OpenFaaS timeout
        )
    except requests.exceptions.Timeout as e:
        print(f"ERROR: Timeout while invoking emitter: {e}. Retrying...", file=sys.stderr)
        time.sleep(5)
        try:
            response = requests.post(
                "http://127.0.0.1:8080/function/emitter",
                data=json.dumps(payload),
                headers={"Content-Type": "application/json"},
                timeout=125
            )
        except Exception as retry_e:
            print(f"Timeout occurred during retry for emitter: {retry_e}", file=sys.stderr)
            return

    except Exception as e:
        print(f"ERROR: Failed to invoke emitter: {e}", file=sys.stderr)
        return

    print("[INFO] Invoked emitter function")
    print(f"Status Code: {response.status_code}")
    print("Response Body:", response.text)

def main():
    config = get_config()

    payload = {
        "start_flag": True,
        "input_queue_name": config["input_queue_name"],
        "worker_queue_name": config["worker_queue_name"]
    }

    # Launch POST request in a background thread
    t = Thread(target=invoke_emitter_function, args=(payload,))
    t.start()

    print("[INFO] Launched async thread for emitter function invocation.")

if __name__ == "__main__":
    main()
