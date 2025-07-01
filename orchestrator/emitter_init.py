import sys
import json
import time
import requests
from utilities import get_config

def main():
    config = get_config()

    # Prepare JSON payload
    request_payload = {
        "start_flag": True,
        "input_queue_name": config["input_queue_name"],
        "worker_queue_name": config["worker_queue_name"]
    }

    try:
        res = requests.post(
            "http://127.0.0.1:8080/function/emitter",
            data=json.dumps(request_payload),
            headers={"Content-Type": "application/json"}
        )
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Failed to connect to the Emitter function: {e}", file=sys.stderr)
        time.sleep(5)  # Wait before retrying
        try:
            res = requests.post(
            "http://127.0.0.1:8080/function/emitter",
            data=json.dumps(request_payload),
            headers={"Content-Type": "application/json"}
            )
        except requests.exceptions.RequestException as retry_e:
            print(f"CRITICAL ERROR: Emitter function retry failed: {retry_e}", file=sys.stderr)
            return {"statusCode": 500, "body": f"Emitter function failure: {retry_e}"}
    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to decode JSON response from Emitter function: {e}", file=sys.stderr)
    except Exception as e:
        print(f"ERROR: Unexpected error while invoking Emitter function: {e}", file=sys.stderr)  
        return

    print("Emitter Response:")
    print(f"Status Code: {res.status_code}")
    print("Response Body:", res.text)

if __name__ == "__main__":
    main()
