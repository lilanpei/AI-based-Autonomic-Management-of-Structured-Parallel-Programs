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
            headers={"Content-Type": "application/json"},
            timeout=125  # Slightly more than function timeout
        )
    except requests.exceptions.Timeout as retry_e:
        print(f"ERROR: Timeout while trying to invoke Emitter function: {retry_e}. Retrying...", file=sys.stderr)
        time.sleep(5)  # Wait before retrying
        try:
            res = requests.post(
            "http://127.0.0.1:8080/function/emitter",
            data=json.dumps(request_payload),
            headers={"Content-Type": "application/json"},
            timeout=125  # Slightly more than function timeout
            )
        except requests.exceptions.Timeout as retry_e:
            print("Timeout occurred during retry. Exiting with error.", file=sys.stderr)
            return {"statusCode": 500, "body": f"Emitter function retry failed: {retry_e}"}

    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to decode JSON response from Emitter function: {e}", file=sys.stderr)
        return {"statusCode": 500, "body": f"Emitter function response parsing failed: {e}"}
    except Exception as e:
        print(f"ERROR: Unexpected error while invoking Emitter function: {e}", file=sys.stderr)  
        return {"statusCode": 500, "body": f"Emitter function invocation failed: {e}"}

    print("Emitter Response:")
    print(f"Status Code: {res.status_code}")
    print("Response Body:", res.text)

if __name__ == "__main__":
    main()
