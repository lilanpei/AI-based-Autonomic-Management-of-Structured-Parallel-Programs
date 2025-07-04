import sys
from utilities import get_config, invoke_function_async

def main():
    if len(sys.argv) != 2:
        print("Usage: python emitter_init.py <start_flag: True|False>")
        sys.exit(1)

    start_flag_input = sys.argv[1].lower()
    if start_flag_input not in ("true", "false"):
        print("ERROR: start_flag must be 'True' or 'False'.")
        sys.exit(1)
    start_flag = start_flag_input == "true"

    config = get_config()

    payload = {
        "start_flag": start_flag,
        "input_queue_name": config["input_queue_name"],
        "worker_queue_name": config["worker_queue_name"]
    }

    invoke_function_async("emitter", payload)

    print("[INFO] Launched async thread for emitter function invocation.")

if __name__ == "__main__":
    main()
