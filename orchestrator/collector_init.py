import sys
from utilities import get_config, invoke_function_async

def main():
    if len(sys.argv) != 3:
        print("Usage: python collector_init.py <start_flag: True|False> <collector_feedback_flag: True|False>")
        sys.exit(1)

    start_flag_input = sys.argv[1].lower()
    feedback_flag_input = sys.argv[2].lower()

    if start_flag_input not in ("true", "false") or feedback_flag_input not in ("true", "false"):
        print("ERROR: Both start_flag and collector_feedback_flag must be 'True' or 'False'.")
        sys.exit(1)

    start_flag = start_flag_input == "true"
    collector_feedback_flag = feedback_flag_input == "true"

    config = get_config()

    payload = {
        "start_flag": start_flag,
        "Feedback_flag": collector_feedback_flag,
        "input_queue_name": config["input_queue_name"],
        "result_queue_name": config["result_queue_name"],
        "output_queue_name": config["output_queue_name"]
    }

    invoke_function_async("collector", payload)

    print("[INFO] Launched async thread for collector function invocation.")

if __name__ == "__main__":
    main()
