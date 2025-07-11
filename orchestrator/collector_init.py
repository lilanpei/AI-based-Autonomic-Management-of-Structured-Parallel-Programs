import sys
from utilities import get_config, invoke_function_async

def parse_bool(arg: str, name: str) -> bool:
    """Parse string to boolean, with error handling."""
    val = arg.strip().lower()
    if val not in ("true", "false"):
        print(f"ERROR: Argument '{name}' must be 'True' or 'False' (case insensitive). Got: {arg}")
        sys.exit(1)
    return val == "true"

def main():
    if len(sys.argv) != 2:
        print("Usage: python collector_init.py <collector_feedback_flag: True|False>")
        sys.exit(1)

    feedback_flag = parse_bool(sys.argv[1], "collector_feedback_flag")

    config = get_config()

    payload = {
        "collector_feedback_flag": feedback_flag,
        "input_queue_name": config["input_queue_name"],
        "result_queue_name": config["result_queue_name"],
        "output_queue_name": config["output_queue_name"]
    }

    invoke_function_async("collector", payload)

    print(f"[INFO] Collector initialized with collector_feedback_flag={feedback_flag}.")

if __name__ == "__main__":
    main()
