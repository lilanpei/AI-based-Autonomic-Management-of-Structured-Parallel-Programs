from utilities import get_config, invoke_function_async

def main():
    config = get_config()

    payload = {
        "input_queue_name": config["input_queue_name"],
        "worker_queue_name": config["worker_queue_name"]
    }

    invoke_function_async("emitter", payload)

    print(f"[INFO] Emitter initialized.")

if __name__ == "__main__":
    main()
