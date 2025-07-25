import subprocess
import os
import argparse
import sys
import time
from utilities import get_utc_now, get_config

def parse_worker_count():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, required=True)
    args, _ = parser.parse_known_args()
    return args.workers

def run_worker_scaler_twice(log_file):
    with open(log_file, 'a') as f:
        for run in [1, 2]:
            print(f"Running worker_scaler.py (run {run}) → {log_file}")
            cmd = ["python", "worker_scaler.py", "-2"]
            process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
            process.wait()
            if process.returncode == 0:
                print(f"✅ Completed worker_scaler.py run {run}")
            else:
                print(f"❌ Failed worker_scaler.py run {run} (exit {process.returncode})")

def get_xargs_log_cmd(function_name, grep_pattern):
    return [
        "bash", "-c",
        f"kubectl get pods -l faas_function={function_name} -n openfaas-fn -o name | "
        f"xargs -P 34 -I{{}} kubectl logs {{}} -n openfaas-fn  | grep '{grep_pattern}'"
    ]

def run_commands_with_logs():
    configuration = get_config()
    log_tags = configuration.get("log_tags", ['.*'])  # Default to match all logs if not specified
    grep_pattern = "\\|".join([f"\\{tag}" for tag in log_tags])  # e.g., \[INFO\]\|\[TIMER\]
    worker_count = parse_worker_count()
    worker_suffix = f"w{worker_count}"
    today = get_utc_now().strftime("%m%d")
    log_dir = f"logs_farm/{today}"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = get_utc_now().strftime("%H%M")

    controller_cmd = {
        "cmd": [
            "bash", "-c",
            f"{' '.join(sys.argv[1:])} | grep '{grep_pattern}'"
        ],
        "log_file": f"{log_dir}/logs_controller_{worker_suffix}_{timestamp}.txt"
    }
    other_cmds = [
        # {
        #     "cmd": ["kubectl", "logs", "-n", "openfaas", "deploy/nats"],
        #     "log_file": f"{log_dir}/logs_nats_{worker_suffix}_{timestamp}.txt"
        # },
        # {
        #     "cmd": ["kubectl", "logs", "-n", "openfaas", "-l", "app=gateway", "-f"],
        #     "log_file": f"{log_dir}/logs_gateway_{worker_suffix}_{timestamp}.txt"
        # },
        {
            "cmd": get_xargs_log_cmd("collector", grep_pattern),
            "log_file": f"{log_dir}/logs_collector_{worker_suffix}_{timestamp}.txt"
        },
        {
            "cmd": get_xargs_log_cmd("emitter", grep_pattern),
            "log_file": f"{log_dir}/logs_emitter_{worker_suffix}_{timestamp}.txt"
        },
        {
            "cmd": get_xargs_log_cmd("worker", grep_pattern),
            "log_file": f"{log_dir}/logs_worker_{worker_suffix}_{timestamp}.txt"
        },
        # {
        #     "cmd": [
        #         "kubectl", "logs",
        #         "-l", "app=queue-worker",
        #         "-n", "openfaas",
        #         "--timestamps",
        #         "--max-log-requests=34"
        #     ],
        #     "log_file": f"{log_dir}/logs_queue-worker_{worker_suffix}_{timestamp}.txt"
        # }
    ]

    run_worker_scale_down_enabled = False
    scaler_log = f"{log_dir}/logs_scaler_{worker_suffix}_{timestamp}.txt"

    try:
        print(f"Log directory: {log_dir}")
        print(f"Worker suffix: {worker_suffix}")

        # Step 1: Run controller command
        with open(controller_cmd["log_file"], "w") as log_file:
            controller_proc = subprocess.Popen(controller_cmd["cmd"], stdout=log_file, stderr=subprocess.STDOUT, text=True)
            print(f"Started controller: {' '.join(controller_cmd['cmd'])} → {controller_cmd['log_file']}")

        if run_worker_scale_down_enabled:
            run_worker_scaler_twice(scaler_log)

        controller_proc.wait()
        if controller_proc.returncode == 0:
            # Step 2: Run all other logging commands
            for cmd_info in other_cmds:
                with open(cmd_info["log_file"], "w") as log_file:
                    proc = subprocess.Popen(cmd_info["cmd"], stdout=log_file, stderr=subprocess.STDOUT, text=True)
                    print(f"Started: {' '.join(cmd_info['cmd'])} → {cmd_info['log_file']}")
            print(f"✅ Completed: {' '.join(controller_cmd['cmd'])}")
        else:
            print(f"❌ Failed (exit {controller_proc.returncode}): {' '.join(controller_cmd['cmd'])} → {log_file}")

    except KeyboardInterrupt:
        print("\n🛑 KeyboardInterrupt: Terminating processes...")
        controller_proc.terminate()
        sys.exit(1)

if __name__ == "__main__":
    run_commands_with_logs()