import subprocess
import os
import argparse
from datetime import datetime
import sys
import time

def parse_worker_count():
    """Parse --workers argument from workflow_controller.py command"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, required=True)
    args, _ = parser.parse_known_args()
    return args.workers

def run_worker_scaler_twice(log_file):
    """Run worker_scaler.py twice, appending to same log file"""
    with open(log_file, 'a') as f:  # 'a' mode for append
        for run in [1, 2]:
            print(f"Running worker_scaler.py (run {run}) → {log_file}")
            cmd = ["python", "worker_scaler.py", "-2"]
            process = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True
            )
            process.wait()
            if process.returncode == 0:
                print(f"✅ Completed worker_scaler.py run {run}")
            else:
                print(f"❌ Failed worker_scaler.py run {run} (exit {process.returncode})")

def run_commands_with_logs():
    worker_count = parse_worker_count()
    worker_suffix = f"w{worker_count}"
    today = datetime.now().strftime("%m%d")
    log_dir = f"logs_farm/{today}"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%H%M")

    # Define all commands except worker_scaler
    commands = [
        {
            "cmd": sys.argv[1:],
            "log_file": f"{log_dir}/logs_controller_{worker_suffix}_{timestamp}.txt"
        },
        {
            "cmd": ["kubectl", "logs", "-n", "openfaas", "deploy/nats"],
            "log_file": f"{log_dir}/logs_nats_{worker_suffix}_{timestamp}.txt"
        },
        {
            "cmd": ["kubectl", "logs", "-n", "openfaas", "-l", "app=gateway", "-f"],
            "log_file": f"{log_dir}/logs_gateway_{worker_suffix}_{timestamp}.txt"
        },
        {
            "cmd": ["kubectl", "logs", "-n", "openfaas", "-l", "app=queue-worker", "-f"],
            "log_file": f"{log_dir}/logs_queue-worker_{worker_suffix}_{timestamp}.txt"
        },
        {
            "cmd": ["kubectl", "logs", "-n", "openfaas-fn", "-l", "faas_function=collector", "-f"],
            "log_file": f"{log_dir}/logs_collector_{worker_suffix}_{timestamp}.txt"
        },
        {
            "cmd": ["kubectl", "logs", "-n", "openfaas-fn", "-l", "faas_function=worker", "-f"],
            "log_file": f"{log_dir}/logs_worker_{worker_suffix}_{timestamp}.txt"
        },
        {
            "cmd": ["kubectl", "logs", "-n", "openfaas-fn", "-l", "faas_function=emitter", "-f"],
            "log_file": f"{log_dir}/logs_emitter_{worker_suffix}_{timestamp}.txt"
        }
    ]

    processes = []
    run_worker_scale_down_enabled = False # True
    scaler_log = f"{log_dir}/logs_scaler_{worker_suffix}_{timestamp}.txt"

    try:
        print(f"Log directory: {log_dir}")
        print(f"Worker suffix: {worker_suffix}")

        # Start all regular commands
        for cmd_info in commands:
            with open(cmd_info["log_file"], "w") as log_file:
                process = subprocess.Popen(
                    cmd_info["cmd"],
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True
                )
                time.sleep(5)  # Ensure log file is created before appending
                processes.append((process, cmd_info["cmd"], cmd_info["log_file"]))
                print(f"Started: {' '.join(cmd_info['cmd'])} → {cmd_info['log_file']}")

        # Run worker_scaler twice sequentially
        if run_worker_scale_down_enabled:
            run_worker_scaler_twice(scaler_log)

        # Wait for other processes
        for process, cmd, log_file in processes:
            process.wait()
            if process.returncode == 0:
                print(f"✅ Completed: {' '.join(cmd)}")
            else:
                print(f"❌ Failed (exit {process.returncode}): {' '.join(cmd)} → {log_file}")

    except KeyboardInterrupt:
        print("\n🛑 KeyboardInterrupt: Terminating processes...")
        for process, _, _ in processes:
            process.terminate()
        sys.exit(1)

if __name__ == "__main__":
    run_commands_with_logs()