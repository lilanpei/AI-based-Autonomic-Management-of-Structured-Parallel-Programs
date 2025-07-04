import subprocess
import argparse
import sys

def run_script(script_name, args=[]):
    cmd = ["python", script_name] + [str(arg) for arg in args]
    print(f"[RUNNING] {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[ERROR] Script {script_name} failed.")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Workflow Controller for AI Task Processing")
    parser.add_argument('--mode', choices=['pipeline', 'farm'], required=True, help='Select execution mode')
    parser.add_argument('--feedback', type=bool, default=False, help='Enable collector feedback')
    parser.add_argument('--tasks', type=int, default=1000, help='Number of tasks to generate')
    parser.add_argument('--workers', type=int, default=1, help='Number of workers (used only in farm mode)')

    args = parser.parse_args()

    # Common startup
    run_script("env_init.py")
    run_script("emitter_init.py", ["True"])  # Start emitter with start_flag=True

    if args.mode == "pipeline":
        run_script("worker_init.py", ["1", "True"])
        run_script("collector_init.py", ["True", "True" if args.feedback else "False"])
        run_script("task_generator.py", [str(args.tasks)])

    elif args.mode == "farm":
        run_script("worker_init.py", [str(args.workers), "True"])
        run_script("collector_init.py", ["True", "True" if args.feedback else "False"])
        run_script("task_generator.py", [str(args.tasks)])

if __name__ == "__main__":
    main()
