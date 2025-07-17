import os
import sys
import time
import yaml
import json
import redis
import requests
import subprocess
import numpy as np
from threading import Thread
from kubernetes import client, config
from kubernetes.client.rest import ApiException

# Module-level cache for configuration
_config = None

def get_config():
    """Load configuration from configuration.yml once."""
    global _config
    if _config is not None:
        return _config

    script_dir = os.path.dirname(__file__)
    config_path = os.path.join(script_dir, 'configuration.yml')
    try:
        with open(config_path, 'r') as f:
            _config = yaml.safe_load(f)
        print("[INFO] Configuration loaded successfully")
    except FileNotFoundError:
        print(f"[ERROR] configuration.yml not found at {config_path}", file=sys.stderr)
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"[ERROR] Error parsing configuration.yml: {e}", file=sys.stderr)
        sys.exit(1)

    return _config

def init_redis_client():
    """
    Initialize Redis client using environment or fallback config.
    """
    config = get_config().get('redis', {})
    host = os.getenv('REDIS_HOSTNAME', config.get('hostname', 'localhost'))
    port = int(os.getenv('REDIS_PORT', config.get('port', 6379)))

    try:
        return redis.Redis(host=host, port=port, decode_responses=True)
    except redis.exceptions.RedisError as e:
        print(f"[ERROR] Redis initialization failed: {e}", file=sys.stderr)
        sys.exit(1)

def get_redis_client_with_retry(retries=3, delay=5):
    """
    Attempts to connect to Redis with retry logic.
    """
    for attempt in range(retries):
        try:
            return init_redis_client()
        except Exception as e:
            print(f"[ERROR] Redis connection failed (Attempt {attempt + 1}/{retries}): {e}")
            time.sleep(delay)
    print("[FATAL] Could not connect to Redis after multiple attempts.")
    sys.exit(1)

def generate_matrix(size):
    """Generate a square matrix of random floats."""
    return np.random.rand(size, size).tolist()

def clear_queues(redis_client=None, queue_names=None):
    """Clear specified Redis queues."""
    config = get_config()
    if queue_names is None:
        queue_names = [
            config.get("input_queue_name"),
            config.get("worker_queue_name"),
            config.get("result_queue_name"),
            config.get("output_queue_name"),
            config.get("control_syn_queue_name"),
            config.get("control_ack_queue_name")
        ]

    redis_client = redis_client or init_redis_client()
    for name in queue_names:
        try:
            count = redis_client.delete(name)
            time.sleep(0.1)  # Allow time for Redis to process
            print(f"[INFO] Cleared Redis Queue '{name}' ({count} items removed).")
        except redis.exceptions.ConnectionError as e:
            print(f"[ERROR] Redis connection failed: {e}", file=sys.stderr)

def scale_function_deployment(replica_count, apps_v1_api=None, deployment_name="worker", namespace="openfaas-fn"):
    """Scale a Kubernetes deployment."""
    if apps_v1_api is None:
        try:
            config.load_incluster_config()
        except:
            try:
                config.load_kube_config()
            except Exception as e:
                print(f"[ERROR] Kubernetes config load failed: {e}", file=sys.stderr)
                sys.exit(1)
        apps_v1_api = client.AppsV1Api()
    body = {"spec": {"replicas": replica_count}}

    try:
        apps_v1_api.patch_namespaced_deployment_scale(name=deployment_name, namespace=namespace, body=body)
        print(f"[INFO] Scaled '{deployment_name}' to {replica_count} replicas.")
    except ApiException as e:
        print(f"[ERROR] Initial scaling failed: {e}", file=sys.stderr)
        # time.sleep(5)
        try:
            apps_v1_api.patch_namespaced_deployment_scale(name=deployment_name, namespace=namespace, body=body)
            print(f"[INFO] Retry succeeded: scaled deployment.")
        except ApiException as e:
            print(f"[CRITICAL] Retry failed: {e}", file=sys.stderr)

def invoke_function_async(function_name, payload, gateway_url="http://127.0.0.1:8080"):
    """Asynchronously invoke OpenFaaS function."""
    url = f"{gateway_url}/async-function/{function_name}"
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        print(f"[INFO] Async invoked '{function_name}'")
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Async invocation failed: {e}", file=sys.stderr)

def async_function(func, *args, **kwargs):
    """
    Runs any function asynchronously with given arguments.

    Args:
        func (callable): The function to execute.
        *args: Positional arguments for the function.
        **kwargs: Keyword arguments for the function.
    """
    thread = Thread(target=func, args=args, kwargs=kwargs)
    # thread.daemon = True  # Optional: thread dies with the main program
    thread.start()

def restart_function(function_name):
    """Restart a Kubernetes function deployment."""
    try:
        subprocess.run(
            ["kubectl", "rollout", "restart", f"deploy/{function_name}", "-n", "openfaas-fn"],
            check=True
        )
        print(f"[INFO] Restarted function '{function_name}'")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Restart failed: {e}", file=sys.stderr)

def get_current_worker_replicas(apps_v1_api=None, namespace="openfaas-fn", deployment_name="worker"):
    """Return current number of worker replicas."""
    if apps_v1_api is None:
        try:
            config.load_incluster_config()
        except:
            try:
                config.load_kube_config()
            except Exception as e:
                print(f"[ERROR] Kubernetes config load failed: {e}", file=sys.stderr)
                sys.exit(1)
        apps_v1_api = client.AppsV1Api()
    deployment = apps_v1_api.read_namespaced_deployment(deployment_name, namespace)
    return deployment.spec.replicas

def send_control_messages(redis_client, control_syn_q, count):
    """Send control messages to Redis SYN queue."""
    message = {
        "type": "SCALE_DOWN",
        "action": "SYN",
        "message": "Scale down request from orchestrator",
        "SYN_timestamp": time.time()
    }

    for _ in range(count):
        try:
            redis_client.lpush(control_syn_q, json.dumps(message))
            print(f"[INFO] Sent control message to '{control_syn_q}'")
        except redis.exceptions.ConnectionError as e:
            print(f"[ERROR] Redis connection error: {e}. Retrying...", file=sys.stderr)
            # time.sleep(5)
            try:
                redis_client = init_redis_client()
                redis_client.lpush(control_syn_q, json.dumps(message))
                print(f"[INFO] Retry succeeded: control message sent.")
            except Exception as e:
                print(f"[CRITICAL] Retry failed: {e}", file=sys.stderr)

def delete_pod_by_name(pod_name, core_v1_api=None, namespace="openfaas-fn"):
    """Delete a pod by name."""
    if core_v1_api is None:
        try:
            config.load_incluster_config()
        except:
            try:
                config.load_kube_config()
            except Exception as e:
                print(f"[ERROR] Kubernetes config load failed: {e}", file=sys.stderr)
                sys.exit(1)
        core_v1_api = client.CoreV1Api()
    try:
        core_v1_api.delete_namespaced_pod(name=pod_name, namespace=namespace, body=client.V1DeleteOptions())
        print(f"[INFO] Deleted pod '{pod_name}'")
    except ApiException as e:
        print(f"[ERROR] Failed to delete pod '{pod_name}': {e}", file=sys.stderr)

def get_worker_pod_names(core_v1_api=None, namespace="openfaas-fn", label_selector="faas_function=worker"):
    """Get a list of worker pod names."""
    if core_v1_api is None:
        try:
            config.load_incluster_config()
        except:
            try:
                config.load_kube_config()
            except Exception as e:
                print(f"[ERROR] Kubernetes config load failed: {e}", file=sys.stderr)
                sys.exit(1)
        core_v1_api = client.CoreV1Api()
    pods = core_v1_api.list_namespaced_pod(namespace=namespace, label_selector=label_selector)
    return [pod.metadata.name for pod in pods.items]

def init_pipeline(feedback_flag, config=None):
    """
    Initialize the pipeline by clearing queues and scaling deployments.
    """
    print("[INFO] Initializing pipeline...")
    print(f"[INFO] Feedback flag: {feedback_flag}")

    if not config:
        config = get_config()

    # Scale worker deployment to 1 replica
    scale_function_deployment(1, apps_v1_api=None, deployment_name="worker", namespace="openfaas-fn")
    time.sleep(5)  # Wait for deployment to stabilize
    scale_queue_worker(3)
    queue_worker_replicas = get_queue_worker_replicas()
    print(f"[INFO] Queue Worker Replicas for pipeline: {queue_worker_replicas}")
    time.sleep(5)  # Wait for queue worker to stabilize
    payload = {
        "input_queue_name": config["input_queue_name"],
        "worker_queue_name": config["worker_queue_name"],
        "result_queue_name": config["result_queue_name"],
        "output_queue_name": config["output_queue_name"],
        "control_syn_queue_name": config["control_syn_queue_name"],
        "control_ack_queue_name": config["control_ack_queue_name"],
        "collector_feedback_flag": feedback_flag
    }

    invoke_function_async("emitter", payload)
    time.sleep(3)
    invoke_function_async("worker", payload)
    time.sleep(3)
    invoke_function_async("collector", payload)
    time.sleep(3)

def init_farm(replicas, feedback_flag, config=None):
    """
    Initialize the farm by clearing queues and scaling deployments.
    """
    print("[INFO] Initializing farm...")
    print(f"[INFO] Number of replicas: {replicas}, Feedback flag: {feedback_flag}")
    if config is None:
        config = get_config()

    # Scale worker deployment to replicas
    scale_function_deployment(replicas, apps_v1_api=None, deployment_name="worker", namespace="openfaas-fn")
    time.sleep(5)  # Wait for deployment to stabilize
    scale_queue_worker(replicas+2)
    queue_worker_replicas = get_queue_worker_replicas()
    print(f"[INFO] Queue Worker Replicas for farm: {queue_worker_replicas}")
    time.sleep(5)  # Wait for queue worker to stabilize
    payload = {
        "input_queue_name": config["input_queue_name"],
        "worker_queue_name": config["worker_queue_name"],
        "result_queue_name": config["result_queue_name"],
        "output_queue_name": config["output_queue_name"],
        "control_syn_queue_name": config["control_syn_queue_name"],
        "control_ack_queue_name": config["control_ack_queue_name"],
        "collector_feedback_flag": feedback_flag
    }

    invoke_function_async("emitter", payload)
    time.sleep(3)

    for i in range(replicas):
        invoke_function_async("worker", payload)
        time.sleep(3)
    invoke_function_async("collector", payload)
    time.sleep(3)

def restart_all_functions():
    """
    Restarts all the functions (emitter, worker, collector).
    """
    for fn in ["emitter", "worker", "collector"]:
        restart_function(fn)
    print("[INFO] Waiting for function deployments to stabilize...")


def scale_deployments_to_single_replica():
    """
    Scales down the emitter, worker, and collector functions to 1 replica each.
    """
    for fn in ["emitter", "worker", "collector"]:
        scale_function_deployment(1, None, fn, "openfaas-fn")
        print(f"[INFO] Scaled down {fn} function deployment to 1 replica.")


def clear_all_queues(redis_client):
    """
    Clears all Redis queues before generating tasks.

    Args:
        redis_client (redis.Redis): Redis client instance.
    """
    clear_queues(redis_client, None)
    print("[INFO] Cleared all Redis queues.")


def initialize_environment():
    """
    Orchestrates environment initialization:
    - Loads config
    - Connects to Redis
    - Scales deployments
    - Clears queues
    """
    config = get_config()
    time.sleep(1)  # Allow time for config to load
    redis_client = get_redis_client_with_retry()
    time.sleep(1)  # Allow time for Redis connection
    scale_deployments_to_single_replica()
    time.sleep(1)  # Allow time for deployments to stabilize
    scale_queue_worker(1)
    time.sleep(1)  # Allow time for queue worker to stabilize
    clear_all_queues(redis_client)
    time.sleep(1)  # Allow time for queues to clear
    print("[INFO] Environment initialization complete.")
    return config, redis_client

def run_script(script_name, args=[]):
    """
    Asynchronously runs the specified script with the provided arguments.

    Args:
        script_name (str): The script to run.
        args (list): List of arguments to pass to the script.
    """
    cmd = ["python", script_name] + [str(arg) for arg in args]
    print(f"[ASYNC RUNNING] {' '.join(cmd)}")
    subprocess.Popen(cmd)

def monitor_worker_replicas():
    """
    Monitors the current number of worker replicas.
    """
    try:
        replicas = get_current_worker_replicas()
        print(f"\n[WORKER STATUS]")
        print(f"  Current worker replicas: {replicas}")
    except Exception as e:
        print(f"[WARNING] Could not retrieve worker replicas: {e}")
    return replicas

def scale_queue_worker(replicas, apps_v1_api=None, namespace="openfaas"):
    """
    Scales the queue-worker deployment to the specified number of replicas.
    Args:
        replicas (int): Number of replicas to scale to.
        apps_v1_api (client.AppsV1Api): Optional pre-initialized API client.
        namespace (str): Kubernetes namespace where the queue-worker is deployed.
    Returns:
        client.V1Scale: The scale object representing the new state.
    """
    if apps_v1_api is None:
        try:
            config.load_incluster_config()
        except:
            try:
                config.load_kube_config()
            except Exception as e:
                print(f"[ERROR] Kubernetes config load failed: {e}", file=sys.stderr)
                sys.exit(1)
        apps_v1_api = client.AppsV1Api()

    deployment_name = "queue-worker"  # Name of the deployment
    body = {
        "spec": {
            "replicas": replicas
        }
    }

    try:
        response = apps_v1_api.patch_namespaced_deployment_scale(
            name=deployment_name,
            namespace=namespace,
            body=body
        )
        print(f"[INFO] Scaled '{deployment_name}' to {replicas} replicas.")
        return response
    except client.exceptions.ApiException as e:
        print(f"[ERROR] Failed to scale deployment: {e}")
        raise


def get_queue_worker_replicas(apps_v1_api=None, namespace="openfaas"):
    """
    Returns the current number of replicas for the queue-worker deployment.

    Args:
        namespace (str): The Kubernetes namespace where queue-worker is deployed.

    Returns:
        int: Number of desired replicas.
    """
    if apps_v1_api is None:
        try:
            config.load_incluster_config()
        except:
            try:
                config.load_kube_config()
            except Exception as e:
                print(f"[ERROR] Kubernetes config load failed: {e}", file=sys.stderr)
                sys.exit(1)
        apps_v1_api = client.AppsV1Api()

    try:
        deployment = apps_v1_api.read_namespaced_deployment(
            name="queue-worker",
            namespace=namespace
        )
        replicas = deployment.spec.replicas
        print(f"[INFO] queue-worker replicas: {replicas}")
        return replicas
    except client.exceptions.ApiException as e:
        print(f"[ERROR] Could not get deployment: {e}")
        raise