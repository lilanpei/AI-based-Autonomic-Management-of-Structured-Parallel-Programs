import os
import sys
import yaml
import json
import time
import redis
import requests
import threading
import subprocess
import numpy as np
from kubernetes import client, config
from kubernetes.client.rest import ApiException

_config = None # Private variable to hold loaded configuration

def get_config():
    """Loads configuration from configuration.yml."""
    global _config
    if _config is None:
        try:
            # Assuming configuration.yml is in the same directory as utilities.py
            script_dir = os.path.dirname(__file__)
            config_path = os.path.join(script_dir, 'configuration.yml')
            with open(config_path, 'r') as f:
                _config = yaml.safe_load(f)
            print("Configuration loaded successfully from configuration.yml")
        except FileNotFoundError:
            print(f"ERROR: configuration.yml not found at {config_path}. Please ensure it's in the same directory as utilities.py.", file=sys.stderr)
            sys.exit(1)
        except yaml.YAMLError as e:
            print(f"ERROR: Error parsing configuration.yml: {e}", file=sys.stderr)
            sys.exit(1)
    return _config


def init_redis_client():
    """
    Initializes and returns a synchronous Redis client using configuration.
    It reads hostname and port from environment variables (REDIS_HOSTNAME, REDIS_PORT)
    or falls back to configuration.yml settings. Exits if connection fails.
    """
    config = get_config()
    redis_settings = config.get('redis', {}) # Get the 'redis' section from config.yml
    
    # Get hostname, prioritizing REDIS_HOSTNAME env var, then config's hostname, then default 'localhost'
    redis_hostname = os.getenv('REDIS_HOSTNAME', redis_settings.get('hostname', 'localhost'))
    
    # Get port, prioritizing REDIS_PORT env var, then config's port, then default 6379.
    # Convert port from config to string for os.getenv default, then to int for redis.Redis.
    redis_port = int(os.getenv('REDIS_PORT', str(redis_settings.get('port', 6379))))

    return redis.Redis(
        host=redis_hostname,
        port=redis_port,
        decode_responses=True
    )

def generate_matrix(size):
    """
    Generates a square matrix of given size with random float values.
    Returns the matrix as a nested Python list (suitable for JSON serialization).
    """
    return np.random.rand(size, size).tolist()

def clear_queues(self, redis_client, queue_names=None):
    """Clears specified Redis queues and the results set."""
    config = get_config()

    if queue_names is None:
        # Clear all relevant queues if no specific names are given
        queue_names = [
            config["input_queue_name"],
            config["worker_queue_name"],
            config["result_queue_name"],
            config["output_queue_name"]
        ]

    if redis_client is None:
        try:
            redis_client = init_redis_client()
            for q_name in queue_names:
                count = redis_client.delete(q_name)
                print(f"Cleared Redis Queue '{q_name}'. Removed {count} items.")
            print("All specified queues and sets cleared.")
        except redis.exceptions.ConnectionError as e:
            print(f"Redis connection error: {str(e)}. Attempting to reinitialize.")
            time.sleep(5)
            try:
                redis_client = init_redis_client() # Reinitialize blocking client
                for q_name in queue_names:
                    count = redis_client.delete(q_name)
                    print(f"Cleared Redis Queue '{q_name}'. Removed {count} items.")
                print("All specified queues and sets cleared.")
            except Exception as init_e:
                print(f"CRITICAL ERROR: Redis reinit failed: {init_e}", file=sys.stderr)
                return {"statusCode": 500, "body": f"Redis failure: {init_e}"}


def scale_function_deployment(replica_count, deployment_name="worker", namespace="openfaas-fn"):
    """
    Scales the Kubernetes deployment to the desired number of replicas.
    """
    try:
        config.load_incluster_config()
    except:
        try:
            config.load_kube_config()  # fallback to local config for testing
        except Exception as e:
            print(f"ERROR: Could not load Kubernetes config: {e}", file=sys.stderr)
            sys.exit(1)

    try:
        api_instance = client.AppsV1Api()
        body = {
            "spec": {
                "replicas": replica_count
            }
        }
        api_instance.patch_namespaced_deployment_scale(
            name=deployment_name,
            namespace=namespace,
            body=body,
        )
        print(f"[INFO] Successfully scaled deployment '{deployment_name}' to {replica_count} replicas.")
    except ApiException as e:
        print(f"ERROR: Failed to scale deployment '{deployment_name}': {e}", file=sys.stderr)
        time.sleep(5)
        try:
            api_instance.patch_namespaced_deployment_scale(
                name=deployment_name,
                namespace=namespace,
                body=body,
            )
            print(f"[INFO] Retry succeeded: deployment scaled.")
        except ApiException as retry_e:
            print(f"CRITICAL ERROR: Retry scaling deployment failed: {retry_e}", file=sys.stderr)
            return {"statusCode": 500, "body": f"Deployment scaling failure: {retry_e}"}


def invoke_function_async(function_name, payload, gateway_url="http://127.0.0.1:8080", timeout=125):
    """
    Asynchronously invokes an OpenFaaS function via /async-function/<function_name>.

    Args:
        function_name (str): Name of the OpenFaaS function (e.g., "emitter", "collector", "worker")
        payload (dict): JSON-serializable data to send
        gateway_url (str): OpenFaaS gateway URL
        timeout (int): Request timeout in seconds
    """
    url = f"{gateway_url}/async-function/{function_name}"
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(url, data=json.dumps(payload), headers=headers, timeout=timeout)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Failed to asynchronously invoke '{function_name}': {e}", file=sys.stderr)
        return

    print(f"[INFO] Async invoked '{function_name}' function")
    print(f"Status Code: {response.status_code}")
    print("Response Body:", response.text)

def restart_function(function_name):
    try:
        subprocess.run(
            ["kubectl", "rollout", "restart", f"deploy/{function_name}", "-n", "openfaas-fn"],
            check=True
        )
        print(f"[INFO] Restarted function: {function_name}")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to restart function '{function_name}': {e}")

def get_current_worker_replicas(namespace="openfaas-fn", deployment_name="worker"):
    config.load_kube_config()
    apps_v1 = client.AppsV1Api()
    deployment = apps_v1.read_namespaced_deployment(deployment_name, namespace)
    return deployment.spec.replicas

def send_control_requests(start_flag, count):
    config_data = get_config()
    worker_q = config_data["worker_queue_name"]
    result_q = config_data["result_queue_name"]

    for _ in range(count):
        payload = {
            "start_flag": start_flag,
            "worker_queue_name": worker_q,
            "result_queue_name": result_q
        }
        # simulate OpenFaaS function trigger
        invoke_function_async("worker", payload)
        print(f"[INFO] Sent control request with start_flag={start_flag}")
