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
            print(f"[INFO] Cleared Redis Queue '{name}' ({count} items removed).")
        except redis.exceptions.ConnectionError as e:
            print(f"[ERROR] Redis connection failed: {e}", file=sys.stderr)

def scale_function_deployment(replica_count, deployment_name="worker", namespace="openfaas-fn"):
    """Scale a Kubernetes deployment."""
    try:
        config.load_incluster_config()
    except:
        try:
            config.load_kube_config()
        except Exception as e:
            print(f"[ERROR] Kubernetes config load failed: {e}", file=sys.stderr)
            sys.exit(1)

    api = client.AppsV1Api()
    body = {"spec": {"replicas": replica_count}}

    try:
        api.patch_namespaced_deployment_scale(name=deployment_name, namespace=namespace, body=body)
        print(f"[INFO] Scaled '{deployment_name}' to {replica_count} replicas.")
    except ApiException as e:
        print(f"[ERROR] Initial scaling failed: {e}", file=sys.stderr)
        time.sleep(5)
        try:
            api.patch_namespaced_deployment_scale(name=deployment_name, namespace=namespace, body=body)
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

def get_current_worker_replicas(namespace="openfaas-fn", deployment_name="worker"):
    """Return current number of worker replicas."""
    config.load_kube_config()
    apps_v1 = client.AppsV1Api()
    deployment = apps_v1.read_namespaced_deployment(deployment_name, namespace)
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
            time.sleep(5)
            try:
                redis_client = init_redis_client()
                redis_client.lpush(control_syn_q, json.dumps(message))
                print(f"[INFO] Retry succeeded: control message sent.")
            except Exception as e:
                print(f"[CRITICAL] Retry failed: {e}", file=sys.stderr)

def delete_pod_by_name(pod_name, namespace="openfaas-fn"):
    """Delete a pod by name."""
    config.load_kube_config()
    v1 = client.CoreV1Api()
    try:
        v1.delete_namespaced_pod(name=pod_name, namespace=namespace, body=client.V1DeleteOptions())
        print(f"[INFO] Deleted pod '{pod_name}'")
    except ApiException as e:
        print(f"[ERROR] Failed to delete pod '{pod_name}': {e}", file=sys.stderr)

def get_worker_pod_names(namespace="openfaas-fn", label_selector="faas_function=worker"):
    """Get a list of worker pod names."""
    config.load_kube_config()
    v1 = client.CoreV1Api()
    pods = v1.list_namespaced_pod(namespace=namespace, label_selector=label_selector)
    return [pod.metadata.name for pod in pods.items]
