import os
import sys
import time
import yaml
import json
import redis
import socket
import requests
import subprocess
import numpy as np
from threading import Thread
from datetime import datetime, timezone
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
    configuration_path = os.path.join(script_dir, 'configuration.yml')
    try:
        with open(configuration_path, 'r') as f:
            _config = yaml.safe_load(f)
        print("[INFO] Configuration loaded successfully")
    except FileNotFoundError:
        print(f"[ERROR] configuration.yml not found at {configuration_path}", file=sys.stderr)
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"[ERROR] Error parsing configuration.yml: {e}", file=sys.stderr)
        sys.exit(1)

    return _config

def get_utc_now():
    """
    Returns the current UTC time as a timezone-aware datetime object.
    """
    return datetime.now(timezone.utc)

def init_redis_client():
    """
    Initialize Redis client using environment or fallback config.
    """
    configuration = get_config().get('redis', {})
    host = os.getenv('REDIS_HOSTNAME', configuration.get('hostname', 'localhost'))
    port = int(os.getenv('REDIS_PORT', configuration.get('port', 6379)))

    try:
        return redis.Redis(host=host, port=port, decode_responses=True)
    except redis.exceptions.RedisError as e:
        print(f"[ERROR] Redis initialization failed: {e}", file=sys.stderr)
        sys.exit(1)

def get_redis_client_with_retry(retries=2, delay=2):
    """
    Attempts to connect to Redis with retry logic.
    """
    for attempt in range(retries):
        try:
            redis_client = init_redis_client()
            redis_client.ping()  # Test connection
            print("[INFO] Redis client initialized.")
            return redis_client
        except Exception as e:
            print(f"[ERROR] Redis connection failed (Attempt {attempt + 1}/{retries}): {e}")
            time.sleep(delay)
            port_forward("redis", "redis-master", 6379, 6379)
    print("[FATAL] Could not connect to Redis after multiple attempts.")
    sys.exit(1)

def generate_matrix(size):
    """Generate a square matrix of random floats."""
    return np.random.rand(size, size).tolist()

def clear_queues(redis_client=None, queue_names=None):
    """Clear specified Redis queues."""
    configuration = get_config()
    if queue_names is None:
        queue_names = [
            configuration.get("input_queue_name"),
            configuration.get("worker_queue_name"),
            configuration.get("result_queue_name"),
            configuration.get("output_queue_name"),
            configuration.get("emitter_control_syn_queue_name"),
            configuration.get("worker_control_syn_queue_name"),
            configuration.get("worker_control_ack_queue_name"),
            configuration.get("collector_control_syn_queue_name"),
            configuration.get("emitter_start_queue_name"),
            configuration.get("worker_start_queue_name"),
            configuration.get("collector_start_queue_name")
        ]

    redis_client = redis_client or get_redis_client_with_retry()
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
        time.sleep(5)  # Wait before retrying
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
        requests.post(url, json=payload, headers=headers)
        print(f"[INFO] Async invoked '{function_name}'")
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Async invocation failed: {e}", file=sys.stderr)


def safe_invoke_function_async(function_name, payload, redis_client, queue_name, replicas=1, gateway_url="http://127.0.0.1:8080", timeout=10, retries=1):
    attempt = 0
    try:
        qlen_pre = redis_client.llen(queue_name)
        print(f"[INFO] Pre-invocation queue length for '{queue_name}': {qlen_pre}")
    except redis.exceptions.ConnectionError as e:
        print(f"[ERROR] Redis connection error: {e}. Retrying...", file=sys.stderr)
        redis_client = get_redis_client_with_retry()
        try:
            qlen_pre = redis_client.llen(queue_name)
            print(f"[INFO] Pre-invocation queue length for '{queue_name}': {qlen_pre}")
        except Exception as e:
            print(f"[CRITICAL] Failed to get pre-invocation queue length: {e}", file=sys.stderr)
            sys.exit(1)
    while attempt <= retries:
        try:
            for _ in range(replicas):
                invoke_function_async(function_name, payload, gateway_url)
                time.sleep(0.1)

            time.sleep(timeout)
            qlen = redis_client.llen(queue_name)
            if qlen == qlen_pre+replicas:
                print(f"[SUCCESS] {qlen}/{qlen_pre+replicas} function invoked.")
                return
            else:
                print(f"[WARNING] Only {qlen}/{qlen_pre+replicas} functions invoked. Retrying…")
        except Exception as e:
            print(f"[ERROR] Invocation error: {e}", file=sys.stderr)
            print("[INFO] Attempting to port-forward to recover...")
            port_forward("openfaas", "gateway", 8080, 8080)
            time.sleep(1)  # Wait for port-forward to stabilize
            port_forward("redis", "redis-master", 6379, 6379)
            time.sleep(1)  # Wait for Redis port-forward to stabilize
            redis_client = get_redis_client_with_retry()  # Reinitialize Redis client
        attempt += 1
        print(f"[INFO] Retrying invocation ({attempt}/{retries})...")
        time.sleep(timeout)

    print(f"[FAILURE] Failed to invoke '{function_name}' after {retries+1} attempts.", file=sys.stderr)

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

def send_control_messages(message, redis_client, control_syn_q, count):
    """Send control messages to Redis SYN queue."""
    for _ in range(count):
        try:
            redis_client.lpush(control_syn_q, json.dumps(message))
            print(f"[INFO] Sent control message to '{control_syn_q}'")
        except redis.exceptions.ConnectionError as e:
            print(f"[ERROR] Redis connection error: {e}. Retrying...", file=sys.stderr)
            time.sleep(1)
            try:
                redis_client = get_redis_client_with_retry()
                redis_client.lpush(control_syn_q, json.dumps(message))
                print(f"[INFO] Retry succeeded: control message sent.")
            except Exception as e:
                print(f"[CRITICAL] Retry failed: {e}", file=sys.stderr)

def delete_pod_by_name(pod_name, namespace="openfaas-fn", core_v1_api=None):
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
        core_v1_api.delete_namespaced_pod(
            name=pod_name,
            namespace=namespace,
            body=client.V1DeleteOptions()
        )
        print(f"[INFO] Deleted pod '{pod_name}'")
    except client.rest.ApiException as e:
        print(f"[ERROR] Failed to delete pod '{pod_name}': {e}", file=sys.stderr)

def get_function_pod_names(function_name, core_v1_api=None, namespace="openfaas-fn"):
    """Get pod names for a specific OpenFaaS function."""
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
    
    label_selector = f"faas_function={function_name}"
    pods = core_v1_api.list_namespaced_pod(namespace=namespace, label_selector=label_selector)
    return [pod.metadata.name for pod in pods.items]

def init_pipeline(redis_client, configuration, payload, function_invoke_timeout, function_invoke_retries, apps_v1_api=None):
    """
    Initialize the pipeline by clearing queues and scaling deployments.
    """
    print("[INFO] Initializing pipeline...")
    if not apps_v1_api:
        try:
            config.load_incluster_config()
        except:
            try:
                config.load_kube_config()
            except Exception as e:
                print(f"[ERROR] Kubernetes config load failed: {e}", file=sys.stderr)
                sys.exit(1)
        apps_v1_api = client.AppsV1Api()
    # Scale worker deployment to 1 replica
    scale_function_deployment(1, apps_v1_api=apps_v1_api, deployment_name="emitter", namespace="openfaas-fn")
    scale_function_deployment(1, apps_v1_api=apps_v1_api, deployment_name="worker", namespace="openfaas-fn")
    scale_function_deployment(1, apps_v1_api=apps_v1_api, deployment_name="collector", namespace="openfaas-fn")
    time.sleep(1)  # Wait for deployment to stabilize
    scale_queue_worker(3, apps_v1_api=apps_v1_api)
    time.sleep(1)  # Wait for queue worker to stabilize
    queue_worker_replicas = get_queue_worker_replicas(apps_v1_api=apps_v1_api)
    print(f"[INFO] Queue Worker Replicas for pipeline: {queue_worker_replicas}")
    time.sleep(1)  # Wait for queue worker to stabilize

    safe_invoke_function_async("emitter", payload, redis_client, configuration.get("emitter_start_queue_name"), replicas=1, timeout=function_invoke_timeout, retries=function_invoke_retries)
    safe_invoke_function_async("worker", payload, redis_client, configuration.get("worker_start_queue_name"), replicas=1, timeout=function_invoke_timeout, retries=function_invoke_retries)
    safe_invoke_function_async("collector", payload, redis_client, configuration.get("collector_start_queue_name"), replicas=1,timeout=function_invoke_timeout, retries=function_invoke_retries)

def init_farm(redis_client, configuration, replicas, payload, function_invoke_timeout, function_invoke_retries, apps_v1_api=None):
    """
    Initialize the farm by clearing queues and scaling deployments.
    """
    print("[INFO] Initializing farm...")
    print(f"[INFO] Scaling to {replicas} worker replicas...")
    if not configuration:
        configuration = get_config()
    if not redis_client:
        redis_client = get_redis_client_with_retry()
    if not apps_v1_api:
        try:
            config.load_incluster_config()
        except:
            try:
                config.load_kube_config()
            except Exception as e:
                print(f"[ERROR] Kubernetes config load failed: {e}", file=sys.stderr)
                sys.exit(1)
        apps_v1_api = client.AppsV1Api()
    # Scale worker deployment to replicas
    scale_function_deployment(1, apps_v1_api=apps_v1_api, deployment_name="emitter", namespace="openfaas-fn")
    scale_function_deployment(replicas, apps_v1_api=apps_v1_api, deployment_name="worker", namespace="openfaas-fn")
    scale_function_deployment(1, apps_v1_api=apps_v1_api, deployment_name="collector", namespace="openfaas-fn")
    time.sleep(1)  # Wait for deployment to stabilize
    scale_queue_worker(replicas+2, apps_v1_api=apps_v1_api)
    time.sleep(1)  # Wait for queue worker to stabilize
    queue_worker_replicas = get_queue_worker_replicas(apps_v1_api=apps_v1_api)
    print(f"[INFO] Queue Worker Replicas for farm: {queue_worker_replicas}")

    safe_invoke_function_async("emitter", payload, redis_client, configuration.get("emitter_start_queue_name"), replicas=1, timeout=function_invoke_timeout, retries=function_invoke_retries)
    safe_invoke_function_async("worker", payload, redis_client, configuration.get("worker_start_queue_name"), replicas=replicas, timeout=function_invoke_timeout, retries=function_invoke_retries)
    safe_invoke_function_async("collector", payload, redis_client, configuration.get("collector_start_queue_name"), replicas=1, timeout=function_invoke_timeout, retries=function_invoke_retries)

def scale_deployments_to_zero_replica(apps_v1_api=None):
    """
    Scales down all deployments to zero replicas.
    This is useful for resetting the environment.
    """
    for fn in ["emitter", "worker", "collector"]:
        scale_function_deployment(0, apps_v1_api=apps_v1_api, deployment_name=fn, namespace="openfaas-fn")
        time.sleep(1)  # Allow time for scaling down
        print(f"[INFO] Scaled down {fn} function deployment to 0 replicas.")
    scale_queue_worker(0, apps_v1_api=apps_v1_api)
    time.sleep(1)  # Allow time for queue worker to scale down
    print("[INFO] Scaled down queue-worker deployment to 0 replicas.")


def clear_all_queues(redis_client):
    """
    Clears all Redis queues before generating tasks.

    Args:
        redis_client (redis.Redis): Redis client instance.
    """
    clear_queues(redis_client, None)
    print("[INFO] Cleared all Redis queues.")

def restart_resources(core_v1_api, program_start_time, namespace = "openfaas"):
    """ Restarts all resources in the specified namespace.
    Deletes all pods, deployments, services, and configmaps.
    """
    # Delete pods by label
    labels = [
        ("app", "gateway"),
        ("app", "nats"),
        ("app", "queue-worker"),
        ("app", "prometheus"),
        ("faas_function", "emitter"),
        ("faas_function", "worker"),
        ("faas_function", "collector"),
    ]
    print(f"[TIMER] Restarting Redis master pod at {(get_utc_now() - program_start_time).total_seconds():.4f}...")
    delete_pod_by_name("redis-master-0", "redis", core_v1_api=core_v1_api)
    print(f"[INFO] Restarting pods in namespace '{namespace}'...")
    for key, val in labels:
        pods = core_v1_api.list_namespaced_pod(namespace=namespace, label_selector=f"{key}={val}")
        for pod in pods.items:
            delete_pod_by_name(pod.metadata.name, namespace=namespace, core_v1_api=core_v1_api)
            time.sleep(1)  # Allow time for pod deletion
            print(f"[INFO] Deleted pod: {pod.metadata.name} ({key}={val})")

def is_port_in_use(port):
    """Check if a local TCP port is already in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0

def port_forward(namespace, svc_name, local_port, remote_port, retries=2, delay=2):
    """
    Starts a kubectl port-forward to a service with retry logic.
    Args:
        namespace (str): Kubernetes namespace.
        svc_name (str): Service name to port-forward.
        local_port (int): Local port.
        remote_port (int): Remote port in the cluster.
        retries (int): Number of retry attempts if port-forward fails or is in use.
        delay (int): Delay in seconds between retries.
    """
    attempt = 0
    while attempt < retries:
        print(f"[INFO] Attempting port-forward for {svc_name} ({local_port} -> {remote_port}) in namespace '{namespace}' (Attempt {attempt+1}/{retries})")
        try:
            proc = subprocess.Popen([
                "kubectl", "port-forward", f"svc/{svc_name}",
                f"{local_port}:{remote_port}", "-n", namespace
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            time.sleep(3)

            if is_port_in_use(local_port):
                print(f"[SUCCESS] Port-forward for {svc_name} is active on port {local_port}.")
                return
            else:
                print(f"[WARNING] Port-forward started but port {local_port} is not responding.")

        except Exception as e:
            print(f"[ERROR] Port-forward failed on attempt {attempt+1}: {e}", file=sys.stderr)

        attempt += 1
        time.sleep(delay)

    print(f"[FAILURE] Could not establish port-forward for {svc_name} after {retries} attempts.", file=sys.stderr)

def initialize_environment(program_start_time):
    """
    Orchestrates environment initialization.
    """
    try:
        config.load_incluster_config()
    except:
        try:
            config.load_kube_config()
        except Exception as e:
            print(f"[ERROR] Kubernetes config load failed: {e}", file=sys.stderr)
            sys.exit(1)
    core_v1_api = client.CoreV1Api()
    apps_v1_api = client.AppsV1Api()

    print(f"[TIMER] Starting environment initialization at {(get_utc_now() - program_start_time).total_seconds():.4f}...")
    scale_deployments_to_zero_replica(apps_v1_api=apps_v1_api)
    time.sleep(10)
    # print(f"[TIMER] Restarting resources at {(get_utc_now() - program_start_time).total_seconds():.4f}...")
    # restart_resources(program_start_time, core_v1_api=core_v1_api)
    # time.sleep(180)
    print(f"[TIMER] Port-forwarding services at {(get_utc_now() - program_start_time).total_seconds():.4f}...")
    port_forward("openfaas", "gateway", 8080, 8080)
    port_forward("redis", "redis-master", 6379, 6379)
    time.sleep(2)
    print(f"[TIMER] Initializing Redis client at {(get_utc_now() - program_start_time).total_seconds():.4f}...")
    redis_client = get_redis_client_with_retry()
    clear_all_queues(redis_client)
    print(f"[TIMER] Environment initialization completed at {(get_utc_now() - program_start_time).total_seconds():.4f}.")
    return get_config(), redis_client

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

def monitor_worker_replicas(apps_v1_api=None):
    """
    Monitors the current number of worker replicas.
    """
    try:
        replicas = get_current_worker_replicas(apps_v1_api=apps_v1_api)
        print(f"\n[WORKER STATUS]")
        print(f"  Current worker replicas: {replicas}")
        return replicas
    except Exception as e:
        print(f"[WARNING] Could not retrieve worker replicas: {e}")

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