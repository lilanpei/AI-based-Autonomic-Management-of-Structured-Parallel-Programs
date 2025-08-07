import os
import re
import sys
import time
import yaml
import json
import redis
import signal
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

def get_redis_client_with_retry(retries=2, delay=1):
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
            if not is_port_in_use(6379):
                print("[INFO] Port 6379 is not in use, attempting to port-forward...")
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
            configuration.get("collector_start_queue_name"),
            configuration.get("workflow_init_syn_queue_name")
        ]

    redis_client = redis_client or get_redis_client_with_retry()
    for name in queue_names:
        try:
            count = redis_client.delete(name)
            print(f"[INFO] Cleared Redis Queue '{name}' ({count} items removed).")
        except redis.exceptions.ConnectionError as e:
            print(f"[ERROR] Redis connection failed: {e}", file=sys.stderr)

def scale_function_deployment(replica_count, apps_v1_api, deployment_name, namespace="openfaas-fn"):
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
        time.sleep(1)  # Wait before retrying
        try:
            apps_v1_api.patch_namespaced_deployment_scale(name=deployment_name, namespace=namespace, body=body)
            print(f"[INFO] Retry succeeded: scaled deployment.")
        except ApiException as e:
            print(f"[ERROR] Retry failed: {e}", file=sys.stderr)

def get_faas_password():
    """Retrieve the OpenFaaS password from Kubernetes secret and decode it."""
    try:
        # Fetch base64-encoded password from Kubernetes
        result = subprocess.run(
            [
                "kubectl", "get", "secret", "-n", "openfaas", "basic-auth",
                "-o", "jsonpath={.data.basic-auth-password}"
            ],
            capture_output=True, text=True, check=True
        )
        encoded_password = result.stdout.strip()

        if not encoded_password:
            raise ValueError("[ERROR] Retrieved empty password from Kubernetes secret.")

        # Decode the password
        decoded_result = subprocess.run(
            ["base64", "--decode"],
            input=encoded_password,
            capture_output=True, text=True, check=True
        )
        return decoded_result.stdout.strip()

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"[ERROR] Failed to retrieve or decode OpenFaaS password: {e.stderr.strip()}")
    except Exception as e:
        raise RuntimeError(f"[ERROR] Unexpected error in get_faas_password(): {e}")

def login_faas_cli(password, max_attempts=10, delay=1):
    """Login to faas-cli with the provided password using password-stdin."""
    print("[INFO] Logging in to faas-cli...")

    for attempt in range(1, max_attempts + 1):
        proc = subprocess.run(
            ["faas-cli", "login", "-u", "admin", "--password-stdin"],
            input=password,
            env={**os.environ, "OPENFAAS_URL": "http://127.0.0.1:8080"},
            capture_output=True,
            text=True
        )

        if proc.returncode == 0:
            print("[INFO] faas-cli login successful.")
            return

        print(f"[WARNING] faas-cli login failed (attempt {attempt}/{max_attempts})")
        time.sleep(delay)

    raise RuntimeError(f"[ERROR] faas-cli login failed after {max_attempts} attempts: {proc.stderr.strip()}")

def run_faas_cli(args, capture_output=False, check=False):
    """
    Runs a faas-cli command with credentials set via environment variables.

    Args:
        args (list): List of CLI arguments, e.g., ["list", "-q"]
        capture_output (bool): If True, captures stdout/stderr.
        check (bool): If True, raises exception on non-zero exit.

    Returns:
        subprocess.CompletedProcess
    """
    # Retrieve password from Kubernetes secret
    password = get_faas_password()

    # Prepare environment with OpenFaaS credentials
    env = os.environ.copy()
    env["OPENFAAS_URL"] = "http://127.0.0.1:8080"
    env["OPENFAAS_USERNAME"] = "admin"
    env["OPENFAAS_PASSWORD"] = password

    # Construct the command
    cmd = ["faas-cli"] + args
    return subprocess.run(
        cmd,
        capture_output=capture_output,
        text=True,
        env=env,
        check=check
    )

def wait_for_gateway_ready(url="http://127.0.0.1:8080/healthz", timeout=30):
    """
    Waits until the OpenFaaS gateway /healthz is reachable and faas-cli is responsive.
    """
    print("[INFO] Waiting for OpenFaaS gateway health endpoint...")

    for second in range(timeout):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                print(f"[INFO] Gateway /healthz passed at {second}s")
                break
            else:
                print(f"[WARNING] /healthz returned status: {response.status_code}")
        except requests.RequestException as e:
            print(f"[WARNING] Exception during /healthz check: {e}")

        time.sleep(1)
    else:
        raise RuntimeError("[WARNING] Gateway /healthz not ready after timeout")

    # Log in to faas-cli
    print("[INFO] Waiting for faas-cli to become responsive...")
    try:
        password = get_faas_password()
        login_faas_cli(password)
    except RuntimeError as e:
        raise RuntimeError(f"[WARNING] faas-cli login failed: {e}")

    # Confirm faas-cli responds to `list`
    for attempt in range(timeout):
        result = run_faas_cli(["list", "-q"], capture_output=True)
        if result.returncode == 0:
            print(f"[INFO] faas-cli is responsive at {attempt+1}/{timeout} attempts")
            return
        else:
            if result.stderr:
                print(f"[WARNING] faas-cli error: {result.stderr.strip()}")
        time.sleep(1)
    else:
        raise RuntimeError("[WARNING] faas-cli not responsive after timeout")

def invoke_function_async(function_name, payload, gateway_url="http://127.0.0.1:8080", timeout=1, retries=3, backoff=1):
    """Asynchronously invoke OpenFaaS function with retry logic."""
    url = f"{gateway_url}/async-function/{function_name}"
    headers = {"Content-Type": "application/json"}

    for attempt in range(retries):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
            if resp.status_code == 202:
                print(f"[INFO] Async invocation sent for '{function_name}' (attempt {attempt+1})")
                return True
            else:
                print(f"[WARNING] Unexpected status {resp.status_code} for '{function_name}' (attempt {attempt+1})")
        except Exception as e:
            print(f"[ERROR] Invocation error (attempt {attempt+1}): {e}")
        time.sleep(backoff)

    print(f"[ERROR] Failed to invoke '{function_name}' after {retries} attempts.")
    return False

def safe_invoke_function_async(
    deployed_function_names,
    payload,
    redis_client,
    queue_name,
    replicas=1,
    gateway_url="http://127.0.0.1:8080",
    timeout=1,
    retries=2
):
    attempt = 0

    # Check Redis queue before starting
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
            print(f"[ERROR] Failed to get pre-invocation queue length: {e}", file=sys.stderr)
            sys.exit(1)

    # Check port-forwarding recovery
    if not is_port_in_use(8080):
        print("[INFO] Gateway port 8080 not in use. Re-forwarding...")
        port_forward("openfaas", "gateway", 8080, 8080)

    if not is_port_in_use(6379):
        print("[INFO] Redis port 6379 not in use. Re-forwarding...")
        port_forward("redis", "redis-master", 6379, 6379)
    count = 0
    for function_instance_name in deployed_function_names:
        count += 1
        print(f"[INFO] Invoking {function_instance_name} ({count}/{replicas})")
        success = invoke_function_async(
            function_instance_name,
            payload,
            gateway_url=gateway_url,
            timeout=5,
            retries=3,
            backoff=1
        )
        if not success:
            print(f"[WARNING] One invocation failed for '{function_instance_name}'")
        time.sleep(0.1)  # Allow some time for the function to process

    while attempt <= retries:
        time.sleep(timeout)
        # Check how many actually landed in the queue
        try:
            qlen_post = redis_client.llen(queue_name)
            print(f"[INFO] Post-invocation queue length: {qlen_post}")
            expected = qlen_pre + replicas
            if qlen_post == expected:
                print(f"[INFO] [SUCCESS] {qlen_post}/{expected} tasks enqueued.")
                return
            elif qlen_post > expected:
                print(f"[WARNING] More tasks enqueued than expected: {qlen_post}/{expected}.")
            else:
                print(f"[WARNING] Only {qlen_post}/{expected} enqueued. Retrying...")
        except redis.exceptions.ConnectionError:
            print(f"[ERROR] Lost Redis connection. Retrying...")
            redis_client = get_redis_client_with_retry()
            if not is_port_in_use(6379):
                print("[INFO] Redis port 6379 not in use. Re-forwarding...")
                port_forward("redis", "redis-master", 6379, 6379)
        except Exception as e:
            print(f"[ERROR] Unexpected Redis error: {e}")

        attempt += 1
        print(f"[INFO] Retry attempt {attempt}/{retries}")

    print(f"[ERROR] Failed to invoke after {retries + 1} attempts.", file=sys.stderr)
    os._exit(1)

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

def get_deployment_replicas(apps_v1_api=None, namespace="openfaas-fn", name_or_prefix=None, exact_match=False):
    """
    Returns total replicas for deployments matching a name or prefix.

    Args:
        apps_v1_api: Kubernetes AppsV1Api client
        namespace (str): Namespace to search deployments in
        name_or_prefix (str): Full name or prefix to match
        exact_match (bool): If True, matches exact name only. If False, matches prefix.

    Returns:
        int: Total replica count for matched deployments
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
        deployments = apps_v1_api.list_namespaced_deployment(namespace=namespace)
        total_replicas = 0
        for deploy in deployments.items:
            name = deploy.metadata.name
            if (exact_match and name == name_or_prefix) or (not exact_match and name.startswith(name_or_prefix)):
                replicas = deploy.spec.replicas or 0
                total_replicas += replicas
        return total_replicas

    except Exception as e:
        print(f"[ERROR] Failed to retrieve deployments in namespace '{namespace}': {e}")
        return 0

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
                print(f"[ERROR] Retry failed: {e}", file=sys.stderr)

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

def delete_pods_by_label(label_selector, namespace="default", core_v1_api=None):
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
        pods = core_v1_api.list_namespaced_pod(namespace=namespace, label_selector=label_selector).items
        if not pods:
            print(f"[INFO] No pods found with label '{label_selector}' in namespace '{namespace}'.")
            return
        for pod in pods:
            pod_name = pod.metadata.name
            core_v1_api.delete_namespaced_pod(
                name=pod_name,
                namespace=namespace,
                body=client.V1DeleteOptions()
            )
            print(f"[INFO] Deleted pod '{pod_name}' in namespace '{namespace}'.")
    except client.rest.ApiException as e:
        print(f"[ERROR] Failed to delete pod(s) with label '{label_selector}': {e}", file=sys.stderr)

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

def init_pipeline(program_start_time, configuration, payload, apps_v1_api=None, core_v1_api=None):
    """
    Initialize the pipeline by clearing queues and scaling deployments.
    """
    print(f"[TIMER] Pipeline initialization started at [{(get_utc_now() - program_start_time).total_seconds():.4f}] seconds.")
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

    deployed_emitter_names = deploy_function(function_name_prefix="emitter", replicas=1, max_retries=3, delay=1)
    deployed_collector_names = deploy_function(function_name_prefix="collector", replicas=1, max_retries=3, delay=1)
    deployed_worker_names = deploy_function(function_name_prefix="worker", replicas=1, max_retries=3, delay=1)

    scale_function_deployment(3, apps_v1_api=apps_v1_api, deployment_name="queue-worker", namespace="openfaas")
    wait_for_pods_ready(f"app=queue-worker", "openfaas", core_v1_api, program_start_time, 3) # Wait for queue worker to stabilize

    queue_worker_replicas = get_deployment_replicas(apps_v1_api=apps_v1_api, namespace="openfaas", name_or_prefix="queue-worker", exact_match=True)
    print(f"[INFO] Queue Worker Replicas for pipeline: {queue_worker_replicas}")

    print(f"[INFO] Initializing Redis client at {(get_utc_now() - program_start_time).total_seconds():.4f}...")
    redis_client = get_redis_client_with_retry()

    print(f"[INFO] Clearing all queues at {(get_utc_now() - program_start_time).total_seconds():.4f}...")
    clear_queues(redis_client, None)

    print(f"[INFO] Invoking functions at {(get_utc_now() - program_start_time).total_seconds():.4f}...")
    safe_invoke_function_async(deployed_emitter_names, payload, redis_client, configuration.get("emitter_start_queue_name"), replicas=1, timeout=configuration.get("first_function_invoke_timeout"), retries=configuration.get("first_function_invoke_retries"))
    safe_invoke_function_async(deployed_worker_names, payload, redis_client, configuration.get("worker_start_queue_name"), replicas=1, timeout=configuration.get("function_invoke_timeout"), retries=configuration.get("function_invoke_retries"))
    safe_invoke_function_async(deployed_collector_names, payload, redis_client, configuration.get("collector_start_queue_name"), replicas=1, timeout=configuration.get("function_invoke_timeout"), retries=configuration.get("function_invoke_retries"))
    print(f"[TIMER] Pipeline initialization completed at [{(get_utc_now() - program_start_time).total_seconds():.4f}] seconds.")

    # Send workflow init completed messages
    message = {
        "type": "WORKFLOW_INIT",
        "action": "SYN",
        "message": "Workflow initialization completed",
        "SYN_timestamp": (get_utc_now() - program_start_time).total_seconds(),
        "program_start_time": str(program_start_time)
    }
    send_control_messages(message, redis_client, configuration.get("workflow_init_syn_queue_name"), 1)

    return redis_client

def init_farm(program_start_time, configuration, replicas, payload, apps_v1_api=None, core_v1_api=None):
    """
    Initialize the farm by clearing queues and scaling deployments.
    """
    print(f"[TIMER] Farm initializing started at [{(get_utc_now() - program_start_time).total_seconds():.4f}] seconds.")
    if not apps_v1_api or not core_v1_api:
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

    deployed_emitter_names = deploy_function(function_name_prefix="emitter", replicas=1, max_retries=3, delay=1)
    deployed_collector_names = deploy_function(function_name_prefix="collector", replicas=1, max_retries=3, delay=1)
    deployed_worker_names = deploy_function(function_name_prefix="worker", replicas=replicas, max_retries=3, delay=1)

    scale_function_deployment(replicas+2, apps_v1_api=apps_v1_api, deployment_name="queue-worker", namespace="openfaas")
    wait_for_pods_ready(f"app=queue-worker", "openfaas", core_v1_api, program_start_time, replicas+2)  # Wait for queue worker to stabilize

    queue_worker_replicas = get_deployment_replicas(apps_v1_api=apps_v1_api, namespace="openfaas", name_or_prefix="queue-worker", exact_match=True)
    print(f"[INFO] Queue Worker Replicas for farm: {queue_worker_replicas}")

    print(f"[INFO] Initializing Redis client at {(get_utc_now() - program_start_time).total_seconds():.4f}...")
    redis_client = get_redis_client_with_retry()

    print(f"[INFO] Clearing all queues at {(get_utc_now() - program_start_time).total_seconds():.4f}...")
    clear_queues(redis_client, None)

    print(f"[INFO] Invoking functions at {(get_utc_now() - program_start_time).total_seconds():.4f}...")
    safe_invoke_function_async(deployed_emitter_names, payload, redis_client, configuration.get("emitter_start_queue_name"), replicas=1, timeout=configuration.get("first_function_invoke_timeout"), retries=configuration.get("first_function_invoke_retries"))
    safe_invoke_function_async(deployed_collector_names, payload, redis_client, configuration.get("collector_start_queue_name"), replicas=1, timeout=configuration.get("function_invoke_timeout"), retries=configuration.get("function_invoke_retries"))
    safe_invoke_function_async(deployed_worker_names, payload, redis_client, configuration.get("worker_start_queue_name"), replicas=replicas, timeout=configuration.get("function_invoke_timeout"), retries=configuration.get("function_invoke_retries"))
    print(f"[TIMER] Farm initialization completed at [{(get_utc_now() - program_start_time).total_seconds():.4f}] seconds.")

    # Send workflow init completed messages
    message = {
        "type": "WORKFLOW_INIT",
        "action": "SYN",
        "message": "Workflow initialization completed",
        "SYN_timestamp": (get_utc_now() - program_start_time).total_seconds(),
        "program_start_time": str(program_start_time)
    }
    send_control_messages(message, redis_client, configuration.get("workflow_init_syn_queue_name"), 1)
    return redis_client

def wait_for_deployment_ready(function_name, timeout=60):
    cmd = [
        "kubectl", "rollout", "status",
        f"deployment/{function_name}",
        "-n", "openfaas-fn",
        f"--timeout={timeout}s"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"[INFO] {function_name} deployment is ready.")
    else:
        print(f"[ERROR] Timeout waiting for {function_name}: {result.stderr}")

def get_existing_function_indices(function_name_prefix):
    """
    Returns a sorted list of all function indices currently deployed (e.g., [1, 2, 3, 5]).
    """
    try:
        result = run_faas_cli(["list", "-q"], capture_output=True)
        all_functions = result.stdout.strip().splitlines()
        pattern = re.compile(fr"^{function_name_prefix}-(\d+)$")
        indices = []

        for fn in all_functions:
            match = pattern.match(fn)
            if match:
                indices.append(int(match.group(1)))

        return sorted(indices)
    except Exception as e:
        print(f"[ERROR] Failed to list {function_name_prefix} functions: {e}")
        return []  # Fail-safe: treat as no functions present


def find_next_available_function_names(function_name_prefix, existing_indices, count):
    """
    Given existing indices and count, find next `count` available function names.
    Reuses missing indices first.
    """
    allocated = []
    index = 1
    existing_set = set(existing_indices)

    while len(allocated) < count:
        if index not in existing_set:
            allocated.append(f"{function_name_prefix}-{index}")
        index += 1

    return allocated

def deploy_function(function_name_prefix, replicas, max_retries=3, delay=1):
    """
    Deploys a specified number of replicas for a function using available worker-X names.
    Includes error handling and reuses freed names if possible.
    """
    try:
        image = get_config().get(f"{function_name_prefix}_image")
        if not image:
            raise ValueError(f"No image specified in configuration for function '{function_name_prefix}_image'")
        print(f"[INFO] Deploying {replicas} new '{function_name_prefix}' function(s) using image '{image}'...")

        existing_indices = get_existing_function_indices(function_name_prefix)
        target_names = find_next_available_function_names(function_name_prefix, existing_indices, replicas)

        for name in target_names:
            try:
                deploy_function_instance(name, image, max_retries, delay)
                wait_for_deployment_ready(name, timeout=60)
            except Exception as e:
                print(f"[ERROR] Failed to deploy or wait for readiness of function '{name}': {e}")

    except Exception as e:
        print(f"[ERROR] Failed to deploy function group '{function_name_prefix}': {e}")

    return target_names

def deploy_function_instance(function_name, image, max_retries=3, delay=1):
    """
    Deploys an OpenFaaS function using faas-cli with retry logic.
    Detects rolling update status and waits before retrying.

    Args:
        function_name (str): Name of the function.
        image (str): Docker image for the function.
        max_retries (int): Maximum retry attempts.
        delay (int): Delay (seconds) between attempts, with exponential backoff.
    """
    cmd = [
        "deploy",
        "--name", function_name,
        "--image", image,
        "--env", "read_timeout=12h",
        "--env", "write_timeout=12h",
        "--env", "exec_timeout=12h",
        "--env", "redis_hostname=redis-master.redis.svc.cluster.local",
        "--env", "redis_port=6379",
        "--annotation", "com.openfaas.scale.min=1",
        "--annotation", "com.openfaas.retry.attempts=0",
        "--lang", "python3-http-skeleton"
    ]

    for attempt in range(1, max_retries + 1):
        print(f"[INFO] Deploying function: {function_name} (Attempt {attempt}/{max_retries})")
        try:
            result = run_faas_cli(cmd, capture_output=True, check=True)
            stdout = result.stdout.strip()
            print(f"[INFO] Deployment stdout: {stdout.strip()}")

            if f"Function {function_name} already exists, attempting rolling-update" in stdout:
                print(f"[INFO] Detected rolling-update for {function_name}, waiting before retry...")
                time.sleep(delay * attempt)  # Exponential wait
                continue

            print(f"[INFO][SUCCESS] Function '{function_name}' deployed successfully.")
            return
        except subprocess.CalledProcessError as e:
            print(f"[WARNING] Attempt {attempt} failed to deploy {function_name}")
            print(f"[WARNING] STDOUT: {e.stdout.strip()}")
            print(f"[WARNING] STDERR: {e.stderr.strip()}")
            if attempt == max_retries:
                raise RuntimeError(f"[FAILURE] Failed to deploy {function_name} after {max_retries} attempts.")
            time.sleep(delay)

    raise RuntimeError(f"[FAILURE] Gave up deploying {function_name} after {max_retries} attempts.")

def scale_deployments_to_zero_replica(program_start_time, apps_v1_api=None, core_v1_api=None):
    """
    Scales down "queue-worker", "gateway", "nats" deployments to zero replicas.
    This is useful for resetting the environment.
    """

    print(f"[INFO] Scaling down queue-worker, gateway, nats deployments to 0 at [{(get_utc_now() - program_start_time).total_seconds():.4f}] seconds.")

    for fn in ["queue-worker", "gateway", "nats"]:
        scale_function_deployment(0, apps_v1_api=apps_v1_api, deployment_name=fn, namespace="openfaas")
        print(f"[INFO] Scaled down {fn} deployment to 0 replicas.")

def wait_for_function_pods_terminated(core_v1_api, namespace="openfaas-fn", attempts=60, delay=1):
    """
    Waits for all pods in the given namespace to terminate, using a fixed number of attempts.

    Args:
        core_v1_api (CoreV1Api): Kubernetes CoreV1Api client.
        namespace (str): Namespace where functions are deployed.
        attempts (int): Number of attempts before giving up.
        delay (int): Delay (seconds) between attempts.
    """
    print("[INFO] Waiting for all function pods to be terminated...")

    for attempt in range(attempts):
        pods = core_v1_api.list_namespaced_pod(namespace=namespace).items
        active_pods = [
            p for p in pods
            if p.status.phase not in ("Succeeded", "Failed") and not p.metadata.deletion_timestamp
        ]

        if not active_pods:
            print(f"[INFO] All function pods terminated after {attempt + 1} attempt(s).")
            return

        print(f"[INFO] Attempt {attempt + 1}/{attempts}: {len(active_pods)} pod(s) ({[p.metadata.name for p in active_pods]}) still terminating...")
        time.sleep(delay)

    print("[WARNING] Some function pods are still running after max attempts.")

def delete_all_functions(core_v1_api):
    result = run_faas_cli(["list", "-q"], capture_output=True)
    functions = result.stdout.strip().splitlines()
    print(f"[INFO] Functions to delete: {functions}")

    for fn in functions:
        if fn:
            print(f"[INFO] Removing function: {fn}")
            run_faas_cli(["remove", fn])

    # Wait for pod termination
    wait_for_function_pods_terminated(core_v1_api)

def restart_resources(program_start_time, core_v1_api, apps_v1_api):
    """
    Restarts "nats", "gateway", "queue-worker" deployments by scaling them down to 0 then scale up to 1 replica.
    This is useful for resetting the environment without explicit deleting the deployments.
    """
    print(f"[INFO] Restarting resources at [{(get_utc_now() - program_start_time).total_seconds():.4f}] seconds.")

    # Scale down queue-worker, gateway, nats deployments to 0 replicas
    scale_deployments_to_zero_replica(program_start_time, apps_v1_api=apps_v1_api, core_v1_api=core_v1_api)

    # Ensure no port-forward processes are running before starting new ones
    if is_port_in_use(8080):
        kill_port_forward_process(8080)
        wait_for_port_free(8080)

    if is_port_in_use(6379):
        kill_port_forward_process(6379)
        wait_for_port_free(6379)

    scale_function_deployment(1, apps_v1_api=apps_v1_api, namespace="openfaas", deployment_name="nats")
    print(f"[INFO] Scaled up nats deployment to 1 replica.")
    wait_for_pods_ready(f"app=nats", "openfaas", core_v1_api, program_start_time, 1)

    scale_function_deployment(1, apps_v1_api=apps_v1_api, namespace="openfaas", deployment_name="gateway")
    print(f"[INFO] Scaled up gateway deployment to 1 replica.")
    wait_for_pods_ready(f"app=gateway", "openfaas", core_v1_api, program_start_time, 1)

    scale_function_deployment(1, apps_v1_api=apps_v1_api, namespace="openfaas", deployment_name="queue-worker")
    print(f"[INFO] Scaled down queue-worker deployment to 1 replica.")
    wait_for_pods_ready(f"app=queue-worker", "openfaas", core_v1_api, program_start_time, 1)

def kill_port_forward_process(port):
    """Find and kill the kubectl port-forward process using a specific port."""
    killed_pids = set()
    try:
        result = subprocess.run(
            ["lsof", "-i", f":{port}"],
            capture_output=True,
            text=True,
            check=False
        )
        for line in result.stdout.splitlines():
            if "kubectl" in line:
                pid = int(line.split()[1])
                if pid not in killed_pids:
                    try:
                        os.kill(pid, signal.SIGTERM)
                        print(f"[INFO] Killed existing kubectl port-forward (PID: {pid}) on port {port}")
                        killed_pids.add(pid)
                        time.sleep(1)
                    except ProcessLookupError:
                        print(f"[INFO] Process {pid} already terminated.")
    except Exception as e:
        print(f"[WARNING] Could not kill existing port-forward: {e}")

def wait_for_port_free(port, timeout=5):
    """Wait until the given port is free or timeout expires."""
    start = time.time()
    while time.time() - start < timeout:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('127.0.0.1', port)) != 0:
                return True
        time.sleep(0.5)
    print(f"[WARNING] Port {port} still in use after {timeout} seconds.")
    return False

def is_port_in_use(port):
    """Check if a local TCP port is already in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0

def port_forward(namespace, svc_name, local_port, remote_port, retries=10, delay=1):
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
            subprocess.Popen([
                "kubectl", "port-forward", f"svc/{svc_name}",
                f"{local_port}:{remote_port}", "-n", namespace
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            time.sleep(0.1)  # Give port-forward some time to establish

            if is_port_in_use(local_port):
                print(f"[INFO] Port-forward for {svc_name} is active on port {local_port}.")

                # Wait for the gateway to be ready if we're forwarding the OpenFaaS gateway
                if svc_name == "gateway" and namespace == "openfaas":
                    wait_for_gateway_ready(timeout=1)

                return
            else:
                print(f"[WARNING] Port-forward started but port {local_port} is not responding.")

        except Exception as e:
            print(f"[WARNING] Port-forward failed on attempt {attempt+1}: {e}", file=sys.stderr)

        attempt += 1
        time.sleep(delay)

    print(f"[ERROR] Could not establish port-forward for {svc_name} after {retries} attempts.", file=sys.stderr)

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

    print(f"[TIMER] Environment initialization started at [{(get_utc_now() - program_start_time).total_seconds():.4f}] seconds.")
    restart_resources(program_start_time, core_v1_api, apps_v1_api)

    print(f"[INFO] Port-forwarding services at {(get_utc_now() - program_start_time).total_seconds():.4f}...")
    port_forward("openfaas", "gateway", 8080, 8080)
    port_forward("redis", "redis-master", 6379, 6379)

    print(f"[INFO] Deleted all functions.")
    delete_all_functions(core_v1_api)
    print(f"[TIMER] Environment initialization completed at [{(get_utc_now() - program_start_time).total_seconds():.4f}] seconds.")

    return get_config()

def run_script(script_name, args=[]):
    """
    Asynchronously runs the specified script with the provided arguments.

    Args:
        script_name (str): The script to run.
        args (list): List of arguments to pass to the script.
    """
    cmd = ["python", script_name] + [str(arg) for arg in args]
    print(f"[INFO] [ASYNC RUNNING] {' '.join(cmd)}")
    subprocess.Popen(cmd)

def monitor_deployment_replicas(apps_v1_api=None, namespace="openfaas-fn", name_or_prefix="worker-", exact_match=False):
    """
    Logs the total replicas for a given deployment name or prefix.

    Args:
        name_or_prefix (str): Full name (e.g., "queue-worker") or prefix (e.g., "worker-")
        exact_match (bool): If True, do exact match instead of prefix match
    """
    try:
        replicas = get_deployment_replicas(
            apps_v1_api=apps_v1_api,
            namespace=namespace,
            name_or_prefix=name_or_prefix,
            exact_match=exact_match
        )
        print(f"[INFO] {replicas} replica(s) for '{name_or_prefix}' (exact_match={exact_match})")
        return replicas
    except Exception as e:
        print(f"[WARNING] Could not retrieve {name_or_prefix} replicas: {e}")

def wait_for_pods_ready(label_selector, namespace, core_v1_api, program_start_time, expected_count=1, timeout=60):
    """
    Wait until the expected number of pods with the given label are Running and Ready,
    and ensure that no pod remains in a non-Running state.
    """
    start_time = (get_utc_now() - program_start_time).total_seconds()
    attempt = 0
    time.sleep(3)  # Allow some time for the deployment to stabilize
    while (get_utc_now() - program_start_time).total_seconds() - start_time < timeout:
        pods = core_v1_api.list_namespaced_pod(
            namespace=namespace, label_selector=label_selector
        ).items

        # Wait until no pods are in non-Running state
        non_running_pods = [pod for pod in pods if pod.status.phase != "Running"]
        if non_running_pods:
            pod_names = [pod.metadata.name for pod in non_running_pods]
            print(f"[INFO] Waiting for non-Running pods to disappear: {pod_names}")
            attempt += 1
            time.sleep(1)
            continue

        # All pods are Running; now check for readiness
        ready_pods = [
            pod for pod in pods
            if all(cs.ready for cs in (pod.status.container_statuses or []))
        ]

        if len(ready_pods) >= expected_count:
            print(f"[INFO] {len(ready_pods)}/{expected_count} pod(s) ready in namespace '{namespace}' with label '{label_selector}'")
            if len(ready_pods) > expected_count:
                print(f"[WARNING] More pods ready than expected: {len(ready_pods)} vs {expected_count}")
            return True

        retrying_time = (get_utc_now() - program_start_time).total_seconds()
        print(f"[INFO] {len(ready_pods)}/{expected_count} {label_selector} pod(s) ready... retrying {attempt} time(s) at {retrying_time:.2f}s")
        attempt += 1
        time.sleep(1)

    raise TimeoutError(f"[ERROR] Timeout waiting for {expected_count} ready pod(s) with label '{label_selector}' in namespace '{namespace}'")
