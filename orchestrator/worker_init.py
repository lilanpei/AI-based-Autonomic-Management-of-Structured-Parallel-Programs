import sys
import time
import json
import requests
from kubernetes import client, config
from kubernetes.client.rest import ApiException
from utilities import get_config

def scale_worker_deployment(replica_count, namespace="openfaas-fn", deployment_name="worker"):
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
            body=body
        )
        print(f"[INFO] Successfully scaled deployment '{deployment_name}' to {replica_count} replicas.")
    except ApiException as e:
        print(f"ERROR: Failed to scale deployment '{deployment_name}': {e}", file=sys.stderr)
        time.sleep(5)
        try:
            api_instance.patch_namespaced_deployment_scale(
                name=deployment_name,
                namespace=namespace,
                body=body
            )
            print(f"[INFO] Retry succeeded: deployment scaled.")
        except ApiException as retry_e:
            print(f"CRITICAL ERROR: Retry scaling deployment failed: {retry_e}", file=sys.stderr)
            sys.exit(1)

def invoke_worker_function(replica_count):
    """
    Invokes the worker function via HTTP POST for each replica.
    """
    config_data = get_config()
    worker_q = config_data["worker_queue_name"]
    output_q = config_data["output_queue_name"]

    for i in range(replica_count):
        payload = {
            "start_flag": True,
            "worker_queue_name": worker_q,
            "result_queue_name": output_q
        }

        try:
            response = requests.post(
                "http://127.0.0.1:8080/function/worker",
                data=json.dumps(payload),
                headers={"Content-Type": "application/json"}
            )
        except requests.exceptions.RequestException as e:
            print(f"ERROR: Request to worker {i} failed: {e}", file=sys.stderr)
            time.sleep(5)
            try:
                response = requests.post(
                    "http://127.0.0.1:8080/function/worker",
                    data=json.dumps(payload),
                    headers={"Content-Type": "application/json"}
                )
            except Exception as retry_e:
                print(f"CRITICAL ERROR: Retry failed for worker {i}: {retry_e}", file=sys.stderr)
                continue  # Skip to next replica

        print(f"[INFO] Invoked worker replica {i} - Status: {response.status_code}")
        print("Response Body:", response.text)

def main():
    if len(sys.argv) != 2:
        print("Usage: python worker_init.py <replica_count>")
        sys.exit(1)

    try:
        replicas = int(sys.argv[1])
        if replicas < 0:
            raise ValueError("Replica count must be non-negative.")
    except ValueError:
        print("ERROR: Invalid replica count. Must be a non-negative integer.")
        sys.exit(1)

    scale_worker_deployment(replicas)
    invoke_worker_function(replicas)

if __name__ == "__main__":
    main()
