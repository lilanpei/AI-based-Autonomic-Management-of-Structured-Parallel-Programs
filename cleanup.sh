#!/bin/bash

set -euo pipefail

echo "[INFO] Deleting alertmanager deployment, service, and configmap..."
kubectl delete deployment alertmanager -n openfaas --ignore-not-found
kubectl delete service alertmanager -n openfaas --ignore-not-found
kubectl delete configmap alertmanager-config -n openfaas --ignore-not-found

# echo "[INFO] Deleting gateway pods..."
# kubectl -n openfaas delete pod -l app=gateway --ignore-not-found

# echo "[INFO] Deleting NATS pods..."
# kubectl -n openfaas delete pod -l app=nats --ignore-not-found

# echo "[INFO] Deleting queue-worker pods..."
# kubectl -n openfaas delete pod -l app=queue-worker --ignore-not-found

echo "[INFO] Scaling gateway, nats, and queue-worker to 0 replicas..."
kubectl -n openfaas scale deployment gateway --replicas=0
kubectl -n openfaas scale deployment nats --replicas=0
kubectl -n openfaas scale deployment queue-worker --replicas=0

echo "[INFO] Deleting Prometheus pods..."
kubectl -n openfaas delete pod -l app=prometheus --ignore-not-found

echo "[INFO] Deleting Redis master pod..."
kubectl delete pod redis-master-0 -n redis --ignore-not-found

echo "[INFO] Scaling gateway, nats, and queue-worker to 1 replicas..."
kubectl -n openfaas scale deployment gateway --replicas=1
kubectl -n openfaas scale deployment nats --replicas=1
kubectl -n openfaas scale deployment queue-worker --replicas=1

sleep 30
kubectl get pods -n openfaas -o wide
echo "[SUCCESS] Cleanup complete."