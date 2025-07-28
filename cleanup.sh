#!/bin/bash

set -euo pipefail

echo "[INFO] Deleting alertmanager deployment, service, and configmap..."
kubectl delete deployment alertmanager -n openfaas --ignore-not-found
kubectl delete service alertmanager -n openfaas --ignore-not-found
kubectl delete configmap alertmanager-config -n openfaas --ignore-not-found

echo "[INFO] Deleting gateway pods..."
kubectl -n openfaas delete pod -l app=gateway --ignore-not-found

echo "[INFO] Deleting NATS pods..."
kubectl -n openfaas delete pod -l app=nats --ignore-not-found

echo "[INFO] Deleting queue-worker pods..."
kubectl -n openfaas delete pod -l app=queue-worker --ignore-not-found

echo "[INFO] Deleting Prometheus pods..."
kubectl -n openfaas delete pod -l app=prometheus --ignore-not-found

echo "[INFO] Deleting Redis master pod..."
kubectl delete pod redis-master-0 -n redis --ignore-not-found

sleep 30
kubectl get pods -n openfaas -o wide
echo "[SUCCESS] Cleanup complete."