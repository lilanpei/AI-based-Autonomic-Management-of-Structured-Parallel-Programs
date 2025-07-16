# Custom OpenFaaS Setup with MaxReplicas Branch

This guide describes how to build, deploy, and verify a custom OpenFaaS setup using the `MaxReplicas` branch of the [faas-netes](https://github.com/lilanpei/faas-netes) and [faas](https://github.com/lilanpei/faas) repositories.

---

## 📥 Clone the Custom Branches

```bash
git clone -b MaxReplicas https://github.com/lilanpei/faas-netes.git
git clone -b MaxReplicas https://github.com/lilanpei/faas.git
```

## 🛠 Build and Push Docker Images
### Gateway (from faas/gateway directory):
```bash
docker build --no-cache -t gateway .
docker tag gateway:latest lilanpei/gateway:latest    # Replace 'lilanpei' with your Docker Hub username
docker push lilanpei/gateway:latest
```
### faas-netes (from faas-netes/ directory):
```bash
docker build --no-cache -t faas-netes .
docker tag faas-netes:latest lilanpei/faas-netes:latest  # Replace 'lilanpei' with your Docker Hub username
docker push lilanpei/faas-netes:latest
```
> ⚠️ Don’t forget to replace lilanpei with your actual Docker Hub username if you’re using your own registry.

## 📦 Add or Update the OpenFaaS Helm Repository
```bash
helm repo add openfaas https://openfaas.github.io/faas-netes/
helm repo update
```
## ⚙️ Get and Customize the Default Helm Values
```bash
helm show values openfaas/openfaas > openfaas-custom-values.yaml
```
Edit openfaas-custom-values.yaml to:
- Use your custom image tags
- Disable Alertmanager
- Increase the number of queue workers (recommended at least 1 for each function)
### Key Edits (example):
```bash
gateway:
  image: lilanpei/gateway:latest     # Your custom gateway image

faasnetes:
  image: lilanpei/faas-netes:latest  # Your custom faas-netes image

# Disable Alertmanager (for openfaas autoscaling)
alertmanager:
  create: false

# Increase number of queue workers (for async processing)
queueWorker:
  replicas: 3
```
###
🚀 Deploy or Upgrade OpenFaaS with Your Custom Images
```bash
helm upgrade openfaas --install openfaas/openfaas \
  --namespace openfaas \
  --create-namespace \
  -f openfaas-custom-values.yaml \
  --set basic_auth=true \
  --set functionNamespace=openfaas-fn \
  --wait
```
### ✅ Verify the Deployment
```bash
kubectl get pods -n openfaas -o wide
```
All pods should be running without errors.

### 📌 Notes
- Ensure Docker images are pushed to a registry accessible by your Kubernetes cluster.
- This setup assumes OpenFaaS Community Edition (openfaasPro: false).
- The MaxReplicas branch contains custom scaling logic and should be used for experimental or research purposes.
