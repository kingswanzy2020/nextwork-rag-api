# Next Steps - Getting Your RAG API Running

Follow these steps in order to get your application working across all environments.

## ✅ Step 1: Configure Ollama to Bind to All Interfaces

**This is REQUIRED for Docker and Kubernetes to work.**

Run the configuration script:

```bash
./configure-ollama.sh
```

This will:
- Configure Ollama to listen on `0.0.0.0:11434` instead of `127.0.0.1:11434`
- Restart the Ollama service
- Verify the configuration

**Verify it worked:**
```bash
sudo ss -tlnp | grep 11434
```

You should see `0.0.0.0:11434` (not `127.0.0.1:11434`)

---

## ✅ Step 2: Test Locally (Optional but Recommended)

Test that your app works locally first:

1. **Start the FastAPI app:**
   ```bash
   python -m uvicorn app:app --reload --host 0.0.0.0 --port 8000
   ```
   Or:
   ```bash
   pip install fastapi uvicorn  # if not already installed
   uvicorn app:app --reload
   ```

2. **Test the API:**
   ```bash
   # Health check
   curl http://localhost:8000/health
   
   # Test query
   curl -X POST http://localhost:8000/query?q=What+is+kubernetes
   
   # Test adding knowledge
   curl -X POST "http://localhost:8000/add?text=Python is a programming language"
   ```

3. **Stop the server** (Ctrl+C) before moving to Docker/Kubernetes

---

## ✅ Step 3: Build Docker Image

Before running in Docker or Kubernetes, build the Docker image:

```bash
docker build -t rag-app .
```

**Verify the image was created:**
```bash
docker images | grep rag-app
```

---

## ✅ Step 4: Test with Docker

1. **Start with Docker Compose:**
   ```bash
   docker-compose up
   ```

   Or manually:
   ```bash
   docker run -p 8000:8000 \
     -e OLLAMA_HOST=host.docker.internal \
     -v $(pwd)/db:/app/db \
     rag-app
   ```

2. **Test the API:**
   ```bash
   curl http://localhost:8000/health
   curl -X POST http://localhost:8000/query?q=What+is+kubernetes
   ```

3. **Stop Docker:**
   ```bash
   docker-compose down
   # or Ctrl+C if running manually
   ```

---

## ✅ Step 5: Deploy to Kubernetes (Minikube)

1. **Ensure Minikube is running:**
   ```bash
   minikube status
   # If not running: minikube start
   ```

2. **Load the Docker image into Minikube:**
   ```bash
   # Build image in minikube's Docker environment
   eval $(minikube docker-env)
   docker build -t rag-app .
   eval $(minikube docker-env -u)  # Reset
   ```

   Or use minikube image load:
   ```bash
   minikube image load rag-app
   ```

3. **Deploy to Kubernetes:**
   ```bash
   kubectl apply -f deployment.yaml
   kubectl apply -f service.yaml
   ```

4. **Check deployment status:**
   ```bash
   kubectl get pods
   kubectl get services
   ```

5. **Get service URL:**
   ```bash
   # Get NodePort URL
   minikube service rag-app-service --url
   
   # Or port-forward for testing
   kubectl port-forward service/rag-app-service 8000:8000
   ```

6. **Test the API:**
   ```bash
   curl http://localhost:8000/health
   curl -X POST http://localhost:8000/query?q=What+is+kubernetes
   ```

7. **View logs:**
   ```bash
   kubectl logs -f deployment/rag-app-deployment
   ```

---

## 🔧 Troubleshooting

### Ollama Connection Issues

**Check Ollama is accessible:**
```bash
# From host
curl http://localhost:11434/api/tags

# From Docker (if inside container)
curl http://host.docker.internal:11434/api/tags

# From Kubernetes pod
kubectl exec -it <pod-name> -- curl http://host.minikube.internal:11434/api/tags
```

**Check Ollama binding:**
```bash
sudo ss -tlnp | grep 11434
# Should show: 0.0.0.0:11434 (not 127.0.0.1:11434)
```

**Check Ollama service status:**
```bash
sudo systemctl status ollama
sudo journalctl -u ollama.service -f
```

### Application Issues

**Check application logs:**
```bash
# Docker
docker-compose logs
docker logs <container-id>

# Kubernetes
kubectl logs deployment/rag-app-deployment
kubectl logs -f deployment/rag-app-deployment
```

**Test environment variables:**
```bash
# Docker
docker-compose exec rag-api env | grep OLLAMA

# Kubernetes
kubectl exec deployment/rag-app-deployment -- env | grep OLLAMA
```

---

## 📋 Quick Reference

| Environment | OLLAMA_HOST | How to Run |
|------------|-------------|------------|
| **Local** | `localhost` (default) | `uvicorn app:app --reload` |
| **Docker** | `host.docker.internal` | `docker-compose up` |
| **Kubernetes** | `host.minikube.internal` | `kubectl apply -f deployment.yaml` |

---

## ✅ Completion Checklist

- [ ] Step 1: Configured Ollama to bind to 0.0.0.0:11434
- [ ] Step 2: Tested locally (optional)
- [ ] Step 3: Built Docker image
- [ ] Step 4: Tested with Docker
- [ ] Step 5: Deployed to Kubernetes (if using)

---

## 🎯 Current Status

You've completed:
- ✅ Updated `app.py` for multi-environment support
- ✅ Created `docker-compose.yml` for Docker
- ✅ Updated `deployment.yaml` for Kubernetes
- ✅ Created configuration script for Ollama

**Next immediate action:** Run `./configure-ollama.sh` to configure Ollama!
