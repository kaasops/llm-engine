# LLM Engine - Multi-Model GPU Serving

Run multiple LLM models concurrently on a single GPU using Ray Serve and KubeRay operator.

## Quick Start

### 1. Install KubeRay Operator

```bash
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm install kuberay-operator kuberay/kuberay-operator --version 1.1.0
```

### 2. Deploy LLM Engine

```bash
kubectl apply -f ray-serve.yaml
```

### 3. Use the API

```bash
# Chat completion
curl -X POST "http://<service-ip>:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

# List available models
curl "http://<service-ip>:8000/v1/models"
```

## What It Does

- **Multiple Models**: Run several LLMs on one GPU simultaneously
- **GPU Efficiency**: Uses [vLLM sleep mode](https://docs.vllm.ai/en/latest/features/sleep_mode.html) to share GPU memory
- **OpenAI Compatible**: Works with any OpenAI-compatible client
- **Auto-scaling**: Automatically scales based on load

**Important**: Ensure you have enough free RAM to offload all LLMs when using sleep mode.

## Configuration

Edit `ray-serve.yaml` to change models:

```yaml
env_vars:
  MODELS: "model1,model2,model3"  # Your models here
```

## Python Client

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://<service-ip>:8000/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## Check Status

```bash
# See if it's running
kubectl get rayservice llm-engine

# View logs
kubectl logs <pod-name>

# Access dashboard
kubectl port-forward service/llm-engine-head-svc 8265:8265
```

## Requirements

- Kubernetes cluster with GPU
- KubeRay operator
- NVIDIA drivers
- Hugging Face model access

## Troubleshooting

- **Out of memory**: Reduce number of models or use smaller models
- **Model not loading**: Check model names and HF_TOKEN
- **Connection issues**: Verify service IP and ports

That's it! You now have multiple LLMs running on one GPU.