# nextwork-rag-api

A Retrieval-Augmented Generation API: FastAPI in front of a Chroma vector store and a
local Ollama model, so answers are grounded in a knowledge base instead of invented.

The repo tracks the same service across three environments — laptop, container, and
Kubernetes — because that progression is where the interesting problems are.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/query?q=...` | Retrieves the closest document from Chroma and asks the model to answer from it |
| `POST` | `/add` | Adds a document to the knowledge base at runtime — no re-running an embedding script |

## Files

| File | Purpose |
|---|---|
| `app.py` | The FastAPI service |
| `embed.py` / `embed_docs.py` | Populate the Chroma collection from `docs/` |
| `docs/` | The knowledge base (`k8s.txt`, `nextwork.txt`) |
| `Dockerfile` | Image build — embeds documents at build time so the container starts ready |
| `docker-compose.yml` | Local container run with the Chroma DB persisted to a volume |
| `configure-ollama.sh` | Rebinds Ollama from `127.0.0.1` to `0.0.0.0` (see below) |
| `deployment.yaml` / `service.yaml` | Kubernetes manifests |
| `semantic_test.py` | Asserts the answer actually contains the expected concept, not just that a 200 came back |
| `.github/workflows/ci.yml` | CI |

## The one thing that breaks everything

Ollama listens on `127.0.0.1:11434` by default. That works from the host and fails from
**every** container, because `127.0.0.1` inside a container is the container. Docker and
Kubernetes both need Ollama bound to `0.0.0.0`:

```bash
./configure-ollama.sh     # writes a systemd override and restarts Ollama
```

`docs/running-across-environments.md` walks through the full sequence for all three
environments.

## Running it

**Locally:**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install fastapi uvicorn chromadb ollama
python embed.py            # builds the Chroma collection into ./db
uvicorn app:app --reload
```

**In Docker:**

```bash
docker compose up --build   # http://localhost:8000
```

`./db` is mounted as a volume, so the embedded knowledge base survives a rebuild.

**On Kubernetes:**

```bash
kubectl apply -f deployment.yaml -f service.yaml
```

## CI without a model

`app.py` honours `USE_MOCK_LLM=1`, which returns the retrieved context directly instead of
calling Ollama. That is what lets CI exercise the retrieval path — the part that can
actually regress — without running an LLM on a build agent.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_HOST` | `localhost` | Where to reach Ollama; `host.docker.internal` from a container |
| `MODEL_NAME` | `tinyllama` | Model to generate with |
| `USE_MOCK_LLM` | `0` | Set to `1` to bypass the model in CI |

No API keys are needed — inference is local, which is also why the documents never leave
the machine.

## Write-ups

Three write-ups cover this repo, one per environment:

- **[Projects / ai-devops / rag-api-fastapi](https://github.com/kingswanzy2020/Projects/tree/main/ai-devops/rag-api-fastapi)** — building the API
- **[Projects / ai-devops / rag-api-docker](https://github.com/kingswanzy2020/Projects/tree/main/ai-devops/rag-api-docker)** — containerizing it
- **[Projects / ai-devops / rag-api-kubernetes](https://github.com/kingswanzy2020/Projects/tree/main/ai-devops/rag-api-kubernetes)** — deploying it to a cluster

The CI pipeline around it is written up at
**[Projects / ci-cd / github-actions-automated-testing](https://github.com/kingswanzy2020/Projects/tree/main/ci-cd/github-actions-automated-testing)**.
