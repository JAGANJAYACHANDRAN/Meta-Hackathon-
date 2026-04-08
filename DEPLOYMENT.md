# GPUScheduler-Env — Deployment Guide

## Prerequisites

- A HuggingFace account — https://huggingface.co
- Git installed locally
- Python 3.10+ with `uv` (or pip)
- Docker (optional, for local testing)

---

## Step 1 — Get a HuggingFace Access Token

1. Go to https://huggingface.co/settings/tokens
2. Click **Create new token** → choose **Fine-grained**
3. Enable these scopes:
   - **Repos → Read access** to all repos under your namespace
   - **Repos → Write access** to all repos under your namespace
   - **Inference → Make calls to the serverless Inference API**
4. Name it (e.g. `meta-hackathon`) and copy the token

---

## Step 2 — Install HF CLI and Login

```bash
# Install the HF CLI globally
uv tool install huggingface-hub

# Login (paste your token when prompted, say Yes to git credentials)
hf auth login
```

Your token is stored at `~/.cache/huggingface/token` — never inside the project.

---

## Step 3 — Create a HuggingFace Space

1. Go to https://huggingface.co/new-space
2. Fill in:
   - **Space name**: `gpu-scheduler-env`
   - **SDK**: Docker
   - **Template**: Blank
   - **Hardware**: CPU Basic (free tier is enough)
   - **Visibility**: Public
3. Click **Create Space**

---

## Step 4 — Clone the Space and Push the Code

```bash
# Clone your new Space (git uses your saved HF token automatically)
cd /path/to/this/repo
git clone https://huggingface.co/spaces/YOUR_USERNAME/gpu-scheduler-env

# Copy the server files into the correct structure
cp gpu_scheduler/server/__init__.py gpu-scheduler-env/server/
cp gpu_scheduler/server/app.py gpu-scheduler-env/server/
cp gpu_scheduler/server/gpu_scheduler_environment.py gpu-scheduler-env/server/
cp gpu_scheduler/server/ENVIRONMENT_DOCS.md gpu-scheduler-env/server/
cp gpu_scheduler/server/requirements.txt gpu-scheduler-env/server/
cp gpu_scheduler/server/Dockerfile gpu-scheduler-env/Dockerfile
cp gpu_scheduler/models.py gpu-scheduler-env/
cp gpu_scheduler/openenv.yaml gpu-scheduler-env/
cp gpu_scheduler/pyproject.toml gpu-scheduler-env/
cp gpu_scheduler/__init__.py gpu-scheduler-env/
cp gpu_scheduler/uv.lock gpu-scheduler-env/

# Push to HuggingFace
cd gpu-scheduler-env
git add .
git commit -m "Initial deploy"
git push
```

HuggingFace will automatically build the Docker image. Build takes 3–5 minutes.

**Important:** The Dockerfile must use port **7860** (HF Spaces requirement). The server/Dockerfile in this repo is already configured correctly.

---

## Step 5 — Verify the Server is Running

Watch the build logs at:
```
https://huggingface.co/spaces/YOUR_USERNAME/gpu-scheduler-env
```

Once the badge turns green, check the health endpoint:
```bash
curl https://YOUR_USERNAME-gpu-scheduler-env.hf.space/health
# Expected: {"status":"healthy"}
```

---

## Step 6 — Configure the .env File

```bash
cd /path/to/this/repo/gpu_scheduler
cp .env.example .env
```

Edit `.env` and fill in:
```
HF_TOKEN=hf_your_actual_token_here
IMAGE_NAME=https://YOUR_USERNAME-gpu-scheduler-env.hf.space
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
```

The `.env` file is gitignored and will never be committed.

---

## Step 7 — Install Dependencies and Run

```bash
cd /path/to/this/repo/gpu_scheduler

# Create venv and install (once)
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
uv pip install openai

# Run inference
python inference.py
```

Total runtime is approximately 8–10 minutes across all three tasks.

---

## Expected Output

```
[START] task=smooth_sailing env=gpu_scheduler model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=SCHEDULE job_000 0 reward=-0.14 done=false error=null
...
[END] success=true steps=24 score=0.569 rewards=...

[START] task=deadline_crunch env=gpu_scheduler model=Qwen/Qwen2.5-72B-Instruct
...
[END] success=true steps=36 score=0.461 rewards=...

[START] task=p0_emergency env=gpu_scheduler model=Qwen/Qwen2.5-72B-Instruct
...
[END] success=false steps=42 score=0.125 rewards=...

[SUMMARY] smooth_sailing=PASS | deadline_crunch=PASS | p0_emergency=FAIL
```

---

## Syncing Changes to HF Space

Every time you update the GitHub repo, sync to the HF Space:

```bash
# 1. Pull latest from GitHub
git pull

# 2. Copy updated files to HF Space
cp gpu_scheduler/server/__init__.py gpu-scheduler-env/server/
cp gpu_scheduler/server/app.py gpu-scheduler-env/server/
cp gpu_scheduler/server/gpu_scheduler_environment.py gpu-scheduler-env/server/
cp gpu_scheduler/models.py gpu-scheduler-env/
cp gpu_scheduler/openenv.yaml gpu-scheduler-env/
cp gpu_scheduler/pyproject.toml gpu-scheduler-env/
cp gpu_scheduler/__init__.py gpu-scheduler-env/
cp gpu_scheduler/uv.lock gpu-scheduler-env/

# 3. Push to HF
cd gpu-scheduler-env
git add .
git commit -m "Sync from main"
git push
```

HuggingFace rebuilds automatically on every push.

---

## Optional — Run with a Local Docker Container

For faster iteration without pushing to HuggingFace every time:

```bash
cd /path/to/this/repo/gpu_scheduler
docker build -t gpu_scheduler-env:latest -f server/Dockerfile .

# Update .env
# IMAGE_NAME=gpu_scheduler-env:latest

python inference.py
```

---

## Environment Variables Reference

| Variable | Required | Description | Default |
|---|---|---|---|
| `HF_TOKEN` | Yes | HuggingFace token for LLM API calls | — |
| `IMAGE_NAME` | Yes | HF Space URL or Docker image tag | — |
| `API_BASE_URL` | No | LLM API endpoint | `https://router.huggingface.co/v1` |
| `MODEL_NAME` | No | LLM model to use | `Qwen/Qwen2.5-72B-Instruct` |
