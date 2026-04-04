# GPUScheduler-Env — Deployment Guide

## Prerequisites

- A HuggingFace account — https://huggingface.co
- Git installed locally
- Python 3.10+ with pip

---

## Step 1 — Get a HuggingFace Access Token

1. Go to https://huggingface.co/settings/tokens
2. Click **New token**
3. Give it **Write** permission
4. Copy and save it — you'll use it as your git password and for LLM API calls

---

## Step 2 — Create a HuggingFace Space

1. Go to https://huggingface.co/new-space
2. Fill in:
   - **Space name**: `gpu-scheduler-env` (or any name you like)
   - **SDK**: Docker
   - **Template**: Blank
   - **Hardware**: CPU Basic (free tier is enough)
   - **Visibility**: Public
3. Click **Create Space**

---

## Step 3 — Clone the Space and Push the Code

```bash
# Clone your new Space (use your HF token as the password when prompted)
git clone https://huggingface.co/spaces/YOUR_USERNAME/gpu-scheduler-env
cd gpu-scheduler-env

# Copy the environment code into it
cp -r /path/to/gpu_scheduler/* .

# Push to HuggingFace
git add .
git commit -m "Initial deploy"
git push
# Enter your HF token as the password when prompted
```

Replace `YOUR_USERNAME` with your HuggingFace username and `/path/to/gpu_scheduler` with wherever the `gpu_scheduler` folder lives on your machine.

HuggingFace will automatically start building the Docker image. Build takes 3–5 minutes.

---

## Step 4 — Verify the Server is Running

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

## Step 5 — Run the Inference Agent

### Install dependencies (once)
```bash
cd /path/to/gpu_scheduler
pip install -e ".[dev]"
pip install openai
```

### Set environment variables
```bash
export HF_TOKEN=hf_your_token_here
export IMAGE_NAME=https://YOUR_USERNAME-gpu-scheduler-env.hf.space
```

### Run
```bash
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

## Updating the Server

Whenever you make code changes and want to redeploy:

```bash
cp -r /path/to/gpu_scheduler/* /path/to/gpu-scheduler-env/
cd /path/to/gpu-scheduler-env
git add .
git commit -m "describe your change"
git push
```

HuggingFace rebuilds and restarts automatically on every push.

---

## Optional — Run with a Local Docker Container

If you have Docker installed and want faster iteration without pushing to HuggingFace every time:

```bash
# Build the image locally
cd /path/to/gpu_scheduler
docker build -t gpu_scheduler-env:latest -f server/Dockerfile .

# Point inference at the local container
export HF_TOKEN=hf_your_token_here
export IMAGE_NAME=gpu_scheduler-env:latest

python inference.py
```

---

## Environment Variables Reference

| Variable | Required | Description | Default |
|---|---|---|---|
| `HF_TOKEN` | Yes | HuggingFace token for LLM API calls | — |
| `IMAGE_NAME` | Yes | URL or Docker image tag of the environment server | — |
| `API_BASE_URL` | No | LLM API endpoint | `https://router.huggingface.co/v1` |
| `MODEL_NAME` | No | LLM model to use | `Qwen/Qwen2.5-72B-Instruct` |
