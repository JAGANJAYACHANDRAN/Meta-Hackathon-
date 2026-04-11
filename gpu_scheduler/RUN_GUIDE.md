# GPU Scheduler — How to Run & See Logs

## One-Time Setup

```bash
cd /Users/jagan.j/Meta-Hackathon-New/gpu_scheduler

# Create virtual environment
python3 -m venv ../venv
source ../venv/bin/activate

# Install the package
pip install -e .
```

## Configure Your Token

Edit the `.env` file in this directory:

```bash
# Open .env and set your HF_TOKEN
nano .env
```

The `.env` file looks like:
```
HF_TOKEN=hf_your_token_here
IMAGE_NAME=https://PACMAN8055-gpu-scheduler-env.hf.space
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
```

---

## Run Inference (see step-by-step logs live)

```bash
cd /Users/jagan.j/Meta-Hackathon-New
source venv/bin/activate
python gpu_scheduler/inference.py
```

This prints live logs to your terminal:
```
[START] task=smooth_sailing env=gpu_scheduler model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=SCHEDULE job_000 0 reward=-0.14 done=false error=null
[STEP] step=2 action=SCHEDULE job_001 1 reward=0.23 done=false error=null
...
[END] success=true steps=24 rewards=-0.14,0.23,...
```

### Save logs to a file while also watching live

```bash
python gpu_scheduler/inference.py 2>&1 | tee logs/run_$(date +%Y%m%d_%H%M%S).log
```

(Create the logs dir first: `mkdir -p logs`)

---

## View HF Space Logs (server-side)

### Container logs (what the server is doing)

```bash
export HF_TOKEN=hf_your_token_here

curl -N \
  -H "Authorization: Bearer $HF_TOKEN" \
  "https://huggingface.co/api/spaces/PACMAN8055/gpu-scheduler-env/logs/run"
```

### Build logs (Docker build output)

```bash
curl -N \
  -H "Authorization: Bearer $HF_TOKEN" \
  "https://huggingface.co/api/spaces/PACMAN8055/gpu-scheduler-env/logs/build"
```

### Check if the Space is healthy

```bash
curl https://PACMAN8055-gpu-scheduler-env.hf.space/health
# Expected: {"status":"healthy"}
```

---

## Quick Test (no LLM needed)

You can test the environment server directly without running the full inference:

```bash
# Check health
curl http://localhost:8000/health

# Open API docs in browser
open https://PACMAN8055-gpu-scheduler-env.hf.space/docs

# Reset environment via API
curl -X POST https://PACMAN8055-gpu-scheduler-env.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task_name": "smooth_sailing"}'
```

---

## Run Locally (without HF Space)

If you want to run the server on your own machine instead of using HF Space:

**Terminal 1 — Start the server:**
```bash
cd /Users/jagan.j/Meta-Hackathon-New
source venv/bin/activate
python -m uvicorn gpu_scheduler.server.app:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 — Run inference against local server:**
```bash
cd /Users/jagan.j/Meta-Hackathon-New
source venv/bin/activate

# Point to local server instead of HF Space
export IMAGE_NAME=http://localhost:8000
python gpu_scheduler/inference.py
```

---

## Scores Reference

| Task | Baseline Score | Pass Threshold |
|---|---|---|
| `smooth_sailing` | ~0.60 | 0.40 |
| `deadline_crunch` | ~0.45 | 0.35 |
| `p0_emergency` | ~0.28 | 0.30 |
