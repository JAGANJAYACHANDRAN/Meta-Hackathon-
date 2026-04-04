---
title: Gpu Scheduler Environment Server
emoji: 🎖️
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# GPUScheduler-Env

**A high-fidelity RL environment for AI-driven GPU cluster orchestration.**

An AI agent manages an 8-node × 8-GPU enterprise cluster running at **$100,000/day**. Every decision — schedule, preempt, or wait — has a direct financial consequence. The agent must learn to minimise "compute burn" across three tasks of escalating difficulty, culminating in a week-long simulation where a 32-GPU P0 emergency arrives mid-run requiring proactively reserved space.

---

## The Problem

Modern GPU clusters are expensive and chaotic. Jobs arrive unpredictably, carry different priorities and deadlines, and compete for shared hardware. Naive schedulers waste millions in idle GPU time. Human operators can't react fast enough at scale.

GPUScheduler-Env gives an RL agent the same information a real cluster orchestrator sees — queue depth, node utilisation, contention metrics, SLA deadlines — and forces it to develop long-horizon strategies under genuine economic pressure.

---

## Cluster Layout

```
8 nodes  ×  8 GPUs  =  64 GPUs total
Cost: $100,000/day  ≈  $4,167/hour  ≈  $65/GPU/hour
```

Each node exposes a **memory contention** metric. When GPU utilisation exceeds 70%, all jobs on that node slow down — up to 50% degradation at full saturation. Packing too many jobs onto one node is a trap.

---

## Action Space

| Action | Parameters | Description |
|---|---|---|
| `SCHEDULE` | `job_id`, `node_id` | Place a queued job on a node (0–7). Jobs needing >8 GPUs auto-span multiple fully-free nodes. |
| `PREEMPT` | `job_id` | Evict a running job. Costs 10× remaining compute value + 1h checkpoint rollback. |
| `WAIT` | — | Advance the simulation clock without scheduling. |

```json
{ "action_type": "SCHEDULE", "job_id": "job_042", "node_id": 3 }
{ "action_type": "PREEMPT",  "job_id": "job_017" }
{ "action_type": "WAIT" }
```

---

## Observation Space

| Field | Type | Description |
|---|---|---|
| `cluster_grid` | `int[8][8]` | GPU occupancy map. −1 = free, else = index into `active_jobs`. |
| `nodes` | `NodeInfo[8]` | Per-node: used/free GPUs, memory contention (0.0–1.0), running job IDs. |
| `active_jobs` | `JobInfo[]` | All running jobs: priority, GPU count, progress %, assigned nodes, deadline. |
| `queue` | `JobInfo[]` | Jobs waiting to be scheduled, with ~4-hour arrival lookahead. |
| `current_hour` | `float` | Simulated hours elapsed this episode. |
| `total_hours` | `float` | Episode horizon (24 / 72 / 168 hours). |
| `compute_burn_so_far` | `float` | Cumulative USD cost this episode. |
| `task_name` | `string` | Active task: `smooth_sailing` \| `deadline_crunch` \| `p0_emergency` |
| `last_action_result` | `string` | Plain-English feedback on the previous action (errors included). |

---

## Reward Function

Reward is continuous — computed every step, not only at episode end.

```
Reward_t = R_progress + R_cost − (R_preemption + R_sla + R_queue)
```

| Component | Type | Formula |
|---|---|---|
| **Progress** | Positive | `Σ (Δprogress_j × priority_weight_j)` per running job |
| **Idle cost** | Negative | `idle_gpus × hours × hourly_rate × 0.3` |
| **Preemption burn** | Negative | `gpu_count × remaining_hours × hourly_rate × 10` |
| **SLA violation** | Negative | `gpu_count × duration × hourly_rate × 5` (one-time on breach) |
| **Queue delay** | Negative | Small per-hour penalty for P0/P1 jobs waiting in queue |

---

## Tasks

### Easy — `smooth_sailing` (24h, 24 steps at 1h/step)
Low-demand window with 12 small P2/P3 jobs, no deadlines. Keep GPUs occupied.
**Grader:** `0.6 × completion_rate + 0.4 × gpu_utilisation`

### Medium — `deadline_crunch` (72h, 36 steps at 2h/step)
~28 mixed P1/P2 jobs, 75% with tight SLA deadlines. Prioritise time-sensitive work.
**Grader:** `0.6 × sla_compliance + 0.4 × completion_rate`

### Hard — `p0_emergency` (168h, 42 steps at 4h/step)
One week of mixed load. At **hour 72**, a **32-GPU P0 gang job** arrives with a 60-hour deadline — requiring 4 fully-free nodes. The agent must drain nodes proactively; preempting costs 10× remaining value.
**Grader:** `0.5 × p0_completed + 0.3 × sla_compliance + 0.2 × gpu_utilisation`

---

## Real-World Tensions

**Preemption burn** — stopping a job wastes all compute since the last checkpoint. The penalty is 10× remaining runtime cost.

**Memory contention** — colocating too many jobs on one node degrades all of them. Spread load even when packing looks locally optimal.

**Gang scheduling** — the 32-GPU P0 job needs 4 *fully* free nodes. Cannot scatter across partially-occupied nodes. Requires look-ahead planning across dozens of steps.

**Checkpoint loss** — preempted jobs lose up to 1 hour of progress on rollback.

---

## Quick Start

```bash
# Install
cd gpu_scheduler
pip install openenv-core

# Start server
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload

# Run baseline inference agent
IMAGE_NAME=gpu_scheduler-env:latest \
HF_TOKEN=hf_... \
python inference.py
```

### Docker

```bash
cd gpu_scheduler/server
docker build -t gpu_scheduler-env:latest .
docker run -p 8000:8000 gpu_scheduler-env:latest
```

### Validate

```bash
pip install openenv-core
openenv validate
```

---

## Inference Script

The baseline `inference.py` runs an LLM agent against all three tasks sequentially.

**Required environment variables:**

| Variable | Description |
|---|---|
| `HF_TOKEN` | HuggingFace / API key |
| `IMAGE_NAME` | Docker image tag for the environment server |
| `API_BASE_URL` | LLM endpoint (default: HuggingFace router) |
| `MODEL_NAME` | Model identifier (default: `Qwen/Qwen2.5-72B-Instruct`) |

**Expected stdout:**
```
[START] task=smooth_sailing env=gpu_scheduler model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=SCHEDULE job_003 0 reward=0.12 done=false error=null
...
[END] success=true steps=24 score=0.621 rewards=0.12,0.18,...

[START] task=deadline_crunch env=gpu_scheduler model=Qwen/Qwen2.5-72B-Instruct
...
[END] success=true steps=36 score=0.487 rewards=...

[START] task=p0_emergency env=gpu_scheduler model=Qwen/Qwen2.5-72B-Instruct
...
[END] success=false steps=42 score=0.284 rewards=...
```

### Enhanced Visual Output

The inference script now includes a **Rich Logger** that provides clear, formatted output for easier debugging:

```bash
# Enable rich output (default)
export VERBOSE_OUTPUT=1
python inference.py
```

**Features:**
- 📊 Formatted tables for each step with progress tracking
- 🚨 Clear error highlighting with contextual hints
- 📈 Episode summaries with color-coded status
- ⚠️ Error categorization and summary table

See [RICH_LOGGER.md](./RICH_LOGGER.md) for full documentation and examples.

**Demo:**
```bash
.venv/bin/python demo_rich_logger.py
```

---

## Baseline Scores

| Task | Score | Notes |
|---|---|---|
| `smooth_sailing` | ~0.60 | Greedy scheduling, good utilisation |
| `deadline_crunch` | ~0.45 | Misses ~30% of deadlines |
| `p0_emergency` | ~0.28 | Often fails to reserve space before hour 72 |

---

## Deploy to Hugging Face Spaces

```bash
# From the gpu_scheduler directory
openenv push --repo-id your-username/gpu-scheduler-env
```

The deployed Space exposes:
- **Web UI** at `/web` — interactive cluster visualiser
- **API docs** at `/docs` — full OpenAPI interface
- **Health check** at `/health`
- **WebSocket** at `/ws` — persistent session endpoint

---

## Project Structure

```
gpu_scheduler/
├── inference.py                      # LLM agent script (hackathon requirement)
├── models.py                         # Typed Pydantic Action/Observation models
├── client.py                         # WebSocket EnvClient
├── __init__.py                       # Package surface
├── openenv.yaml                      # OpenEnv spec + task definitions
├── pyproject.toml
├── README.md
└── server/
    ├── app.py                        # FastAPI server (HTTP + WebSocket)
    ├── gpu_scheduler_environment.py  # Core simulation engine
    ├── Dockerfile
    └── requirements.txt
```

---

## Why This Matters

GPU scheduling is a billion-dollar problem. Major cloud providers and ML labs spend enormous engineering effort on cluster orchestrators that are still largely rule-based. An RL agent that reasons about priorities, contention, and economic trade-offs over long horizons could meaningfully outperform hand-crafted heuristics — and this environment provides a rigorous testbed to develop and evaluate such agents.
