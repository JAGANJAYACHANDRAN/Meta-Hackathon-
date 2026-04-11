# GPUScheduler-Env

**A high-fidelity RL environment for AI-driven GPU cluster orchestration.**

An AI agent manages an 8-node × 8-GPU enterprise cluster running at **$100,000/day**. Every decision — schedule, preempt, or wait — has a direct financial consequence. The agent must minimise "compute burn" across three tasks of escalating difficulty, culminating in a week-long simulation where a 32-GPU P0 emergency arrives mid-run requiring proactively reserved space.

---

## The Problem

Modern GPU clusters are expensive and chaotic. Jobs arrive unpredictably, carry different priorities and deadlines, and compete for shared hardware. Naive schedulers waste millions in idle GPU time. Human operators can't react fast enough at scale.

GPUScheduler-Env gives an AI agent the same information a real cluster orchestrator sees — queue depth, node utilisation, contention metrics, SLA deadlines — and forces it to develop long-horizon strategies under genuine economic pressure.

---

## Cluster Layout

```
8 nodes  ×  8 GPUs  =  64 GPUs total
Cost: $100,000/day  ≈  $4,167/hour  ≈  $65/GPU/hour
```

Each node exposes a **memory contention** metric. Contention degrades job progress via a smooth quadratic curve: `progress_rate = 1.0 − 0.4 × contention²`. At full saturation (8/8 GPUs on a node), jobs run 40% slower. Spreading load across nodes is critical.

---

## Action Space

| Action | Parameters | Description |
|---|---|---|
| `SCHEDULE` | `job_id`, `node_id` | Place a queued job on a node (0–7). Jobs needing >8 GPUs (gang jobs) auto-span multiple fully-free nodes. |
| `PREEMPT` | `job_id` | Evict a running job back to queue. Incurs a burn penalty + up to 2h checkpoint rollback. P0 jobs **cannot** be preempted. |
| `WAIT` | — | Advance the simulation clock without scheduling. |

### Examples

```json
{ "action_type": "SCHEDULE", "job_id": "job_042", "node_id": 3 }
{ "action_type": "PREEMPT",  "job_id": "job_017" }
{ "action_type": "WAIT" }
```

### Gang Jobs (>8 GPUs)

When scheduling a gang job (e.g. 32 GPUs = 4 nodes), the agent specifies one fully-empty anchor `node_id`. The environment automatically selects the remaining fully-free nodes. If not enough fully-free nodes exist, the action fails.

---

## Observation Space

| Field | Type | Description |
|---|---|---|
| `cluster_grid` | `int[8][8]` | GPU occupancy map. −1 = free, else = index into `active_jobs`. |
| `nodes` | `NodeInfo[8]` | Per-node: used/free GPUs, memory contention (0.0–1.0), running job IDs. |
| `active_jobs` | `JobInfo[]` | All running jobs: priority, GPU count, progress %, assigned nodes, deadline. |
| `queue` | `JobInfo[]` | Jobs ready to be scheduled now. |
| `upcoming_jobs` | `JobInfo[]` | Preview of jobs arriving within the lookahead window (6h/12h/24h by task). **Not yet schedulable** — for planning only. |
| `current_hour` | `float` | Simulated hours elapsed this episode. |
| `total_hours` | `float` | Episode horizon (24 / 72 / 168 hours). |
| `compute_burn_so_far` | `float` | Cumulative USD cost this episode. |
| `task_name` | `string` | Active task identifier. |
| `last_action_result` | `string` | Plain-English feedback on the previous action (errors included). |
| `score` | `float \| null` | Normalised grader score [0.0, 1.0]. Only populated when `done=True`. |

### JobInfo Fields

| Field | Description |
|---|---|
| `job_id` | Unique identifier (e.g. `job_042`) |
| `priority` | 0–3 (P0 = critical, P3 = spot/lowest) |
| `gpu_count` | GPUs needed (1–64). Values >8 require gang scheduling. |
| `duration_hours` | Total runtime to complete |
| `progress` | Fraction completed (0.0–1.0) |
| `deadline_hour` | Absolute hour by which the job must finish (`null` = no SLA) |
| `status` | `queued` \| `running` \| `completed` \| `preempted` \| `upcoming` |
| `assigned_nodes` | Node IDs currently occupied (empty if queued) |
| `arrival_hour` | When the job entered the queue |

---

## Reward Function

Reward is continuous and bounded to **[0.0, 1.0]** — computed every step, not only at episode end.

A reward of **0.5 is neutral** (no progress, no penalty). Values above 0.5 = net-positive scheduling; below 0.5 = waste or violations.

```
raw = R_progress − (R_idle + R_preemption + R_sla + R_queue)
Reward_t = clamp(0.5 + raw, 0.0, 1.0)
```

| Component | Sign | Formula |
|---|---|---|
| **Job progress** | + | `Σ (Δprogress × priority_weight)` per running job. Weights: P0=0.50, P1=0.40, P2=0.25, P3=0.10 |
| **Idle GPU cost** | − | `idle_gpus × hours × rate × hourly_rate × scale`. Rate = **0.20** when queue has work, **0.05** when empty |
| **Preemption burn** | − | `gpu_count × min(elapsed_hours, 2.0) × hourly_rate × 0.3 × scale`. Capped at 2h checkpoint interval |
| **SLA violation** | − | First miss: `gpu_count × duration × hourly_rate × 3.0 × scale`. Each step overdue: `gpu_count × hours × hourly_rate × 0.5 × scale` |
| **Queue delay** | − | `gpu_count × hours × priority_factor × hourly_rate × 0.1 × scale`. Factor: P0=0.50, P1=0.25, P2/P3=0 |

### SLA Partial Credit

Jobs that finish late still receive partial credit toward the SLA compliance metric:

| Finish Time | SLA Credit |
|---|---|
| On time (before deadline) | 1.0 |
| Up to 4h late | 0.5 |
| 4–8h late | 0.25 |
| Beyond 8h late | 0.0 |

---

## Tasks

Three graded tasks of escalating difficulty, each with a deterministic job schedule (seeded RNG):

### Easy — `smooth_sailing` (24h, 24 steps at 1h/step)

~24 P1–P3 jobs with moderate GPU demand, some with deadlines. Keep GPUs occupied and complete jobs efficiently.

**Grader:** `0.8 × completion_rate + 0.2 × gpu_utilisation`
**Pass threshold:** 0.40 | **Lookahead:** 6h

### Medium — `deadline_crunch` (72h, 36 steps at 2h/step)

~30 mixed P1–P3 jobs with bursty arrivals. 75% carry tight SLA deadlines. P3 spot jobs can be sacrificed to meet P1 deadlines.

**Grader:** `0.6 × sla_compliance + 0.4 × completion_rate`
**Pass threshold:** 0.35 | **Lookahead:** 12h

### Hard — `p0_emergency` (168h, 42 steps at 4h/step)

One week of mixed load (~40 jobs). At **hour 72**, a **32-GPU P0 gang job** arrives with a 60-hour deadline — requiring 4 fully-free nodes simultaneously. The agent must proactively drain nodes before the emergency arrives.

**Grader:** `0.5 × p0_completed + 0.3 × sla_compliance + 0.2 × gpu_utilisation`
**Pass threshold:** 0.30 | **Lookahead:** 24h

---

## Key Mechanics

**Memory contention** — colocating too many jobs on one node degrades all of them via a smooth quadratic curve (`1 − 0.4 × contention²`). At full node utilisation, jobs run 40% slower.

**Gang scheduling** — jobs requiring >8 GPUs need multiple *fully free* nodes. The agent schedules one anchor node; the environment auto-assigns the rest.

**Preemption burn** — stopping a job wastes compute since the last checkpoint (up to 2h of work). The penalty scales with GPU count and elapsed time.

**Checkpoint rollback** — preempted jobs lose up to 2 hours of progress when returned to queue.

**Dynamic arrivals** — jobs arrive throughout the episode on a pre-determined schedule. The `upcoming_jobs` field gives the agent a lookahead window to plan ahead.

---

## Quick Start

### Prerequisites

- Python 3.10+
- A HuggingFace account and token

### Install

```bash
git clone https://github.com/JAGANJAYACHANDRAN/Meta-Hackathon-.git
cd Meta-Hackathon-

# Create virtual environment
uv venv --python 3.12 .venv
source .venv/bin/activate

# Install the gpu_scheduler package
cd gpu_scheduler
uv pip install -e ".[dev]"
uv pip install openai
```

### Configure

```bash
cd gpu_scheduler
cp .env.example .env
```

Edit `.env`:
```
HF_TOKEN=hf_your_token_here
IMAGE_NAME=https://PACMAN8055-gpu-scheduler-env.hf.space
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
```

### Run Inference

```bash
# Run all three tasks
python inference.py

# Run a specific task only
python inference.py --task smooth_sailing
python inference.py --task p0_emergency
```

### Run with Local LLM (Ollama)

```bash
# Start Ollama with your model
ollama serve

# Point inference at localhost
# In .env:
API_BASE_URL=http://localhost:11434/v1
MODEL_NAME=qwen2.5:7b
IMAGE_NAME=https://PACMAN8055-gpu-scheduler-env.hf.space
```

### Run with Docker (local server)

```bash
cd gpu_scheduler/server
docker build -t gpu_scheduler-env:latest .
docker run -p 7860:7860 gpu_scheduler-env:latest

# In .env, set:
# IMAGE_NAME=gpu_scheduler-env:latest
```

---

## Expected Output

```
[START] task=smooth_sailing env=gpu_scheduler model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=SCHEDULE job_000 0 reward=0.45 done=false error=null
[STEP] step=2 action=SCHEDULE job_001 1 reward=0.52 done=false error=null
...
[END] success=true steps=24 rewards=0.45,0.52,...

[START] task=deadline_crunch env=gpu_scheduler model=Qwen/Qwen2.5-72B-Instruct
...
[END] success=true steps=36 rewards=...

[START] task=p0_emergency env=gpu_scheduler model=Qwen/Qwen2.5-72B-Instruct
...
[END] success=false steps=42 rewards=...
```

---

## Environment Variables

| Variable | Required | Description | Default |
|---|---|---|---|
| `HF_TOKEN` | Yes | HuggingFace / API key for LLM access | — |
| `IMAGE_NAME` | Yes | HF Space URL or Docker image tag | — |
| `API_BASE_URL` | No | LLM API endpoint | `https://router.huggingface.co/v1` |
| `MODEL_NAME` | No | LLM model identifier | `Qwen/Qwen2.5-72B-Instruct` |
| `TEMPERATURE` | No | LLM sampling temperature | `0.3` |
| `MAX_TOKENS` | No | Max tokens for LLM response | `4096` |
| `GPU_SCHEDULER_TASK` | No | Override task on `reset()` | `smooth_sailing` |

---

## Project Structure

```
Meta-Hackathon-/
├── inference.py                         # LLM agent script (entry point)
├── gpu_scheduler/
│   ├── __init__.py                      # Package surface
│   ├── models.py                        # Pydantic Action/Observation models
│   ├── client.py                        # WebSocket client (with keepalive)
│   ├── openenv.yaml                     # OpenEnv spec + task definitions
│   ├── pyproject.toml                   # Package config
│   ├── README.md                        # Environment README (for HF Space)
│   └── server/
│       ├── app.py                       # FastAPI server (HTTP + WebSocket)
│       ├── gpu_scheduler_environment.py # Core simulation engine
│       ├── ENVIRONMENT_DOCS.md          # Detailed function-level docs
│       ├── Dockerfile                   # HF Spaces deployment (port 7860)
│       └── requirements.txt
├── DEPLOYMENT.md                        # Step-by-step HF Spaces deployment guide
├── gpu-scheduler-env/                   # HF Space clone (synced separately)
└── test_*.py                            # Test suites
```

---

## Deployment

The environment server is deployed as a Docker container on **HuggingFace Spaces**:

**Live:** [https://huggingface.co/spaces/PACMAN8055/gpu-scheduler-env](https://huggingface.co/spaces/PACMAN8055/gpu-scheduler-env)

The Space exposes:
- **Web UI** at `/web`
- **API docs** at `/docs`
- **Health check** at `/health`
- **WebSocket** at `/ws` — persistent session endpoint for the agent

See [DEPLOYMENT.md](DEPLOYMENT.md) for the full step-by-step guide.

---

## How It Works

```
┌──────────────┐        WebSocket        ┌─────────────────────────┐
│              │  ───── reset() ───────►  │                         │
│  inference.py│                          │  gpu_scheduler_         │
│  (LLM Agent) │  ◄── observation ─────  │  environment.py         │
│              │                          │  (Simulation Engine)    │
│  Prompt LLM  │  ───── step(action) ──►  │                         │
│  Parse action│                          │  • Generate jobs        │
│  Log results │  ◄── obs, reward, done   │  • Track GPU state      │
│              │                          │  • Compute rewards      │
└──────────────┘                          │  • Grade performance    │
                                          └─────────────────────────┘
```

1. `inference.py` connects to the environment via WebSocket
2. Calls `reset(task_name)` to start an episode
3. Formats the observation into a prompt and sends it to the LLM
4. Parses the LLM's response into a `SCHEDULE`, `PREEMPT`, or `WAIT` action
5. Calls `step(action)` and receives the next observation + reward
6. Repeats until `done=True`, then reports the grader score

---

## Why This Matters

GPU scheduling is a billion-dollar problem. Major cloud providers and ML labs spend enormous engineering effort on cluster orchestrators that are still largely rule-based. An AI agent that reasons about priorities, contention, and economic trade-offs over long horizons could meaningfully outperform hand-crafted heuristics — and this environment provides a rigorous testbed to develop and evaluate such agents.
