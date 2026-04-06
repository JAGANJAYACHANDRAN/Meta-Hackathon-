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
| `PREEMPT` | `job_id` | Evict a running job. Costs 2× wasted checkpoint work + up to 2h progress rollback. P0 jobs cannot be preempted. |
| `WAIT` | — | Advance the simulation clock without scheduling. |
| `BATCH` | `sub_actions[]` | Execute multiple SCHEDULE/PREEMPT actions **atomically within one timestep**. The clock advances only once after all sub-actions complete. |

### Single actions
```json
{ "action_type": "SCHEDULE", "job_id": "job_042", "node_id": 3 }
{ "action_type": "PREEMPT",  "job_id": "job_017" }
{ "action_type": "WAIT" }
```

### Batch action (atomic)
```json
{
  "action_type": "BATCH",
  "sub_actions": [
    { "action_type": "PREEMPT",  "job_id": "job_003" },
    { "action_type": "PREEMPT",  "job_id": "job_008" },
    { "action_type": "SCHEDULE", "job_id": "job_020", "node_id": 0 }
  ]
}
```

All sub-actions in a BATCH execute at the same simulated timestamp. This enables coordinated strategies like preempting multiple low-priority jobs and immediately scheduling a high-priority gang job — with no idle GPU penalties between sub-actions.

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

Reward is continuous and bounded to **[0.0, 1.0]** — computed every step, not only at episode end.

A reward of **0.5 is neutral** (no progress, no penalty). Values above 0.5 indicate net-positive scheduling; below 0.5 indicates waste or violations.

```
raw = R_progress − (R_idle + R_preemption + R_sla + R_queue)
Reward_t = clamp(0.5 + raw, 0.0, 1.0)
```

| Component | Type | Formula |
|---|---|---|
| **Progress** | Positive | `Σ (Δprogress_j × priority_weight_j)` per running job. Weights: P0=0.5×, P1=0.4×, P2=0.25×, P3=0.1× |
| **Idle cost** | Negative | `idle_gpus × hours × hourly_rate × rate`. Rate = 0.08 when queue has work, 0.02 when empty |
| **Preemption burn** | Negative | `gpu_count × min(elapsed_hours, 2.0) × hourly_rate × 0.3`. Capped at 2h checkpoint interval |
| **SLA violation** | Negative | Initial: `gpu_count × duration × hourly_rate × 0.15` (one-time). Continuing: `gpu_count × hours × hourly_rate × 0.05` per step |
| **Queue delay** | Negative | `gpu_count × hours × factor × hourly_rate × 0.03`. Factor: P0=0.5, P1=0.25, P2/P3=0 |

For BATCH actions, all sub-action rewards (e.g. preemption burns) are summed, then one time-step reward is computed. Since all sub-actions execute atomically, freed GPUs are immediately available for scheduling within the same step — no intermediate idle penalties.

---

## Tasks

### Easy — `smooth_sailing` (24h, 24 steps at 1h/step)
Moderate-demand window with ~24 P1–P3 jobs, some with deadlines. Keep GPUs occupied.
**Grader:** `0.8 × completion_rate + 0.2 × gpu_utilisation`

### Medium — `deadline_crunch` (72h, 36 steps at 2h/step)
~28 mixed P1/P2 jobs, 75% with tight SLA deadlines. Prioritise time-sensitive work.
**Grader:** `0.6 × sla_compliance + 0.4 × completion_rate`

### Hard — `p0_emergency` (168h, 42 steps at 4h/step)
One week of mixed load. At **hour 72**, a **32-GPU P0 gang job** arrives with a 60-hour deadline — requiring 4 fully-free nodes. The agent must drain nodes proactively; preempting costs 10× remaining value.
**Grader:** `0.5 × p0_completed + 0.3 × sla_compliance + 0.2 × gpu_utilisation`

### Hard — `batch_priority_inversion` (48h, 24 steps at 2h/step)
A 48-hour scenario testing **atomic batch preemption**. Eight P3 background jobs occupy the cluster (hours 0-4), then 10 P1 jobs with tight deadlines arrive (hours 12-20). The agent must use BATCH actions to preempt multiple low-priority jobs and schedule high-priority work atomically, avoiding idle-GPU penalties between actions. Strategic preemption is key — thrashing destroys the score.
**Grader:** `0.7 × sla_compliance + 0.3 × preemption_efficiency`
**Pass threshold:** 0.50

### Hard — `batch_gang_scheduling` (96h, 32 steps at 3h/step)
A 96-hour multi-gang scheduling challenge. Two P0 gang jobs arrive at different times: a **16-GPU job at hour 24** (needs 2 fully-free nodes) and a **24-GPU job at hour 60** (needs 3 fully-free nodes). The agent must coordinate BATCH operations to preempt multiple jobs and free entire nodes while maintaining P1 SLAs and overall utilization. Requires multi-horizon planning: preparing for gang job #2 while gang job #1 is running.
**Grader:** `0.4 × gang_completion + 0.3 × sla_compliance + 0.2 × utilisation + 0.1 × preemption_efficiency`
**Pass threshold:** 0.55

---

## Real-World Tensions

**Preemption burn** — stopping a job wastes compute since the last checkpoint (up to 2h). The penalty is scaled to pull the step reward toward 0.

**Memory contention** — colocating too many jobs on one node degrades all of them (quadratic: `1 - 0.4 × contention²`). Spread load even when packing looks locally optimal.

**Gang scheduling** — the 32-GPU P0 job needs 4 *fully* free nodes. Cannot scatter across partially-occupied nodes. Use BATCH to preempt multiple jobs and schedule the gang job atomically in one step.

**Checkpoint loss** — preempted jobs lose up to 2 hours of progress on rollback.

**Atomic batching** — BATCH actions let the agent preempt + schedule in one timestep with zero idle-GPU waste between actions. Critical for the P0 emergency scenario where multiple nodes must be freed simultaneously.

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
[STEP] step=1 action=BATCH[SCHEDULE job_003 0, SCHEDULE job_004 2] reward=0.12 done=false error=null
[STEP] step=2 action=SCHEDULE job_005 1 reward=0.18 done=false error=null
...
[END] success=true steps=24 score=0.621 rewards=0.12,0.18,...

[START] task=deadline_crunch env=gpu_scheduler model=Qwen/Qwen2.5-72B-Instruct
...
[END] success=true steps=36 score=0.487 rewards=...

[START] task=p0_emergency env=gpu_scheduler model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=18 action=BATCH[PREEMPT job_003, PREEMPT job_008, SCHEDULE job_020 0] reward=-1.24 done=false error=null
...
[END] success=false steps=42 score=0.284 rewards=...
```

The LLM agent outputs multiple actions per turn using a structured `REASON + ACTIONS` format. Multiple SCHEDULE/PREEMPT actions are wrapped into a single BATCH and executed atomically (one clock tick). Single actions are sent directly without the BATCH wrapper.

---

## Baseline Scores

| Task | Score | Notes |
|---|---|---|
| `smooth_sailing` | ~0.60 | Greedy scheduling, good utilisation |
| `deadline_crunch` | ~0.45 | Misses ~30% of deadlines |
| `p0_emergency` | ~0.28 | Often fails to reserve space before hour 72 |
| `batch_priority_inversion` | TBD | New test case for atomic batch preemption |
| `batch_gang_scheduling` | TBD | New test case for multi-gang coordination |

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
