---
title: Gpu Scheduler Environment Server
emoji: 🎖️
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
app_port: 7860
base_path: /web
tags:
  - openenv
---

# GPUScheduler-Env

**A high-fidelity RL environment for AI-driven GPU cluster orchestration.**

An AI agent manages an 8-node × 8-GPU enterprise cluster running at **$100,000/day**. Every decision — schedule, preempt, or wait — has a direct financial consequence. The agent must minimise "compute burn" across five tasks of escalating difficulty, from a calm 24-hour window to coordinated multi-gang scheduling requiring atomic batch operations.

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

Each node exposes a **memory contention** metric. Contention degrades job progress via a smooth quadratic curve: `progress_rate = 1.0 − 0.4 × contention²`. At full saturation (8/8 GPUs), jobs run 40% slower. Spreading load across nodes is critical.

---

## Action Space

| Action | Parameters | Description |
|---|---|---|
| `SCHEDULE` | `job_id`, `node_id` | Place a queued job on a node (0–7). Jobs needing >8 GPUs (gang jobs) auto-span multiple fully-free nodes. |
| `PREEMPT` | `job_id` | Evict a running job back to queue. Incurs a burn penalty + up to 2h checkpoint rollback. P0 jobs **cannot** be preempted. |
| `WAIT` | — | Advance the simulation clock without scheduling. |
| `BATCH` | `sub_actions[]` | Execute multiple `SCHEDULE`/`PREEMPT` actions atomically in one timestep. The clock advances only once after all sub-actions are applied. |

### Examples

```json
{ "action_type": "SCHEDULE", "job_id": "job_042", "node_id": 3 }
{ "action_type": "PREEMPT",  "job_id": "job_017" }
{ "action_type": "WAIT" }
{ "action_type": "BATCH", "sub_actions": [
    { "action_type": "PREEMPT", "job_id": "job_003" },
    { "action_type": "SCHEDULE", "job_id": "job_020", "node_id": 0 }
]}
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
| `upcoming_jobs` | `JobInfo[]` | Preview of jobs arriving within the lookahead window (6–24h by task). **Not yet schedulable** — for planning only. |
| `current_hour` | `float` | Simulated hours elapsed this episode. |
| `total_hours` | `float` | Episode horizon (24 / 48 / 72 / 96 / 168 hours). |
| `compute_burn_so_far` | `float` | Cumulative USD cost this episode. |
| `task_name` | `string` | Active task: `smooth_sailing` \| `deadline_crunch` \| `p0_emergency` \| `batch_priority_inversion` \| `batch_gang_scheduling` |
| `last_action_result` | `string` | Plain-English feedback on the previous action (errors included). |
| `score` | `float \| null` | Normalised grader score [0.0, 1.0]. Only populated when `done=True`. |

---

## Reward Function

Reward is continuous and bounded to **[0.0, 1.0]** — computed every step, not only at episode end.

**0.5 is neutral** (no progress, no penalty). Above 0.5 = net-positive; below = waste or violations.

```
raw = R_progress − (R_idle + R_preemption + R_sla + R_queue)
Reward_t = clamp(0.5 + raw, 0.0, 1.0)
```

| Component | Sign | Formula |
|---|---|---|
| **Job progress** | + | `Σ (Δprogress × priority_weight)`. Weights: P0=0.50, P1=0.40, P2=0.25, P3=0.10 |
| **Idle GPU cost** | − | `idle_gpus × hours × hourly_rate × scale × rate`. Rate = 0.08 (queue non-empty) or 0.02 (empty) |
| **Preemption burn** | − | `gpu_count × min(elapsed, 2h) × hourly_rate × 0.3 × scale` |
| **SLA violation** | − | Initial: `gpu_count × duration × hourly_rate × 0.15 × scale` one-time. Continuing: `gpu_count × hours × hourly_rate × 0.05 × scale` per step overdue |
| **Queue delay** | − | `gpu_count × hours × factor × hourly_rate × scale × 0.03`. Factor: P0=0.50, P1=0.25; P2/P3 = no penalty |

### SLA Partial Credit

| Finish Time | Credit |
|---|---|
| On time | 1.0 |
| ≤ 4h late | 0.5 |
| 4–8h late | 0.25 |
| > 8h late | 0.0 |

---

## Tasks

### Easy — `smooth_sailing` (24h, 24 steps at 1h/step)

~24 P1–P3 jobs with moderate demand, some with deadlines. Keep GPUs occupied.

**Grader:** `0.8 × completion_rate + 0.2 × gpu_utilisation` | **Pass:** ≥ 0.40

### Medium — `deadline_crunch` (72h, 36 steps at 2h/step)

~30 mixed P1–P3 jobs with bursty arrivals. 75% carry tight SLA deadlines.

**Grader:** `0.6 × sla_compliance + 0.4 × completion_rate` | **Pass:** ≥ 0.35

### Hard — `p0_emergency` (168h, 42 steps at 4h/step)

One week of mixed load. At **hour 72**, a **32-GPU P0 gang job** arrives (deadline: hour 132) requiring 4 fully-free nodes.

**Grader:** `0.5 × p0_completed + 0.3 × sla_compliance + 0.2 × gpu_utilisation` | **Pass:** ≥ 0.30

### Hard — `batch_priority_inversion` (48h, 24 steps at 2h/step)

P3 background jobs occupy the cluster early. At ~hour 12, multiple P1 jobs with tight deadlines arrive. The agent must use `BATCH` actions to atomically preempt low-priority jobs and schedule high-priority work, avoiding idle-GPU penalties between operations.

**Grader:** `0.7 × sla_compliance + 0.3 × preemption_efficiency` | **Pass:** ≥ 0.50

Preemption efficiency = `max(0, 1.0 − preemption_count × 0.1)` — penalises excessive preemptions.

### Hard — `batch_gang_scheduling` (96h, 32 steps at 3h/step)

Two P0 gang jobs arrive mid-episode (16-GPU at hour 24, 24-GPU at hour 60). The agent must proactively reserve node capacity via `BATCH` operations — coordinating preemptions to free entire nodes while maintaining P1 SLAs and utilisation.

**Grader:** `0.4 × gang_completion + 0.3 × sla_compliance + 0.2 × gpu_utilisation + 0.1 × (1 − preemption_waste)` | **Pass:** ≥ 0.55

Gang completion is measured as completed gang jobs (≥16 GPUs) out of 2 total. Preemption waste = total wasted hours / total preempted job durations.

---

## Key Mechanics

**Memory contention** — quadratic degradation: `1 − 0.4 × contention²`. At full node load, 40% slowdown.

**Gang scheduling** — jobs >8 GPUs need multiple *fully free* nodes. Agent picks an anchor; environment assigns the rest.

**Preemption burn** — evicting a job wastes up to 2h of compute (checkpoint rollback). Penalty scales with GPU count.

**Dynamic arrivals** — jobs arrive on a deterministic schedule. `upcoming_jobs` gives a planning lookahead window.

---

## Quick Start

```bash
# Install
cd gpu_scheduler
pip install -e ".[dev]"
pip install openai

# Configure (edit .env with your tokens)
cp .env.example .env

# Run all tasks
python ../inference.py

# Run a single task
python ../inference.py --task smooth_sailing
```

### Docker

```bash
cd gpu_scheduler/server
docker build -t gpu_scheduler-env:latest .
docker run -p 7860:7860 gpu_scheduler-env:latest
```

---

## Deployed Environment

**Live:** [https://PACMAN8055-gpu-scheduler-env.hf.space](https://PACMAN8055-gpu-scheduler-env.hf.space)

| Endpoint | Description |
|---|---|
| `/web` | Interactive cluster visualiser |
| `/docs` | OpenAPI docs |
| `/health` | Health check |
| `/ws` | WebSocket session endpoint |

---

## Project Structure

```
gpu_scheduler/
├── models.py                        # Pydantic Action/Observation models
├── client.py                        # WebSocket client (with keepalive)
├── __init__.py                      # Package surface
├── openenv.yaml                     # OpenEnv spec + task definitions
├── pyproject.toml
└── server/
    ├── app.py                       # FastAPI server (HTTP + WebSocket)
    ├── gpu_scheduler_environment.py # Core simulation engine
    ├── ENVIRONMENT_DOCS.md          # Detailed function-level reference
    ├── Dockerfile                   # HF Spaces deployment (port 7860)
    └── requirements.txt
```
