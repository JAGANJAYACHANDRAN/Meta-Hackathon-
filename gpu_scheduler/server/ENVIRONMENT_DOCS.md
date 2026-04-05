# GPU Scheduler Environment — Function-by-Function Reference

## Overview

`gpu_scheduler_environment.py` is the **core simulation engine**. It models an 8-node × 8-GPU cluster costing $100,000/day. An LLM agent interacts with it via `reset()` and `step()` calls to schedule jobs, and the environment returns observations + rewards.

---

## Constants

| Constant | Value | Purpose |
|----------|-------|---------|
| `NUM_NODES` | 8 | Physical nodes in the cluster |
| `GPUS_PER_NODE` | 8 | GPUs per node |
| `TOTAL_GPUS` | 64 | Total GPU capacity |
| `DAILY_COST_USD` | $100,000 | Full-cluster daily operating cost |
| `HOURLY_RATE_CLUSTER` | ~$4,167 | Hourly cost at full capacity |
| `HOURLY_RATE_PER_GPU` | ~$65 | Cost per GPU per hour |
| `REWARD_SCALE` | 1/HOURLY_RATE_CLUSTER | Normalises raw dollar rewards into RL-friendly range |
| `CONTENTION_DEGRADATION` | 0.4 | Smooth quadratic: progress_rate = 1 - 0.4 × contention² |
| `PREEMPTION_BURN_MULTIPLIER` | 2.0 | Preemption costs 2× the wasted checkpoint work |
| `PREEMPTION_CHECKPOINT_HOURS` | 2.0 | Assumed checkpoint interval (hours) — caps wasted work |
| `PRIORITY_WEIGHTS` | {0: 2.0, 1: 1.5, 2: 1.0, 3: 0.5} | P0 progress gives 2× more reward than P3 |
| `PRIORITY_QUEUE_FACTORS` | {0: 2.0, 1: 1.0, 2: 0.0, 3: 0.0} | Only P0/P1 jobs incur queue-delay penalty |

---

## Task Configurations (`TASK_CONFIGS`)

| Task | Total Hours | Hours/Step | Agent Steps | Lookahead | Seed | Description |
|------|-------------|------------|-------------|-----------|------|-------------|
| `smooth_sailing` | 24h | 1.0 | 24 | 6h | 42 | ~24 P1–P3 jobs with moderate GPU demand, some with deadlines. |
| `deadline_crunch` | 72h | 2.0 | 36 | 12h | 137 | ~30 P1–P3 jobs with bursty arrivals. P3 spot jobs can be sacrificed. |
| `p0_emergency` | 168h | 4.0 | 42 | 24h | 999 | ~40 mixed jobs + a 32-GPU P0 emergency arriving at hour 72. |

---

## Module-Level Functions

### `_make_job_id(index: int) -> str`

**Purpose:** Generate a zero-padded job identifier.

**Input:** Integer index (e.g., `42`)
**Output:** String like `"job_042"`

---

### `_generate_job_schedule(task_name: str, rng: random.Random) -> List[Dict]`

**Purpose:** Pre-generate the **complete deterministic job arrival schedule** for an entire episode. Uses a seeded RNG so every run of the same task produces identical jobs.

**Returns:** List of job dicts sorted by `arrival_hour`. Each dict contains:
- `job_id`, `priority`, `priority_label`, `gpu_count`, `duration_hours`, `deadline_hour`, `arrival_hour`

**Per-task generation logic:**

| Task | Jobs | GPU Sizes | Priorities | Deadlines | Special |
|------|------|-----------|------------|-----------|---------|
| `smooth_sailing` | 24 | 2, 4, 8 | P1 (~20%), P2 (~60%), P3 (~20%) | ~35% of P1/P2 have deadlines | Fixed arrival hours with some bursts |
| `deadline_crunch` | 30 | 2, 4, 8 | P1 (~37%), P2 (~50%), P3 (~13%) | P1: 85%, P2: 65%, P3: none | Random arrivals over 0–68h |
| `p0_emergency` | 40 + 1 | 1, 2, 4, 8 | P1, P2, P3 | 50% have deadlines | **Plus `job_P0_EMERGENCY`**: 32 GPUs, P0, arrives hour 72, deadline hour 132 |

---

## Class: `GpuSchedulerEnvironment`

Implements the OpenEnv `Environment` interface. Each WebSocket session gets its own instance (`SUPPORTS_CONCURRENT_SESSIONS = True`).

---

### `__init__(self) -> None`

**Purpose:** Allocate all instance variable containers. Does NOT set up task-specific state — that happens in `reset()`.

**State containers initialized:**

| Variable | Type | Purpose |
|----------|------|---------|
| `_state` | `State` | OpenEnv state (episode_id, step_count) |
| `_task_name` | `str` | Current task name |
| `_total_hours` | `float` | Episode duration |
| `_hours_per_step` | `float` | Simulated hours per agent step |
| `_lookahead_hours` | `float` | How far ahead the agent can see upcoming jobs |
| `_current_hour` | `float` | Current simulation clock |
| `_node_jobs` | `Dict[int, List[str]]` | node_id → list of job_ids running on it |
| `_node_gpu_used` | `Dict[int, int]` | node_id → number of GPUs allocated (source of truth) |
| `_job_schedule` | `List[Dict]` | Full pre-generated arrival plan |
| `_queue` | `List[JobInfo]` | Jobs waiting to be scheduled |
| `_active_jobs` | `Dict[str, JobInfo]` | job_id → currently running JobInfo |
| `_completed_jobs` | `List[JobInfo]` | Finished jobs |
| `_preempted_jobs` | `List[JobInfo]` | Jobs that were evicted |
| `_compute_burn_usd` | `float` | Cumulative cost in USD |
| `_cumulative_reward` | `float` | Total reward across episode |
| `_last_action_result` | `str` | Human-readable feedback shown to LLM |

**Grader metrics:** `_total_jobs_spawned`, `_sla_jobs_total`, `_sla_jobs_met` (float — supports partial credit), `_p0_job_completed`, `_cumulative_gpu_hrs_used`, `_cumulative_gpu_hrs_avail`

---

### `reset(self, task_name: Optional[str] = None) -> GpuSchedulerObservation`

**Purpose:** Start a fresh episode. Wipes all state, loads task config, generates job schedule, releases hour-0 jobs.

**Task resolution order:**
1. `task_name` kwarg (from `inference.py`)
2. `GPU_SCHEDULER_TASK` env var
3. `"smooth_sailing"` default

**Flow:**
1. Look up task in `TASK_CONFIGS`
2. Wipe ALL episode state (nodes, jobs, metrics)
3. Create seeded RNG → call `_generate_job_schedule()`
4. Call `_release_arriving_jobs(0.0, 0.0)` to put hour-0 jobs in queue
5. Return initial observation via `_build_observation(reward=0.0, done=False)`

**Returns:** `GpuSchedulerObservation` showing an empty cluster + any hour-0 queued jobs + upcoming arrivals within the lookahead window.

---

### `step(self, action: GpuSchedulerAction) -> GpuSchedulerObservation`

**Purpose:** Execute one agent decision and advance the simulation clock.

**Flow:**
1. Increment `step_count`
2. `_apply_action(action)` → immediate reward (0.0 for valid, negative for invalid/preempt)
3. `_advance_time(hours_per_step)` → time-driven reward (progress, burns, SLA, queue delay)
4. Sum rewards, accumulate into `_cumulative_reward`
5. Check if done: clock expired OR all work completed
6. Return observation via `_build_observation()`

**Returns:** `GpuSchedulerObservation` with updated state, reward, done flag. If `done=True`, grader score is embedded.

---

### `state` (property)

**Purpose:** Return current `State(episode_id, step_count)`. Used by OpenEnv's `GET /state` endpoint.

---

## Action Handlers

### `_apply_action(self, action: GpuSchedulerAction) -> float`

**Purpose:** Dispatch to the correct handler based on `action.action_type`.

| Action Type | Handler | Return |
|-------------|---------|--------|
| `WAIT` | Sets `_last_action_result` to "WAIT — advancing clock." | `0.0` |
| `SCHEDULE` | `_do_schedule(job_id, node_id)` | `0.0` success, `-0.1` invalid |
| `PREEMPT` | `_do_preempt(job_id)` | Moderate negative (burn penalty) |
| Unknown | Treated as WAIT | `0.0` |

Always sets `_last_action_result` so the LLM sees feedback.

---

### `_get_free_gpus(self, node_id: int) -> int`

**Purpose:** Return available GPUs on a node.

**Formula:** `GPUS_PER_NODE (8) - _node_gpu_used[node_id]`

`_node_gpu_used` is the **single source of truth** — never recomputed from job lists.

---

### `_do_schedule(self, job_id, node_id) -> float`

**Purpose:** Place a queued job onto one or more nodes.

**Validation checks (each returns `-0.1` penalty on failure):**
1. `job_id` must exist in `_queue`
2. `node_id` must be 0–7
3. Node must have enough free GPUs

**Two placement modes:**

#### Single-node jobs (gpu_count ≤ 8):
- Place on the specified `node_id`
- Allocate `gpu_count` GPUs on that node

#### Gang jobs (gpu_count > 8, e.g., 32-GPU P0):
- Needs `ceil(gpu_count / 8)` **fully free** nodes (all 8 GPUs available)
- Agent's `node_id` = preferred anchor node
- Environment auto-selects remaining nodes from fully-free pool
- Each selected node contributes all 8 GPUs

**On success:**
- Updates `_node_gpu_used` and `_node_jobs`
- Moves job from `_queue` → `_active_jobs`
- Sets job status to `"running"` with `assigned_nodes`
- Returns `0.0`

---

### `_do_preempt(self, job_id) -> float`

**Purpose:** Immediately halt a running job and return it to the queue.

**Validation:**
- Job must be in `_active_jobs` (else `-0.1`)
- P0 jobs are **non-preemptible** (else `-0.5`)

**Penalties applied:**
1. **Preemption Burn (sunk-cost + priority aware):**
   `gpu_count × min(elapsed_hours, 2.0) × hourly_rate × PREEMPTION_BURN_MULTIPLIER × (1 + progress × SUNK_COST_MULTIPLIER) × (1 + (2 − priority×0.5)) × REWARD_SCALE`
   Wasted work is capped at `PREEMPTION_CHECKPOINT_HOURS` (2h), modelling real checkpoint intervals.
2. **Checkpoint Rollback:** Job loses up to 2 hours of progress (rolls back to last checkpoint).

**On success:**
- Frees GPU slots on all assigned nodes
- Moves job from `_active_jobs` → front of `_queue`
- Adds to `_preempted_jobs` list
- Returns moderate negative reward (proportional to invested work, not remaining work)

---

## Time Advancement

### `_advance_time(self, hours: float) -> float`

**Purpose:** Advance the simulation clock and process all side effects. This is where most of the reward signal comes from.

**Executes several operations in strict order:**

#### 1. Update progress for all running jobs (dense reward)
- Compute contention on each job's nodes via `_average_contention()`
- **Smooth quadratic degradation:** `progress_rate = 1.0 - 0.4 × contention²`
  - At contention 0.5 (4/8 GPUs): 10% slowdown
  - At contention 0.75 (6/8 GPUs): 22.5% slowdown
  - At contention 1.0 (8/8 GPUs): 40% slowdown
- Progress delta: `(hours / duration_hours) × progress_rate` → stored on `JobInfo.progress_this_step`; `effective_rate = progress_rate`
- **Escalation:** after long `hours_waiting`, the effective priority index moves toward P0 (stronger progress reward weight).
- **Reward += (hours/duration) × progress_rate × PRIORITY_WEIGHTS[effective_priority] × CONTENTION_REWARD_FACTOR**
- Mark jobs that hit 100% as newly completed

#### 2. Finalise completed jobs
- Remove from `_active_jobs`, free GPU slots on all nodes
- Set status to `"completed"`
- SLA compliance (supports partial credit):
  - On-time (finished before deadline): **+1.0** SLA credit
  - Up to 4h late: **+0.5** SLA credit
  - 4–8h late: **+0.25** SLA credit
  - Beyond 8h late: no credit
- Track P0 emergency completion for grader

#### 3. Advance the clock
- `_current_hour = min(old_hour + hours, _total_hours)`

#### 4. SLA violations (quadratic lateness, capped per step)
- For every running/queued job past its deadline (after a small grace `MIN_PENALTY_HOURS`):
  - Let `late = min(current_hour − deadline − grace, SLA_LATE_CAP_HOURS)`
  - **Penalty** ≈ `late² × gpu_count × hourly_rate × SLA_QUADRATIC_MULTIPLIER × REWARD_SCALE`, capped by `MAX_SLA_PENALTY` each step.

#### 5. Release newly-arrived jobs
- Calls `_release_arriving_jobs(old_hour, new_hour)`
- Moves jobs from master schedule → visible queue

#### 6a. Queue-delay penalty
- For each queued job: `hours_in_queue += hours`, `hours_waiting += hours`
- For each active job: `hours_waiting += hours`
- P0 jobs: penalty = `gpu_count × hours × 2.0 × hourly_rate × REWARD_SCALE × 0.1`
- P1 jobs: penalty = `gpu_count × hours × 1.0 × hourly_rate × REWARD_SCALE × 0.1`
- P2/P3: no queue penalty
- **Starvation:** queued jobs with `hours_waiting > STARVATION_THRESHOLD_HOURS` pay an extra mild quadratic penalty (capped per step).

#### 6b. Idle-GPU opportunity cost (graduated rent)
- `idle_gpus = 64 - active_gpus`
- **Idle rate** depends on queue length: empty queue → `IDLE_GPU_BASE_RATE`; short queue → `IDLE_GPU_QUEUE_RATE`; long queue (`> IDLE_BACKLOG_THRESHOLD`) → `IDLE_GPU_BACKLOG_RATE`.
- If any P0/P1 job is queued, the idle rate is multiplied by `IDLE_PRIORITY_MULTIPLIER`.
- **Penalty = idle_gpus × hours × hourly_rate × REWARD_SCALE × idle_rate**
- Tracks `_cumulative_gpu_hrs_used` and `_cumulative_gpu_hrs_avail` for grader

#### 6c. Fragmentation tax
- Partially used nodes that cannot fit common GPU counts pay an hourly tax via `_calculate_fragmentation_penalty(hours)`.

---

### `_release_arriving_jobs(self, from_hour: float, to_hour: float) -> None`

**Purpose:** Move jobs from the master schedule into the visible queue when their `arrival_hour` falls within `[from_hour, to_hour]`.

**Deduplication:** Builds a set of already-seen job IDs (from queue + active + completed + preempted) to prevent double-releasing.

**For each new arrival:**
- Creates a `JobInfo` object from the raw dict
- Appends to `_queue`
- Increments `_total_jobs_spawned`
- If job has a deadline: increments `_sla_jobs_total`

---

## Utility Helpers

### `_average_contention(self, node_ids: List[int]) -> float`

**Purpose:** Compute average memory contention across nodes a job occupies.

**Formula:** `mean(used_gpus[n] / 8 for n in node_ids)`

**Range:** 0.0 (empty) to 1.0 (fully saturated). Used in the smooth quadratic degradation formula.

---

### `_all_work_done(self) -> bool`

**Purpose:** Check if the episode can terminate early (before time runs out).

**Returns `True` when ALL three conditions met:**
1. Every job in the master schedule has arrived (`arrival_hour ≤ current_hour`)
2. No active (running) jobs remain
3. Queue is empty

---

## Task Graders

### `_compute_grader_score(self) -> float`

**Purpose:** Compute the final normalised score (0.0–1.0) when `done=True`.

**Metrics used:**
- `completion_rate` = completed_jobs / total_jobs_spawned
- `utilisation` = cumulative_gpu_hrs_used / cumulative_gpu_hrs_available
- `sla_rate` = sla_jobs_met / sla_jobs_total (supports fractional credit for late completions)

**Per-task scoring formula:**

| Task | Formula | What it rewards |
|------|---------|-----------------|
| `smooth_sailing` | **80% completion + 20% utilisation** | Complete jobs efficiently |
| `deadline_crunch` | **60% SLA compliance + 40% completion** | Meet deadlines above all |
| `p0_emergency` | **50% P0 done (binary) + 30% SLA + 20% utilisation** | Handle the emergency, respect SLAs, stay efficient |

**Pass thresholds:** smooth_sailing ≥ 0.40, deadline_crunch ≥ 0.35, p0_emergency ≥ 0.30

---

## Observation Builder

### `_build_observation(self, reward, done) -> GpuSchedulerObservation`

**Purpose:** Assemble the full typed observation from current internal state.

**Builds:**

1. **8×8 cluster grid** — Each cell holds the index of the job occupying that GPU slot (`-1` = free). Allows the agent to visualise layout.

2. **Per-node `NodeInfo`** — For each of 8 nodes:
   - `node_id`, `total_gpus` (8), `used_gpus`, `free_gpus`
   - `memory_contention` = used_gpus / 8
   - `running_jobs` = list of job_ids on this node

3. **Active jobs** — Snapshot of all currently running `JobInfo` objects

4. **Queue** — Copy of `_queue` (jobs waiting to be scheduled, ready now)

5. **Upcoming jobs** — Preview of jobs arriving within the lookahead window (6h/12h/24h depending on task). Status set to `"upcoming"`. NOT schedulable yet — for planning only.

6. **Clock & economics** — `current_hour`, `total_hours`, `compute_burn_so_far`

7. **Metadata** — step_count, jobs_spawned, jobs_completed, jobs_preempted, sla_met, sla_total

8. **Score** — Only populated when `done=True` via `_compute_grader_score()`

---

## Complete Data Flow (One Step)

```
Agent sends: GpuSchedulerAction(action_type="SCHEDULE", job_id="job_003", node_id=2)
                                    │
                                    ▼
                        step(action) called
                                    │
                    ┌───────────────┼───────────────┐
                    ▼                               ▼
           _apply_action()                   _advance_time()
                    │                               │
           _do_schedule()               ┌───────────┼───────────┐
           validates job_id             │           │           │
           validates node GPUs          │           │           │
           moves job: queue→active      │           │           │
           updates _node_gpu_used       ▼           ▼           ▼
           returns 0.0 or -0.1    progress      SLA check   release
                                  update       (graduated)  new jobs
                    │                               │
                    └───────────────┼───────────────┘
                                    ▼
                         total_reward = sum
                                    │
                                    ▼
                       _build_observation()
                       (includes upcoming_jobs
                        from lookahead window)
                                    │
                                    ▼
                    GpuSchedulerObservation returned
                    (with reward, done, score if terminal)
```

---

## Reward Summary Table

| Source | When | Formula | Sign |
|--------|------|---------|------|
| Job progress | Every step, per running job | `(hours/duration) × progress_rate × priority_weight` | **+** |
| Invalid action | Bad SCHEDULE/PREEMPT | Fixed `-0.1` or `-0.5` (P0 preempt attempt) | **−** |
| Preemption burn | PREEMPT action | `gpu_count × min(elapsed_hrs, 2.0) × hourly_rate × 2.0 × scale` | **−** |
| SLA violation (initial) | Job first misses deadline | `gpu_count × duration × hourly_rate × 3.0 × scale` | **−** |
| SLA violation (continuing) | Each step while overdue | `gpu_count × hours × hourly_rate × 0.5 × scale` | **−** |
| Queue delay | P0/P1 sitting in queue | `gpu_count × hours × priority_factor × hourly_rate × scale × 0.1` | **−** |
| Idle GPU cost (with work) | Queue non-empty | `idle_gpus × hours × hourly_rate × scale × 0.2` | **−** |
| Idle GPU cost (no work) | Queue empty | `idle_gpus × hours × hourly_rate × scale × 0.05` | **−** |
