# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
GPU Scheduler Environment — Core Simulation Engine
====================================================
Simulates an 8-node × 8-GPU cluster ($100,000/day) where an agent
must schedule, preempt, and wait to minimise "compute burn".

Three tasks of escalating difficulty:
    EASY   — smooth_sailing:   24h,  low-demand FIFO scheduling
    MEDIUM — deadline_crunch:  72h,  overlapping P1/P2 jobs with tight SLAs
    HARD   — p0_emergency:    168h,  32-GPU P0 gang job arrives at hour 72
"""

import os
import random
from typing import Any, Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

# Handles both package-relative import (normal) and direct/Docker import path
try:
    from ..models import (
        ActionType,
        GpuSchedulerAction,
        GpuSchedulerObservation,
        JobInfo,
        NodeInfo,
    )
except ImportError:
    from models import (                          # direct run / Docker fallback
        ActionType,
        GpuSchedulerAction,
        GpuSchedulerObservation,
        JobInfo,
        NodeInfo,
    )


# ---------------------------------------------------------------------------
# Cluster geometry — fixed for the entire competition
# ---------------------------------------------------------------------------
NUM_NODES     = 8                            # physical nodes in the cluster
GPUS_PER_NODE = 8                            # GPUs per node
TOTAL_GPUS    = NUM_NODES * GPUS_PER_NODE    # 64 total

# ---------------------------------------------------------------------------
# Economic constants
# The cluster burns $100,000/day when fully loaded.
# We derive per-GPU per-hour cost from that top-line number.
# ---------------------------------------------------------------------------
DAILY_COST_USD      = 100_000.0                          # $100k/day at full 64-GPU capacity
HOURLY_RATE_CLUSTER = DAILY_COST_USD / 24.0              # ~$4,167/hour total cluster
HOURLY_RATE_PER_GPU = HOURLY_RATE_CLUSTER / TOTAL_GPUS   # ~$65/GPU/hour

# Scales reward into a manageable RL range (avoids huge raw-dollar magnitudes)
REWARD_SCALE = 1.0 / HOURLY_RATE_CLUSTER

# ---------------------------------------------------------------------------
# Simulation tuning knobs
# ---------------------------------------------------------------------------
CONTENTION_THRESHOLD       = 0.7   # above this ratio, job progress rate degrades
PREEMPTION_BURN_MULTIPLIER = 10.0  # preemption costs 10× the remaining compute value
MAX_CONTENTION_SLOWDOWN    = 0.5   # at 100% contention, jobs run at 50% speed

# P0 progress contributes 2× more reward than P3 (encourages prioritising critical work)
PRIORITY_WEIGHTS: Dict[int, float] = {0: 2.0, 1: 1.5, 2: 1.0, 3: 0.5}

# Queue delay penalty only fires for high-priority jobs sitting idle in queue
PRIORITY_QUEUE_FACTORS: Dict[int, float] = {0: 2.0, 1: 1.0, 2: 0.0, 3: 0.0}

# ---------------------------------------------------------------------------
# Task configuration table
# hours_per_step controls how many simulated hours each agent step covers,
# keeping total LLM API calls within the 20-minute runtime budget.
# ---------------------------------------------------------------------------
TASK_CONFIGS: Dict[str, Dict] = {
    "smooth_sailing": {
        "total_hours":    24.0,
        "hours_per_step":  1.0,    # 24 agent steps total
        "seed":           42,
        "description":    "Low-demand 24h window. Keep GPUs busy, avoid waste.",
    },
    "deadline_crunch": {
        "total_hours":    72.0,
        "hours_per_step":  2.0,    # 36 agent steps total
        "seed":           137,
        "description":    "72h with overlapping P1/P2 jobs and tight deadlines.",
    },
    "p0_emergency": {
        "total_hours":   168.0,
        "hours_per_step":  4.0,    # 42 agent steps total
        "seed":           999,
        "description":    "One-week run. A 32-GPU P0 job arrives at hour 72. Plan ahead.",
    },
}


# ---------------------------------------------------------------------------
# Job generation helpers (module-level, not methods)
# ---------------------------------------------------------------------------

def _make_job_id(index: int) -> str:
    """Return a short zero-padded job identifier, e.g. 'job_042'."""
    return f"job_{index:03d}"


def _generate_job_schedule(task_name: str, rng: random.Random) -> List[Dict]:
    """
    Pre-generate the complete deterministic job arrival schedule for one episode.

    Using a fixed RNG seed ensures the same task always produces the identical
    job sequence — which is critical for reproducible grader scores across
    multiple inference runs.

    Jobs are returned as plain dicts (not JobInfo objects) so they can be safely
    converted to fresh JobInfo instances at arrival time, avoiding shared-object bugs.

    Args:
        task_name: One of 'smooth_sailing', 'deadline_crunch', 'p0_emergency'.
        rng:       Seeded random.Random instance (seed comes from TASK_CONFIGS).

    Returns:
        List of job dicts sorted by arrival_hour, consumed by _release_arriving_jobs.
    """
    jobs: List[Dict] = []
    idx = 0

    if task_name == "smooth_sailing":
        # ~12 small, simple jobs spread evenly over 24 hours.
        # No deadlines — agent just needs to keep GPUs occupied efficiently.
        arrival_hours = [0, 1, 2, 4, 5, 7, 9, 11, 14, 17, 20, 23]
        for hour in arrival_hours:
            gpu_count = rng.choice([1, 2, 2, 4])       # small jobs only
            priority  = rng.choice([2, 2, 3])           # P2 / P3 mix
            jobs.append({
                "job_id":         _make_job_id(idx),
                "priority":       priority,
                "priority_label": f"P{priority}",
                "gpu_count":      gpu_count,
                "duration_hours": round(rng.uniform(2.0, 8.0), 1),
                "deadline_hour":  None,                 # no SLA on this task
                "arrival_hour":   float(hour),
            })
            idx += 1

    elif task_name == "deadline_crunch":
        # ~28 mixed P1/P2 jobs with bursty arrivals; 75% carry tight deadlines.
        # Agent must juggle SLA compliance AND utilisation across a 72h window.
        raw_arrivals = sorted(rng.uniform(0.0, 68.0) for _ in range(28))
        for hour in raw_arrivals:
            gpu_count = rng.choice([2, 4, 4, 8])
            priority  = rng.choice([1, 1, 2, 2, 2])
            duration  = round(rng.uniform(3.0, 16.0), 1)
            # Tight deadline: arrival time + full runtime + small grace window
            has_dl   = rng.random() < 0.75
            deadline = round(hour + duration + rng.uniform(2.0, 8.0), 1) if has_dl else None
            jobs.append({
                "job_id":         _make_job_id(idx),
                "priority":       priority,
                "priority_label": f"P{priority}",
                "gpu_count":      gpu_count,
                "duration_hours": duration,
                "deadline_hour":  deadline,
                "arrival_hour":   round(hour, 1),
            })
            idx += 1

    elif task_name == "p0_emergency":
        # 168h (1 week): mixed P1–P3 background load throughout,
        # THEN at hour 72: THE EMERGENCY — a 32-GPU P0 gang job.
        # Agent must proactively drain 4 nodes before hour 72 to avoid
        # needing to preempt jobs (which carries a massive burn penalty).
        raw_arrivals = sorted(rng.uniform(0.0, 140.0) for _ in range(40))
        for hour in raw_arrivals:
            gpu_count = rng.choice([1, 2, 4, 4, 8])
            priority  = rng.choice([1, 2, 2, 3, 3])
            duration  = round(rng.uniform(4.0, 48.0), 1)
            has_dl    = rng.random() < 0.5
            deadline  = round(hour + duration + rng.uniform(4.0, 16.0), 1) if has_dl else None
            jobs.append({
                "job_id":         _make_job_id(idx),
                "priority":       priority,
                "priority_label": f"P{priority}",
                "gpu_count":      gpu_count,
                "duration_hours": duration,
                "deadline_hour":  deadline,
                "arrival_hour":   round(hour, 1),
            })
            idx += 1

        # *** THE P0 EMERGENCY — deterministic, always at exactly hour 72 ***
        # Needs 4 fully-free nodes (32 GPUs). Deadline = hour 132 (48h runtime + 12h grace).
        jobs.append({
            "job_id":         "job_P0_EMERGENCY",
            "priority":       0,
            "priority_label": "P0",
            "gpu_count":      32,
            "duration_hours": 48.0,
            "deadline_hour":  132.0,    # arrival 72 + runtime 48 + grace 12
            "arrival_hour":   72.0,
        })

        # Re-sort after injecting the emergency job (it may not be the last entry)
        jobs.sort(key=lambda j: j["arrival_hour"])

    return jobs


# ---------------------------------------------------------------------------
# Environment class
# ---------------------------------------------------------------------------

class GpuSchedulerEnvironment(Environment):
    """
    GPU Cluster Scheduler Simulation Environment.

    Implements the full OpenEnv Environment interface:
        reset()  → GpuSchedulerObservation   (fresh episode, empty cluster)
        step()   → GpuSchedulerObservation   (with reward + done flag)
        state    → State                      (episode_id, step_count)

    Internal state:
        _node_jobs:     node_id → [job_ids]   (which jobs occupy each node)
        _node_gpu_used: node_id → int         (GPUs consumed per node — source of truth)
        _active_jobs:   job_id  → JobInfo     (currently running jobs)
        _queue:         [JobInfo]              (waiting to be scheduled)

    Task selection: reset() reads the GPU_SCHEDULER_TASK environment variable,
    which inference.py sets before each call to env.reset().
    """

    # Enables multiple concurrent WebSocket sessions — each gets its own instance
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        """
        Allocate all instance variable containers.

        reset() must always be called before step() — __init__ only creates the
        containers; it does not set up any task-specific state. This matches the
        OpenEnv contract where reset() marks the true start of an episode.
        """
        self._state = State(episode_id=str(uuid4()), step_count=0)

        # Task configuration — populated by reset()
        self._task_name:      str   = "smooth_sailing"
        self._total_hours:    float = 24.0
        self._hours_per_step: float = 1.0
        self._current_hour:   float = 0.0

        # Cluster state: one entry per physical node (0–7)
        self._node_jobs:     Dict[int, List[str]] = {i: [] for i in range(NUM_NODES)}
        self._node_gpu_used: Dict[int, int]       = {i: 0  for i in range(NUM_NODES)}

        # Job lifecycle containers
        self._job_schedule:   List[Dict]          = []   # full pre-generated arrival plan
        self._queue:          List[JobInfo]        = []   # jobs waiting to be scheduled
        self._active_jobs:    Dict[str, JobInfo]   = {}   # job_id → running JobInfo
        self._completed_jobs: List[JobInfo]        = []
        self._preempted_jobs: List[JobInfo]        = []

        # Episode-level metrics consumed by the task graders
        self._total_jobs_spawned:       int   = 0
        self._sla_jobs_total:           int   = 0
        self._sla_jobs_met:             int   = 0
        self._p0_job_completed:         bool  = False
        self._cumulative_gpu_hrs_used:  float = 0.0
        self._cumulative_gpu_hrs_avail: float = 0.0
        self._compute_burn_usd:         float = 0.0
        self._cumulative_reward:        float = 0.0

        # Plain-English result of the last action (shown to the LLM each step)
        self._last_action_result: str = ""

    # ------------------------------------------------------------------
    # OpenEnv interface — reset / step / state
    # ------------------------------------------------------------------

    def reset(self, task_name: Optional[str] = None) -> GpuSchedulerObservation:
        """
        Start a fresh episode for the specified task.

        Task resolution order (first non-None wins):
          1. task_name kwarg  — passed directly by inference.py via env.reset(task_name=...)
          2. GPU_SCHEDULER_TASK env var — fallback for local direct-server usage
          3. "smooth_sailing" — safe default

        Loads the task configuration, seeds the RNG, pre-generates the complete
        job arrival schedule, releases any hour-0 jobs into the visible queue,
        and returns the initial observation.

        Args:
            task_name: One of 'smooth_sailing', 'deadline_crunch', 'p0_emergency'.
                       If None, falls back to GPU_SCHEDULER_TASK env var or default.

        Returns:
            GpuSchedulerObservation of an empty cluster with hour-0 jobs in queue.
        """
        # Resolve task: kwarg > env var > default
        if task_name is None:
            task_name = os.getenv("GPU_SCHEDULER_TASK", "smooth_sailing")
        if task_name not in TASK_CONFIGS:
            task_name = "smooth_sailing"    # safe fallback

        cfg = TASK_CONFIGS[task_name]
        self._task_name      = task_name
        self._total_hours    = cfg["total_hours"]
        self._hours_per_step = cfg["hours_per_step"]
        self._current_hour   = 0.0

        # Wipe all episode state
        self._state           = State(episode_id=str(uuid4()), step_count=0)
        self._node_jobs       = {i: [] for i in range(NUM_NODES)}
        self._node_gpu_used   = {i: 0  for i in range(NUM_NODES)}
        self._queue           = []
        self._active_jobs     = {}
        self._completed_jobs  = []
        self._preempted_jobs  = []
        self._last_action_result = "Episode started. Cluster is empty."

        # Wipe grader metrics
        self._total_jobs_spawned       = 0
        self._sla_jobs_total           = 0
        self._sla_jobs_met             = 0
        self._p0_job_completed         = False
        self._cumulative_gpu_hrs_used  = 0.0
        self._cumulative_gpu_hrs_avail = 0.0
        self._compute_burn_usd         = 0.0
        self._cumulative_reward        = 0.0

        # Fixed seed → identical job schedule every run → reproducible grader scores
        rng = random.Random(cfg["seed"])
        self._job_schedule = _generate_job_schedule(task_name, rng)

        # Jobs arriving at exactly hour 0 go straight into the queue
        self._release_arriving_jobs(from_hour=0.0, to_hour=0.0)

        return self._build_observation(reward=0.0, done=False)

    def step(self, action: GpuSchedulerAction) -> GpuSchedulerObservation:
        """
        Execute one agent decision and advance the simulation clock.

        Step sequence:
            1. Apply the agent's action (SCHEDULE / PREEMPT / WAIT)
               Returns an immediate reward delta for the action itself.
            2. Advance the simulation clock by hours_per_step:
               - Update progress for all running jobs (with contention degradation)
               - Finalise completed jobs and check their SLA status
               - Penalise any jobs that have blown past their deadline
               - Release newly-arrived jobs into the visible queue
               - Apply queue-delay penalty for P0/P1 jobs waiting too long
               - Deduct idle-GPU opportunity cost
            3. Sum action reward + time reward; accumulate into episode total
            4. Check terminal conditions (clock expired or all work done)
            5. Return the new observation (grader score embedded in metadata if done)

        Args:
            action: Typed GpuSchedulerAction from the agent.

        Returns:
            GpuSchedulerObservation with updated cluster state, step reward, done flag.
        """
        self._state.step_count += 1

        # 1. Instant reward/penalty from the action itself
        reward_action = self._apply_action(action)

        # 2. Time-driven reward: progress, costs, SLA violations, queue delay
        reward_time = self._advance_time(self._hours_per_step)

        total_reward = reward_action + reward_time
        self._cumulative_reward += total_reward

        # 3. Termination check
        done = (
            self._current_hour >= self._total_hours
            or self._all_work_done()
        )

        return self._build_observation(reward=total_reward, done=done)

    @property
    def state(self) -> State:
        """
        Return the current OpenEnv State (episode_id + step_count).

        Consumed by the GET /state HTTP endpoint and used by openenv validate
        to confirm the environment is live and correctly tracking steps.
        """
        return self._state

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def _apply_action(self, action: GpuSchedulerAction) -> float:
        """
        Dispatch the agent's action to the appropriate handler.

        Always sets self._last_action_result with a human-readable outcome
        so the LLM can see exactly what happened and learn from mistakes.

        Args:
            action: The typed action from the agent.

        Returns:
            Immediate reward delta: 0.0 for WAIT, negative for invalid actions.
        """
        if action.action_type == ActionType.WAIT:
            self._last_action_result = "WAIT — advancing clock."
            return 0.0

        if action.action_type == ActionType.SCHEDULE:
            return self._do_schedule(action.job_id, action.node_id)

        if action.action_type == ActionType.PREEMPT:
            return self._do_preempt(action.job_id)

        # Catch-all for unexpected action types
        self._last_action_result = (
            f"UNKNOWN action type '{action.action_type}' — treated as WAIT."
        )
        return 0.0

    def _get_free_gpus(self, node_id: int) -> int:
        """
        Return the number of GPUs currently available on a given node.

        _node_gpu_used is the single source of truth for GPU allocation.
        It is updated atomically in _do_schedule, _do_preempt, and when
        jobs complete in _advance_time — never recomputed from job lists.

        Args:
            node_id: Node index in range [0, NUM_NODES).

        Returns:
            Integer count of free GPU slots on that node.
        """
        return GPUS_PER_NODE - self._node_gpu_used[node_id]

    def _do_schedule(
        self,
        job_id:  Optional[str],
        node_id: Optional[int],
    ) -> float:
        """
        Attempt to place a queued job onto one or more nodes.

        Single-node jobs (gpu_count <= 8):
            Placed on the node specified by node_id, provided it has enough
            free GPUs. Returns error + small penalty if not.

        Gang jobs (gpu_count > 8, e.g. the 32-GPU P0 emergency):
            Needs N = ceil(gpu_count / 8) FULLY free nodes (all 8 GPUs each).
            The agent's node_id acts as a preferred anchor; the environment
            auto-selects the remaining N-1 nodes from any other fully-free ones.
            This keeps the action space simple for the LLM while modelling the
            real bin-packing challenge of gang scheduling.

        Invalid actions receive a -0.1 penalty (episode continues; LLM sees error).

        Args:
            job_id:  ID of the queued job to place.
            node_id: Target node (0–7); used as anchor hint for gang jobs.

        Returns:
            0.0 on successful placement, -0.1 on any validation failure.
        """
        # --- Validate: job must be in the queue ---
        queue_job = next((j for j in self._queue if j.job_id == job_id), None)
        if queue_job is None:
            self._last_action_result = (
                f"INVALID: '{job_id}' not found in queue "
                f"(may be running, completed, or not yet arrived)."
            )
            return -0.1

        gpu_count    = queue_job.gpu_count
        nodes_needed = (gpu_count + GPUS_PER_NODE - 1) // GPUS_PER_NODE  # ceiling division

        if nodes_needed == 1:
            # ---- Single-node placement ----
            if node_id is None or node_id not in range(NUM_NODES):
                self._last_action_result = (
                    f"INVALID: node_id {node_id} is out of range. Must be 0–7."
                )
                return -0.1

            if self._get_free_gpus(node_id) < gpu_count:
                self._last_action_result = (
                    f"INVALID: node {node_id} has {self._get_free_gpus(node_id)} "
                    f"free GPUs but '{job_id}' needs {gpu_count}."
                )
                return -0.1

            # Allocate GPUs and record placement
            self._node_gpu_used[node_id] += gpu_count
            self._node_jobs[node_id].append(job_id)
            assigned_nodes = [node_id]

        else:
            # ---- Gang-scheduled multi-node placement ----
            # Each assigned node must contribute all 8 GPUs to the job
            fully_free = [
                n for n in range(NUM_NODES)
                if self._get_free_gpus(n) == GPUS_PER_NODE
            ]

            if len(fully_free) < nodes_needed:
                self._last_action_result = (
                    f"INVALID: '{job_id}' [{queue_job.priority_label}] needs "
                    f"{nodes_needed} fully-free nodes ({gpu_count} GPUs total). "
                    f"Only {len(fully_free)} fully-free node(s) available right now. "
                    f"WAIT for running jobs to finish, or PREEMPT lower-priority jobs."
                )
                return -0.1

            # Honour the agent's preferred anchor node if it is free
            if node_id is not None and node_id in fully_free:
                fully_free.remove(node_id)
                selected = [node_id] + fully_free[:nodes_needed - 1]
            else:
                selected = fully_free[:nodes_needed]

            assigned_nodes = selected
            for n in assigned_nodes:
                self._node_gpu_used[n] += GPUS_PER_NODE   # full node consumed
                self._node_jobs[n].append(job_id)

        # --- Move job: queue → active ---
        self._queue = [j for j in self._queue if j.job_id != job_id]
        queue_job.status         = "running"
        queue_job.assigned_nodes = assigned_nodes
        self._active_jobs[job_id] = queue_job

        self._last_action_result = (
            f"Scheduled '{job_id}' [{queue_job.priority_label}] "
            f"on node(s) {assigned_nodes} ({gpu_count} GPUs)."
        )
        return 0.0

    def _do_preempt(self, job_id: Optional[str]) -> float:
        """
        Immediately halt a running job and return it to the front of the queue.

        Applies two penalties:
          1. Preemption Burn Penalty:
               gpu_count × remaining_hours × hourly_rate × PREEMPTION_BURN_MULTIPLIER
             Represents wasted electricity and compute investment.
          2. Checkpoint Loss:
               Job loses up to 1 simulated hour of progress (models the real cost
               of rolling back to the last checkpoint and re-loading state).

        P0 jobs are protected — attempting to preempt one returns an error with
        an elevated penalty to strongly discourage the behaviour in the LLM.

        Args:
            job_id: ID of the currently running job to evict.

        Returns:
            Large negative float (burn penalty), or -0.1/-0.5 for invalid actions.
        """
        job = self._active_jobs.get(job_id)
        if job is None:
            self._last_action_result = (
                f"INVALID: '{job_id}' is not currently running."
            )
            return -0.1

        if job.priority == 0:
            # P0 jobs are mission-critical and non-preemptible
            self._last_action_result = (
                f"INVALID: P0 job '{job_id}' is non-preemptible. "
                f"Wait for lower-priority jobs to finish naturally."
            )
            return -0.5    # elevated penalty for even attempting this

        # --- Burn penalty: cost of all wasted GPU-hours ---
        remaining_hours = job.duration_hours * (1.0 - job.progress)
        burn_penalty = (
            job.gpu_count
            * remaining_hours
            * HOURLY_RATE_PER_GPU
            * PREEMPTION_BURN_MULTIPLIER
            * REWARD_SCALE
        )

        # --- Checkpoint loss: roll back up to 1 hour of progress ---
        loss_fraction   = min(1.0 / max(job.duration_hours, 1.0), job.progress)
        job.progress      = max(0.0, job.progress - loss_fraction)
        job.elapsed_hours = max(0.0, job.elapsed_hours - 1.0)

        # --- Free cluster resources ---
        gpus_per_node = GPUS_PER_NODE if len(job.assigned_nodes) > 1 else job.gpu_count
        for n in job.assigned_nodes:
            self._node_gpu_used[n] -= gpus_per_node
            if job_id in self._node_jobs[n]:
                self._node_jobs[n].remove(job_id)

        # --- Return to front of queue for rapid rescheduling ---
        job.status         = "queued"
        job.assigned_nodes = []
        self._queue.insert(0, job)
        del self._active_jobs[job_id]
        self._preempted_jobs.append(job)

        self._last_action_result = (
            f"PREEMPTED '{job_id}' [{job.priority_label}] — "
            f"burn ${burn_penalty / REWARD_SCALE:,.0f} | "
            f"progress rolled back to {job.progress:.1%}."
        )
        return -burn_penalty

    # ------------------------------------------------------------------
    # Time advancement
    # ------------------------------------------------------------------

    def _advance_time(self, hours: float) -> float:
        """
        Advance the simulation clock by `hours` and process all side effects.

        Executes six operations in strict order to avoid clock-skew bugs:
            1. Update progress for all running jobs (contention-adjusted rate)
            2. Detect jobs that reached 100% progress → mark as completed
            3. Advance the wall clock
            4. Detect SLA violations (deadline now in the past, job not done)
            5. Release newly-arrived jobs from the master schedule into the queue
            6. Apply queue-delay penalty for P0/P1 waiting in queue
               Deduct idle-GPU opportunity cost

        Args:
            hours: Simulated hours to advance (1.0, 2.0, or 4.0 depending on task).

        Returns:
            Combined float reward from all time-driven components.
        """
        reward = 0.0

        # 1. Update progress for every running job --------------------------------
        newly_completed: List[str] = []

        for job_id, job in list(self._active_jobs.items()):
            # Average GPU utilisation on the nodes this job occupies
            contention = self._average_contention(job.assigned_nodes)

            # Contention above threshold slows progress proportionally
            if contention > CONTENTION_THRESHOLD:
                excess        = (contention - CONTENTION_THRESHOLD) / (1.0 - CONTENTION_THRESHOLD)
                progress_rate = 1.0 - MAX_CONTENTION_SLOWDOWN * excess
            else:
                progress_rate = 1.0    # full speed below threshold

            delta             = (hours / job.duration_hours) * progress_rate
            job.progress      = min(1.0, job.progress + delta)
            job.elapsed_hours += hours * progress_rate

            # Progress reward weighted by job priority
            reward += delta * PRIORITY_WEIGHTS[job.priority]

            if job.progress >= 1.0:
                newly_completed.append(job_id)

        # 2. Finalise completed jobs BEFORE advancing the clock -------------------
        #    This ensures the SLA deadline comparison uses the correct end-of-step time.
        old_hour = self._current_hour
        new_hour = min(old_hour + hours, self._total_hours)

        for job_id in newly_completed:
            job           = self._active_jobs.pop(job_id)
            job.status    = "completed"

            # Free the GPU slots this job held on each assigned node
            gpus_per_node = GPUS_PER_NODE if len(job.assigned_nodes) > 1 else job.gpu_count
            for n in job.assigned_nodes:
                self._node_gpu_used[n] -= gpus_per_node
                if job_id in self._node_jobs[n]:
                    self._node_jobs[n].remove(job_id)
            job.assigned_nodes = []

            self._completed_jobs.append(job)

            # SLA compliance: did the job finish before its deadline?
            if job.deadline_hour is not None and new_hour <= job.deadline_hour:
                self._sla_jobs_met += 1

            # Track the special P0 emergency job for the hard-task grader
            if job_id == "job_P0_EMERGENCY":
                self._p0_job_completed = True

        # 3. Advance the simulation clock -----------------------------------------
        self._current_hour = new_hour

        # 4. SLA violations: penalise running/queued jobs past their deadline ------
        for job in list(self._active_jobs.values()) + list(self._queue):
            if job.deadline_hour and self._current_hour > job.deadline_hour:
                sla_penalty = (
                    job.gpu_count
                    * job.duration_hours
                    * HOURLY_RATE_PER_GPU
                    * 5.0           # SLA violation multiplier
                    * REWARD_SCALE
                )
                reward -= sla_penalty
                job.deadline_hour = None    # clear so we don't double-penalise next step

        # 5. Release newly-arrived jobs into the visible queue --------------------
        self._release_arriving_jobs(from_hour=old_hour, to_hour=self._current_hour)

        # 6a. Queue-delay penalty for high-priority jobs left waiting -------------
        for job in self._queue:
            job.hours_in_queue += hours
            factor = PRIORITY_QUEUE_FACTORS.get(job.priority, 0.0)
            if factor > 0.0:
                # Small per-step nudge to dispatch critical jobs promptly
                reward -= (
                    job.gpu_count * hours * factor
                    * HOURLY_RATE_PER_GPU * REWARD_SCALE * 0.1
                )

        # 6b. Idle-GPU opportunity cost: unused GPUs are wasted money -------------
        active_gpus = sum(self._node_gpu_used.values())
        idle_gpus   = TOTAL_GPUS - active_gpus

        # Track cumulative utilisation for grader score computation
        self._cumulative_gpu_hrs_used  += active_gpus * hours
        self._cumulative_gpu_hrs_avail += TOTAL_GPUS  * hours
        self._compute_burn_usd         += TOTAL_GPUS  * hours * HOURLY_RATE_PER_GPU

        # Penalise idle slots at 30% of full cost (opportunity cost, not full burn)
        reward -= idle_gpus * hours * HOURLY_RATE_PER_GPU * REWARD_SCALE * 0.3

        return reward

    def _release_arriving_jobs(self, from_hour: float, to_hour: float) -> None:
        """
        Move jobs from the master schedule into the visible queue when they arrive.

        Checks all jobs in self._job_schedule whose arrival_hour falls within
        [from_hour, to_hour]. Converts them to JobInfo objects and appends to
        self._queue. Also increments SLA tracking counters for the grader.

        Uses a set of already-seen job IDs to prevent double-releasing.

        Args:
            from_hour: Inclusive start of the arrival window.
            to_hour:   Inclusive end of the arrival window.
        """
        # Build the set of job IDs that have already entered any lifecycle stage
        already_seen: set = (
            {j.job_id for j in self._queue}
            | set(self._active_jobs.keys())
            | {j.job_id for j in self._completed_jobs}
            | {j.job_id for j in self._preempted_jobs}
        )

        for job_dict in self._job_schedule:
            if from_hour <= job_dict["arrival_hour"] <= to_hour:
                if job_dict["job_id"] not in already_seen:
                    job = JobInfo(**job_dict)
                    self._queue.append(job)
                    self._total_jobs_spawned += 1
                    if job.deadline_hour is not None:
                        self._sla_jobs_total += 1
                    already_seen.add(job.job_id)

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def _average_contention(self, node_ids: List[int]) -> float:
        """
        Compute the average memory contention across a set of nodes.

        Contention = used_gpus / GPUS_PER_NODE per node.
        Averaged across all nodes a job occupies.
        Values above CONTENTION_THRESHOLD (0.7) trigger progress rate penalties.

        Args:
            node_ids: List of node IDs a job is currently occupying.

        Returns:
            Average contention float in [0.0, 1.0]. Returns 0.0 for empty list.
        """
        if not node_ids:
            return 0.0
        return sum(
            self._node_gpu_used[n] / GPUS_PER_NODE for n in node_ids
        ) / len(node_ids)

    def _all_work_done(self) -> bool:
        """
        Return True when no further work remains in the episode.

        Triggers early termination only when:
            - All jobs in the master schedule have been released (arrived)
            - The running cluster is idle (no active jobs)
            - The queue is empty (no pending jobs)

        A perfect run where the agent schedules everything efficiently will
        end the episode before the time limit, signalling maximum efficiency.
        """
        all_released = all(
            j["arrival_hour"] <= self._current_hour
            for j in self._job_schedule
        )
        return all_released and not self._active_jobs and not self._queue

    # ------------------------------------------------------------------
    # Task graders — produce the normalised 0.0–1.0 score
    # ------------------------------------------------------------------

    def _compute_grader_score(self) -> float:
        """
        Compute the normalised task score injected into result.info when done=True.

        Each task weights different objectives to match its design intent:

        smooth_sailing:
            60% job completion rate  +  40% GPU utilisation
            → proves the agent can keep hardware busy without waste.

        deadline_crunch:
            60% SLA compliance  +  40% job completion rate
            → proves the agent prioritises time-sensitive work.

        p0_emergency:
            50% P0 job completed (binary)  +  30% SLA compliance  +  20% utilisation
            → proves the agent can reserve resources for a critical emergency.

        Returns:
            Score in [0.0, 1.0], rounded to 4 decimal places.
        """
        total       = max(self._total_jobs_spawned, 1)
        completed   = len(self._completed_jobs)

        completion_rate = completed / total
        utilisation     = (
            self._cumulative_gpu_hrs_used
            / max(self._cumulative_gpu_hrs_avail, 1.0)
        )
        sla_rate = self._sla_jobs_met / max(self._sla_jobs_total, 1)

        if self._task_name == "smooth_sailing":
            score = 0.6 * completion_rate + 0.4 * utilisation

        elif self._task_name == "deadline_crunch":
            score = 0.6 * sla_rate + 0.4 * completion_rate

        elif self._task_name == "p0_emergency":
            p0_done = 1.0 if self._p0_job_completed else 0.0
            score   = 0.5 * p0_done + 0.3 * sla_rate + 0.2 * utilisation

        else:
            score = completion_rate    # safe fallback for unknown tasks

        return round(max(0.0, min(score, 1.0)), 4)

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _build_observation(
        self,
        reward: float = 0.0,
        done:   bool  = False,
    ) -> GpuSchedulerObservation:
        """
        Assemble the full typed GpuSchedulerObservation from current internal state.

        Builds:
          - 8×8 cluster grid: each cell holds the index of the job occupying that
            GPU slot (−1 = free), allowing the agent to visualise cluster layout.
          - Per-node NodeInfo objects with live GPU counts and contention metrics.
          - Snapshot of active jobs and current queue.
          - Simulation clock and cumulative economic burn.
          - Episode metadata (step count, job counters).
          - Grader score in metadata["score"] when done=True, so inference.py
            can extract it via result.info.get("score").

        Args:
            reward: Step reward to embed in the observation base class.
            done:   Whether this is the terminal observation for this episode.

        Returns:
            Fully-populated GpuSchedulerObservation.
        """
        # --- 8×8 cluster grid ---
        active_list = list(self._active_jobs.values())
        job_to_idx  = {j.job_id: i for i, j in enumerate(active_list)}

        grid: List[List[int]] = []
        for node_id in range(NUM_NODES):
            row      = [-1] * GPUS_PER_NODE
            gpu_slot = 0
            for job_id in self._node_jobs[node_id]:
                job  = self._active_jobs.get(job_id)
                if job is None:
                    continue
                idx   = job_to_idx.get(job_id, -1)
                # Gang jobs use all 8 GPUs per node; single-node jobs use job.gpu_count
                gpus  = GPUS_PER_NODE if len(job.assigned_nodes) > 1 else job.gpu_count
                gpus  = min(gpus, GPUS_PER_NODE - gpu_slot)  # don't overflow the row
                for g in range(gpus):
                    row[gpu_slot + g] = idx
                gpu_slot += gpus
            grid.append(row)

        # --- Per-node NodeInfo with live contention ---
        nodes: List[NodeInfo] = [
            NodeInfo(
                node_id           = n,
                total_gpus        = GPUS_PER_NODE,
                used_gpus         = self._node_gpu_used[n],
                free_gpus         = self._get_free_gpus(n),
                memory_contention = round(self._node_gpu_used[n] / GPUS_PER_NODE, 3),
                running_jobs      = list(self._node_jobs[n]),
            )
            for n in range(NUM_NODES)
        ]

        # --- Metadata: always include episode stats; inject score when terminal ---
        meta: Dict[str, Any] = {
            "step_count":     self._state.step_count,
            "jobs_spawned":   self._total_jobs_spawned,
            "jobs_completed": len(self._completed_jobs),
            "jobs_preempted": len(self._preempted_jobs),
            "sla_met":        self._sla_jobs_met,
            "sla_total":      self._sla_jobs_total,
        }
        # Terminal score: populate the observation-level score field so it
        # survives OpenEnv's serialize_observation (which strips metadata).
        terminal_score: Optional[float] = None
        if done:
            terminal_score     = self._compute_grader_score()
            meta["score"]      = terminal_score   # also in metadata for completeness

        return GpuSchedulerObservation(
            cluster_grid        = grid,
            nodes               = nodes,
            active_jobs         = active_list,
            queue               = list(self._queue),
            current_hour        = round(self._current_hour, 2),
            total_hours         = self._total_hours,
            compute_burn_so_far = round(self._compute_burn_usd, 2),
            task_name           = self._task_name,
            last_action_result  = self._last_action_result,
            done                = done,
            reward              = reward,
            metadata            = meta,
            score               = terminal_score,
        )
