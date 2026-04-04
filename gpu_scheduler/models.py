# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
GpuScheduler Data Models
=========================
Typed Pydantic models that define every piece of data flowing between
the agent and the environment server.  Three layers:

  ActionType     — enum of the three discrete agent actions
  GpuSchedulerAction    — the action the agent sends to /step
  JobInfo        — snapshot of one job (queued or running)
  NodeInfo       — snapshot of one physical node (8 GPUs)
  GpuSchedulerObservation — the full state the agent sees after each step

All models inherit from OpenEnv base types so they are automatically
schema-exported, validated at the HTTP boundary, and compatible with
`openenv validate`.
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Action space
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    """
    The three discrete operations an agent can perform each step.

    SCHEDULE — place a queued job onto a specific node.
    PREEMPT  — immediately evict a running job back to the queue.
               Triggers a large penalty (preemption burn).
    WAIT     — advance the simulation clock one step without scheduling.
               Useful when the cluster is full or a better slot is coming.
    """
    SCHEDULE = "SCHEDULE"
    PREEMPT  = "PREEMPT"
    WAIT     = "WAIT"


class GpuSchedulerAction(Action):
    """
    Action submitted by the agent at each decision step.

    Fields are optional depending on action_type:
        SCHEDULE → requires job_id + node_id
        PREEMPT  → requires job_id only
        WAIT     → no additional fields needed

    The environment server validates combinations and returns an error in
    the observation's `last_action_result` field if invalid.
    """

    action_type: ActionType = Field(
        ...,
        description="Which operation to perform: SCHEDULE, PREEMPT, or WAIT",
    )
    job_id: Optional[str] = Field(
        default=None,
        description="ID of the job to schedule or preempt. Required for SCHEDULE/PREEMPT.",
    )
    node_id: Optional[int] = Field(
        default=None,
        ge=0,
        le=7,
        description="Target node index (0–7) for SCHEDULE actions.",
    )


# ---------------------------------------------------------------------------
# Sub-models embedded inside the Observation
# ---------------------------------------------------------------------------

class JobInfo(BaseModel):
    """
    Complete snapshot of one job at a given simulation step.

    Covers all lifecycle states: queued → running → completed / preempted.
    Both the queue preview and the active job list use this same model.
    """

    job_id: str = Field(
        ...,
        description="Unique identifier, e.g. 'job_00042'",
    )
    priority: int = Field(
        ...,
        ge=0,
        le=3,
        description="Integer priority: 0=P0 (highest/critical), 3=P3 (lowest/spot)",
    )
    priority_label: str = Field(
        ...,
        description="Human-readable label: 'P0', 'P1', 'P2', or 'P3'",
    )
    gpu_count: int = Field(
        ...,
        ge=1,
        le=64,
        description=(
            "Number of GPUs required. Values >8 span multiple nodes "
            "(e.g. 32 GPUs = 4 nodes — a gang-scheduled job)."
        ),
    )
    duration_hours: float = Field(
        ...,
        gt=0,
        description="Total runtime needed to complete this job, in simulated hours.",
    )
    elapsed_hours: float = Field(
        default=0.0,
        ge=0.0,
        description="Hours of compute already spent on this job (resets after preemption penalty).",
    )
    progress: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Fraction of work completed (0.0 → 1.0). "
            "Degrades below nominal rate when node memory contention is high."
        ),
    )
    status: str = Field(
        default="queued",
        description="Lifecycle state: 'queued' | 'running' | 'completed' | 'preempted'",
    )
    deadline_hour: Optional[float] = Field(
        default=None,
        description=(
            "Absolute simulation hour by which the job must finish. "
            "None = no SLA. Missing deadline triggers SLA violation penalty."
        ),
    )
    hours_in_queue: float = Field(
        default=0.0,
        ge=0.0,
        description="Hours this job has been waiting in queue. Used to compute queueing delay penalty.",
    )
    assigned_nodes: List[int] = Field(
        default_factory=list,
        description="Node IDs the job is currently occupying (empty if queued).",
    )
    arrival_hour: float = Field(
        default=0.0,
        ge=0.0,
        description="Simulated hour at which this job arrived in the cluster queue.",
    )


class NodeInfo(BaseModel):
    """
    Snapshot of one physical node's resource state.

    Each node has 8 physical GPUs.  Memory contention rises as more
    GPU-hours are colocated on the same PCIe bus, degrading job progress
    rates above the 0.7 contention threshold.
    """

    node_id: int = Field(
        ...,
        ge=0,
        le=7,
        description="Node index in the 8-node cluster (0–7).",
    )
    total_gpus: int = Field(
        default=8,
        description="Physical GPU capacity of this node (always 8 in this cluster).",
    )
    used_gpus: int = Field(
        default=0,
        ge=0,
        le=8,
        description="GPUs currently allocated to running jobs.",
    )
    free_gpus: int = Field(
        default=8,
        ge=0,
        le=8,
        description="GPUs available for scheduling (= total_gpus - used_gpus).",
    )
    memory_contention: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Normalised contention level (0.0 = no contention, 1.0 = fully saturated). "
            "Above 0.7 all jobs on this node suffer a progress rate penalty."
        ),
    )
    running_jobs: List[str] = Field(
        default_factory=list,
        description="job_ids of all jobs currently executing on this node.",
    )


# ---------------------------------------------------------------------------
# Observation — what the agent sees after every step / reset
# ---------------------------------------------------------------------------

class GpuSchedulerObservation(Observation):
    """
    Full environment snapshot delivered to the agent after each step.

    Structured into four logical sections:
      1. Cluster topology   — 8×8 grid + per-node resource/contention metrics
      2. Job lists          — active (running) jobs + lookahead queue
      3. Simulation clock   — current hour, episode horizon, burn so far
      4. Action feedback    — plain-text result of the last action taken

    The LLM agent should use all four sections to decide its next action.
    """

    # ------------------------------------------------------------------
    # 1. Cluster topology
    # ------------------------------------------------------------------
    cluster_grid: List[List[int]] = Field(
        ...,
        description=(
            "8×8 integer matrix. "
            "Each cell is the GPU slot index within a node row. "
            "Value = -1 (free) or a job index into active_jobs list."
        ),
    )
    nodes: List[NodeInfo] = Field(
        ...,
        description="Per-node resource snapshot for all 8 nodes.",
    )

    # ------------------------------------------------------------------
    # 2. Job lists
    # ------------------------------------------------------------------
    active_jobs: List[JobInfo] = Field(
        default_factory=list,
        description="All jobs currently executing on the cluster.",
    )
    queue: List[JobInfo] = Field(
        default_factory=list,
        description=(
            "Jobs waiting to be scheduled. "
            "Includes a ~4-hour lookahead window of upcoming arrivals "
            "so the agent can plan proactively."
        ),
    )

    # ------------------------------------------------------------------
    # 3. Simulation clock & economics
    # ------------------------------------------------------------------
    current_hour: float = Field(
        default=0.0,
        ge=0.0,
        description="Simulated wall-clock hour elapsed since episode start.",
    )
    total_hours: float = Field(
        default=24.0,
        gt=0,
        description="Total duration of this episode (24 / 72 / 168 hours by task).",
    )
    compute_burn_so_far: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Cumulative operational cost in USD accrued this episode. "
            "Cluster burns ~$4,167/hour at full 64-GPU capacity ($100k/day)."
        ),
    )

    # ------------------------------------------------------------------
    # 4. Task metadata & action feedback
    # ------------------------------------------------------------------
    task_name: str = Field(
        default="smooth_sailing",
        description="Which of the three tasks is running: smooth_sailing | deadline_crunch | p0_emergency",
    )
    last_action_result: str = Field(
        default="",
        description=(
            "Plain-English outcome of the previous action, e.g. "
            "'Scheduled job_003 on node 2 (6 GPUs)' or "
            "'INVALID: node 5 has only 2 free GPUs but job needs 4'."
        ),
    )
    score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Normalised grader score in [0.0, 1.0] — only populated when done=True. "
            "None at all intermediate steps. Used by inference.py to report episode result."
        ),
    )
