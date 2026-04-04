#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
GPUScheduler-Env Inference Script
==================================
Runs an LLM-driven scheduling agent against all three environment tasks
and emits structured stdout logs required by the hackathon evaluator.

MANDATORY ENVIRONMENT VARIABLES
--------------------------------
  HF_TOKEN      — HuggingFace / API key for LLM access
  IMAGE_NAME    — Docker image tag for the GpuScheduler environment server
  API_BASE_URL  — LLM API endpoint  (default: HuggingFace inference router)
  MODEL_NAME    — Model identifier  (default: Qwen/Qwen2.5-72B-Instruct)

STDOUT FORMAT (one block per task)
------------------------------------
  [START] task=<name> env=gpu_scheduler model=<model>
  [STEP]  step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>

TASK SCHEDULE
--------------
  smooth_sailing   — 24 simulated hours, 24 agent steps  (1 h/step)
  deadline_crunch  — 72 simulated hours, 36 agent steps  (2 h/step)
  p0_emergency     — 168 simulated hours, 42 agent steps (4 h/step)

The step granularity is intentional: the environment advances the
simulation clock by N hours per step() call, which keeps LLM API
calls within the 20-minute runtime budget while preserving rich
multi-step trajectories.

RUN
----
  IMAGE_NAME=gpu_scheduler-env:latest \\
  HF_TOKEN=hf_...                     \\
  python inference.py

  Run a subset of tasks (comma-separated names):
  INFERENCE_ONLY=p0_emergency python inference.py
"""

import asyncio
import os
import re
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI

# Load .env file from the project root (same directory as this script)
_env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(_env_path)

# Import typed client + models from the gpu_scheduler package
from gpu_scheduler import (
    ActionType,
    GpuSchedulerAction,
    GpuSchedulerEnv,
    GpuSchedulerObservation,
    JobInfo,
)

# Import rich logger for enhanced output
try:
    from gpu_scheduler.rich_logger import (
        log_step_table,
        log_episode_start,
        log_episode_end,
        log_error_summary,
    )
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# ---------------------------------------------------------------------------
# Environment variables — loaded from .env file or shell environment
# ---------------------------------------------------------------------------
IMAGE_NAME   = os.getenv("IMAGE_NAME")          # HF Space URL or Docker image
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
BENCHMARK    = "gpu_scheduler"

# Comma-separated task names to run (subset of TASKS). Example: INFERENCE_ONLY=p0_emergency
INFERENCE_ONLY = os.getenv("INFERENCE_ONLY", "").strip()

# ---------------------------------------------------------------------------
# Task configuration
# (task_name, max_agent_steps, grader_success_threshold)
#
# max_agent_steps controls how many LLM calls happen per episode.
# The environment internally advances (total_hours / max_steps) simulated
# hours per step, so every step still sees meaningful state changes.
# ---------------------------------------------------------------------------
TASKS: List[Tuple[str, int, float]] = [
    ("smooth_sailing",  24, 0.40),   # Easy:   24h sim,  1 h/step
    ("deadline_crunch", 36, 0.35),   # Medium: 72h sim,  2 h/step
    ("p0_emergency",    42, 0.30),   # Hard:  168h sim,  4 h/step
]

# LLM sampling config — lower temperature = more deterministic scheduling
TEMPERATURE = 0.3
MAX_TOKENS  = 200   # Enough for action line + brief chain-of-thought


# ---------------------------------------------------------------------------
# System prompt — tells the LLM exactly what it is and how to act
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = textwrap.dedent("""
    You are an EXPERT AI GPU cluster scheduler managing a $100K/day cluster.
    Goal: MAXIMIZE GPU UTILIZATION while meeting SLAs and P0 emergencies (idle GPUs are heavily penalized).
    Missing deadlines causes very large negative rewards (SLA penalties).

    === DEADLINE MATH (do this every step) ===

    For each queued/running job with a deadline:
      slack      = deadline_hour - current_hour
      work_left  = duration_hours * (1 - progress)
    Interpretation:
      slack < 0              → already late — schedule or expect SLA penalty
      slack < work_left      → will miss finish — highest priority
      slack < work_left*1.2  → CRITICAL (barely time to finish)
      slack < work_left*1.5  → URGENT
    If the observation shows [OVERDUE], [CRITICAL], or [URGENT] on a job, prefer scheduling it
    over lower-priority work that has slack.

    === MANDATORY SCHEDULING ALGORITHM ===

    Every step, follow this decision tree:

    1. CHECK QUEUE: Are there jobs waiting?
       → YES: Proceed to step 2
       → NO: If nothing to do, WAIT

    2. PRIORITIZE: Consider the JOB QUEUE (sorted by urgency in the observation) in this order:
       a) [OVERDUE] / [CRITICAL] queued jobs — SCHEDULE first if they fit any node
       b) P0 jobs (emergency) — when in queue and 4 fully-free nodes exist
       c) [URGENT] and P1 with tight slack
       d) Other jobs; P2/P3 by hours_in_queue

    3. MATCH RESOURCES: For the chosen job:
       a) Read gpu_count from the queue row
       b) Scan NODES: pick node_id where free_gpus >= gpu_count (single-node jobs)
       c) GANG JOB (gpu_count > 8): need enough fully-free nodes. Use GANG SCHEDULING line.

    4. TAKE ACTION:
       → Resources fit: SCHEDULE <job_id> <node_id>  (node_id is integer 0–7 only)
       → P0/gang blocked by lower-priority work: PREEMPT a P2/P3 running job if needed
       → Truly no placement: WAIT only if nothing fits

    5. NEVER:
       - Repeat the same SCHEDULE after INVALID — read Last action and change node or job
       - Use prose like "Node 0" on line 1 — use SCHEDULE job_000 0
       - Ignore DEADLINE CRITICAL section or [CRITICAL] queue tags

    === P0 EMERGENCY (task p0_emergency) ===

    job_p0_emergency arrives at simulated hour 72; needs 4 fully-free nodes (32 GPUs).
    From hour ~56 onward (P0 PREP in observation): avoid filling those nodes with long-running P2/P3;
    before hour 72, PREEMPT low-priority runners if GANG SCHEDULING shows fewer than 4 fully-free nodes.
    When job_p0_emergency is in JOB QUEUE: SCHEDULE job_p0_emergency <anchor> on any fully-free node id.

    === CRITICAL RULES ===

    1. Prefer SCHEDULE over WAIT whenever any queued job can be placed
    2. Match job gpu_count to node free_gpus before SCHEDULE
    3. Job IDs must match JOB QUEUE exactly (typically lowercase job_…)

    ACTIONS (exactly one on line 1)
    -------------------------------
    SCHEDULE <job_id> <node_id>
    PREEMPT <job_id>
    WAIT

    OUTPUT: Line 1 = action only; optional brief reasoning after that.
""").strip()


# ---------------------------------------------------------------------------
# Observation formatter — converts the rich observation object to plain text
# that the LLM can reason over
# ---------------------------------------------------------------------------

def _slack_and_work_left(job: JobInfo, current_hour: float) -> Tuple[float, float]:
    """Time until deadline and remaining compute hours (queued jobs use progress 0)."""
    if job.deadline_hour is None:
        return 0.0, 0.0
    slack = job.deadline_hour - current_hour
    work_left = job.duration_hours * (1.0 - job.progress)
    return slack, work_left


def _deadline_urgency_sort_key(job: JobInfo, current_hour: float) -> float:
    """Lower = more urgent (negative if cannot finish before deadline at current rate)."""
    if job.deadline_hour is None:
        return 99999.0
    slack, work_left = _slack_and_work_left(job, current_hour)
    return slack - work_left


def _deadline_urgency_tag(job: JobInfo, current_hour: float) -> str:
    """Short tag for queue/running rows: overdue / critical / urgent."""
    if job.deadline_hour is None:
        return ""
    slack, work_left = _slack_and_work_left(job, current_hour)
    if slack < 0:
        return " [OVERDUE]"
    if work_left > 0 and slack < work_left:
        return " [CRITICAL]"
    if work_left > 0 and slack < work_left * 1.2:
        return " [CRITICAL]"
    if work_left > 0 and slack < work_left * 1.5:
        return " [URGENT]"
    return ""


def _node_capacity_label(used_gpus: int, total_gpus: int) -> str:
    """EMPTY / PARTIAL / FILLED — single source of truth for node status strings."""
    if used_gpus == 0:
        return "EMPTY"
    if used_gpus >= total_gpus:
        return "FILLED"
    return "PARTIAL"


def format_nodes_status_summary(obs: GpuSchedulerObservation) -> str:
    """
    One-line grouping of node ids by capacity state (for terminal / Rich output).
    Mirrors labels used in format_observation().
    """
    empty: List[str] = []
    partial: List[str] = []
    filled: List[str] = []
    for n in obs.nodes:
        tag = _node_capacity_label(n.used_gpus, n.total_gpus)
        if tag == "EMPTY":
            empty.append(str(n.node_id))
        elif tag == "FILLED":
            filled.append(str(n.node_id))
        else:
            partial.append(str(n.node_id))
    parts: List[str] = []
    if empty:
        parts.append(f"Empty: {','.join(empty)}")
    if partial:
        parts.append(f"Partial: {','.join(partial)}")
    if filled:
        parts.append(f"Filled: {','.join(filled)}")
    return " │ ".join(parts) if parts else "(no node data)"


def format_observation(obs: GpuSchedulerObservation, step: int) -> str:
    """
    Render a GpuSchedulerObservation as a compact, human-readable block.

    Structured into four sections the LLM can scan quickly:
      1. Episode header (step, hour, burn)
      2. Node table (GPUs used/free + contention bar)
      3. Running jobs table
      4. Queue table (with deadlines highlighted)

    Args:
        obs:  The observation returned by env.step() or env.reset().
        step: Current agent step number (for the header).

    Returns:
        Multi-line string ready to embed in the LLM user prompt.
    """
    lines: List[str] = []

    # --- Header ---
    hours_left = obs.total_hours - obs.current_hour
    lines += [
        f"=== Step {step} | Hour {obs.current_hour:.0f} / {obs.total_hours:.0f} "
        f"({hours_left:.0f}h remaining) ===",
        f"Compute Burn: ${obs.compute_burn_so_far:,.0f}",
        f"Last action: {obs.last_action_result or '(start of episode)'}",
        "",
    ]

    # --- Node table ---
    lines.append(
        "NODES  (node_id | used/total GPUs | contention | status | running jobs)"
    )
    for node in obs.nodes:
        # Visual contention bar: ▓ per 10% contention, ░ for free headroom
        filled  = int(node.memory_contention * 10)
        bar     = "▓" * filled + "░" * (10 - filled)
        # Flag nodes that are in the danger zone for contention degradation
        warn    = " ⚠ CONTENTION" if node.memory_contention >= 0.7 else ""
        fill_status = f" [{_node_capacity_label(node.used_gpus, node.total_gpus)}]"
        jobs_str = ", ".join(node.running_jobs) if node.running_jobs else "idle"
        lines.append(
            f"  Node {node.node_id}: {node.used_gpus}/{node.total_gpus} GPUs"
            f" | [{bar}] {node.memory_contention:.2f}{warn}{fill_status}"
            f" | {jobs_str}"
        )

    fully_free_ids = [n.node_id for n in obs.nodes if n.free_gpus == 8]
    ffc = getattr(obs, "fully_free_nodes_count", len(fully_free_ids))
    lines += [
        "",
        f"GANG SCHEDULING: {ffc}/4 nodes fully free (need 4 for a 32-GPU gang job)",
        f"  Fully free node ids: {fully_free_ids if fully_free_ids else 'NONE — wait, preempt, or spread load'}",
    ]
    up_p0 = getattr(obs, "upcoming_p0_jobs", None) or []
    if up_p0:
        lines.append(f"  Upcoming P0 within ~8h: {', '.join(up_p0)}")
    if obs.task_name == "p0_emergency" and 56.0 <= obs.current_hour < 72.0:
        hours_until_p0 = 72.0 - obs.current_hour
        lines.append(
            f"  P0 PREP: job_p0_emergency arrives in ~{hours_until_p0:.0f}h — need 4 fully-free nodes; "
            f"clear/preempt P2/P3 on reserved footprint if GANG count is low."
        )
    err = getattr(obs, "last_action_error_code", None)
    if err:
        lines.append(f"  Last action error code: {err}")

    # --- Running jobs table ---
    lines += ["", "RUNNING JOBS  (job_id | priority | GPUs | progress | deadline | slack vs work)"]
    if obs.active_jobs:
        for job in obs.active_jobs:
            dl = f"h{job.deadline_hour:.0f}" if job.deadline_hour else "no deadline"
            tag = _deadline_urgency_tag(job, obs.current_hour)
            slack_w = ""
            if job.deadline_hour is not None:
                s, w = _slack_and_work_left(job, obs.current_hour)
                slack_w = f" | {s:+.1f}h to dl, ~{w:.1f}h work left"
            urgent = ""
            if job.deadline_hour and (job.deadline_hour - obs.current_hour) < (
                job.duration_hours * (1 - job.progress)
            ):
                urgent = " AT_RISK"
            lines.append(
                f"  {job.job_id:12s} [{job.priority_label}] | {job.gpu_count:2d} GPUs"
                f" | {job.progress:5.1%} done | nodes={job.assigned_nodes}"
                f" | {dl}{tag}{urgent}{slack_w}"
            )
    else:
        lines.append("  (no jobs currently running)")

    # --- Deadline-critical queued jobs (most urgent first) ---
    critical_queued: List[Tuple[JobInfo, float, float]] = []
    for job in obs.queue:
        if job.deadline_hour is None:
            continue
        s, w = _slack_and_work_left(job, obs.current_hour)
        if w > 0 and s < w * 1.5:
            critical_queued.append((job, s, w))
    if critical_queued:
        critical_queued.sort(key=lambda t: t[1])
        lines += ["", "DEADLINE CRITICAL (queued — schedule if a node fits)"]
        for job, s, w in critical_queued[:8]:
            tag = _deadline_urgency_tag(job, obs.current_hour)
            lines.append(
                f"  {job.job_id} [{job.priority_label}] {tag}: "
                f"{s:.1f}h until deadline, ~{w:.1f}h work left — prioritize over loose P2/P3"
            )

    # --- Queue table (sorted by deadline urgency) ---
    lines += ["", "JOB QUEUE  (sorted by deadline urgency | id | pri | GPUs | dur | queued | deadline)"]
    if obs.queue:
        sorted_q = sorted(obs.queue, key=lambda j: _deadline_urgency_sort_key(j, obs.current_hour))
        for job in sorted_q[:20]:
            dl = f"h{job.deadline_hour:.0f}" if job.deadline_hour else "none"
            tag = _deadline_urgency_tag(job, obs.current_hour)
            slack_w = ""
            if job.deadline_hour is not None:
                s, w = _slack_and_work_left(job, obs.current_hour)
                slack_w = f" | slack {s:.1f}h, work ~{w:.1f}h"
            lines.append(
                f"  {job.job_id:12s} [{job.priority_label}] | {job.gpu_count:2d} GPUs"
                f" | {job.duration_hours:.0f}h runtime"
                f" | q {job.hours_in_queue:.1f}h | {dl}{tag}{slack_w}"
            )
        if len(sorted_q) > 20:
            lines.append(f"  ... ({len(sorted_q) - 20} more jobs in queue)")
    else:
        lines.append("  (queue is empty)")

    lines.append(format_recommended_action_section(obs))
    lines += ["", "=== SMART HINTS ===", suggest_smart_action(obs)]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Action parser — extracts a valid GpuSchedulerAction from raw LLM text
# ---------------------------------------------------------------------------

def parse_action(response: str) -> GpuSchedulerAction:
    """
    Parse the LLM's text response into a typed GpuSchedulerAction.

    Reads only the first line of the response (the rest is optional reasoning).
    Falls back to WAIT on any parse failure so the episode never crashes.

    Patterns recognised (case-insensitive):
        SCHEDULE <job_id> <node_id>
        PREEMPT  <job_id>
        WAIT

    Args:
        response: Raw string from the LLM completion.

    Returns:
        A valid GpuSchedulerAction — never raises.
    """
    # Only parse the first line; ignore chain-of-thought below it
    first_line = response.strip().split("\n")[0].strip()

    # SCHEDULE job_id node_id  (allow "node_0", "Node 3", plain integer; case-insensitive)
    m = re.match(
        r"SCHEDULE\s+(\S+)\s+(?:node_\s*|node\s+)?(\d+)\b",
        first_line,
        re.IGNORECASE,
    )
    if m:
        return GpuSchedulerAction(
            action_type="SCHEDULE",
            job_id=m.group(1).lower().rstrip(".,);"),
            node_id=int(m.group(2)),
        )

    # PREEMPT job_id
    m = re.match(r"PREEMPT\s+(\S+)", first_line, re.IGNORECASE)
    if m:
        return GpuSchedulerAction(
            action_type="PREEMPT",
            job_id=m.group(1).lower(),
        )

    # WAIT (or anything else — safe default)
    return GpuSchedulerAction(action_type="WAIT")


def format_executed_action(action: GpuSchedulerAction) -> str:
    """Single-line description of the action actually sent to the environment."""
    if action.action_type == ActionType.WAIT:
        return "WAIT"
    if action.action_type == ActionType.SCHEDULE:
        return f"SCHEDULE {action.job_id} Node {action.node_id}"
    if action.action_type == ActionType.PREEMPT:
        return f"PREEMPT {action.job_id}"
    return str(action.action_type)


def suggest_smart_action(obs: GpuSchedulerObservation) -> str:
    """Heuristic hints appended to the user prompt (non-binding)."""
    hints: List[str] = []

    p0_jobs = [j for j in obs.queue if j.priority == 0]
    if p0_jobs:
        p0 = p0_jobs[0]
        need = (p0.gpu_count + 7) // 8
        fully_free = [n for n in obs.nodes if n.free_gpus == 8]
        hints.append(
            f"P0 in queue: {p0.job_id} needs {need} fully-free node(s), {len(fully_free)} ready now."
        )
        if len(fully_free) >= need:
            hints.append(
                f"  Ready: schedule on anchor node {fully_free[0].node_id} "
                f"(env picks other fully-free nodes for gang jobs)."
            )
        else:
            hints.append(
                f"  Short {need - len(fully_free)} node(s): consider PREEMPT on P2/P3 or WAIT."
            )

    urgent: List[str] = []
    q_sorted = sorted(
        obs.queue,
        key=lambda j: _deadline_urgency_sort_key(j, obs.current_hour),
    )
    for job in q_sorted:
        if not job.deadline_hour:
            continue
        rem = job.duration_hours * (1.0 - job.progress)
        slack = job.deadline_hour - obs.current_hour
        if rem > 0 and slack < rem * 1.5:
            urgent.append(f"{job.job_id} ({slack:.1f}h to dl, ~{rem:.1f}h work)")
        if len(urgent) >= 5:
            break
    if urgent:
        hints.append("Deadline pressure: " + "; ".join(urgent[:5]))

    if (
        obs.task_name == "p0_emergency"
        and 56.0 <= obs.current_hour < 72.0
        and not p0_jobs
    ):
        ffc_n = len([n for n in obs.nodes if n.free_gpus == 8])
        hints.append(
            f"P0 prep window (h{obs.current_hour:.0f}): target 4 fully-free nodes before h72; "
            f"currently {ffc_n} fully-free."
        )

    low_pri_run = [j for j in obs.active_jobs if j.priority >= 2]
    if p0_jobs and len([n for n in obs.nodes if n.free_gpus == 8]) < 4 and low_pri_run:
        hints.append(f"Preempt candidates (lower priority): {low_pri_run[0].job_id}")

    if not hints:
        return "No critical hints."
    return "\n".join(hints)


def _queue_priority_score(job: JobInfo, current_hour: float) -> float:
    """Higher score = schedule sooner (used for hints and heuristic fallback)."""
    base = 1000.0 * (4 - job.priority)
    if job.deadline_hour is not None:
        slack = job.deadline_hour - current_hour
        remaining = job.duration_hours * (1.0 - job.progress)
        if remaining <= 0:
            pass
        elif slack <= 0:
            base += 10000.0
        elif slack < remaining:
            base += 5000.0
        elif slack < remaining * 1.2:
            base += 2000.0
        elif slack < remaining * 1.5:
            base += 1000.0
        elif slack < remaining * 2.0:
            base += 500.0
    base += job.hours_in_queue * 10.0
    return base


def format_recommended_action_section(obs: GpuSchedulerObservation) -> str:
    """One concrete SCHEDULE line when placement is possible; otherwise guidance."""
    lines: List[str] = ["", "=== RECOMMENDED ACTION ==="]

    if obs.task_name == "p0_emergency" and 56.0 <= obs.current_hour < 72.0:
        p0_in_queue = any(j.job_id == "job_p0_emergency" for j in obs.queue)
        ffc = getattr(
            obs,
            "fully_free_nodes_count",
            sum(1 for n in obs.nodes if n.free_gpus == 8),
        )
        if not p0_in_queue and ffc < 4:
            low_pri = [j for j in obs.active_jobs if j.priority >= 2]
            if low_pri:
                lines.append(
                    f"P0 lands at hour 72 — need 4 fully-free nodes; only {ffc} now. "
                    f"Strongly consider: PREEMPT {low_pri[0].job_id} (or another P2/P3 victim)."
                )

    if not obs.queue:
        lines.append("No jobs in queue. WAIT for arrivals.")
        return "\n".join(lines)

    sorted_queue = sorted(
        obs.queue,
        key=lambda j: _queue_priority_score(j, obs.current_hour),
        reverse=True,
    )

    for job in sorted_queue:
        if job.gpu_count <= 8:
            eligible = [n.node_id for n in obs.nodes if n.free_gpus >= job.gpu_count]
            if not eligible:
                continue
            nid = eligible[0]
            free = next(n.free_gpus for n in obs.nodes if n.node_id == nid)
            urgency = ""
            if job.deadline_hour is not None:
                slack = job.deadline_hour - obs.current_hour
                remaining = job.duration_hours * (1.0 - job.progress)
                if slack <= 0 or (remaining > 0 and slack < remaining):
                    urgency = "DEADLINE CRITICAL - "
                elif remaining > 0 and slack < remaining * 1.5:
                    urgency = "URGENT DEADLINE - "
            lines.append(
                f"{urgency}SCHEDULE {job.job_id} {nid} "
                f"({job.gpu_count} GPUs; node {nid} has {free} free)"
            )
            return "\n".join(lines)

        nodes_needed = (job.gpu_count + 7) // 8
        ffc = getattr(
            obs,
            "fully_free_nodes_count",
            sum(1 for n in obs.nodes if n.free_gpus == 8),
        )
        if ffc >= nodes_needed:
            free_node = next(n.node_id for n in obs.nodes if n.free_gpus == 8)
            lines.append(
                f"SCHEDULE {job.job_id} {free_node} "
                f"(gang job; {ffc}/{nodes_needed} fully-free nodes ready)"
            )
            return "\n".join(lines)
        # This gang job does not fit yet; a lower-priority smaller job might — keep scanning.
        continue

    lines.append(
        "No queued job fits current cluster capacity. WAIT or use PREEMPT to free nodes."
    )
    return "\n".join(lines)


def get_heuristic_action(obs: GpuSchedulerObservation) -> Optional[GpuSchedulerAction]:
    """
    Pick a valid SCHEDULE when the queue has at least one placeable job.

    Returns None when WAIT is appropriate (empty queue or no fit).
    """
    if not obs.queue:
        return None

    sorted_queue = sorted(
        obs.queue,
        key=lambda j: _queue_priority_score(j, obs.current_hour),
        reverse=True,
    )

    for job in sorted_queue:
        if job.gpu_count <= 8:
            for node in obs.nodes:
                if node.free_gpus >= job.gpu_count:
                    return GpuSchedulerAction(
                        action_type=ActionType.SCHEDULE,
                        job_id=job.job_id,
                        node_id=node.node_id,
                    )
        else:
            nodes_needed = (job.gpu_count + 7) // 8
            fully_free = [n.node_id for n in obs.nodes if n.free_gpus == 8]
            if len(fully_free) >= nodes_needed:
                return GpuSchedulerAction(
                    action_type=ActionType.SCHEDULE,
                    job_id=job.job_id,
                    node_id=fully_free[0],
                )

    return None


def action_to_log_line(action: GpuSchedulerAction) -> str:
    """First-line action string for logs / prompt override."""
    if action.action_type == ActionType.WAIT:
        return "WAIT"
    if action.action_type == ActionType.PREEMPT:
        return f"PREEMPT {action.job_id or ''}".strip()
    # SCHEDULE
    return f"SCHEDULE {action.job_id} {action.node_id}"


def _maybe_override_wait_with_heuristic(
    action: GpuSchedulerAction,
    obs: GpuSchedulerObservation,
    raw: str,
    reason_prefix: str,
) -> Tuple[GpuSchedulerAction, str]:
    """If action is WAIT but work is placeable, substitute heuristic SCHEDULE."""
    if action.action_type != ActionType.WAIT or not obs.queue:
        return action, raw
    heuristic = get_heuristic_action(obs)
    if heuristic is None:
        return action, raw
    line = action_to_log_line(heuristic)
    first = raw.strip().split("\n")[0].strip() if raw.strip() else "WAIT"
    new_raw = f"{line}\n[{reason_prefix}: was: {first}]"
    return heuristic, new_raw


# ---------------------------------------------------------------------------
# LLM call — builds the full prompt and returns (action, raw_response)
# ---------------------------------------------------------------------------

def _sync_chat_completion(client: OpenAI, user_prompt: str) -> str:
    """Blocking OpenAI chat call — run via asyncio.to_thread from async agents."""
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        stream=False,
    )
    return (completion.choices[0].message.content or "WAIT").strip()


async def get_llm_action(
    client: OpenAI,
    obs: GpuSchedulerObservation,
    step: int,
    history: List[str],
) -> Tuple[GpuSchedulerAction, str]:
    """
    Query the LLM for the next scheduling decision.

    Includes the last 3 history entries so the model can see the recent
    trajectory and avoid repeating useless WAIT actions.

    The HTTP request runs in a thread pool so the asyncio event loop can
    service WebSocket keepalives while waiting on slow local LLMs (e.g. Ollama).

    Args:
        client:  OpenAI-compatible client pointed at the configured endpoint.
        obs:     Current observation from the environment.
        step:    Current agent step index (used in the observation header).
        history: Rolling log of recent (step, action, reward) strings.

    Returns:
        Tuple of (parsed GpuSchedulerAction, raw LLM response string).
    """
    # Build the state block for this step
    state_text = format_observation(obs, step)

    # Append recent trajectory so the LLM sees cause→effect chains
    history_block = "\n".join(history[-3:]) if history else "No prior steps."

    recent_errors = [h for h in history[-5:] if "INVALID" in h]
    error_block = ""
    if recent_errors:
        error_block = (
            "RECENT ERRORS (do not repeat):\n"
            + "\n".join(recent_errors[-3:])
            + "\n\n"
        )

    user_prompt = (
        f"{state_text}\n\n"
        f"{error_block}"
        f"RECENT TRAJECTORY:\n{history_block}\n\n"
        f"What is your next action? (Line 1: action only)"
    )

    try:
        raw = await asyncio.to_thread(_sync_chat_completion, client, user_prompt)
        action = parse_action(raw)
        return _maybe_override_wait_with_heuristic(
            action, obs, raw, "OVERRIDE schedulable work"
        )

    except Exception as exc:
        # Never crash the episode on an API error — prefer heuristic over blind WAIT
        print(f"[DEBUG] LLM request failed at step {step}: {exc}", flush=True)
        fallback = GpuSchedulerAction(action_type=ActionType.WAIT)
        raw_fb = "WAIT (api-error fallback)"
        return _maybe_override_wait_with_heuristic(
            fallback, obs, raw_fb, f"OVERRIDE API error ({type(exc).__name__})"
        )


# ---------------------------------------------------------------------------
# Structured stdout loggers — format required by the hackathon evaluator
# ---------------------------------------------------------------------------

def log_start(task: str, env_name: str, model: str) -> None:
    """Emit the mandatory [START] line at the beginning of each episode."""
    print(f"[START] task={task} env={env_name} model={model}", flush=True)
    
    # Also display rich episode start if available
    if RICH_AVAILABLE:
        log_episode_start(task=task, env=env_name, model=model)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str],
    current_hour: float = 0.0,
    total_hours: float = 0.0,
    nodes_summary: Optional[str] = None,
) -> None:
    """Emit one [STEP] line immediately after env.step() returns."""
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} "
        f"reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )
    
    # Also display rich table if available
    if RICH_AVAILABLE:
        log_step_table(
            step=step,
            action=action,
            reward=reward,
            done=done,
            error=error,
            current_hour=current_hour,
            total_hours=total_hours,
            nodes_summary=nodes_summary,
        )


def log_end(
    success: bool,
    steps: int,
    score: float,
    rewards: List[float],
) -> None:
    """Emit the mandatory [END] line after env.close() or episode completion."""
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )
    
    # Also display rich episode end if available
    if RICH_AVAILABLE:
        total_reward = sum(rewards) if rewards else 0.0
        log_episode_end(
            success=success,
            steps=steps,
            score=score,
            total_reward=total_reward,
        )


# ---------------------------------------------------------------------------
# Single task runner
# ---------------------------------------------------------------------------

async def run_task(
    client: OpenAI,
    env: GpuSchedulerEnv,
    task_name: str,
    max_steps: int,
    success_threshold: float,
) -> bool:
    """
    Run one complete episode of the given task and emit structured logs.

    Flow:
        1. Reset the environment (selects the task via GPU_SCHEDULER_TASK env var)
        2. Loop: get LLM action → step env → log → check done
        3. Extract final score from result.info (provided by the task grader)
        4. Emit [END] and return whether score >= success_threshold

    Args:
        client:            OpenAI-compatible LLM client.
        env:               Connected GpuSchedulerEnv instance.
        task_name:         One of: smooth_sailing | deadline_crunch | p0_emergency
        max_steps:         Maximum agent steps before the episode is force-ended.
        success_threshold: Minimum grader score to count as success.

    Returns:
        True if the episode's grader score >= success_threshold.
    """
    rewards:     List[float] = []
    steps_taken: int         = 0
    score:       float       = 0.0
    success:     bool        = False
    history:     List[str]   = []
    errors:      List[tuple[int, str]] = []  # Track errors for summary

    log_start(task=task_name, env_name=BENCHMARK, model=MODEL_NAME)

    try:
        # Pass task_name directly so it works inside Docker containers too
        result = await env.reset(task_name=task_name)
        obs: GpuSchedulerObservation = result.observation

        for step in range(1, max_steps + 1):
            # Episode may end early (e.g. all jobs completed ahead of schedule)
            if result.done:
                break

            # --- Get action from LLM ---
            action, raw_response = await get_llm_action(client, obs, step, history)
            # Log what the env actually executed (may differ from LLM line if parse adjusted)
            action_str = format_executed_action(action)

            # --- Execute action in the environment ---
            result = await env.step(action)
            obs    = result.observation

            reward = result.reward or 0.0
            done   = result.done
            # last_action_result carries validation errors (e.g. "invalid node_id")
            error  = obs.last_action_result if "INVALID" in (obs.last_action_result or "") else None

            rewards.append(reward)
            steps_taken = step
            
            # Track errors for summary
            if error:
                errors.append((step, error))

            # Pass additional info to log_step for rich display
            log_step(
                step=step,
                action=action_str,
                reward=reward,
                done=done,
                error=error,
                current_hour=obs.current_hour,
                total_hours=obs.total_hours,
                nodes_summary=format_nodes_status_summary(obs),
            )

            # Rolling history: compact one-liner per step for the LLM context window
            history.append(
                f"Step {step}: {action_str} → reward {reward:+.2f} | "
                f"h{obs.current_hour:.0f}/{obs.total_hours:.0f} | "
                f"{obs.last_action_result or 'ok'}"
            )

            if done:
                break

        # --- Extract grader score from terminal observation ---
        # The environment's task grader computes a normalised 0.0–1.0 score
        # and stores it in observation.score when done=True (serialised as a
        # plain observation field so it survives OpenEnv's serialize_observation).
        raw_score = getattr(result.observation, "score", None)
        score = float(raw_score) if raw_score is not None else 0.0
        score = max(0.0, min(score, 1.0))   # safety clamp to [0, 1]

        success = score >= success_threshold

    except Exception as exc:
        # Catch unexpected errors; still emit a valid [END] line
        print(f"[DEBUG] Task '{task_name}' crashed: {exc}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
        
        # Display error summary if available
        if RICH_AVAILABLE and errors:
            log_error_summary(errors)

    return success


# ---------------------------------------------------------------------------
# Main entry point — runs all three tasks sequentially
# ---------------------------------------------------------------------------

async def main() -> None:
    """
    Connect to the GpuScheduler environment container and run all three tasks.

    Tasks run sequentially on the same container instance.  The environment
    server resets its internal state on each env.reset() call, so there is
    no cross-contamination between episodes.

    Total expected runtime: < 15 minutes on a 2-vCPU / 8 GB machine
    (24 + 36 + 42 = 102 LLM calls × ~5 s each ≈ 8.5 minutes + env overhead).
    """
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # If IMAGE_NAME looks like a URL, connect directly; otherwise use Docker
    if IMAGE_NAME and IMAGE_NAME.startswith("http"):
        env = GpuSchedulerEnv(base_url=IMAGE_NAME)
    else:
        env = await GpuSchedulerEnv.from_docker_image(IMAGE_NAME)

    try:
        results: Dict[str, bool] = {}

        tasks_to_run = list(TASKS)
        if INFERENCE_ONLY:
            allowed = {x.strip() for x in INFERENCE_ONLY.split(",") if x.strip()}
            tasks_to_run = [t for t in TASKS if t[0] in allowed]
            if not tasks_to_run:
                print(
                    f"[ERROR] INFERENCE_ONLY={INFERENCE_ONLY!r} matched no tasks. "
                    f"Known: {[t[0] for t in TASKS]}",
                    flush=True,
                )
                return

        for task_name, max_steps, threshold in tasks_to_run:
            passed = await run_task(
                client=client,
                env=env,
                task_name=task_name,
                max_steps=max_steps,
                success_threshold=threshold,
            )
            results[task_name] = passed
            # Blank line between task blocks for readability
            print("", flush=True)

        # Final summary (informational — not part of the mandatory format)
        all_passed = all(results.values())
        summary    = " | ".join(f"{t}={'PASS' if v else 'FAIL'}" for t, v in results.items())
        print(f"[SUMMARY] {summary} | all_passed={str(all_passed).lower()}", flush=True)

    finally:
        # Always clean up the Docker container, even on exception
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error (container cleanup): {e}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
