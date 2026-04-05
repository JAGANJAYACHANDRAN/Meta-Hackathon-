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
          Optional indented "state" block on following lines (sim hour, nodes,
          running vs queued, GPUs per node).
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

OPTIONAL LOG FILE (full stdout mirror for inspection)
------------------------------------------------------
  INFERENCE_LOG_FILE   — path to write every printed line (still prints to
                         terminal). Set to off / false / 0 to disable.
                         If unset, defaults to:
                         gpu_scheduler/logs/inference_YYYYMMDD_HHMMSS.log
"""

import asyncio
import os
import re
import sys
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, TextIO, Tuple

from dotenv import load_dotenv
from openai import OpenAI

# Load .env file from the project root (same directory as this script)
_env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(_env_path)

# Import typed client + models from the gpu_scheduler package
from gpu_scheduler import (
    GpuSchedulerAction,
    GpuSchedulerEnv,
    GpuSchedulerObservation,
    SubAction,
)

# ---------------------------------------------------------------------------
# Environment variables — loaded from .env file or shell environment
# ---------------------------------------------------------------------------
IMAGE_NAME   = os.getenv("IMAGE_NAME")          # HF Space URL or Docker image
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
BENCHMARK    = "gpu_scheduler"

# Mirror stdout to a log file (see module docstring)
_INFERENCE_LOG_FP: Optional[TextIO] = None
_ORIG_STDOUT: Optional[TextIO] = None


class _TeeStdout:
    """Write to the real terminal and to an open log file."""

    def __init__(self, terminal: TextIO, log_file: TextIO) -> None:
        self._terminal = terminal
        self._log_file = log_file

    def write(self, s: str) -> int:
        self._terminal.write(s)
        self._log_file.write(s)
        self._terminal.flush()
        self._log_file.flush()
        return len(s)

    def flush(self) -> None:
        self._terminal.flush()
        self._log_file.flush()

    def __getattr__(self, name: str):
        return getattr(self._terminal, name)


def _setup_inference_log_file() -> Optional[Path]:
    """
    Tee stdout to a log file. Returns the path written, or None if disabled.
    """
    global _INFERENCE_LOG_FP, _ORIG_STDOUT
    raw = os.getenv("INFERENCE_LOG_FILE", "").strip().lower()
    if raw in ("0", "false", "no", "off", "disable", "disabled"):
        return None

    script_dir = Path(__file__).resolve().parent
    if raw:
        log_path = Path(raw).expanduser()
        if not log_path.is_absolute():
            log_path = Path.cwd() / log_path
    else:
        log_dir = script_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"inference_{datetime.now():%Y%m%d_%H%M%S}.log"

    log_path.parent.mkdir(parents=True, exist_ok=True)
    _INFERENCE_LOG_FP = open(log_path, "w", encoding="utf-8")
    _ORIG_STDOUT = sys.stdout
    sys.stdout = _TeeStdout(_ORIG_STDOUT, _INFERENCE_LOG_FP)
    return log_path


def _teardown_inference_log_file() -> None:
    global _INFERENCE_LOG_FP, _ORIG_STDOUT
    if _ORIG_STDOUT is not None:
        sys.stdout = _ORIG_STDOUT
        _ORIG_STDOUT = None
    if _INFERENCE_LOG_FP is not None:
        _INFERENCE_LOG_FP.close()
        _INFERENCE_LOG_FP = None

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
MAX_TOKENS  = 512   # Increased for multi-action batches + reasoning


# ---------------------------------------------------------------------------
# System prompt — tells the LLM exactly what it is and how to act
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = textwrap.dedent("""
    You are an AI GPU cluster scheduler. Your goal is to MINIMIZE total
    compute burn (cluster costs $100,000 per day = ~$4,167/hour at full capacity).

    CLUSTER LAYOUT
    ---------------
    8 nodes (node 0-7), each with 8 GPUs = 64 GPUs total.

    JOB PRIORITIES
    ---------------
    P0 - mission-critical; NEVER preempt; schedule ASAP even if it blocks others.
    P1 - high priority; try to avoid preemption but not required; respect deadlines.
    P2 - standard; preemption allowed for P0/P1 emergencies.
    P3 - low/spot; first to be preempted when resources are needed.

    AVAILABLE ACTIONS
    ------------------
    SCHEDULE <job_id> <node_id>
        Place a queued job on node 0-7. The node must have enough free GPUs.
        For gang jobs (>8 GPUs): the job will be placed on multiple nodes.
        For this to happen you have to free up the required number of nodes.
        You are allowed to PREEMPT non-P0 jobs (P1/P2/P3) to free the required number of nodes.
        Calculate how many nodes are needed: ceil(gpu_count / 8).
        PREEMPT enough low-priority jobs first to free those nodes, then issue a single
        SCHEDULE <job_id> <one_free_node_id>. The system will automatically assign
        the remaining nodes to the gang job.
    PREEMPT <job_id>
        Evict a running job back to queue. Penalty = 2x wasted work.
        Cannot preempt P0 jobs. Useful to free space for higher-priority work(P0 and P1).
    WAIT
        Advance the simulation clock without acting. Use ONLY when the queue
        is empty AND no preemptions are beneficial.

    CRITICAL RULES
    ---------------
    1. SCHEDULE job_ids MUST come from the SCHEDULABLE JOBS list — use ONLY
       the EXACT IDs shown there (e.g. job_007). NEVER invent job IDs.
    2. PREEMPT job_ids MUST come from the RUNNING JOBS list.
    3. GIVE THE EXACT JOB ID WHEN SCHEDULING OR PREEMPTING.
    4. If SCHEDULABLE JOBS is empty and no preemptions needed, use WAIT.
    5. Never re-schedule an already-running job.
    6. Pick node_ids with enough free GPUs for the job's gpu_count.
    7. Never preempt P0 jobs.
    8. PREEMPT is EXTREMELY EXPENSIVE. Only preempt when ALL of these are true:
       a. A higher-priority job is in the queue that CANNOT fit on any node
       b. No other way to free GPUs (no jobs finishing soon)
       c. The preempted job is low-priority (P3 first, then P2)
       NEVER preempt to "reshuffle" or "optimize". NEVER preempt a job you
       just scheduled. Each preemption carries a large burn penalty.
    9. NEVER preempt and reschedule the same jobs back and forth. This creates
       a loop that destroys your score. If you preempted a job, let it run
       after rescheduling — do NOT preempt it again.
    10. The TASK NAME is never a job_id. "p0_emergency", "smooth_sailing",
        "deadline_crunch" are task names — they will NEVER appear in job lists.

    MULTI-ACTION BATCHING (ATOMIC)
    --------------------------------
    You may output MULTIPLE actions per turn. All actions in one turn
    execute ATOMICALLY within a single timestep — the clock advances
    only ONCE after all actions are applied. This means:
    - No idle GPU penalties between actions in the same batch
    - Preempt + schedule in one turn = zero wasted idle time
    - All state changes (GPU allocation/release) happen at the same sim hour

    Use batching to:
    - Schedule ALL queued jobs that fit on available nodes in one turn
    - Preempt multiple low-priority jobs AND schedule a high-priority job
      in the same turn (no idle penalty on freed GPUs)
    - Execute coordinated multi-step strategies atomically

    Track GPU changes mentally across your batch:
    - After SCHEDULE: that node loses GPUs
    - After PREEMPT: that job's node(s) regain GPUs
    - Plan subsequent actions accounting for these changes
    - WAIT should be the ONLY action if nothing else is needed (ends the turn)

    REWARD SIGNALS (bounded 0.0–1.0, where 0.5 = neutral)
    ----------------------------------------------------
    + Progress: proportional to job completion per step (P0=0.5x, P1=0.4x, P2=0.25x, P3=0.1x)
    - Idle GPU cost: penalty for unused GPUs (higher when schedulable work is waiting)
    - Preemption burn: gpu_count × wasted_hours × rate × 0.3
    - SLA violation: initial penalty + continuing penalty while overdue
    - Queue delay: penalty per hour P0/P1 jobs wait in queue
    Reward > 0.5 = good step, < 0.5 = net-negative step

    RESPONSE FORMAT (STRICT)
    -------------------------
    You MUST respond in EXACTLY this format:

    REASON: <one-line explanation of your decision>
    ACTIONS:
    <action_1>
    <action_2>
    ...

    Where each action is one of:
        SCHEDULE job_NNN X
        PREEMPT job_NNN
        WAIT

    Rules:
    - REASON line is mandatory (one line only).
    - ACTIONS: header is mandatory.
    - At least one action is required.
    - One action per line after ACTIONS: header.
    - WAIT must be the last action if present (ends the batch).
    - No extra text, numbering, bullets, or commentary in the action lines.

    EXAMPLE RESPONSES
    ------------------
    Example 1 - schedule multiple jobs:
    REASON: 3 jobs queued, all fit on available nodes
    ACTIONS:
    SCHEDULE job_007 3
    SCHEDULE job_012 5
    SCHEDULE job_001 0

    Example 2 - preempt to make room for P0:
    REASON: P0 job_020 needs 32 GPUs, preempting P3 jobs on nodes 0-3
    ACTIONS:
    PREEMPT job_003
    PREEMPT job_008
    PREEMPT job_011
    SCHEDULE job_020 0

    Example 3 - nothing to do:
    REASON: queue empty, waiting for next arrivals
    ACTIONS:
    WAIT
""").strip()


# ---------------------------------------------------------------------------
# Observation formatter — converts the rich observation object to plain text
# that the LLM can reason over
# ---------------------------------------------------------------------------

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

    # --- Header (avoid exposing task_name as a schedulable entity) ---
    hours_left = obs.total_hours - obs.current_hour
    lines += [
        f"=== Step {step} | Hour {obs.current_hour:.0f} / {obs.total_hours:.0f} "
        f"({hours_left:.0f}h remaining) ===",
        f"Compute Burn: ${obs.compute_burn_so_far:,.0f}",
        f"Last action: {obs.last_action_result or '(start of episode)'}",
        "",
    ]

    # --- Node table ---
    lines.append("NODES  (node_id | used/total GPUs | contention | running jobs)")
    for node in obs.nodes:
        # Visual contention bar: ▓ per 10% contention, ░ for free headroom
        filled  = int(node.memory_contention * 10)
        bar     = "▓" * filled + "░" * (10 - filled)
        # Flag nodes where quadratic contention degradation is significant (>15%)
        warn    = " ⚠ HIGH CONTENTION" if node.memory_contention >= 0.6 else ""
        jobs_str = ", ".join(node.running_jobs) if node.running_jobs else "idle"
        lines.append(
            f"  Node {node.node_id}: {node.used_gpus}/{node.total_gpus} GPUs"
            f" | [{bar}] {node.memory_contention:.2f}{warn}"
            f" | {jobs_str}"
        )

    # --- Running jobs table ---
    lines += ["", "RUNNING JOBS  (job_id | priority | GPUs | progress | deadline)"]
    if obs.active_jobs:
        for job in obs.active_jobs:
            dl   = f"deadline h{job.deadline_hour:.0f}" if job.deadline_hour else "no deadline"
            # Highlight jobs in danger of missing their SLA
            urgent = ""
            if job.deadline_hour and (job.deadline_hour - obs.current_hour) < (job.duration_hours * (1 - job.progress)):
                urgent = " ⚠ AT RISK"
            lines.append(
                f"  {job.job_id:12s} [{job.priority_label}] | {job.gpu_count:2d} GPUs"
                f" | {job.progress:5.1%} done | nodes={job.assigned_nodes}"
                f" | {dl}{urgent}"
            )
    else:
        lines.append("  (no jobs currently running)")

    # --- Queue table ---
    lines += ["", "JOB QUEUE  (job_id | priority | GPUs needed | duration | queued_for | deadline)"]
    if obs.queue:
        for job in obs.queue:
            dl   = f"deadline h{job.deadline_hour:.0f}" if job.deadline_hour else "no deadline"
            lines.append(
                f"  {job.job_id:12s} [{job.priority_label}] | {job.gpu_count:2d} GPUs"
                f" | {job.duration_hours:.0f}h runtime"
                f" | queued {job.hours_in_queue:.1f}h | {dl}"
            )
    else:
        lines.append("  (queue is empty)")

    # --- Upcoming arrivals table ---
    if obs.upcoming_jobs:
        lines += ["", "UPCOMING ARRIVALS (arriving soon — NOT yet schedulable, plan ahead)"]
        for job in obs.upcoming_jobs:
            dl   = f"deadline h{job.deadline_hour:.0f}" if job.deadline_hour else "no deadline"
            lines.append(
                f"  {job.job_id:12s} [{job.priority_label}] | {job.gpu_count:2d} GPUs"
                f" | {job.duration_hours:.0f}h runtime"
                f" | arrives h{job.arrival_hour:.0f} | {dl}"
            )

    # --- Schedulable summary — explicit list of valid job_ids ---
    if obs.queue:
        job_ids = [job.job_id for job in obs.queue]
        lines += ["", f"SCHEDULABLE JOBS (use ONLY these with SCHEDULE): {', '.join(job_ids)}"]
    else:
        lines += ["", "SCHEDULABLE JOBS: none — you MUST respond with WAIT"]

    # --- Available nodes summary ---
    free_nodes = [f"node {n.node_id}({n.free_gpus} free)" for n in obs.nodes if n.free_gpus > 0]
    if free_nodes:
        lines.append(f"AVAILABLE NODES: {', '.join(free_nodes)}")
    else:
        lines.append("AVAILABLE NODES: none — cluster is full, use WAIT or PREEMPT")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Action parser — extracts a valid GpuSchedulerAction from raw LLM text
# ---------------------------------------------------------------------------

def parse_single_action(line: str) -> Optional[GpuSchedulerAction]:
    """
    Parse a single action line into a GpuSchedulerAction.

    Returns None if the line is not a valid action.
    """
    line = line.strip()
    if not line:
        return None

    # SCHEDULE job_id node_id  (job_id can be job_NNN or job_P0_EMERGENCY etc.)
    m = re.match(r"SCHEDULE\s+(job_\w+)\s+(\d+)", line, re.IGNORECASE)
    if m:
        return GpuSchedulerAction(
            action_type="SCHEDULE",
            job_id=m.group(1),
            node_id=int(m.group(2)),
        )

    # PREEMPT job_id
    m = re.match(r"PREEMPT\s+(job_\w+)", line, re.IGNORECASE)
    if m:
        return GpuSchedulerAction(
            action_type="PREEMPT",
            job_id=m.group(1),
        )

    # WAIT
    if re.match(r"WAIT\b", line, re.IGNORECASE):
        return GpuSchedulerAction(action_type="WAIT")

    return None


def parse_actions(response: str) -> List[GpuSchedulerAction]:
    """
    Parse the LLM's structured response into a list of GpuSchedulerActions.

    Expected format:
        REASON: <text>
        ACTIONS:
        <action_1>
        <action_2>
        ...

    Falls back to parsing any recognizable action lines if the format is
    not strictly followed. Always returns at least [WAIT].
    """
    actions: List[GpuSchedulerAction] = []
    lines = response.strip().split("\n")

    # Find the ACTIONS: header and parse lines after it
    action_section = False
    for line in lines:
        stripped = line.strip()
        if re.match(r"ACTIONS\s*:", stripped, re.IGNORECASE):
            action_section = True
            continue
        if action_section:
            parsed = parse_single_action(stripped)
            if parsed:
                actions.append(parsed)
                # WAIT ends the batch
                if parsed.action_type == "WAIT":
                    break

    # Fallback: if no ACTIONS: header found, try parsing every line
    if not actions:
        for line in lines:
            parsed = parse_single_action(line.strip())
            if parsed:
                actions.append(parsed)
                if parsed.action_type == "WAIT":
                    break

    # Ultimate fallback: WAIT
    if not actions:
        actions.append(GpuSchedulerAction(action_type="WAIT"))

    return actions


# ---------------------------------------------------------------------------
# LLM call — builds the full prompt and returns (action, raw_response)
# ---------------------------------------------------------------------------

def validate_action(
    action: GpuSchedulerAction,
    obs: GpuSchedulerObservation,
) -> Optional[GpuSchedulerAction]:
    """
    Validate an LLM-proposed action against the current observation.

    Returns the action if valid, or None if it should be skipped.
    Does NOT auto-correct — the batch may contain subsequent valid actions.
    """
    if action.action_type == "SCHEDULE":
        queue_ids = {job.job_id for job in obs.queue}
        if action.job_id not in queue_ids:
            return None
        if action.node_id is None or action.node_id < 0 or action.node_id > 7:
            return None
        node = next((n for n in obs.nodes if n.node_id == action.node_id), None)
        if node is None or node.free_gpus <= 0:
            return None
        return action

    if action.action_type == "PREEMPT":
        running_ids = {job.job_id for job in obs.active_jobs}
        if action.job_id not in running_ids:
            return None
        # Don't preempt P0 jobs (env rejects it anyway with penalty)
        p0_ids = {job.job_id for job in obs.active_jobs if job.priority == 0}
        if action.job_id in p0_ids:
            return None
        return action

    # WAIT is always valid
    return action


def validate_batch(
    actions: List[GpuSchedulerAction],
    obs: GpuSchedulerObservation,
    recently_preempted: Optional[set] = None,
) -> List[GpuSchedulerAction]:
    """
    Pre-validate a batch of actions against the initial observation.

    Simulates GPU state changes across the batch so that sequential
    PREEMPT→SCHEDULE chains are validated correctly. Drops invalid
    actions but keeps the rest. Always returns at least [WAIT].

    recently_preempted: set of job_ids preempted in the last N steps.
    If a job was recently preempted, block re-preemption to prevent loops.
    """
    valid: List[GpuSchedulerAction] = []
    if recently_preempted is None:
        recently_preempted = set()

    # Track simulated GPU changes: node_id -> free_gpus
    free_gpus = {n.node_id: n.free_gpus for n in obs.nodes}
    scheduled_ids: set = set()  # jobs moved from queue to running in this batch
    preempted_ids: set = set()  # jobs moved from running to queue in this batch
    queue_ids = {job.job_id for job in obs.queue}
    running_ids = {job.job_id for job in obs.active_jobs}
    p0_ids = {job.job_id for job in obs.active_jobs if job.priority == 0}
    # Map job_id -> gpu_count for queue jobs
    job_gpus = {job.job_id: job.gpu_count for job in obs.queue}
    # Map job_id -> (gpu_count, assigned_nodes) for running jobs
    running_info = {
        job.job_id: (job.gpu_count, job.assigned_nodes)
        for job in obs.active_jobs
    }

    for action in actions:
        if action.action_type == "WAIT":
            valid.append(action)
            break  # WAIT ends the batch

        if action.action_type == "SCHEDULE":
            jid = action.job_id
            nid = action.node_id
            # Must be in original queue or preempted back to queue in this batch
            if jid not in queue_ids and jid not in preempted_ids:
                continue
            if jid in scheduled_ids:
                continue  # already scheduled in this batch
            if nid is None or nid < 0 or nid > 7:
                continue
            gpus_needed = job_gpus.get(jid)
            if gpus_needed is None:
                # Might be a preempted job re-entering queue
                if jid in running_info:
                    gpus_needed = running_info[jid][0]
                else:
                    continue
            # For gang jobs, just check if anchor node is free
            node_gpus = min(gpus_needed, 8)
            if free_gpus.get(nid, 0) < node_gpus:
                continue
            # Accept and simulate the allocation
            free_gpus[nid] -= node_gpus
            scheduled_ids.add(jid)
            valid.append(action)

        elif action.action_type == "PREEMPT":
            jid = action.job_id
            if jid not in running_ids or jid in preempted_ids:
                continue
            if jid in p0_ids:
                continue
            # Block preemption loops: don't re-preempt recently preempted jobs
            if jid in recently_preempted:
                continue
            # Accept and simulate freeing GPUs
            if jid in running_info:
                gpu_count, nodes = running_info[jid]
                gpus_per_node = gpu_count // max(len(nodes), 1)
                for nid in nodes:
                    free_gpus[nid] = free_gpus.get(nid, 0) + gpus_per_node
            preempted_ids.add(jid)
            valid.append(action)

    if not valid:
        valid.append(GpuSchedulerAction(action_type="WAIT"))

    return valid


def get_llm_actions(
    client: OpenAI,
    obs: GpuSchedulerObservation,
    step: int,
    history: List[str],
    max_steps: int = 42,
    recently_preempted: Optional[set] = None,
) -> Tuple[List[GpuSchedulerAction], str, List[str]]:
    """
    Query the LLM for a batch of scheduling actions.

    The LLM sees the full cluster state and can output multiple
    actions (schedules and/or preemptions) to be executed sequentially.
    """
    state_text = format_observation(obs, step)

    history_block = "\n".join(history[-5:]) if history else "No prior steps."

    schedulable = ", ".join(j.job_id for j in obs.queue)
    preemptable = ", ".join(
        j.job_id for j in obs.active_jobs if j.priority > 0
    )

    # Compute hours_per_step from observation
    hours_per_step = obs.total_hours / max_steps if max_steps else 1.0
    steps_remaining = max_steps - step + 1 if max_steps else 0

    user_prompt = (
        f"{state_text}\n\n"
        f"STEP BUDGET: {steps_remaining} steps left | "
        f"each step advances clock by {hours_per_step:.0f}h\n\n"
        f"RECENT TRAJECTORY:\n{history_block}\n\n"
        f"SCHEDULABLE job_ids (ONLY these are valid for SCHEDULE): "
        f"{schedulable or 'NONE — use WAIT'}\n"
        f"PREEMPTABLE job_ids (non-P0 running): {preemptable or 'NONE'}\n\n"
        f"⚠ Use ONLY the exact job_ids listed above. "
        f"PREEMPT only if a queued job cannot fit without freeing GPUs. "
        f"Respond in REASON + ACTIONS format."
    )

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        raw = (completion.choices[0].message.content or "WAIT").strip()
        actions = parse_actions(raw)
        proposed_strs = []
        for a in actions:
            if a.action_type == "SCHEDULE":
                proposed_strs.append(f"SCHEDULE {a.job_id} {a.node_id}")
            elif a.action_type == "PREEMPT":
                proposed_strs.append(f"PREEMPT {a.job_id}")
            else:
                proposed_strs.append("WAIT")
        actions = validate_batch(actions, obs, recently_preempted)
        # Cap batch size to avoid burning too many steps at once.
        # Leave at least 2 steps for the LLM to react to state changes.
        max_batch = max(1, steps_remaining - 2)
        if len(actions) > max_batch:
            actions = actions[:max_batch]
        return actions, raw, proposed_strs

    except Exception as exc:
        print(f"[DEBUG] LLM request failed at step {step}: {exc}", flush=True)
        return [GpuSchedulerAction(action_type="WAIT")], "WAIT (api-error fallback)", []


# ---------------------------------------------------------------------------
# Structured stdout loggers — format required by the hackathon evaluator
# ---------------------------------------------------------------------------

_STEP_LABEL_W = 18  # width for right-aligned labels in the state block
# Human-readable step trace (does not affect hackathon [STEP] line format)
_STEP_LOG_SEPARATOR = "=" * 72
_STEP_LOG_TRAIL_BLANK_LINES = 10


def _print_env_step_trace(
    env_step: int,
    llm_raw: str,
    batch_size: int,
    batch_index: int,
    batch_action_strings: List[str],
    executed_action: str,
) -> None:
    """LLM text + batch metadata, framed with === lines (stdout only)."""
    print(_STEP_LOG_SEPARATOR, flush=True)
    print(
        f"LLM RAW RESPONSE (env step {env_step}, "
        f"batch action {batch_index}/{batch_size}):",
        flush=True,
    )
    print(llm_raw, flush=True)
    print(_STEP_LOG_SEPARATOR, flush=True)
    print("STEP DETAILS:", flush=True)
    print(f"  actions_in_batch: {batch_size}", flush=True)
    print(f"  full_batch: [{', '.join(batch_action_strings)}]", flush=True)
    print(f"  executed_this_step: {executed_action}", flush=True)
    print(_STEP_LOG_SEPARATOR, flush=True)


def _step_state_line(label: str, value: str) -> str:
    """One indented line: label right-aligned, colon, value."""
    return f"      {label:>{_STEP_LABEL_W}} : {value}"


def format_step_state_block(obs: GpuSchedulerObservation) -> str:
    """
    Multi-line cluster summary after env.step (human-readable; printed below [STEP]).

    Right-aligned labels, spaced GPU columns, sim clock from observation.
    """
    nodes_sorted = sorted(obs.nodes, key=lambda n: n.node_id)
    occupied = sum(1 for n in nodes_sorted if n.used_gpus > 0)
    gpu_parts = [
        f"n{n.node_id} {n.used_gpus:>2}/{n.total_gpus}" for n in nodes_sorted
    ]
    # Align continuation with first line's value column (6 spaces + label + " : ")
    value_indent = 6 + _STEP_LABEL_W + 3
    cont = " " * value_indent
    gpu_line1 = "   ".join(gpu_parts[:4])
    gpu_line2 = "   ".join(gpu_parts[4:]) if len(gpu_parts) > 4 else ""
    lines = [
        _step_state_line(
            "sim_hour",
            f"{obs.current_hour:.0f} / {obs.total_hours:.0f}",
        ),
        _step_state_line("occupied_nodes", str(occupied)),
        _step_state_line(
            "jobs",
            f"running {len(obs.active_jobs)}    queued {len(obs.queue)}",
        ),
        _step_state_line("GPUs per node", gpu_line1),
    ]
    if gpu_line2:
        lines.append(f"{cont}{gpu_line2}")
    return "\n".join(lines)


def log_start(task: str, env_name: str, model: str) -> None:
    """Emit the mandatory [START] line at the beginning of each episode."""
    print(f"[START] task={task} env={env_name} model={model}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str],
    obs: Optional[GpuSchedulerObservation] = None,
) -> None:
    """Emit one [STEP] line immediately after env.step() returns."""
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} "
        f"reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )
    if obs is not None:
        print(format_step_state_block(obs), flush=True)


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


# ---------------------------------------------------------------------------
# Single task runner
# ---------------------------------------------------------------------------

def _build_batch_action(
    actions: List[GpuSchedulerAction],
) -> Tuple[GpuSchedulerAction, str]:
    """
    Convert a list of parsed actions into a single BATCH GpuSchedulerAction.

    If the list contains only a WAIT (or is empty), returns a plain WAIT action.
    Otherwise wraps all SCHEDULE/PREEMPT actions into a BATCH with sub_actions.

    Returns:
        (action, action_str) — the action to send + human-readable summary.
    """
    # Filter out WAIT from the middle — only meaningful at the end
    non_wait = [a for a in actions if a.action_type != "WAIT"]

    if not non_wait:
        return GpuSchedulerAction(action_type="WAIT"), "WAIT"

    if len(non_wait) == 1:
        # Single action — send directly, no BATCH wrapper needed
        a = non_wait[0]
        if a.action_type == "SCHEDULE":
            return a, f"SCHEDULE {a.job_id} {a.node_id}"
        else:
            return a, f"PREEMPT {a.job_id}"

    # Multiple actions — wrap in BATCH
    subs = []
    parts = []
    for a in non_wait:
        if a.action_type == "SCHEDULE":
            subs.append(SubAction(
                action_type="SCHEDULE",
                job_id=a.job_id,
                node_id=a.node_id,
            ))
            parts.append(f"SCHEDULE {a.job_id} {a.node_id}")
        elif a.action_type == "PREEMPT":
            subs.append(SubAction(
                action_type="PREEMPT",
                job_id=a.job_id,
            ))
            parts.append(f"PREEMPT {a.job_id}")

    batch_action = GpuSchedulerAction(
        action_type="BATCH",
        sub_actions=subs,
    )
    action_str = f"BATCH[{', '.join(parts)}]"
    return batch_action, action_str


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
        1. Reset the environment
        2. Query LLM for a batch of actions
        3. Wrap batch into a single BATCH action (or WAIT/single action)
        4. Execute via one env.step() — all sub-actions apply atomically,
           clock advances once
        5. Repeat until done or max_steps reached

    Each LLM call = one env step = one clock tick. Multiple SCHEDULE/PREEMPT
    actions within that step all happen at the same simulated timestamp.
    """
    rewards:     List[float] = []
    steps_taken: int         = 0
    score:       float       = 0.0
    success:     bool        = False
    history:     List[str]   = []
    # Track recently preempted jobs to prevent preempt-schedule-preempt loops.
    # Maps job_id -> step_number when preempted. Expires after PREEMPT_COOLDOWN steps.
    preempt_history: Dict[str, int] = {}
    PREEMPT_COOLDOWN = 5  # blocks re-preemption for 5 steps

    log_start(task=task_name, env_name=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(task_name=task_name)
        obs: GpuSchedulerObservation = result.observation

        for step in range(1, max_steps + 1):
            if result.done:
                break

            # Build recently_preempted set (jobs within cooldown window)
            recently_preempted = {
                jid for jid, s in preempt_history.items()
                if step - s <= PREEMPT_COOLDOWN
            }

            # Get batch of actions from LLM
            parsed_actions, raw_response, proposed_strs = get_llm_actions(
                client, obs, step, history, max_steps, recently_preempted
            )

            # WAIT guard: if LLM wants to WAIT but there are schedulable jobs
            # that fit on available nodes, auto-schedule the best one instead
            if (
                len(parsed_actions) == 1
                and parsed_actions[0].action_type == "WAIT"
                and obs.queue
            ):
                # Find best schedulable job that fits
                for job in sorted(obs.queue, key=lambda j: (j.priority, -j.hours_in_queue)):
                    best_node = max(obs.nodes, key=lambda n: n.free_gpus)
                    needed = min(job.gpu_count, 8)
                    if best_node.free_gpus >= needed:
                        parsed_actions = [GpuSchedulerAction(
                            action_type="SCHEDULE",
                            job_id=job.job_id,
                            node_id=best_node.node_id,
                        )]
                        raw_response += f"\n[AUTO-CORRECTED] WAIT→SCHEDULE {job.job_id} {best_node.node_id} (queue not empty)"
                        break

            # Build action strings for the validated+executed batch (for logging)
            batch_plan = []
            for a in parsed_actions:
                if a.action_type == "SCHEDULE":
                    batch_plan.append(f"SCHEDULE {a.job_id} {a.node_id}")
                elif a.action_type == "PREEMPT":
                    batch_plan.append(f"PREEMPT {a.job_id}")
                else:
                    batch_plan.append("WAIT")

            # Detect actions the LLM proposed but were dropped as invalid
            dropped = [s for s in proposed_strs if s not in batch_plan]

            # Convert to single env action (BATCH, single, or WAIT)
            action, action_str = _build_batch_action(parsed_actions)

            _print_env_step_trace(
                env_step=step,
                llm_raw=raw_response,
                batch_size=len(parsed_actions),
                batch_index=1,
                batch_action_strings=batch_plan,
                executed_action=action_str,
            )

            # Single env.step() — all actions execute atomically, clock advances once
            result = await env.step(action)
            obs = result.observation

            reward = result.reward or 0.0
            done = result.done
            error = (
                obs.last_action_result
                if "INVALID" in (obs.last_action_result or "")
                else None
            )

            rewards.append(reward)
            steps_taken = step

            log_step(
                step=step,
                action=action_str,
                reward=reward,
                done=done,
                error=error,
                obs=obs,
            )
            print(_STEP_LOG_SEPARATOR, flush=True)
            print("\n" * _STEP_LOG_TRAIL_BLANK_LINES, end="", flush=True)

            # Track preemptions in this step for cooldown
            for a in parsed_actions:
                if a.action_type == "PREEMPT":
                    preempt_history[a.job_id] = step

            # Rolling history — include dropped-action feedback so LLM can self-correct
            dropped_note = (
                f" | DROPPED (invalid): {', '.join(dropped)}" if dropped else ""
            )
            history.append(
                f"Step {step}: {action_str} → reward {reward:+.2f} | "
                f"h{obs.current_hour:.0f}/{obs.total_hours:.0f} | "
                f"{obs.last_action_result or 'ok'}{dropped_note}"
            )

            if done:
                break

        # Extract grader score
        raw_score = getattr(result.observation, "score", None)
        score = float(raw_score) if raw_score is not None else 0.0
        score = max(0.0, min(score, 1.0))

        success = score >= success_threshold

    except Exception as exc:
        print(f"[DEBUG] Task '{task_name}' crashed: {exc}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

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
    log_path = _setup_inference_log_file()
    if log_path is not None:
        print(f"[LOG] Full run output also written to: {log_path}", flush=True)

    try:
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

        # If IMAGE_NAME looks like a URL, connect directly; otherwise use Docker
        if IMAGE_NAME and IMAGE_NAME.startswith("http"):
            env = GpuSchedulerEnv(base_url=IMAGE_NAME)
        else:
            env = await GpuSchedulerEnv.from_docker_image(IMAGE_NAME)

        try:
            results: Dict[str, bool] = {}

            for task_name, max_steps, threshold in TASKS:
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
            summary = " | ".join(
                f"{t}={'PASS' if v else 'FAIL'}" for t, v in results.items()
            )
            print(
                f"[SUMMARY] {summary} | all_passed={str(all_passed).lower()}",
                flush=True,
            )

        finally:
            # Always clean up the Docker container, even on exception
            try:
                await env.close()
            except Exception as e:
                print(
                    f"[DEBUG] env.close() error (container cleanup): {e}",
                    flush=True,
                )
    finally:
        _teardown_inference_log_file()


if __name__ == "__main__":
    asyncio.run(main())
