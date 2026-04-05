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
    GpuSchedulerAction,
    GpuSchedulerEnv,
    GpuSchedulerObservation,
)

# ---------------------------------------------------------------------------
# Environment variables — loaded from .env file or shell environment
# ---------------------------------------------------------------------------
IMAGE_NAME   = os.getenv("IMAGE_NAME")          # HF Space URL or Docker image
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
BENCHMARK    = "gpu_scheduler"

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
    You are an AI GPU cluster scheduler. Your goal is to MINIMIZE total
    compute burn (cluster costs $100,000 per day = ~$4,167/hour at full capacity).

    CLUSTER LAYOUT
    ---------------
    8 nodes (node 0–7), each with 8 GPUs = 64 GPUs total.

    JOB PRIORITIES
    ---------------
    P0 — mission-critical; NEVER preempt; schedule ASAP even if it blocks others
    P1 — high priority; avoid preemption; respect deadlines
    P2 — standard; preemption allowed only for P0 emergencies
    P3 — low/spot; first to be preempted if resources are needed

    ACTIONS (respond with EXACTLY ONE action per turn, on the first line)
    -----------------------------------------------------------------------
      SCHEDULE <job_id> <node_id>   Place a queued job on node 0–7.
                                    The node must have enough free GPUs.
                                    Jobs needing >8 GPUs span multiple nodes
                                    (e.g. 32-GPU job → any 4 nodes with ≥8 free GPUs each).
      PREEMPT <job_id>              Evict a running job. Burn scales with wasted checkpoint hours,
                                    job progress sunk so far, and priority (preempting P1 hurts more than P3).
                                    P0 jobs cannot be preempted.
      WAIT                          Advance the clock ONLY when the schedulable queue is empty, the cluster
                                    cannot place any queued job, or you need time for running jobs to finish
                                    to free nodes for a large gang job.

    CRITICAL RULES
    ---------------
    • job_id MUST come from the "SCHEDULABLE JOBS" list in the observation.
    • Job IDs always look like "job_000", "job_001", etc. NEVER use anything else.
    • Do NOT use task names, node names, or any other text as a job_id.
    • If SCHEDULABLE JOBS is empty, you MUST respond with WAIT.
    • Do NOT re-schedule a job that is already running (listed under RUNNING JOBS).
    • Pick a node_id (0–7) that has enough free GPUs for the job.

    REWARD SIGNALS (dense, every step; align actions with these)
    -----------------------------------------------------------
    + Progress reward:     Each step adds (Δt / job duration) × priority weight × contention factor.
                           Keeping jobs running earns immediate credit; contention slows effective rate.
    − Idle GPU “rent”:     Unused GPUs are taxed hourly. Rate is LOW when the queue is empty, HIGHER
                           when jobs are waiting, SEVERE when the backlog is large; it DOUBLES if any P0/P1
                           is still in the queue. Idle capacity + backlog is one of the worst outcomes.
    − SLA lateness:        After a small grace past deadline, penalty grows like (hours late)² × GPU count
                           (capped per step so signals stay stable).
    − Fragmentation tax:   Partially filled nodes that block common shapes (e.g. a full 8-GPU placement)
                           incur a small ongoing penalty—prefer cleaner packing when you have a choice.
    − Preemption burn:    Scales with wasted checkpoint hours, progress already made, and job priority.
    − Queue delay:         Extra nudge while P0/P1 sit in the queue; long waits add starvation pressure.

    STRATEGY TIPS
    --------------
    • If SCHEDULABLE JOBS is non-empty and some node has enough free GPUs for a job, SCHEDULE now—do
      not WAIT. Waiting burns idle-GPU rent while the queue is non-empty.
    • Prefer nodes with MORE free GPUs when placing a job to limit memory contention on that node.
    • For multi-node jobs (>8 GPUs): the environment needs that many fully free nodes; pick an anchor
      node_id that is fully free or let the env pick from fully free nodes.
    • Watch deadline_hour and prioritise jobs closest to missing SLA.
    • Use UPCOMING ARRIVALS to reserve capacity for a large gang job (e.g. 32-GPU P0); only then WAIT
      strategically for running jobs to finish or use PREEMPT on lower priority.
    • Preemption is for freeing space for higher-priority or gang jobs—not as a substitute for scheduling
      when nodes are idle.
    • WAIT only when the schedulable queue is empty, the cluster truly cannot place any queued job,
      or you are deliberately clearing nodes for an imminent large job.

    OUTPUT FORMAT
    --------------
    Line 1: EXACTLY one action — SCHEDULE job_XXX N / PREEMPT job_XXX / WAIT
    Do NOT output anything else on line 1. No explanations, no apologies.

    Example responses:
      SCHEDULE job_007 3
      PREEMPT job_002
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

    # SCHEDULE job_id node_id
    m = re.match(r"SCHEDULE\s+(\S+)\s+(\d+)", first_line, re.IGNORECASE)
    if m:
        return GpuSchedulerAction(
            action_type="SCHEDULE",
            job_id=m.group(1).lower(),
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


# ---------------------------------------------------------------------------
# LLM call — builds the full prompt and returns (action, raw_response)
# ---------------------------------------------------------------------------

def validate_action(
    action: GpuSchedulerAction,
    obs: GpuSchedulerObservation,
) -> GpuSchedulerAction:
    """
    Validate an LLM-proposed action against the current observation.

    If the action references an invalid job_id (not in queue) or an
    invalid node (not enough free GPUs), attempt to auto-correct by
    scheduling the highest-priority queued job on the best available node.
    Falls back to WAIT if no valid scheduling is possible.
    """
    queue_ids = {job.job_id for job in obs.queue}
    free_by_node = {n.node_id: n.free_gpus for n in obs.nodes}

    if action.action_type == "SCHEDULE":
        job_valid = action.job_id in queue_ids
        node_valid = (
            action.node_id is not None
            and action.node_id in free_by_node
            and free_by_node[action.node_id] > 0
        )
        if job_valid and node_valid:
            return action

        # Auto-correct: pick highest-priority queued job + best node
        if obs.queue:
            best_job = sorted(obs.queue, key=lambda j: (j.priority, -j.hours_in_queue))[0]
            best_node = max(obs.nodes, key=lambda n: n.free_gpus)
            if best_node.free_gpus >= best_job.gpu_count:
                return GpuSchedulerAction(
                    action_type="SCHEDULE",
                    job_id=best_job.job_id,
                    node_id=best_node.node_id,
                )

        return GpuSchedulerAction(action_type="WAIT")

    if action.action_type == "PREEMPT":
        running_ids = {job.job_id for job in obs.active_jobs}
        if action.job_id in running_ids:
            return action
        return GpuSchedulerAction(action_type="WAIT")

    return action


def get_llm_action(
    client: OpenAI,
    obs: GpuSchedulerObservation,
    step: int,
    history: List[str],
) -> Tuple[GpuSchedulerAction, str]:
    """
    Query the LLM for the next scheduling decision.

    Includes the last 3 history entries so the model can see the recent
    trajectory and avoid repeating useless WAIT actions.
    """
    state_text = format_observation(obs, step)

    history_block = "\n".join(history[-3:]) if history else "No prior steps."
    user_prompt = (
        f"{state_text}\n\n"
        f"RECENT TRAJECTORY:\n{history_block}\n\n"
        f"Respond with exactly one action on the first line. "
        f"Valid job_ids: {', '.join(j.job_id for j in obs.queue) or 'NONE (use WAIT)'}"
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
        action = parse_action(raw)
        action = validate_action(action, obs)
        return action, raw

    except Exception as exc:
        print(f"[DEBUG] LLM request failed at step {step}: {exc}", flush=True)
        return GpuSchedulerAction(action_type="WAIT"), "WAIT (api-error fallback)"


# ---------------------------------------------------------------------------
# Structured stdout loggers — format required by the hackathon evaluator
# ---------------------------------------------------------------------------

_STEP_LABEL_W = 18  # width for right-aligned labels in the state block


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
            action, raw_response = get_llm_action(client, obs, step, history)
            # Use only the first line as the logged action string
            action_str = raw_response.split("\n")[0].strip()

            # --- Execute action in the environment ---
            result = await env.step(action)
            obs    = result.observation

            reward = result.reward or 0.0
            done   = result.done
            # last_action_result carries validation errors (e.g. "invalid node_id")
            error  = obs.last_action_result if "INVALID" in (obs.last_action_result or "") else None

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
