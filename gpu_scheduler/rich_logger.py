#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Rich Logger for GPU Scheduler
==============================
One row per step: Step | Action | Reward | Progress | Status | Detail.
"""

import os
import re
from typing import Optional, Tuple

from rich.console import Console

# Initialize Rich console
console = Console()

# Check if verbose output is enabled
VERBOSE_OUTPUT = os.getenv("VERBOSE_OUTPUT", "1") == "1"

# Dashboard column widths (fixed-width pipe table; wide terminals only)
_ACTION_W = 27
_REWARD_W = 8
_PROGRESS_W = 10
_STATUS_W = 14
_DETAIL_W = 36


def _truncate(s: str, max_len: int) -> str:
    s = s.strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def _status_and_detail(error: Optional[str], done: bool) -> Tuple[str, str]:
    """Returns (status label, detail text). When error is None, detail is '-'."""
    if not error:
        label = "✅ OK" + (" [done]" if done else "")
        return label, "-"

    text = error.replace("INVALID: ", "").strip()

    if "not found in queue" in error:
        return "❌ Not found", "Check queue / running / arrivals"
    if "out of range" in error:
        return "❌ Bad node", "node_id must be 0–7"
    if "free GPUs" in error and "needs" in error:
        m = re.search(r"node\s+(\d+)\s+has\s+(\d+)\s+free GPUs", text)
        if m:
            detail = f"node {m.group(1)} has {m.group(2)} free; need more"
        else:
            detail = "Try another node or wait"
        return "❌ No GPUs", _truncate(detail, _DETAIL_W)
    if "fully-free nodes" in error:
        return "❌ No nodes", "Gang schedule needs idle nodes"
    if "non-preemptible" in error:
        return "❌ Protected", "Cannot preempt P0"
    if "not currently running" in error:
        return "❌ Not running", "Preempt only running jobs"

    return "❌ Error", _truncate(text, _DETAIL_W)


def log_step_table(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str],
    current_hour: float = 0.0,
    total_hours: float = 0.0,
    nodes_summary: Optional[str] = None,
) -> None:
    """One table row per step; optional dim line with node Empty/Partial/Filled ids."""
    if not VERBOSE_OUTPUT:
        return

    action_display = action
    if len(action_display) > _ACTION_W:
        action_display = action_display[: _ACTION_W - 3] + "..."

    reward_style = "green" if reward >= 0 else "red"
    reward_plain = format(reward, f"+{_REWARD_W}.2f")

    if total_hours > 0:
        progress_plain = f"{current_hour:.0f}/{total_hours:.0f}h"
    else:
        progress_plain = "-"
    progress_plain = progress_plain[:_PROGRESS_W].ljust(_PROGRESS_W)

    status_label, detail_plain = _status_and_detail(error, done)
    detail_cell = _truncate(detail_plain, _DETAIL_W).ljust(_DETAIL_W)

    if error:
        status_pad = _truncate(status_label, _STATUS_W).ljust(_STATUS_W)
        status_display = f"[red]{status_pad}[/red]"
    else:
        status_pad = _truncate(status_label, _STATUS_W).ljust(_STATUS_W)
        status_display = f"[green]{status_pad}[/green]"

    detail_display = f"[dim]{detail_cell}[/dim]"

    line = (
        f"{step:2d} │ [yellow]{action_display:<{_ACTION_W}}[/yellow] │ "
        f"[{reward_style}]{reward_plain}[/] │ {progress_plain} │ "
        f"{status_display} │ {detail_display}"
    )
    console.print(line, overflow="ellipsis", no_wrap=True, soft_wrap=False)
    if nodes_summary:
        console.print(
            f"{'':2s}   [dim]Nodes — {nodes_summary}[/dim]",
            overflow="ellipsis",
            no_wrap=True,
            soft_wrap=False,
        )


def log_episode_start(task: str, env: str, model: str) -> None:
    """Episode header plus dashboard column labels for step rows."""
    if not VERBOSE_OUTPUT:
        return

    console.print()
    console.rule(f"[bold cyan]Task: {task}[/bold cyan]", style="cyan")
    console.print(f"[dim]Model: {model} | Env: {env}[/dim]\n")

    header = (
        f"[bold]{'St':>2} │ {'Action':<{_ACTION_W}} │ "
        f"{'Reward':>{_REWARD_W}} │ {'Progress':^{_PROGRESS_W}} │ "
        f"{'Status':<{_STATUS_W}} │ {'Detail':<{_DETAIL_W}}[/bold]"
    )
    sep = (
        f"[dim]{'─' * 5}┼{'─' * _ACTION_W}┼{'─' * _REWARD_W}┼"
        f"{'─' * _PROGRESS_W}┼{'─' * _STATUS_W}┼{'─' * _DETAIL_W}[/dim]"
    )
    console.print(header, overflow="ellipsis", no_wrap=True, soft_wrap=False)
    console.print(sep, overflow="ellipsis", no_wrap=True, soft_wrap=False)


def log_episode_end(
    success: bool,
    steps: int,
    score: float,
    total_reward: float,
) -> None:
    """One-line episode summary to match dashboard density."""
    if not VERBOSE_OUTPUT:
        return

    score_style = "green" if score >= 0.7 else "yellow" if score >= 0.4 else "red"
    reward_style = "green" if total_reward >= 0 else "red"
    status_chip = "[bold green]✓ PASS[/bold green]" if success else "[bold red]✗ FAIL[/bold red]"

    console.print(
        f"\n[bold cyan]── Done[/bold cyan] │ {status_chip} │ "
        f"steps={steps} │ score=[{score_style}]{score:.3f}[/] │ "
        f"Σreward=[{reward_style}]{total_reward:+.2f}[/]\n"
    )


def log_error_summary(errors: list[tuple[int, str]]) -> None:
    """One line per error: step, category, truncated message."""
    if not VERBOSE_OUTPUT or not errors:
        return

    console.print(f"\n[bold red]Errors ({len(errors)})[/bold red]")
    hdr = f"[bold]{'St':>2} │ {'Type':<18} │ Detail[/bold]"
    sep = f"[dim]{'─' * 5}┼{'─' * 18}┼{'─' * 50}[/dim]"
    console.print(hdr, overflow="ellipsis", no_wrap=True, soft_wrap=False)
    console.print(sep, overflow="ellipsis", no_wrap=True, soft_wrap=False)

    for step, error_msg in errors:
        error_type = _classify_error(error_msg)[:18]
        clean_msg = error_msg.replace("INVALID: ", "").strip()
        if len(clean_msg) > 48:
            clean_msg = clean_msg[:45] + "..."
        line = (
            f"{step:2d} │ [yellow]{error_type:<18}[/yellow] │ [dim]{clean_msg}[/dim]"
        )
        console.print(line, overflow="ellipsis", no_wrap=True, soft_wrap=False)

    console.print()


def _classify_error(error: str) -> str:
    """Short error type label for the episode error summary."""
    if "not found in queue" in error:
        return "JOB_NOT_IN_QUEUE"
    elif "out of range" in error:
        return "INVALID_NODE_ID"
    elif "free GPUs" in error and "needs" in error:
        return "INSUFFICIENT_GPUS"
    elif "fully-free nodes" in error:
        return "INSUFFICIENT_NODES"
    elif "non-preemptible" in error:
        return "P0_PROTECTED"
    elif "not currently running" in error:
        return "JOB_NOT_RUNNING"
    return "UNKNOWN"
