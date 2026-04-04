# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

from __future__ import annotations

from gpu_scheduler import ActionType, GpuSchedulerObservation, JobInfo, NodeInfo
from gpu_scheduler.inference import (
    SYSTEM_PROMPT,
    format_observation,
    format_recommended_action_section,
    get_heuristic_action,
    parse_action,
)
from gpu_scheduler.server.gpu_scheduler_environment import GpuSchedulerEnvironment


def _empty_grid() -> list[list[int]]:
    return [[-1] * 8 for _ in range(8)]


def _idle_nodes() -> list[NodeInfo]:
    return [
        NodeInfo(
            node_id=i,
            used_gpus=0,
            free_gpus=8,
            memory_contention=0.0,
            running_jobs=[],
        )
        for i in range(8)
    ]


def test_heuristic_schedules_when_capacity_available() -> None:
    obs = GpuSchedulerObservation(
        cluster_grid=_empty_grid(),
        nodes=_idle_nodes(),
        queue=[
            JobInfo(
                job_id="job_000",
                priority=2,
                priority_label="P2",
                gpu_count=4,
                duration_hours=5.0,
            )
        ],
        active_jobs=[],
        current_hour=0.0,
        total_hours=24.0,
        fully_free_nodes_count=8,
    )
    action = get_heuristic_action(obs)
    assert action is not None
    assert action.action_type == ActionType.SCHEDULE
    assert action.job_id == "job_000"
    assert action.node_id in range(8)


def test_system_prompt_includes_scheduling_algorithm() -> None:
    assert "MANDATORY SCHEDULING" in SYSTEM_PROMPT
    assert "CHECK QUEUE" in SYSTEM_PROMPT
    assert "NEVER:" in SYSTEM_PROMPT
    assert "DEADLINE MATH" in SYSTEM_PROMPT
    assert "job_p0_emergency" in SYSTEM_PROMPT


def test_observation_includes_gang_metadata() -> None:
    env = GpuSchedulerEnvironment()
    obs = env.reset(task_name="p0_emergency")
    assert hasattr(obs, "fully_free_nodes_count")
    assert obs.fully_free_nodes_count >= 0
    assert isinstance(obs.upcoming_p0_jobs, list)


def test_recommended_action_section_suggests_schedule() -> None:
    obs = GpuSchedulerObservation(
        cluster_grid=_empty_grid(),
        nodes=_idle_nodes(),
        queue=[
            JobInfo(
                job_id="job_007",
                priority=1,
                priority_label="P1",
                gpu_count=2,
                duration_hours=4.0,
                deadline_hour=10.0,
            )
        ],
        active_jobs=[],
        current_hour=1.0,
        total_hours=72.0,
        fully_free_nodes_count=8,
    )
    text = format_recommended_action_section(obs)
    assert "SCHEDULE job_007" in text
    assert "===" in text


def test_parse_schedule_accepts_node_aliases() -> None:
    a = parse_action("SCHEDULE job_001 Node 3")
    assert a.action_type == ActionType.SCHEDULE
    assert a.job_id == "job_001"
    assert a.node_id == 3
    b = parse_action("SCHEDULE job_002 node_5")
    assert b.node_id == 5


def test_format_observation_deadline_critical_section() -> None:
    obs = GpuSchedulerObservation(
        cluster_grid=_empty_grid(),
        nodes=_idle_nodes(),
        queue=[
            JobInfo(
                job_id="job_tight",
                priority=2,
                priority_label="P2",
                gpu_count=2,
                duration_hours=10.0,
                deadline_hour=5.0,
                progress=0.0,
            ),
        ],
        active_jobs=[],
        current_hour=2.0,
        total_hours=72.0,
        task_name="deadline_crunch",
    )
    block = format_observation(obs, step=1)
    assert "DEADLINE CRITICAL" in block
    assert "[URGENT]" in block or "[CRITICAL]" in block or "[OVERDUE]" in block


def test_format_observation_p0_prep_banner() -> None:
    obs = GpuSchedulerObservation(
        cluster_grid=_empty_grid(),
        nodes=_idle_nodes(),
        queue=[],
        active_jobs=[],
        current_hour=60.0,
        total_hours=168.0,
        task_name="p0_emergency",
    )
    block = format_observation(obs, step=1)
    assert "P0 PREP" in block


def test_format_observation_contains_recommended_section() -> None:
    obs = GpuSchedulerObservation(
        cluster_grid=_empty_grid(),
        nodes=_idle_nodes(),
        queue=[
            JobInfo(
                job_id="job_001",
                priority=3,
                priority_label="P3",
                gpu_count=1,
                duration_hours=3.0,
            )
        ],
        active_jobs=[],
        current_hour=0.0,
        total_hours=24.0,
    )
    block = format_observation(obs, step=1)
    assert "RECOMMENDED ACTION" in block
    assert "SCHEDULE job_001" in block
