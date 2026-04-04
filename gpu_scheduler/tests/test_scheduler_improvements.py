# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

from __future__ import annotations

from models import ActionErrorCode, GpuSchedulerAction, JobInfo
from server.gpu_scheduler_environment import GpuSchedulerEnvironment, ResourceManager


def test_job_id_p0_lowercase_at_arrival() -> None:
    env = GpuSchedulerEnvironment()
    env.reset(task_name="p0_emergency")
    env._current_hour = 72.0
    env._release_arriving_jobs(from_hour=72.0, to_hour=72.0)
    p0 = [j for j in env._queue if j.priority == 0]
    assert len(p0) == 1
    assert p0[0].job_id == "job_p0_emergency"


def test_smooth_priority_no_large_jump() -> None:
    env = GpuSchedulerEnvironment()
    env.reset()
    job = JobInfo(
        job_id="test",
        priority=1,
        priority_label="P1",
        gpu_count=8,
        duration_hours=10.0,
        deadline_hour=20.0,
    )
    env._current_hour = 12.0
    s1 = env._compute_dynamic_priority(job)
    env._current_hour = 12.5
    s2 = env._compute_dynamic_priority(job)
    assert abs(s1 - s2) < 150.0


def test_deadline_aware_preemption_ordering() -> None:
    env = GpuSchedulerEnvironment()
    env.reset()
    env._current_hour = 10.0
    job_loose = JobInfo(
        job_id="victim1",
        priority=2,
        priority_label="P2",
        gpu_count=4,
        duration_hours=10.0,
        progress=0.5,
    )
    job_tight = JobInfo(
        job_id="victim2",
        priority=2,
        priority_label="P2",
        gpu_count=4,
        duration_hours=10.0,
        progress=0.5,
        deadline_hour=env._current_hour + 6.0,
    )
    env._active_jobs = {"victim1": job_loose, "victim2": job_tight}
    cands = env._find_preemption_candidates(target_priority=0)
    costs = {jid: c for jid, _, c in cands}
    assert costs["victim2"] > costs["victim1"] * 2.4


def test_observation_gang_fields_after_reset() -> None:
    env = GpuSchedulerEnvironment()
    obs = env.reset(task_name="p0_emergency")
    assert obs.fully_free_nodes_count == 8
    assert isinstance(obs.upcoming_p0_jobs, list)


def test_resource_manager_best_fit_and_gang() -> None:
    used = {0: 0, 1: 4, 2: 6, 3: 8, 4: 0, 5: 8, 6: 8, 7: 8}
    out = ResourceManager.find_best_fit_nodes(
        4, used, num_nodes=8, gpus_per_node=8, strategy="best_fit"
    )
    assert out == [1]

    out_g = ResourceManager.find_best_fit_nodes(
        16, used, num_nodes=8, gpus_per_node=8, strategy="gang_reserve"
    )
    assert set(out_g or []) == {0, 4}


def test_schedule_invalid_sets_error_code() -> None:
    env = GpuSchedulerEnvironment()
    env.reset()
    act = GpuSchedulerAction(action_type="SCHEDULE", job_id="nosuchjob", node_id=0)
    obs = env.step(act)
    assert obs.last_action_error_code == ActionErrorCode.JOB_NOT_IN_QUEUE
    assert env._invalid_action_count >= 1


def test_schedule_accepts_matching_job_id_case() -> None:
    env = GpuSchedulerEnvironment()
    obs = env.reset(task_name="smooth_sailing")
    if not obs.queue:
        return
    jid = obs.queue[0].job_id.upper()
    act = GpuSchedulerAction(action_type="SCHEDULE", job_id=jid, node_id=0)
    obs2 = env.step(act)
    assert obs2.last_action_error_code is None
    assert "Scheduled" in (obs2.last_action_result or "")
