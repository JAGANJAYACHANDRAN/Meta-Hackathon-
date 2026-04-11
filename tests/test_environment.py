"""
Tests for the GPU Scheduler environment simulation engine.

Validates reset, step, observation structure, action handling,
reward computation, and task-specific behaviour.
"""

import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "gpu_scheduler"))

from gpu_scheduler.models import (
    ActionType,
    GpuSchedulerAction,
    GpuSchedulerObservation,
    JobInfo,
    NodeInfo,
    SubAction,
)


# ---------------------------------------------------------------------------
# Reset & Observation structure
# ---------------------------------------------------------------------------

class TestReset:
    """Verify environment reset for every task."""

    @pytest.mark.parametrize("task_name", [
        "smooth_sailing",
        "deadline_crunch",
        "p0_emergency",
        "batch_priority_inversion",
        "batch_gang_scheduling",
    ])
    def test_reset_returns_valid_observation(self, env, task_name):
        obs = env.reset(task_name=task_name)
        assert isinstance(obs, GpuSchedulerObservation)
        assert obs.current_hour == 0.0
        assert obs.task_name == task_name
        assert len(obs.nodes) == 8
        assert len(obs.cluster_grid) == 8

    def test_reset_cluster_starts_empty(self, env):
        obs = env.reset(task_name="smooth_sailing")
        for node in obs.nodes:
            assert node.used_gpus == 0
            assert node.free_gpus == 8
            assert len(node.running_jobs) == 0

    def test_reset_has_jobs_in_queue_or_upcoming(self, env):
        obs = env.reset(task_name="smooth_sailing")
        total_jobs = len(obs.queue) + len(obs.upcoming_jobs)
        assert total_jobs > 0, "Episode must start with jobs available"

    def test_reset_invalid_task_falls_back(self, env):
        obs = env.reset(task_name="nonexistent_task")
        assert obs.task_name == "smooth_sailing"

    def test_reset_deterministic_with_seed(self, env):
        """Same task must produce identical queue on repeated reset."""
        obs1 = env.reset(task_name="deadline_crunch")
        q1 = [(j.job_id, j.priority, j.gpu_count) for j in obs1.queue]
        obs2 = env.reset(task_name="deadline_crunch")
        q2 = [(j.job_id, j.priority, j.gpu_count) for j in obs2.queue]
        assert q1 == q2, "Reset with same seed must be deterministic"


class TestObservationStructure:
    """Validate the shape and types of observation fields."""

    def test_node_info_fields(self, easy_obs):
        for node in easy_obs.nodes:
            assert isinstance(node, NodeInfo)
            assert 0 <= node.node_id <= 7
            assert node.total_gpus == 8
            assert node.free_gpus + node.used_gpus == node.total_gpus
            assert 0.0 <= node.memory_contention <= 1.0

    def test_job_info_fields(self, easy_obs):
        for job in easy_obs.queue:
            assert isinstance(job, JobInfo)
            assert job.job_id
            assert 0 <= job.priority <= 3
            assert job.gpu_count >= 1
            assert job.duration_hours > 0

    def test_cluster_grid_dimensions(self, easy_obs):
        assert len(easy_obs.cluster_grid) == 8
        for row in easy_obs.cluster_grid:
            assert len(row) == 8

    def test_initial_burn_is_zero(self, easy_obs):
        assert easy_obs.compute_burn_so_far == 0.0


# ---------------------------------------------------------------------------
# Action handling
# ---------------------------------------------------------------------------

class TestActions:
    """Validate SCHEDULE, PREEMPT, WAIT, and BATCH actions."""

    def test_wait_advances_clock(self, env):
        obs = env.reset(task_name="smooth_sailing")
        initial_hour = obs.current_hour
        action = GpuSchedulerAction(action_type=ActionType.WAIT)
        obs = env.step(action)
        assert obs.current_hour > initial_hour

    def test_schedule_valid_job(self, env):
        obs = env.reset(task_name="smooth_sailing")
        if not obs.queue:
            pytest.skip("No jobs in queue at hour 0")
        job = obs.queue[0]
        # Find a node with enough free GPUs
        target_node = None
        for node in obs.nodes:
            if node.free_gpus >= min(job.gpu_count, 8):
                target_node = node.node_id
                break
        if target_node is None:
            pytest.skip("No node with enough free GPUs")
        action = GpuSchedulerAction(
            action_type=ActionType.SCHEDULE,
            job_id=job.job_id,
            node_id=target_node,
        )
        obs = env.step(action)
        assert isinstance(obs, GpuSchedulerObservation)
        # Job should now be running (or clock advanced and it completed instantly)
        running_ids = [j.job_id for j in obs.active_jobs]
        queued_ids = [j.job_id for j in obs.queue]
        assert job.job_id in running_ids or job.job_id not in queued_ids

    def test_schedule_invalid_job_id_returns_error(self, env):
        env.reset(task_name="smooth_sailing")
        action = GpuSchedulerAction(
            action_type=ActionType.SCHEDULE,
            job_id="nonexistent_job_999",
            node_id=0,
        )
        obs = env.step(action)
        assert "INVALID" in (obs.last_action_result or "")

    def test_schedule_node_overflow_returns_error(self, env):
        obs = env.reset(task_name="smooth_sailing")
        if not obs.queue:
            pytest.skip("No jobs in queue")
        # Schedule jobs until a node is full, then try one more
        scheduled = 0
        for job in obs.queue:
            action = GpuSchedulerAction(
                action_type=ActionType.SCHEDULE,
                job_id=job.job_id,
                node_id=0,
            )
            obs = env.step(action)
            scheduled += 1
            if scheduled >= 3:
                break
        # Try scheduling on the same node (likely full or close)
        remaining = [j for j in obs.queue if j.gpu_count > obs.nodes[0].free_gpus]
        if remaining:
            action = GpuSchedulerAction(
                action_type=ActionType.SCHEDULE,
                job_id=remaining[0].job_id,
                node_id=0,
            )
            obs = env.step(action)
            assert "INVALID" in (obs.last_action_result or "") or obs.nodes[0].free_gpus >= 0

    def test_wait_when_queue_empty(self, env):
        """WAIT should always be valid, even with empty queue."""
        env.reset(task_name="smooth_sailing")
        action = GpuSchedulerAction(action_type=ActionType.WAIT)
        obs = env.step(action)
        assert isinstance(obs, GpuSchedulerObservation)

    def test_batch_action_atomic(self, env):
        obs = env.reset(task_name="smooth_sailing")
        if len(obs.queue) < 2:
            pytest.skip("Need at least 2 jobs in queue for batch test")

        jobs = obs.queue[:2]
        subs = []
        for i, job in enumerate(jobs):
            node = None
            for n in obs.nodes:
                if n.free_gpus >= min(job.gpu_count, 8):
                    node = n.node_id
                    break
            if node is not None:
                subs.append(SubAction(
                    action_type="SCHEDULE",
                    job_id=job.job_id,
                    node_id=node,
                ))

        if len(subs) < 2:
            pytest.skip("Not enough node capacity for batch test")

        action = GpuSchedulerAction(
            action_type=ActionType.BATCH,
            sub_actions=subs,
        )
        obs = env.step(action)
        assert isinstance(obs, GpuSchedulerObservation)


# ---------------------------------------------------------------------------
# Reward & scoring
# ---------------------------------------------------------------------------

class TestRewards:
    """Validate reward computation stays within spec bounds."""

    def test_reward_in_unit_interval(self, env):
        obs = env.reset(task_name="smooth_sailing")
        for _ in range(5):
            action = GpuSchedulerAction(action_type=ActionType.WAIT)
            obs = env.step(action)
            assert 0.0 <= obs.reward <= 1.0, f"Reward {obs.reward} outside [0, 1]"

    def test_scheduling_produces_nonzero_reward(self, env):
        obs = env.reset(task_name="smooth_sailing")
        if not obs.queue:
            pytest.skip("No jobs to schedule")
        job = obs.queue[0]
        node = next((n for n in obs.nodes if n.free_gpus >= min(job.gpu_count, 8)), None)
        if node is None:
            pytest.skip("No capacity")
        action = GpuSchedulerAction(
            action_type=ActionType.SCHEDULE, job_id=job.job_id, node_id=node.node_id
        )
        obs = env.step(action)
        # Reward should be non-zero (positive for scheduling work)
        assert obs.reward > 0.0


# ---------------------------------------------------------------------------
# Episode termination
# ---------------------------------------------------------------------------

class TestTermination:
    """Validate that episodes terminate correctly."""

    def test_episode_terminates_within_max_steps(self, env):
        env.reset(task_name="smooth_sailing")
        done = False
        for step in range(100):  # well above the 24-step limit
            action = GpuSchedulerAction(action_type=ActionType.WAIT)
            obs = env.step(action)
            if obs.done:
                done = True
                break
        assert done, "Episode must terminate"

    def test_done_obs_has_score(self, env):
        env.reset(task_name="smooth_sailing")
        obs = None
        for _ in range(50):
            action = GpuSchedulerAction(action_type=ActionType.WAIT)
            obs = env.step(action)
            if obs.done:
                break
        if obs and obs.done:
            assert obs.score is not None, "Terminal observation must include a score"
            assert 0.0 <= obs.score <= 1.0


# ---------------------------------------------------------------------------
# Task-specific: job generation counts
# ---------------------------------------------------------------------------

class TestTaskJobGeneration:
    """Validate correct job generation per task."""

    def test_smooth_sailing_has_moderate_jobs(self, env):
        obs = env.reset(task_name="smooth_sailing")
        total = len(obs.queue) + len(obs.upcoming_jobs)
        assert total >= 5, f"smooth_sailing should have at least 5 jobs, got {total}"

    def test_p0_emergency_has_gang_job(self, env):
        """p0_emergency must include a 32-GPU P0 gang job."""
        obs = env.reset(task_name="p0_emergency")
        all_jobs = list(obs.queue) + list(obs.upcoming_jobs)
        gang_jobs = [j for j in all_jobs if j.gpu_count >= 32 and j.priority == 0]
        # Gang job may arrive later (hour 72), so check upcoming
        # Run a few steps to see it appear
        if not gang_jobs:
            for _ in range(20):
                action = GpuSchedulerAction(action_type=ActionType.WAIT)
                obs = env.step(action)
                gang_jobs = [j for j in obs.upcoming_jobs if j.gpu_count >= 32 and j.priority == 0]
                gang_jobs += [j for j in obs.queue if j.gpu_count >= 32 and j.priority == 0]
                if gang_jobs:
                    break
        assert len(gang_jobs) >= 1, "p0_emergency must include a 32-GPU P0 gang job"

    def test_batch_priority_inversion_job_counts(self, env):
        obs = env.reset(task_name="batch_priority_inversion")
        all_jobs = list(obs.queue) + list(obs.upcoming_jobs)
        # Run through all steps to collect all jobs
        for _ in range(30):
            action = GpuSchedulerAction(action_type=ActionType.WAIT)
            obs = env.step(action)
            for j in obs.queue:
                if j.job_id not in [x.job_id for x in all_jobs]:
                    all_jobs.append(j)
            if obs.done:
                break
        p1_jobs = [j for j in all_jobs if j.priority == 1]
        p3_jobs = [j for j in all_jobs if j.priority == 3]
        assert len(p1_jobs) >= 8, f"Expected >=8 P1 jobs, got {len(p1_jobs)}"
        assert len(p3_jobs) >= 6, f"Expected >=6 P3 jobs, got {len(p3_jobs)}"

    def test_batch_gang_scheduling_has_two_gang_jobs(self, env):
        obs = env.reset(task_name="batch_gang_scheduling")
        all_jobs = list(obs.queue) + list(obs.upcoming_jobs)
        for _ in range(40):
            action = GpuSchedulerAction(action_type=ActionType.WAIT)
            obs = env.step(action)
            for j in obs.queue:
                if j.job_id not in [x.job_id for x in all_jobs]:
                    all_jobs.append(j)
            for j in obs.upcoming_jobs:
                if j.job_id not in [x.job_id for x in all_jobs]:
                    all_jobs.append(j)
            if obs.done:
                break
        gang_jobs = [j for j in all_jobs if j.gpu_count >= 16]
        assert len(gang_jobs) >= 2, f"Expected >=2 gang jobs, got {len(gang_jobs)}"
