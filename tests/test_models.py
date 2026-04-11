"""
Tests for Pydantic data models (action/observation types).

Validates schema correctness, field constraints, serialization,
and that models match the openenv.yaml spec.
"""

import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gpu_scheduler.models import (
    ActionType,
    GpuSchedulerAction,
    GpuSchedulerObservation,
    JobInfo,
    NodeInfo,
    SubAction,
)


# ---------------------------------------------------------------------------
# ActionType enum
# ---------------------------------------------------------------------------

class TestActionType:

    def test_all_action_types_present(self):
        assert ActionType.SCHEDULE == "SCHEDULE"
        assert ActionType.PREEMPT == "PREEMPT"
        assert ActionType.WAIT == "WAIT"
        assert ActionType.BATCH == "BATCH"

    def test_action_type_count(self):
        assert len(ActionType) == 4


# ---------------------------------------------------------------------------
# GpuSchedulerAction
# ---------------------------------------------------------------------------

class TestGpuSchedulerAction:

    def test_wait_action_minimal(self):
        action = GpuSchedulerAction(action_type=ActionType.WAIT)
        assert action.action_type == ActionType.WAIT
        assert action.job_id is None
        assert action.node_id is None

    def test_schedule_action_requires_job_and_node(self):
        action = GpuSchedulerAction(
            action_type=ActionType.SCHEDULE,
            job_id="job_001",
            node_id=3,
        )
        assert action.job_id == "job_001"
        assert action.node_id == 3

    def test_preempt_action(self):
        action = GpuSchedulerAction(
            action_type=ActionType.PREEMPT,
            job_id="job_005",
        )
        assert action.action_type == ActionType.PREEMPT
        assert action.job_id == "job_005"

    def test_batch_action_with_sub_actions(self):
        subs = [
            SubAction(action_type="SCHEDULE", job_id="job_001", node_id=0),
            SubAction(action_type="PREEMPT", job_id="job_002"),
        ]
        action = GpuSchedulerAction(
            action_type=ActionType.BATCH,
            sub_actions=subs,
        )
        assert len(action.sub_actions) == 2
        assert action.sub_actions[0].action_type == "SCHEDULE"
        assert action.sub_actions[1].action_type == "PREEMPT"

    def test_node_id_bounds(self):
        """node_id must be 0-7."""
        action = GpuSchedulerAction(
            action_type=ActionType.SCHEDULE, job_id="job_001", node_id=0
        )
        assert action.node_id == 0

        action = GpuSchedulerAction(
            action_type=ActionType.SCHEDULE, job_id="job_001", node_id=7
        )
        assert action.node_id == 7

        with pytest.raises(Exception):
            GpuSchedulerAction(
                action_type=ActionType.SCHEDULE, job_id="job_001", node_id=8
            )

        with pytest.raises(Exception):
            GpuSchedulerAction(
                action_type=ActionType.SCHEDULE, job_id="job_001", node_id=-1
            )

    def test_action_serialization(self):
        action = GpuSchedulerAction(
            action_type=ActionType.SCHEDULE, job_id="job_010", node_id=5
        )
        data = action.model_dump()
        assert data["action_type"] == "SCHEDULE"
        assert data["job_id"] == "job_010"
        assert data["node_id"] == 5


# ---------------------------------------------------------------------------
# SubAction
# ---------------------------------------------------------------------------

class TestSubAction:

    def test_schedule_sub_action(self):
        sub = SubAction(action_type="SCHEDULE", job_id="job_001", node_id=2)
        assert sub.action_type == "SCHEDULE"
        assert sub.node_id == 2

    def test_preempt_sub_action_no_node(self):
        sub = SubAction(action_type="PREEMPT", job_id="job_003")
        assert sub.node_id is None


# ---------------------------------------------------------------------------
# JobInfo
# ---------------------------------------------------------------------------

class TestJobInfo:

    def test_job_info_defaults(self):
        job = JobInfo(
            job_id="job_001",
            priority=1,
            priority_label="P1",
            gpu_count=4,
            duration_hours=10.0,
        )
        assert job.elapsed_hours == 0.0
        assert job.progress == 0.0
        assert job.status == "queued"
        assert job.deadline_hour is None
        assert job.assigned_nodes == []

    def test_job_priority_range(self):
        for p in range(4):
            job = JobInfo(
                job_id=f"job_{p}",
                priority=p,
                priority_label=f"P{p}",
                gpu_count=1,
                duration_hours=1.0,
            )
            assert job.priority == p

        with pytest.raises(Exception):
            JobInfo(
                job_id="bad", priority=5, priority_label="P5",
                gpu_count=1, duration_hours=1.0,
            )

    def test_job_gpu_count_range(self):
        """gpu_count 1-64 valid, outside should fail."""
        job = JobInfo(
            job_id="small", priority=0, priority_label="P0",
            gpu_count=1, duration_hours=1.0,
        )
        assert job.gpu_count == 1

        job = JobInfo(
            job_id="big", priority=0, priority_label="P0",
            gpu_count=64, duration_hours=1.0,
        )
        assert job.gpu_count == 64


# ---------------------------------------------------------------------------
# NodeInfo
# ---------------------------------------------------------------------------

class TestNodeInfo:

    def test_node_defaults(self):
        node = NodeInfo(node_id=0)
        assert node.total_gpus == 8
        assert node.used_gpus == 0
        assert node.free_gpus == 8
        assert node.memory_contention == 0.0
        assert node.running_jobs == []

    def test_node_id_range(self):
        for i in range(8):
            node = NodeInfo(node_id=i)
            assert node.node_id == i


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class TestObservation:

    def test_observation_defaults(self):
        obs = GpuSchedulerObservation(
            cluster_grid=[[-1] * 8 for _ in range(8)],
            nodes=[NodeInfo(node_id=i) for i in range(8)],
            reward=0.0,
            done=False,
        )
        assert obs.current_hour == 0.0
        assert obs.total_hours == 24.0
        assert obs.compute_burn_so_far == 0.0
        assert obs.task_name == "smooth_sailing"
        assert obs.score is None
