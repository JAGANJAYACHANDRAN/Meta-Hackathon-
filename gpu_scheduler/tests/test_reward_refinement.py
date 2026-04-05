# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Unit tests for reward / penalty refinements (Phases 1–4)."""

from __future__ import annotations

import sys
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from pydantic import BaseModel


def _install_openenv_stubs() -> None:
    """Meta OpenEnv isn't on PyPI under the same API; stub enough to import the sim."""

    class _EnvAction(BaseModel):
        pass

    class _EnvObservation(BaseModel):
        """Mirror minimal OpenEnv observation fields used by GpuSchedulerObservation."""

        done: bool = False
        reward: float = 0.0

    @dataclass
    class _State:
        episode_id: str = ""
        step_count: int = 0

    class _Environment:
        pass

    def _ensure(name: str) -> ModuleType:
        if name not in sys.modules:
            sys.modules[name] = ModuleType(name)
        return sys.modules[name]

    _ensure("openenv")
    _ensure("openenv.core")
    _ensure("openenv.core.env_server")
    ifaces = _ensure("openenv.core.env_server.interfaces")
    ifaces.Environment = _Environment  # type: ignore[attr-defined]
    types_mod = _ensure("openenv.core.env_server.types")
    types_mod.Action = _EnvAction  # type: ignore[attr-defined]
    types_mod.Observation = _EnvObservation  # type: ignore[attr-defined]
    types_mod.State = _State  # type: ignore[attr-defined]


_install_openenv_stubs()

# Repo root (Meta-Hackathon-); load server + models without importing gpu_scheduler/__init__.py
_ROOT = Path(__file__).resolve().parents[2]
_pkg = _ROOT / "gpu_scheduler"
_srv = _pkg / "server"
for p in (_srv, _pkg):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import gpu_scheduler_environment as gse  # noqa: E402
from models import ActionType, GpuSchedulerAction, JobInfo  # noqa: E402

GpuSchedulerEnvironment = gse.GpuSchedulerEnvironment
IDLE_GPU_BASE_RATE = gse.IDLE_GPU_BASE_RATE
IDLE_GPU_BACKLOG_RATE = gse.IDLE_GPU_BACKLOG_RATE
IDLE_GPU_QUEUE_RATE = gse.IDLE_GPU_QUEUE_RATE
PRIORITY_WEIGHTS = gse.PRIORITY_WEIGHTS


class TestRewardRefinement(unittest.TestCase):
    """Regression and design checks for refined rewards."""

    def test_validate_reward_mechanism(self) -> None:
        env = GpuSchedulerEnvironment()
        checks = env._validate_reward_mechanism()
        self.assertTrue(all(checks.values()), checks)

    def test_phase1_dense_progress_fields(self) -> None:
        env = GpuSchedulerEnvironment()
        obs = env.reset("smooth_sailing")
        self.assertGreater(len(obs.queue), 0)
        job_id = obs.queue[0].job_id
        env.step(
            GpuSchedulerAction(action_type=ActionType.SCHEDULE, job_id=job_id, node_id=0)
        )
        active = env._active_jobs[job_id]
        self.assertGreater(active.progress_this_step, 0.0)
        self.assertGreater(active.effective_rate, 0.0)
        self.assertLessEqual(active.progress_this_step, 1.0)

    def test_phase1_p0_progress_dominates_p3_same_delta(self) -> None:
        self.assertGreater(PRIORITY_WEIGHTS[0], PRIORITY_WEIGHTS[3])

    def test_phase2_idle_rates_ordered(self) -> None:
        self.assertLess(IDLE_GPU_BASE_RATE, IDLE_GPU_QUEUE_RATE)
        self.assertLessEqual(IDLE_GPU_QUEUE_RATE, IDLE_GPU_BACKLOG_RATE)

    def test_phase3_quadratic_sla_penalty_increases_with_lateness(self) -> None:
        env = GpuSchedulerEnvironment()
        env.reset("smooth_sailing")
        j = JobInfo(
            job_id="sla_probe",
            priority=2,
            priority_label="P2",
            gpu_count=4,
            duration_hours=100.0,
            deadline_hour=0.0,
            arrival_hour=0.0,
            status="running",
            assigned_nodes=[0],
        )
        env._active_jobs[j.job_id] = j
        env._node_gpu_used[0] = 4
        env._node_jobs[0] = [j.job_id]
        env._current_hour = 5.0
        pen_lo = -env._advance_time(1.0)
        env.reset("smooth_sailing")
        env._active_jobs[j.job_id] = j
        env._node_gpu_used[0] = 4
        env._node_jobs[0] = [j.job_id]
        env._current_hour = 15.0
        pen_hi = -env._advance_time(1.0)
        self.assertGreater(pen_hi, pen_lo)

    def test_phase4a_fragmentation_penalty_partial_node(self) -> None:
        env = GpuSchedulerEnvironment()
        env.reset("smooth_sailing")
        env._node_gpu_used[0] = 7
        env._node_jobs[0] = ["phantom"]
        pen = env._calculate_fragmentation_penalty(1.0)
        self.assertGreater(pen, 0.0)
        env._node_gpu_used[0] = 0
        env._node_jobs[0] = []
        self.assertEqual(env._calculate_fragmentation_penalty(1.0), 0.0)

    def test_phase4b_preempt_penalty_higher_with_more_progress(self) -> None:
        env = GpuSchedulerEnvironment()
        env.reset("smooth_sailing")
        low = JobInfo(
            job_id="low_p",
            priority=3,
            priority_label="P3",
            gpu_count=4,
            duration_hours=20.0,
            progress=0.1,
            elapsed_hours=2.0,
            status="running",
            assigned_nodes=[0],
        )
        high = low.model_copy(update={"job_id": "high_p", "progress": 0.9})
        env._node_gpu_used[0] = 4
        env._active_jobs[low.job_id] = low
        env._node_jobs[0] = [low.job_id]
        pen_low = -env._do_preempt(low.job_id)
        env.reset("smooth_sailing")
        env._node_gpu_used[0] = 4
        env._active_jobs[high.job_id] = high
        env._node_jobs[0] = [high.job_id]
        pen_high = -env._do_preempt(high.job_id)
        self.assertGreater(pen_high, pen_low)

    def test_integration_wait_only_until_done_all_tasks(self) -> None:
        """All three tasks should terminate under passive WAIT (bounded steps)."""
        for task in ("smooth_sailing", "deadline_crunch", "p0_emergency"):
            env = GpuSchedulerEnvironment()
            obs = env.reset(task)
            for _ in range(600):
                obs = env.step(
                    GpuSchedulerAction(action_type=ActionType.WAIT)
                )
                if obs.done:
                    break
            self.assertTrue(obs.done, msg=f"{task} did not finish within step budget")
            self.assertIsNotNone(obs.score)
            self.assertGreaterEqual(obs.score, 0.0)
            self.assertLessEqual(obs.score, 1.0)


if __name__ == "__main__":
    unittest.main()
