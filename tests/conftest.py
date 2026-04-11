"""Shared fixtures for the GPU Scheduler test suite."""

import pytest
import sys
import os

# Ensure project root and gpu_scheduler package are importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "gpu_scheduler"))


@pytest.fixture
def env():
    """Fresh GpuSchedulerEnvironment instance for each test."""
    from gpu_scheduler.server.gpu_scheduler_environment import GpuSchedulerEnvironment
    return GpuSchedulerEnvironment()


@pytest.fixture
def easy_obs(env):
    """Observation from a reset smooth_sailing episode."""
    return env.reset(task_name="smooth_sailing")


@pytest.fixture
def medium_obs(env):
    """Observation from a reset deadline_crunch episode."""
    return env.reset(task_name="deadline_crunch")


@pytest.fixture
def hard_obs(env):
    """Observation from a reset p0_emergency episode."""
    return env.reset(task_name="p0_emergency")
