# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
GpuScheduler Environment — public package surface.

Importing from this package gives you everything needed to connect
an agent to the running environment server:

    from gpu_scheduler import GpuSchedulerEnv, GpuSchedulerAction, GpuSchedulerObservation

    env = await GpuSchedulerEnv.from_docker_image(image_name)
    result = await env.reset()
    result = await env.step(GpuSchedulerAction(action_type="WAIT"))
"""

# client.py — the WebSocket EnvClient (formerly inference.py)
from .client import GpuSchedulerEnv

# models.py — all typed Pydantic models for actions and observations
from .models import (
    ActionErrorCode,
    ActionType,
    GpuSchedulerAction,
    GpuSchedulerObservation,
    JobInfo,
    NodeInfo,
)

__all__ = [
    "GpuSchedulerEnv",
    "GpuSchedulerAction",
    "GpuSchedulerObservation",
    "JobInfo",
    "NodeInfo",
    "ActionType",
    "ActionErrorCode",
]
