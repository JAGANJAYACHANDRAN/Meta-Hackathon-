# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
GpuScheduler Environment Client
================================
WebSocket-based client that connects to the running GpuScheduler server.
Handles action serialization (Python → JSON) and result deserialization
(JSON → typed Pydantic models) for the full scheduler action/observation space.

Usage:
    # Against a running local server
    env = GpuSchedulerEnv(base_url="http://localhost:8000")
    result = await env.reset()
    result = await env.step(GpuSchedulerAction(action_type="WAIT"))
    await env.close()

    # Against a Docker container (auto-starts, auto-cleans up)
    env = await GpuSchedulerEnv.from_docker_image(os.getenv("IMAGE_NAME"))
"""

from typing import Dict, List, Optional

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

# NOTE: models.py is built in Step 2 — these imports become valid after that.
from .models import (
    GpuSchedulerAction,
    GpuSchedulerObservation,
    JobInfo,
    NodeInfo,
)


class GpuSchedulerEnv(
    EnvClient[GpuSchedulerAction, GpuSchedulerObservation, State]
):
    """
    Typed WebSocket client for the GpuScheduler Environment.

    Extends OpenEnv's EnvClient with GPU-scheduler-specific serialization.
    Each instance maintains a single persistent WebSocket session on the server,
    so state is fully isolated between concurrent agent runs.

    The client handles three action types:
        SCHEDULE(job_id, node_id) — place a queued job onto a node
        PREEMPT(job_id)           — evict a running job back to queue
        WAIT                      — advance the simulation clock one step
    """

    def _step_payload(self, action: GpuSchedulerAction) -> Dict:
        """
        Serialize a GpuSchedulerAction into the JSON body sent to POST /step.

        Only includes fields relevant to the chosen action_type so the server
        can cleanly validate partial actions (e.g. WAIT has no job_id).

        Args:
            action: Typed action from the agent.

        Returns:
            Dict that will be JSON-encoded and sent to the server.
        """
        payload: Dict = {"action_type": action.action_type}

        # Conditionally include optional fields — avoids None noise in JSON
        if action.job_id is not None:
            payload["job_id"] = action.job_id
        if action.node_id is not None:
            payload["node_id"] = action.node_id

        return payload

    def _parse_result(self, payload: Dict) -> StepResult[GpuSchedulerObservation]:
        """
        Deserialize the server's JSON response into a typed StepResult.

        The server returns a flat dict; we reconstruct the nested Pydantic
        models (NodeInfo, JobInfo) that make up the observation.

        Args:
            payload: Raw JSON dict from the server step/reset response.

        Returns:
            StepResult containing a fully-typed GpuSchedulerObservation.
        """
        obs_data = payload.get("observation", {})

        # --- Reconstruct nested node objects ---
        nodes = [
            NodeInfo(**n) for n in obs_data.get("nodes", [])
        ]

        # --- Reconstruct active jobs and queue ---
        active_jobs = [
            JobInfo(**j) for j in obs_data.get("active_jobs", [])
        ]
        queue = [
            JobInfo(**j) for j in obs_data.get("queue", [])
        ]

        # --- Build the full observation object ---
        observation = GpuSchedulerObservation(
            cluster_grid=obs_data.get("cluster_grid", [[-1] * 8 for _ in range(8)]),
            nodes=nodes,
            active_jobs=active_jobs,
            queue=queue,
            current_hour=obs_data.get("current_hour", 0.0),
            total_hours=obs_data.get("total_hours", 24.0),
            compute_burn_so_far=obs_data.get("compute_burn_so_far", 0.0),
            task_name=obs_data.get("task_name", "unknown"),
            last_action_result=obs_data.get("last_action_result", ""),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Deserialize the GET /state response into an OpenEnv State object.

        Args:
            payload: Raw JSON dict from the server state response.

        Returns:
            State with episode_id and current step_count.
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
