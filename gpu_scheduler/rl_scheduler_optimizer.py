# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Lightweight scheduler scoring-weight tuner (gradient-free).

Enable with ``ENABLE_RL_OPTIMIZER=1``. Optional deps: ``numpy``, ``scipy``
(``pip install -e ".[rl]"`` from the ``gpu_scheduler`` directory).
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, Optional

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore

try:
    from scipy.optimize import differential_evolution
except ImportError:  # pragma: no cover
    differential_evolution = None


class SimplifiedScoringOptimizer:
    """
    Maintains a 5-D weight vector used to score nodes for single-GPU-node jobs.
    Periodically refines weights with differential evolution when scipy is present.
    """

    def __init__(self, initial_weights: Optional["np.ndarray"] = None) -> None:
        if np is None:
            self.current_weights = [0.2, 0.2, 0.3, 0.2, 0.1]
        elif initial_weights is None:
            self.current_weights = np.array([0.2, 0.2, 0.3, 0.2, 0.1], dtype=float)
        else:
            self.current_weights = np.asarray(initial_weights, dtype=float)

        self.episode_history: deque = deque(maxlen=50)

    def get_current_weights(self) -> Any:
        return self.current_weights

    def compute_node_score(
        self,
        node_free_gpus: int,
        node_contention: float,
        job_priority: int,
        job_progress: float,
        weights: Any,
    ) -> float:
        w = weights
        if np is not None:
            w = np.asarray(weights, dtype=float)

        scores: list[float] = []
        scores.append((node_free_gpus / 8.0) * float(w[0]))
        scores.append((1.0 - node_free_gpus / 8.0) * float(w[1]))
        scores.append((1.0 - node_contention) * float(w[2]))
        priority_boost = (3 - job_priority) / 3.0
        scores.append(priority_boost * float(w[3]))
        scores.append(job_progress * float(w[4]) * 0.5)
        return float(sum(scores))

    def evaluate_weights(self, weights: Any, episode_metrics: Dict) -> float:
        """Return negative combined score (for minimisers)."""
        score = (
            float(episode_metrics.get("score", 0.0)) * 0.5
            + float(episode_metrics.get("utilization", 0.0)) * 0.3
            + float(episode_metrics.get("sla_rate", 0.0)) * 0.2
        )
        return -score

    def optimize_weights(self, num_iterations: int = 30) -> Any:
        if (
            np is None
            or differential_evolution is None
            or len(self.episode_history) < 10
        ):
            return self.current_weights

        def objective(flat_weights: Any) -> float:
            vals: list[float] = []
            for ep in list(self.episode_history)[-10:]:
                vals.append(self.evaluate_weights(flat_weights, ep))
            return sum(vals) / max(len(vals), 1)

        bounds = [(0.0, 1.0)] * 5
        result = differential_evolution(
            objective,
            bounds,
            maxiter=num_iterations,
            seed=42,
            polish=False,
        )
        return result.x

    def update(self, episode_metrics: Dict[str, Any]) -> None:
        self.episode_history.append(dict(episode_metrics))
        if len(self.episode_history) < 10 or len(self.episode_history) % 10 != 0:
            return

        new_w = self.optimize_weights()
        if np is not None:
            old_s = -self.evaluate_weights(self.current_weights, episode_metrics)
            new_s = -self.evaluate_weights(new_w, episode_metrics)
            if new_s > old_s:
                self.current_weights = new_w
