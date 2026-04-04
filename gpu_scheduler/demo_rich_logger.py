#!/usr/bin/env python3
"""
Demo script to show the rich logger output without running the full inference.
"""

import sys
import os

# Add parent directory to path so we can import gpu_scheduler
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rich_logger import (
    log_episode_start,
    log_step_table,
    log_episode_end,
    log_error_summary,
)

def demo():
    """Demonstrate the rich logger with sample data."""
    
    # Episode start
    log_episode_start(task="p0_emergency", env="gpu_scheduler", model="Qwen/Qwen2.5-72B-Instruct")
    
    # Example steps with various scenarios
    
    # Step 1: Error - job not in queue
    log_step_table(
        step=1,
        action="SCHEDULE p0_emergency 0",
        reward=-1.30,
        done=False,
        error="INVALID: 'p0_emergency' not found in queue (may be running, completed, or not yet arrived).",
        current_hour=4.0,
        total_hours=168.0,
    )
    
    # Step 2: Successful action
    log_step_table(
        step=2,
        action="SCHEDULE job_001 1",
        reward=0.23,
        done=False,
        error=None,
        current_hour=8.0,
        total_hours=168.0,
    )
    
    # Step 3: Wait action (negative reward)
    log_step_table(
        step=3,
        action="WAIT",
        reward=-0.05,
        done=False,
        error=None,
        current_hour=12.0,
        total_hours=168.0,
    )
    
    # Step 4: Error - invalid node
    log_step_table(
        step=4,
        action="SCHEDULE job_002 10",
        reward=-0.10,
        done=False,
        error="INVALID: node_id 10 is out of range. Must be 0–7.",
        current_hour=16.0,
        total_hours=168.0,
    )
    
    # Step 5: Error - insufficient GPUs
    log_step_table(
        step=5,
        action="SCHEDULE large_job 3",
        reward=-0.10,
        done=False,
        error="INVALID: node 3 has 2 free GPUs but 'large_job' needs 8.",
        current_hour=20.0,
        total_hours=168.0,
    )
    
    # Step 6: Error - trying to preempt P0
    log_step_table(
        step=6,
        action="PREEMPT critical_p0",
        reward=-0.50,
        done=False,
        error="INVALID: P0 job 'critical_p0' is non-preemptible. Wait for lower-priority jobs to finish naturally.",
        current_hour=24.0,
        total_hours=168.0,
    )
    
    # Step 7: Successful high-reward action
    log_step_table(
        step=7,
        action="SCHEDULE p0_emergency 0",
        reward=1.85,
        done=False,
        error=None,
        current_hour=28.0,
        total_hours=168.0,
    )
    
    # Episode end with error summary
    errors = [
        (1, "INVALID: 'p0_emergency' not found in queue (may be running, completed, or not yet arrived)."),
        (4, "INVALID: node_id 10 is out of range. Must be 0–7."),
        (5, "INVALID: node 3 has 2 free GPUs but 'large_job' needs 8."),
        (6, "INVALID: P0 job 'critical_p0' is non-preemptible. Wait for lower-priority jobs to finish naturally."),
    ]
    
    log_error_summary(errors)
    
    log_episode_end(
        success=False,
        steps=7,
        score=0.425,
        total_reward=-0.77,
    )

if __name__ == "__main__":
    demo()
