#!/usr/bin/env python3
"""
Quick validation script for the two new batch test cases.
Tests that the environment can reset and generate jobs for both new tasks.
"""

import sys
import os

# Add the gpu_scheduler module to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'gpu_scheduler'))

from gpu_scheduler.server.gpu_scheduler_environment import GpuSchedulerEnvironment

def test_task(task_name: str):
    """Test that a task can be reset and has valid job generation."""
    print(f"\n{'='*60}")
    print(f"Testing task: {task_name}")
    print(f"{'='*60}")
    
    try:
        env = GpuSchedulerEnvironment()
        obs = env.reset(task_name=task_name)
        
        print(f"✓ Task reset successfully")
        print(f"✓ Current hour: {obs.current_hour}")
        print(f"✓ Total hours: {obs.total_hours}")
        print(f"✓ Task name: {obs.task_name}")
        print(f"✓ Queue size: {len(obs.queue)}")
        print(f"✓ Upcoming jobs: {len(obs.upcoming_jobs)}")
        
        # Check that jobs were generated
        if len(obs.queue) == 0 and len(obs.upcoming_jobs) == 0:
            print(f"✗ WARNING: No jobs in queue or upcoming!")
            return False
            
        # Print sample jobs
        if obs.queue:
            print(f"\nSample queued job:")
            job = obs.queue[0]
            print(f"  - {job.job_id}: P{job.priority}, {job.gpu_count} GPUs, {job.duration_hours}h")
            
        if obs.upcoming_jobs:
            print(f"\nSample upcoming job:")
            job = obs.upcoming_jobs[0]
            print(f"  - {job.job_id}: P{job.priority}, {job.gpu_count} GPUs, arrives at h{job.arrival_hour}")
        
        # Test taking a step
        from gpu_scheduler.models import GpuSchedulerAction, ActionType
        action = GpuSchedulerAction(action_type=ActionType.WAIT)
        obs = env.step(action)
        
        print(f"\n✓ Step executed successfully")
        print(f"✓ Reward: {obs.reward:.4f}")
        print(f"✓ Done: {obs.done}")
        print(f"✓ Current hour after step: {obs.current_hour}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Test both new tasks."""
    print("\n" + "="*60)
    print("VALIDATION: New Batch Test Cases")
    print("="*60)
    
    results = {}
    
    # Test both new tasks
    results["batch_priority_inversion"] = test_task("batch_priority_inversion")
    results["batch_gang_scheduling"] = test_task("batch_gang_scheduling")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    for task, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {task}")
    
    all_passed = all(results.values())
    print(f"\n{'='*60}")
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print(f"{'='*60}\n")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        print(f"{'='*60}\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
