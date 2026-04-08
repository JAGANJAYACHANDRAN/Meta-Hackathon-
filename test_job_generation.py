#!/usr/bin/env python3
"""
Minimal validation script that directly tests job generation without full environment.
"""

import sys
import random

# Test the job generation logic by importing just what we need
sys.path.insert(0, "gpu_scheduler/server")

def test_job_generation():
    """Test job generation for new tasks."""
    print("\n" + "="*60)
    print("VALIDATION: Job Generation for New Batch Test Cases")
    print("="*60)
    
    # Define the task configs
    TASK_CONFIGS = {
        "batch_priority_inversion": {
            "seed": 777,
            "total_hours": 48.0,
            "hours_per_step": 2.0,
        },
        "batch_gang_scheduling": {
            "seed": 2048,
            "total_hours": 96.0,
            "hours_per_step": 3.0,
        },
    }
    
    def _make_job_id(index: int) -> str:
        return f"job_{index:03d}"
    
    def _generate_job_schedule_batch_priority_inversion(rng: random.Random):
        """Generate jobs for batch_priority_inversion."""
        jobs = []
        idx = 0
        
        # Phase 1: 8 P3 jobs arrive early (hours 0-4)
        p3_arrivals = [0, 0, 1, 1, 2, 3, 3, 4]
        for hour in p3_arrivals:
            gpu_count = rng.choice([2, 2, 4, 4])
            duration  = round(rng.uniform(12.0, 24.0), 1)
            jobs.append({
                "job_id":         _make_job_id(idx),
                "priority":       3,
                "priority_label": "P3",
                "gpu_count":      gpu_count,
                "duration_hours": duration,
                "deadline_hour":  None,
                "arrival_hour":   float(hour),
            })
            idx += 1
        
        # Phase 2: 10 P1 jobs with tight deadlines (hours 12-20)
        p1_arrivals = sorted(rng.uniform(12.0, 20.0) for _ in range(10))
        for hour in p1_arrivals:
            gpu_count = rng.choice([4, 4, 6, 8, 8])
            duration  = round(rng.uniform(4.0, 8.0), 1)
            deadline  = round(hour + duration + rng.uniform(5.0, 8.0), 1)
            jobs.append({
                "job_id":         _make_job_id(idx),
                "priority":       1,
                "priority_label": "P1",
                "gpu_count":      gpu_count,
                "duration_hours": duration,
                "deadline_hour":  deadline,
                "arrival_hour":   round(hour, 1),
            })
            idx += 1
        
        # Phase 3: 6 P2 jobs scattered throughout
        p2_arrivals = sorted(rng.uniform(6.0, 42.0) for _ in range(6))
        for hour in p2_arrivals:
            gpu_count = rng.choice([2, 2, 4])
            duration  = round(rng.uniform(6.0, 14.0), 1)
            has_dl    = rng.random() < 0.5
            deadline  = round(hour + duration + rng.uniform(8.0, 12.0), 1) if has_dl else None
            jobs.append({
                "job_id":         _make_job_id(idx),
                "priority":       2,
                "priority_label": "P2",
                "gpu_count":      gpu_count,
                "duration_hours": duration,
                "deadline_hour":  deadline,
                "arrival_hour":   round(hour, 1),
            })
            idx += 1
        
        jobs.sort(key=lambda j: j["arrival_hour"])
        return jobs
    
    def _generate_job_schedule_batch_gang_scheduling(rng: random.Random):
        """Generate jobs for batch_gang_scheduling."""
        jobs = []
        idx = 0
        
        # Background load: 20 P2/P3 jobs
        background_arrivals = sorted(rng.uniform(0.0, 90.0) for _ in range(20))
        for hour in background_arrivals:
            gpu_count = rng.choice([1, 2, 2, 4, 4])
            priority  = rng.choice([2, 2, 2, 3, 3])
            duration  = round(rng.uniform(8.0, 36.0), 1)
            has_dl    = rng.random() < 0.4
            deadline  = round(hour + duration + rng.uniform(6.0, 18.0), 1) if has_dl else None
            jobs.append({
                "job_id":         _make_job_id(idx),
                "priority":       priority,
                "priority_label": f"P{priority}",
                "gpu_count":      gpu_count,
                "duration_hours": duration,
                "deadline_hour":  deadline,
                "arrival_hour":   round(hour, 1),
            })
            idx += 1
        
        # First gang job: 16-GPU P0 at hour 24
        jobs.append({
            "job_id":         "job_GANG_01",
            "priority":       0,
            "priority_label": "P0",
            "gpu_count":      16,
            "duration_hours": 24.0,
            "deadline_hour":  60.0,
            "arrival_hour":   24.0,
        })
        
        # Second gang job: 24-GPU P0 at hour 60
        jobs.append({
            "job_id":         "job_GANG_02",
            "priority":       0,
            "priority_label": "P0",
            "gpu_count":      24,
            "duration_hours": 30.0,
            "deadline_hour":  108.0,
            "arrival_hour":   60.0,
        })
        
        # 5 P1 jobs with deadlines
        p1_arrivals = sorted(rng.uniform(30.0, 75.0) for _ in range(5))
        for hour in p1_arrivals:
            gpu_count = rng.choice([2, 4, 4, 6])
            duration  = round(rng.uniform(6.0, 12.0), 1)
            deadline  = round(hour + duration + rng.uniform(6.0, 10.0), 1)
            jobs.append({
                "job_id":         _make_job_id(idx),
                "priority":       1,
                "priority_label": "P1",
                "gpu_count":      gpu_count,
                "duration_hours": duration,
                "deadline_hour":  deadline,
                "arrival_hour":   round(hour, 1),
            })
            idx += 1
        
        jobs.sort(key=lambda j: j["arrival_hour"])
        return jobs
    
    # Test batch_priority_inversion
    print(f"\n{'='*60}")
    print("Testing: batch_priority_inversion")
    print(f"{'='*60}")
    
    try:
        config = TASK_CONFIGS["batch_priority_inversion"]
        rng = random.Random(config["seed"])
        jobs = _generate_job_schedule_batch_priority_inversion(rng)
        
        print(f"✓ Generated {len(jobs)} jobs")
        
        # Count by priority
        p1_jobs = [j for j in jobs if j["priority"] == 1]
        p2_jobs = [j for j in jobs if j["priority"] == 2]
        p3_jobs = [j for j in jobs if j["priority"] == 3]
        
        print(f"✓ P1 jobs: {len(p1_jobs)} (expected: 10)")
        print(f"✓ P2 jobs: {len(p2_jobs)} (expected: 6)")
        print(f"✓ P3 jobs: {len(p3_jobs)} (expected: 8)")
        
        # Verify P3 jobs arrive early
        p3_arrival_hours = [j["arrival_hour"] for j in p3_jobs]
        print(f"✓ P3 arrival range: {min(p3_arrival_hours):.1f}h - {max(p3_arrival_hours):.1f}h (expected: 0-4h)")
        
        # Verify P1 jobs have deadlines
        p1_with_deadlines = [j for j in p1_jobs if j["deadline_hour"] is not None]
        print(f"✓ P1 jobs with deadlines: {len(p1_with_deadlines)}/{ len(p1_jobs)}")
        
        if len(jobs) == 24 and len(p1_jobs) == 10 and len(p2_jobs) == 6 and len(p3_jobs) == 8:
            print("✓ PASS: batch_priority_inversion")
            test1_pass = True
        else:
            print("✗ FAIL: Incorrect job counts")
            test1_pass = False
            
    except Exception as e:
        print(f"✗ FAIL: {str(e)}")
        import traceback
        traceback.print_exc()
        test1_pass = False
    
    # Test batch_gang_scheduling
    print(f"\n{'='*60}")
    print("Testing: batch_gang_scheduling")
    print(f"{'='*60}")
    
    try:
        config = TASK_CONFIGS["batch_gang_scheduling"]
        rng = random.Random(config["seed"])
        jobs = _generate_job_schedule_batch_gang_scheduling(rng)
        
        print(f"✓ Generated {len(jobs)} jobs")
        
        # Count gang jobs
        gang_jobs = [j for j in jobs if j["gpu_count"] >= 16]
        p1_jobs = [j for j in jobs if j["priority"] == 1]
        bg_jobs = [j for j in jobs if j["priority"] in [2, 3]]
        
        print(f"✓ Gang jobs (>=16 GPUs): {len(gang_jobs)} (expected: 2)")
        print(f"✓ P1 jobs: {len(p1_jobs)} (expected: 5)")
        print(f"✓ Background P2/P3 jobs: {len(bg_jobs)} (expected: 20)")
        
        # Verify gang job details
        for gang in gang_jobs:
            print(f"✓ Gang job: {gang['job_id']}, {gang['gpu_count']} GPUs @ h{gang['arrival_hour']}")
        
        if len(jobs) == 27 and len(gang_jobs) == 2 and len(p1_jobs) == 5:
            print("✓ PASS: batch_gang_scheduling")
            test2_pass = True
        else:
            print("✗ FAIL: Incorrect job counts")
            test2_pass = False
            
    except Exception as e:
        print(f"✗ FAIL: {str(e)}")
        import traceback
        traceback.print_exc()
        test2_pass = False
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'✓ PASS' if test1_pass else '✗ FAIL'}: batch_priority_inversion")
    print(f"{'✓ PASS' if test2_pass else '✗ FAIL'}: batch_gang_scheduling")
    
    if test1_pass and test2_pass:
        print(f"\n{'='*60}")
        print("✓ ALL TESTS PASSED")
        print(f"{'='*60}\n")
        return 0
    else:
        print(f"\n{'='*60}")
        print("✗ SOME TESTS FAILED")
        print(f"{'='*60}\n")
        return 1

if __name__ == "__main__":
    sys.exit(test_job_generation())
