# Rich Logger for GPU Scheduler

This enhanced logging system provides clear, visually appealing output for the GPU Scheduler inference runs, making it much easier to understand what's happening at each step and identify problems.

## Features

- **Rich Table Display**: Each step is shown in a formatted table with clear field labels
- **Color-Coded Status**: 
  - ✓ Green for successful actions
  - ✗ Red for errors
  - Rewards colored green (positive) or red (negative)
- **Error Highlighting**: Errors are displayed in a prominent red panel with the problem clearly identified
- **Helpful Hints**: Contextual hints are automatically shown for each error type
- **Error Summary**: At the end of each episode, all errors are summarized in a table with categorization
- **Progress Tracking**: Shows current hour / total hours with percentage
- **Episode Summary**: Clean display of final results (success status, steps, score, total reward)

## Usage

### Environment Variable

The rich logger is controlled by the `VERBOSE_OUTPUT` environment variable:

```bash
# Enable rich output (default)
export VERBOSE_OUTPUT=1
python inference.py

# Disable rich output (show only standard logs)
export VERBOSE_OUTPUT=0
python inference.py
```

### Standard Output Format Preserved

The rich logger **does not** interfere with the standard `[START]`, `[STEP]`, and `[END]` log format required by the hackathon evaluator. Both formats are output simultaneously:

- Standard logs go to stdout as before (for automated grading)
- Rich formatted output is displayed in addition (for human readability)

## Example Output

### Step with Error

```
 Field      Value                                                               
 Step       1                                                                   
 Action     SCHEDULE p0_emergency 0                                             
 Reward     -1.30                                                               
 Done       false                                                               
 Progress   4 / 168 hours (2.4%)                                                
 Status     ✗ ERROR                                                             
╭───────────────────────────── ⚠ Problem Detected ─────────────────────────────╮
│ 'p0_emergency' not found in queue (may be running, completed, or not yet     │
│ arrived).                                                                    │
╰──────────────────────────────────────────────────────────────────────────────╯
💡 Hint: The job may be already running, completed, or hasn't arrived yet. 
Check the queue state in the observation.
```

### Step with Success

```
 Field      Value                                                               
 Step       2                                                                   
 Action     SCHEDULE job_001 1                                                  
 Reward     +0.23                                                               
 Done       false                                                               
 Progress   8 / 168 hours (4.8%)                                                
 Status     ✓ OK                                                                
```

### Error Summary

At the end of each episode with errors:

```
Errors Summary: 4 error(s) occurred
┏━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ S… ┃ Error Type       ┃ Message                                              ┃
┡━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 1  │ JOB_NOT_IN_QUEUE │ 'p0_emergency' not found in queue (may be running,   │
│ 4  │ INVALID_NODE_ID  │ node_id 10 is out of range. Must be 0–7.             │
│ 5  │ INSUFFICIENT_GP… │ node 3 has 2 free GPUs but 'large_job' needs 8.      │
│ 6  │ P0_PROTECTED     │ P0 job 'critical_p0' is non-preemptible...           │
└────┴──────────────────┴──────────────────────────────────────────────────────┘
```

### Episode Summary

```
─────────────────────────────── Episode Complete ───────────────────────────────
 Status        ✗ FAILED                                                         
 Steps         7                                                                
 Score         0.425                                                            
 Total Reward  -0.77                                                            
```

## Error Types

The logger automatically categorizes errors into types:

- **JOB_NOT_IN_QUEUE**: Trying to schedule a job that's not in the queue
- **INVALID_NODE_ID**: Node ID is out of range (must be 0-7)
- **INSUFFICIENT_GPUS**: Not enough free GPUs on the target node
- **INSUFFICIENT_NODES**: Not enough fully-free nodes for gang scheduling
- **P0_PROTECTED**: Attempting to preempt a P0 job (not allowed)
- **JOB_NOT_RUNNING**: Trying to preempt a job that isn't running

## Demo

To see the rich logger in action without running a full inference:

```bash
cd gpu_scheduler
.venv/bin/python demo_rich_logger.py
```

This will show example output for various scenarios including errors and successful actions.

## Technical Details

- **Module**: `gpu_scheduler/rich_logger.py`
- **Dependencies**: `rich >= 13.0.0` (added to `pyproject.toml`)
- **Integration**: Automatically imported in `inference.py` with graceful fallback if `rich` is not available

## Benefits

1. **Faster Debugging**: Errors are immediately visible with clear explanations
2. **Better Understanding**: Progress tracking helps understand simulation state
3. **Error Patterns**: Error summary table makes it easy to spot recurring issues
4. **Professional Output**: Clean, modern terminal UI improves the development experience
5. **No Interference**: Standard log format is preserved for automated evaluation
