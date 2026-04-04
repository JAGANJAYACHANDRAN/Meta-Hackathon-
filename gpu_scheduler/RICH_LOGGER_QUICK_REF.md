# Rich Logger Quick Reference

## Enable/Disable

```bash
# Enable rich output (default)
export VERBOSE_OUTPUT=1

# Disable rich output
export VERBOSE_OUTPUT=0
```

## Demo

```bash
cd gpu_scheduler
.venv/bin/python demo_rich_logger.py
```

## What You Get

### ✓ Success Step
- Green checkmark
- Positive reward in green
- Progress percentage

### ✗ Error Step
- Red X symbol
- Negative reward in red
- Error panel with problem description
- Helpful hint specific to error type

### Episode Summary
- Overall status (success/failed)
- Total steps taken
- Final score (color-coded)
- Total reward

### Error Summary Table
- Lists all errors from the episode
- Categorizes by type
- Shows step number and message

## Error Types

| Type | Hint |
|------|------|
| `JOB_NOT_IN_QUEUE` | Check queue state - job may be running, completed, or not arrived |
| `INVALID_NODE_ID` | Use node_id between 0 and 7 |
| `INSUFFICIENT_GPUS` | Wait for jobs to complete or use different node |
| `INSUFFICIENT_NODES` | Gang scheduling requires fully idle nodes |
| `P0_PROTECTED` | P0 jobs cannot be preempted - only preempt P1/P2 |
| `JOB_NOT_RUNNING` | Cannot preempt job that isn't running |

## Files

- `rich_logger.py` - Core module
- `demo_rich_logger.py` - Demo script
- `RICH_LOGGER.md` - Full documentation
- `IMPLEMENTATION_SUMMARY.md` - Technical details

## Integration

The rich logger integrates seamlessly with `inference.py`:
- Standard `[START]`, `[STEP]`, `[END]` logs are preserved
- Rich output is displayed in addition for human readability
- Graceful fallback if rich library is unavailable
