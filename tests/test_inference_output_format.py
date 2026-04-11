"""
Tests that inference.py stdout output strictly follows the OpenEnv spec.

Spec format (exactly three line types on stdout):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
"""

import re
import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Regex patterns matching the exact spec
START_RE = re.compile(
    r"^\[START\] task=\S+ env=\S+ model=\S+$"
)
STEP_RE = re.compile(
    r"^\[STEP\] step=\d+ action=\S.* reward=\d+\.\d{2} done=(?:true|false) error=(?:null|\S.*)$"
)
END_RE = re.compile(
    r"^\[END\] success=(?:true|false) steps=\d+ rewards=[\d.,]+$"
)
VALID_LINE_RE = re.compile(r"^\[(START|STEP|END)\] ")


class TestStartLineFormat:
    """Validate [START] line generation."""

    def test_start_line_matches_spec(self):
        from inference import log_start
        import io
        buf = io.StringIO()
        old = sys.stdout
        sys.stdout = buf
        try:
            log_start(task="smooth_sailing", env_name="gpu_scheduler", model="Qwen/Qwen2.5-72B-Instruct")
        finally:
            sys.stdout = old
        line = buf.getvalue().strip()
        assert START_RE.match(line), f"[START] line does not match spec: {line!r}"

    def test_start_contains_required_fields(self):
        from inference import log_start
        import io
        buf = io.StringIO()
        old = sys.stdout
        sys.stdout = buf
        try:
            log_start(task="test_task", env_name="gpu_scheduler", model="test-model")
        finally:
            sys.stdout = old
        line = buf.getvalue().strip()
        assert "task=test_task" in line
        assert "env=gpu_scheduler" in line
        assert "model=test-model" in line


class TestStepLineFormat:
    """Validate [STEP] line generation."""

    def test_step_line_matches_spec(self):
        from inference import log_step
        import io
        buf = io.StringIO()
        old = sys.stdout
        sys.stdout = buf
        try:
            log_step(step=1, action="WAIT", reward=0.50, done=False, error=None)
        finally:
            sys.stdout = old
        line = buf.getvalue().strip()
        assert STEP_RE.match(line), f"[STEP] line does not match spec: {line!r}"

    def test_step_reward_two_decimals(self):
        from inference import log_step
        import io
        buf = io.StringIO()
        old = sys.stdout
        sys.stdout = buf
        try:
            log_step(step=5, action="SCHEDULE job_001 2", reward=0.123456, done=False, error=None)
        finally:
            sys.stdout = old
        line = buf.getvalue().strip()
        assert "reward=0.12" in line, "Reward must be formatted to 2 decimal places"

    def test_step_done_lowercase(self):
        from inference import log_step
        import io
        buf = io.StringIO()
        old = sys.stdout
        sys.stdout = buf
        try:
            log_step(step=3, action="WAIT", reward=0.0, done=True, error=None)
        finally:
            sys.stdout = old
        line = buf.getvalue().strip()
        assert "done=true" in line, "done must be lowercase boolean"
        assert "done=True" not in line, "done must not be Python-style True"

    def test_step_error_null_when_none(self):
        from inference import log_step
        import io
        buf = io.StringIO()
        old = sys.stdout
        sys.stdout = buf
        try:
            log_step(step=1, action="WAIT", reward=0.0, done=False, error=None)
        finally:
            sys.stdout = old
        line = buf.getvalue().strip()
        assert "error=null" in line

    def test_step_error_string_when_present(self):
        from inference import log_step
        import io
        buf = io.StringIO()
        old = sys.stdout
        sys.stdout = buf
        try:
            log_step(step=2, action="SCHEDULE job_001 5", reward=0.0, done=False, error="INVALID: node full")
        finally:
            sys.stdout = old
        line = buf.getvalue().strip()
        assert "error=INVALID:" in line

    def test_step_single_line_no_newlines(self):
        from inference import log_step
        import io
        buf = io.StringIO()
        old = sys.stdout
        sys.stdout = buf
        try:
            log_step(step=1, action="WAIT", reward=0.5, done=False, error=None, obs=None)
        finally:
            sys.stdout = old
        lines = [l for l in buf.getvalue().split("\n") if l.strip()]
        assert len(lines) == 1, f"[STEP] must be exactly one line, got {len(lines)}"

    def test_step_sanitizes_multiline_error(self):
        from inference import log_step
        import io
        buf = io.StringIO()
        old = sys.stdout
        sys.stdout = buf
        try:
            log_step(
                step=4,
                action="SCHEDULE job_001 2",
                reward=0.0,
                done=False,
                error="INVALID: node full\nretry later",
            )
        finally:
            sys.stdout = old
        line = buf.getvalue().strip()
        assert STEP_RE.match(line), f"[STEP] line does not match spec: {line!r}"
        assert "\n" not in line
        assert "error=INVALID: node full retry later" in line


class TestEndLineFormat:
    """Validate [END] line generation."""

    def test_end_line_matches_spec(self):
        from inference import log_end
        import io
        buf = io.StringIO()
        old = sys.stdout
        sys.stdout = buf
        try:
            log_end(success=True, steps=24, score=0.75, rewards=[0.0, 0.12, 1.0])
        finally:
            sys.stdout = old
        line = buf.getvalue().strip()
        assert END_RE.match(line), f"[END] line does not match spec: {line!r}"

    def test_end_no_score_field(self):
        """The spec does NOT include a score= field in [END]."""
        from inference import log_end
        import io
        buf = io.StringIO()
        old = sys.stdout
        sys.stdout = buf
        try:
            log_end(success=True, steps=10, score=0.85, rewards=[0.5, 0.6])
        finally:
            sys.stdout = old
        line = buf.getvalue().strip()
        assert "score=" not in line, "[END] line must NOT contain score= field per spec"

    def test_end_success_lowercase(self):
        from inference import log_end
        import io
        buf = io.StringIO()
        old = sys.stdout
        sys.stdout = buf
        try:
            log_end(success=False, steps=5, score=0.2, rewards=[0.1])
        finally:
            sys.stdout = old
        line = buf.getvalue().strip()
        assert "success=false" in line
        assert "success=False" not in line

    def test_end_rewards_comma_separated_two_decimals(self):
        from inference import log_end
        import io
        buf = io.StringIO()
        old = sys.stdout
        sys.stdout = buf
        try:
            log_end(success=True, steps=3, score=0.9, rewards=[0.0, 0.5, 1.0])
        finally:
            sys.stdout = old
        line = buf.getvalue().strip()
        assert "rewards=0.00,0.50,1.00" in line

    def test_end_line_still_valid_when_rewards_empty(self):
        from inference import log_end
        import io
        buf = io.StringIO()
        old = sys.stdout
        sys.stdout = buf
        try:
            log_end(success=False, steps=0, score=0.0, rewards=[])
        finally:
            sys.stdout = old
        line = buf.getvalue().strip()
        assert END_RE.match(line), f"[END] line does not match spec: {line!r}"
        assert "rewards=0.00" in line


class TestNoStdoutNoise:
    """Verify that debug/info/warn/error output goes to stderr, not stdout."""

    def test_log_debug_goes_to_stderr(self):
        from inference import _log_debug
        import io
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = stdout_buf
        sys.stderr = stderr_buf
        try:
            _log_debug("[DEBUG] test message")
        finally:
            sys.stdout = old_out
            sys.stderr = old_err
        assert stdout_buf.getvalue() == "", "Debug output must not appear on stdout"
        assert "[DEBUG] test message" in stderr_buf.getvalue()


class TestDiagnosticConsoleBehavior:
    """Verify default terminal output stays strict unless verbose is enabled."""

    def test_default_log_setup_hides_stderr_from_terminal(self, monkeypatch, tmp_path):
        import io
        import inference

        log_path = tmp_path / "inference.log"
        terminal_buf = io.StringIO()
        old_stderr = sys.stderr

        monkeypatch.setenv("INFERENCE_LOG_FILE", str(log_path))
        monkeypatch.delenv("INFERENCE_VERBOSE", raising=False)
        sys.stderr = terminal_buf

        try:
            inference._setup_inference_log_file()
            inference._log_debug("[DEBUG] hidden from terminal")
        finally:
            inference._teardown_inference_log_file()
            sys.stderr = old_stderr

        assert terminal_buf.getvalue() == ""
        assert "[DEBUG] hidden from terminal" in log_path.read_text()

    def test_verbose_log_setup_mirrors_stderr_to_terminal(self, monkeypatch, tmp_path):
        import io
        import inference

        log_path = tmp_path / "inference.log"
        terminal_buf = io.StringIO()
        old_stderr = sys.stderr

        monkeypatch.setenv("INFERENCE_LOG_FILE", str(log_path))
        monkeypatch.setenv("INFERENCE_VERBOSE", "1")
        sys.stderr = terminal_buf

        try:
            inference._setup_inference_log_file()
            inference._log_debug("[DEBUG] mirrored to terminal")
        finally:
            inference._teardown_inference_log_file()
            sys.stderr = old_stderr

        assert "[DEBUG] mirrored to terminal" in terminal_buf.getvalue()
        assert "[DEBUG] mirrored to terminal" in log_path.read_text()
