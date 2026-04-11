"""
Tests that the project meets OpenEnv hackathon submission requirements.

Checks file structure, environment variable handling, OpenAI client usage,
openenv.yaml validity, and Dockerfile configuration.
"""

import os
import re
import sys

import pytest
import yaml

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)


class TestProjectStructure:
    """Verify required files exist in the expected locations."""

    def test_inference_py_in_root(self):
        assert os.path.isfile(os.path.join(ROOT, "inference.py")), \
            "inference.py must be in the project root"

    def test_dockerfile_exists(self):
        assert os.path.isfile(os.path.join(ROOT, "Dockerfile")), \
            "Dockerfile must exist in the project root"

    def test_requirements_txt_exists(self):
        assert os.path.isfile(os.path.join(ROOT, "requirements.txt")), \
            "requirements.txt must exist in the project root"

    def test_openenv_yaml_exists(self):
        assert os.path.isfile(os.path.join(ROOT, "gpu_scheduler", "openenv.yaml")), \
            "openenv.yaml must exist in the gpu_scheduler package"

    def test_readme_exists(self):
        assert os.path.isfile(os.path.join(ROOT, "README.md")), \
            "README.md must exist in the project root"

    def test_gpu_scheduler_package(self):
        pkg = os.path.join(ROOT, "gpu_scheduler")
        assert os.path.isdir(pkg), "gpu_scheduler/ package directory must exist"
        assert os.path.isfile(os.path.join(pkg, "__init__.py"))
        assert os.path.isfile(os.path.join(pkg, "models.py"))
        assert os.path.isfile(os.path.join(pkg, "client.py"))

    def test_server_package(self):
        srv = os.path.join(ROOT, "gpu_scheduler", "server")
        assert os.path.isdir(srv), "gpu_scheduler/server/ must exist"
        assert os.path.isfile(os.path.join(srv, "app.py"))
        assert os.path.isfile(os.path.join(srv, "gpu_scheduler_environment.py"))

    def test_tests_directory_exists(self):
        assert os.path.isdir(os.path.join(ROOT, "tests")), \
            "tests/ directory must exist"


class TestInferenceEnvVars:
    """Verify inference.py handles environment variables per spec."""

    def test_api_base_url_has_default(self):
        """API_BASE_URL must have a default value."""
        from inference import _get_api_base_url
        # Clear env var to check default
        old = os.environ.pop("API_BASE_URL", None)
        try:
            url = _get_api_base_url()
            assert url, "API_BASE_URL must have a non-empty default"
            assert url.startswith("http"), "Default must be a valid URL"
        finally:
            if old is not None:
                os.environ["API_BASE_URL"] = old

    def test_model_name_has_default(self):
        """MODEL_NAME must have a default value."""
        from inference import _get_model_name
        old = os.environ.pop("MODEL_NAME", None)
        try:
            name = _get_model_name()
            assert name, "MODEL_NAME must have a non-empty default"
        finally:
            if old is not None:
                os.environ["MODEL_NAME"] = old

    def test_hf_token_is_mandatory(self):
        """Missing HF_TOKEN (and API_KEY) must raise ValueError."""
        from inference import _make_openai_client
        old_hf = os.environ.pop("HF_TOKEN", None)
        old_api = os.environ.pop("API_KEY", None)
        try:
            with pytest.raises((ValueError, RuntimeError)):
                _make_openai_client()
        finally:
            if old_hf is not None:
                os.environ["HF_TOKEN"] = old_hf
            if old_api is not None:
                os.environ["API_KEY"] = old_api


class TestOpenAIClientUsage:
    """Verify inference.py uses the OpenAI client."""

    def test_imports_openai(self):
        source = open(os.path.join(ROOT, "inference.py")).read()
        assert "from openai import OpenAI" in source, \
            "Must use OpenAI client: `from openai import OpenAI`"

    def test_no_alternative_sdks(self):
        source = open(os.path.join(ROOT, "inference.py")).read()
        assert "import anthropic" not in source
        assert "import google.generativeai" not in source
        assert "import cohere" not in source

    def test_no_direct_http_llm_calls(self):
        source = open(os.path.join(ROOT, "inference.py")).read()
        assert "httpx.get(" not in source
        assert "requests.get(" not in source


class TestOpenEnvYaml:
    """Validate openenv.yaml structure and content."""

    @pytest.fixture
    def spec(self):
        path = os.path.join(ROOT, "gpu_scheduler", "openenv.yaml")
        with open(path) as f:
            return yaml.safe_load(f)

    def test_spec_version(self, spec):
        assert spec.get("spec_version") == 1

    def test_has_name(self, spec):
        assert spec.get("name") == "gpu_scheduler"

    def test_has_port(self, spec):
        assert spec.get("port") == 7860

    def test_has_tasks(self, spec):
        tasks = spec.get("tasks", [])
        assert len(tasks) >= 3, "Must define at least 3 tasks"

    def test_task_names(self, spec):
        task_names = {t["name"] for t in spec["tasks"]}
        assert "smooth_sailing" in task_names
        assert "deadline_crunch" in task_names
        assert "p0_emergency" in task_names

    def test_tasks_have_required_fields(self, spec):
        for task in spec["tasks"]:
            assert "name" in task
            assert "difficulty" in task
            assert "description" in task
            assert "episode_length" in task
            assert "success_threshold" in task

    def test_env_vars_section(self, spec):
        env_vars = spec.get("env_vars", {})
        assert "API_BASE_URL" in env_vars
        assert "MODEL_NAME" in env_vars
        assert env_vars["API_BASE_URL"].get("default"), \
            "API_BASE_URL must have a default in openenv.yaml"
        assert env_vars["MODEL_NAME"].get("default"), \
            "MODEL_NAME must have a default in openenv.yaml"

    def test_action_space_defined(self, spec):
        assert "action" in spec
        props = spec["action"].get("properties", {})
        assert "action_type" in props
        enum_vals = props["action_type"].get("enum", [])
        assert "SCHEDULE" in enum_vals
        assert "PREEMPT" in enum_vals
        assert "WAIT" in enum_vals

    def test_observation_space_defined(self, spec):
        assert "observation" in spec
        props = spec["observation"].get("properties", {})
        assert "cluster_grid" in props
        assert "nodes" in props
        assert "queue" in props
        assert "current_hour" in props


class TestDockerfile:
    """Validate Dockerfile meets spec."""

    @pytest.fixture
    def dockerfile(self):
        path = os.path.join(ROOT, "Dockerfile")
        with open(path) as f:
            return f.read()

    def test_exposes_port_7860(self, dockerfile):
        assert "EXPOSE 7860" in dockerfile

    def test_has_healthcheck(self, dockerfile):
        assert "HEALTHCHECK" in dockerfile

    def test_runs_uvicorn(self, dockerfile):
        assert "uvicorn" in dockerfile

    def test_port_7860_in_cmd(self, dockerfile):
        assert "7860" in dockerfile
