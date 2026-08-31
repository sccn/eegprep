from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from eegprep.cli.core import EEGPrepCLIError
from eegprep.cli.discovery import skills_list
from eegprep.mcp import server, tools
from tests.fixtures import SAMPLE_DATASET_PATH


ROOT = Path(__file__).resolve().parents[1]


def _run_module(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT / 'src'}{os.pathsep}{env.get('PYTHONPATH', '')}"
    return subprocess.run(
        [sys.executable, "-m", "eegprep.mcp.server", *args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_mcp_capabilities_expose_agent_contract():
    payload = tools.capabilities()

    assert payload["status"] == "ok"
    assert "eegprep_run_cli_command" in payload["tools"]
    assert payload["command_policy"]["shell_execution"] is False
    assert payload["command_policy"]["writes_require_allow_write"] is True
    assert "resample" in payload["command_policy"]["write_commands"]


def test_mcp_agent_guide_is_bundled_and_discoverable():
    payload = tools.agent_guide()
    skill_names = {item["name"] for item in skills_list()["skills"]}

    assert payload["status"] == "ok"
    assert payload["name"] == "eegprep-mcp"
    assert "eegprep_plan_cli_command" in payload["content"]
    assert "eegprep-mcp" in skill_names


def test_inspect_dataset_sections_are_bounded():
    summary = tools.inspect_eeg_dataset(str(SAMPLE_DATASET_PATH))
    events = tools.inspect_eeg_dataset(str(SAMPLE_DATASET_PATH), section="events", limit=3)
    channels = tools.inspect_eeg_dataset(str(SAMPLE_DATASET_PATH), section="channels", limit=2)
    ica = tools.inspect_eeg_dataset(str(SAMPLE_DATASET_PATH), section="ica")

    assert summary["status"] == "ok"
    assert summary["n_channels"] > 0
    assert events["count"] > events["returned"]
    assert events["returned"] == 3
    assert events["truncated"] is True
    assert channels["returned"] == 2
    assert channels["truncated"] is True
    assert "has_ica" in ica


def test_validate_dataset_returns_cli_contract():
    payload = tools.validate_eeg_dataset(str(SAMPLE_DATASET_PATH))

    assert payload["status"] in {"ok", "warning"}
    assert payload["schema_version"] == "eegprep.validate.v1"
    assert payload["can_continue"] is True


def test_command_schema_and_examples_delegate_to_cli_discovery():
    schema = tools.command_schema("resample")
    examples = tools.command_examples("pipeline")

    assert schema["status"] == "ok"
    assert schema["schema"]["schema_version"] == "eegprep.schema.command.resample.v1"
    assert examples["status"] == "ok"
    assert any("pipeline plan" in item for item in examples["examples"])


def test_plan_cli_command_detects_read_only_and_write_modes(tmp_path):
    read_only = tools.plan_cli_command(["validate", str(SAMPLE_DATASET_PATH), "--json"])
    pipeline_plan = tools.plan_cli_command(["pipeline", "plan", str(tmp_path / "preprocess.yaml"), "--json"])
    bids_validate = tools.plan_cli_command(["bids", "validate", str(tmp_path), "--json"])
    write = tools.plan_cli_command(
        ["resample", str(SAMPLE_DATASET_PATH), "--freq", "64", "--output", str(tmp_path / "out.set"), "--json"]
    )
    overwrite = tools.plan_cli_command(["resample", str(SAMPLE_DATASET_PATH), "--freq", "64", "--overwrite", "--json"])

    assert read_only["writes_files"] is False
    assert pipeline_plan["writes_files"] is False
    assert bids_validate["writes_files"] is False
    assert write["writes_files"] is True
    assert write["requires_allow_write"] is True
    assert overwrite["overwrites_files"] is True
    assert overwrite["requires_allow_overwrite"] is True


def test_run_cli_command_requires_json_and_write_confirmation(tmp_path):
    with pytest.raises(EEGPrepCLIError, match="requires --json"):
        tools.run_cli_command(["validate", str(SAMPLE_DATASET_PATH)])

    with pytest.raises(EEGPrepCLIError, match="may write files"):
        tools.run_cli_command(
            ["resample", str(SAMPLE_DATASET_PATH), "--freq", "64", "--output", str(tmp_path / "out.set"), "--json"]
        )


def test_run_cli_command_executes_read_only_cli_in_process():
    payload = tools.run_cli_command(["validate", str(SAMPLE_DATASET_PATH), "--json"])

    assert payload["status"] == "ok"
    assert payload["exit_code"] == 0
    assert payload["result"]["schema_version"] == "eegprep.validate.v1"
    assert payload["result"]["can_continue"] is True
    assert json.loads(payload["stdout"])["schema_version"] == "eegprep.validate.v1"


def test_run_cli_command_rejects_non_eegprep_subcommands():
    with pytest.raises(EEGPrepCLIError) as excinfo:
        tools.plan_cli_command(["python", "-c", "print('no')"])

    assert excinfo.value.code == "COMMAND_NOT_ALLOWED"


def test_server_registers_tools_resources_and_prompts():
    pytest.importorskip("mcp.server.fastmcp")
    app = server.build_server()

    tool_names = {item.name for item in asyncio.run(app.list_tools())}
    resource_uris = {str(item.uri) for item in asyncio.run(app.list_resources())}
    prompt_names = {item.name for item in asyncio.run(app.list_prompts())}

    assert "eegprep_inspect_dataset" in tool_names
    assert "eegprep_run_cli_command" in tool_names
    assert "eegprep://capabilities" in resource_uris
    assert "eegprep://agent-guide" in resource_uris
    assert "eegprep_preprocess_plan" in prompt_names


def test_server_call_boundary_returns_structured_error():
    payload = server._call(tools.inspect_eeg_dataset, str(ROOT / "missing.set"))

    assert payload["status"] == "error"
    assert payload["code"] == "INPUT_FILE_NOT_FOUND"


def test_mcp_module_help_and_version_are_usable():
    help_result = _run_module("--help")
    version_result = _run_module("--version")

    assert help_result.returncode == 0
    assert "Run the EEGPrep Model Context Protocol server" in help_result.stdout
    assert "eegprep-mcp" in help_result.stdout
    assert version_result.returncode == 0
    assert version_result.stdout.startswith("eegprep-mcp ")
