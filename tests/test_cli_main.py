from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import yaml

from eegprep.cli.core import EEGPrepCLIError, command_error, emit_command_result
from tests.fixtures import SAMPLE_DATASET_PATH


ROOT = Path(__file__).resolve().parents[1]


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT / 'src'}{os.pathsep}{env.get('PYTHONPATH', '')}"
    return subprocess.run(
        [sys.executable, "-m", "eegprep.cli", *args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _json_stdout(result: subprocess.CompletedProcess[str]) -> dict:
    return json.loads(result.stdout)


def test_batch_run_without_output_dir_preserves_config_relative_outputs(tmp_path):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config = config_dir / "pipeline.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "schema_version": "eegprep.pipeline.v1",
                "input": {"path": str(SAMPLE_DATASET_PATH), "format": "eeglab"},
                "output": {"directory": "relative-out"},
                "steps": [{"name": "qc"}],
            }
        ),
        encoding="utf-8",
    )

    result = _run_cli("batch", "run", str(SAMPLE_DATASET_PATH), "--pipeline", str(config), "--json")

    assert result.returncode == 0, result.stderr
    payload = _json_stdout(result)
    assert payload["status"] == "ok"
    assert (config_dir / "relative-out" / "qc.json").is_file()
    assert Path(payload["results"][0]["manifest"]).is_file()


def test_structured_command_error_preserves_non_default_exit_code(capsys):
    error = EEGPrepCLIError("CONFIG_SCHEMA_ERROR", "bad config", exit_code=2)

    result = command_error("pipeline run", error)
    exit_code = emit_command_result(result, json_output=True)

    # A structured-result usage error (exit_code=2) must not be silently downgraded to 1, so the
    # structured path matches the exception path for the same error class.
    assert result["exit_code"] == 2
    assert exit_code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["exit_code"] == 2
    assert payload["error"]["code"] == "CONFIG_SCHEMA_ERROR"
