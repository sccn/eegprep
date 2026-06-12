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


def test_help_has_agent_start_section():
    result = _run_cli("--help")

    assert result.returncode == 0
    assert "Start here (for AI agents):" in result.stdout
    assert "eegprep skills get eegprep-cli" in result.stdout
    assert "migrate" in result.stdout
    assert "eeglab              Inspect EEGLAB history" not in result.stdout


def test_capabilities_schema_examples_and_skill_are_json_readable():
    capabilities = _run_cli("capabilities", "--json")
    schema = _run_cli("schema", "command", "filter", "--json")
    examples = _run_cli("examples", "pipeline", "--json")
    skill = _run_cli("skills", "get", "eegprep-cli", "--json")

    assert capabilities.returncode == 0
    commands = _json_stdout(capabilities)["commands"]
    assert "filter" in commands
    assert "batch" in commands
    assert "migrate" in commands
    assert "eeglab" not in commands
    assert schema.returncode == 0
    assert _json_stdout(schema)["schema"]["schema_version"] == "eegprep.schema.command.filter.v1"
    assert examples.returncode == 0
    assert any("pipeline run" in item for item in _json_stdout(examples)["examples"])
    assert skill.returncode == 0
    assert "Agent Rules" in _json_stdout(skill)["content"]


def test_transform_command_schemas_define_every_required_property():
    for command in ("resample", "rereference", "filter", "clean", "epoch", "ica"):
        schema = _json_stdout(_run_cli("schema", "command", command, "--json"))["schema"]
        properties = set(schema["properties"])
        missing = [name for name in schema["required"] if name not in properties]
        assert not missing, f"{command} required params absent from properties: {missing}"
        # --output is conditionally required (allowed to be omitted with --overwrite), so it must
        # not appear in the unconditional required list and the alternative must be advertised.
        assert "output" not in schema["required"], command
        assert schema["anyOf"] == [{"required": ["output"]}, {"required": ["overwrite"]}], command


def test_every_advertised_capability_has_schema_and_examples():
    capabilities = _json_stdout(_run_cli("capabilities", "--json"))["commands"]

    for command in capabilities:
        schema = _run_cli("schema", "command", command, "--json")
        examples = _run_cli("examples", command, "--json")
        assert schema.returncode == 0, f"schema missing for {command}: {schema.stdout}{schema.stderr}"
        assert examples.returncode == 0, f"examples missing for {command}: {examples.stdout}{examples.stderr}"


def test_json_parse_errors_return_stable_error_code():
    result = _run_cli("schema", "command", "--json")

    assert result.returncode == 2
    payload = _json_stdout(result)
    assert payload["status"] == "error"
    assert payload["code"] == "CONFIG_SCHEMA_ERROR"


def test_batch_run_dry_run_and_qc_pipeline(tmp_path):
    config = tmp_path / "pipeline.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "schema_version": "eegprep.pipeline.v1",
                "input": {"path": str(SAMPLE_DATASET_PATH), "format": "eeglab"},
                "output": {"directory": str(tmp_path / "unused")},
                "steps": [{"name": "qc"}],
            }
        ),
        encoding="utf-8",
    )
    output_root = tmp_path / "batch-out"

    dry_run = _run_cli(
        "batch",
        "run",
        str(SAMPLE_DATASET_PATH),
        "--pipeline",
        str(config),
        "--output-dir",
        str(output_root),
        "--dry-run",
        "--json",
    )
    assert dry_run.returncode == 0, dry_run.stderr
    dry_payload = _json_stdout(dry_run)
    assert dry_payload["status"] == "ok"
    assert dry_payload["subjects_total"] == 1
    assert dry_payload["results"][0]["pipeline_result"]["dry_run"] is True
    assert not output_root.exists()

    run = _run_cli(
        "batch",
        "run",
        str(SAMPLE_DATASET_PATH),
        "--pipeline",
        str(config),
        "--output-dir",
        str(output_root),
        "--json",
    )
    assert run.returncode == 0, run.stderr
    payload = _json_stdout(run)
    assert payload["status"] == "ok"
    assert payload["subjects_success"] == 1
    assert (output_root / "eeglab_data" / "qc.json").is_file()
    assert (output_root / "eeglab_data" / "eegprep_manifest.json").is_file()


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


def test_skill_get_human_output_is_markdown_not_result_envelope():
    result = _run_cli("skills", "get", "eegprep-cli")

    assert result.returncode == 0
    assert result.stdout.startswith("# EEGPrep CLI Agent Guide")
    assert '"status"' not in result.stdout


def test_inspect_and_validate_sample_data_json():
    inspect = _run_cli("inspect", "dataset", str(SAMPLE_DATASET_PATH), "--json")
    events = _run_cli("inspect", "events", str(SAMPLE_DATASET_PATH), "--json")
    channels = _run_cli("inspect", "channels", str(SAMPLE_DATASET_PATH), "--json")
    ica = _run_cli("inspect", "ica", str(SAMPLE_DATASET_PATH), "--json")
    validate = _run_cli("validate", str(SAMPLE_DATASET_PATH), "--json")

    assert inspect.returncode == 0
    assert _json_stdout(inspect)["n_channels"] > 0
    assert events.returncode == 0
    assert "events" in _json_stdout(events)
    assert channels.returncode == 0
    assert _json_stdout(channels)["count"] > 0
    assert ica.returncode == 0
    assert "has_ica" in _json_stdout(ica)
    assert validate.returncode == 0
    assert _json_stdout(validate)["can_continue"] is True


def test_missing_input_returns_stable_error_code():
    result = _run_cli("validate", str(ROOT / "does-not-exist.set"), "--json")

    assert result.returncode == 1
    payload = _json_stdout(result)
    assert payload["status"] == "error"
    assert payload["code"] == "INPUT_FILE_NOT_FOUND"


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


def _write_run_config(tmp_path, name):
    config = tmp_path / name
    config.write_text(
        yaml.safe_dump(
            {
                "schema_version": "eegprep.pipeline.v1",
                "input": {"path": str(SAMPLE_DATASET_PATH), "format": "eeglab"},
                "output": {"directory": str(tmp_path / f"out_{name}")},
                "steps": [{"name": "resample", "freq": 128}, {"name": "qc"}],
            }
        ),
        encoding="utf-8",
    )
    return config


def test_pipeline_run_verbose_and_quiet_flags_take_effect(tmp_path):
    verbose = _run_cli("pipeline", "run", str(_write_run_config(tmp_path, "verbose.yaml")), "--json", "--verbose")
    quiet = _run_cli("pipeline", "run", str(_write_run_config(tmp_path, "quiet.yaml")), "--json", "--quiet")

    assert verbose.returncode == 0, verbose.stderr
    assert quiet.returncode == 0, quiet.stderr
    # The flags must actually drive logging: --verbose emits progress chatter, --quiet suppresses it.
    verbose_stderr = [line for line in verbose.stderr.splitlines() if line.strip()]
    quiet_stderr = [line for line in quiet.stderr.splitlines() if line.strip()]
    assert verbose_stderr, "expected --verbose to emit progress logging on stderr"
    assert len(quiet_stderr) < len(verbose_stderr)
    # In both modes the single JSON line agents parse must stay clean on stdout.
    assert len([line for line in verbose.stdout.splitlines() if line.strip()]) == 1
    assert len([line for line in quiet.stdout.splitlines() if line.strip()]) == 1
    assert _json_stdout(verbose)["status"] == "ok"
    assert _json_stdout(quiet)["status"] == "ok"
