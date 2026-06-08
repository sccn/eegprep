from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from eegprep.cli.commands import transforms
from eegprep.functions.popfunc.pop_loadset import pop_loadset
from eegprep.functions.popfunc.pop_saveset import pop_saveset
from tests.fixtures import SAMPLE_DATASET_PATH, create_test_eeg


ROOT = Path(__file__).resolve().parents[1]


def _run_transform_cli(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT / 'src'}{os.pathsep}{env.get('PYTHONPATH', '')}"
    return subprocess.run(
        [sys.executable, "-m", "eegprep.cli.commands.transforms", *args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _run_main_cli(*args: str) -> subprocess.CompletedProcess[str]:
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


def test_transform_subcommands_register_for_main_dispatcher():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    transforms.register_subcommands(subparsers)

    assert {"resample", "rereference", "filter", "clean", "epoch", "ica"} <= set(subparsers.choices)


def test_resample_writes_dataset_manifest_and_clean_json_stdout(tmp_path):
    output = tmp_path / "resampled.set"

    result = _run_transform_cli(
        "resample",
        str(SAMPLE_DATASET_PATH),
        "--freq",
        "128",
        "--output",
        str(output),
        "--json",
    )

    assert result.returncode == 0, result.stderr
    payload = _json_stdout(result)
    assert payload["status"] == "ok"
    assert payload["schema_version"] == transforms.RESULT_SCHEMA_VERSION
    assert payload["command"] == "resample"
    assert payload["summary"]["srate"] == 128
    assert "pop_resample" in payload["history"]
    assert output.exists()

    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    assert payload["manifest"]["path"] == str(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == transforms.MANIFEST_SCHEMA_VERSION
    assert manifest["parameters"]["freq"] == 128
    assert "pop_resample" in manifest["history"]
    assert any(record["path"].endswith("eeglab_data.fdt") for record in manifest["input_files"])

    loaded = pop_loadset(str(output))
    assert loaded["srate"] == 128
    assert "pop_resample" in str(loaded.get("history"))


def test_transform_requires_output_unless_overwrite_is_explicit():
    result = _run_transform_cli(
        "resample",
        str(SAMPLE_DATASET_PATH),
        "--freq",
        "128",
        "--json",
    )

    assert result.returncode == 1
    payload = _json_stdout(result)
    assert payload["status"] == "error"
    assert payload["error"]["code"] == "OUTPUT_REQUIRED"
    assert result.stdout.strip().startswith("{")


def test_main_dispatcher_reaches_transform_command():
    result = _run_main_cli(
        "resample",
        str(SAMPLE_DATASET_PATH),
        "--freq",
        "128",
        "--json",
    )

    assert result.returncode == 1
    payload = _json_stdout(result)
    assert payload["status"] == "error"
    assert payload["code"] == "OUTPUT_REQUIRED"
    assert result.stderr == ""


def test_transform_refuses_existing_output_without_overwrite(tmp_path):
    output = tmp_path / "existing.set"
    output.write_text("already here", encoding="utf-8")

    result = _run_transform_cli(
        "resample",
        str(SAMPLE_DATASET_PATH),
        "--freq",
        "128",
        "--output",
        str(output),
        "--json",
    )

    assert result.returncode == 1
    payload = _json_stdout(result)
    assert payload["error"]["code"] == "OUTPUT_EXISTS"
    assert payload["error"]["path"] == str(output)
    assert output.read_text(encoding="utf-8") == "already here"


def test_transform_refuses_output_manifest_path_collision(tmp_path):
    output = tmp_path / "resampled.set"

    result = _run_transform_cli(
        "resample",
        str(SAMPLE_DATASET_PATH),
        "--freq",
        "128",
        "--output",
        str(output),
        "--manifest",
        str(output),
        "--json",
    )

    assert result.returncode == 1
    payload = _json_stdout(result)
    assert payload["error"]["code"] == "OUTPUT_PATH_COLLISION"
    assert not output.exists()


def test_epoch_command_runs_against_saved_eeg_dataset(tmp_path):
    input_path = tmp_path / "events.set"
    output_path = tmp_path / "epochs.set"
    eeg = create_test_eeg(n_channels=3, n_samples=600, srate=100)
    eeg["event"] = [
        {"type": "S1", "latency": 200, "duration": 0},
        {"type": "S2", "latency": 300, "duration": 0},
        {"type": "S1", "latency": 450, "duration": 0},
    ]
    pop_saveset(eeg, str(input_path))

    result = _run_transform_cli(
        "epoch",
        str(input_path),
        "--event-type",
        "S1",
        "--tmin",
        "-0.1",
        "--tmax",
        "0.2",
        "--output",
        str(output_path),
        "--json",
    )

    assert result.returncode == 0, result.stderr
    payload = _json_stdout(result)
    assert payload["status"] == "ok"
    assert payload["summary"]["trials"] == 2
    assert "pop_epoch" in payload["history"]

    loaded = pop_loadset(str(output_path))
    assert loaded["trials"] == 2
    assert "pop_epoch" in str(loaded.get("history"))
