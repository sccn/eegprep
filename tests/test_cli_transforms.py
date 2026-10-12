from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from eegprep.cli.core import MANIFEST_SCHEMA_VERSION, read_manifest
from eegprep.cli.commands import transforms
from eegprep.functions.popfunc.pop_loadset import pop_loadset
from tests.fixtures import SAMPLE_DATASET_PATH


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


def _json_stdout(result: subprocess.CompletedProcess[str]) -> dict:
    return json.loads(result.stdout)


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
    stored_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert {record["path"] for record in stored_manifest["output_files"]} == {"resampled.set"}
    manifest = read_manifest(manifest_path)
    assert manifest["schema_version"] == MANIFEST_SCHEMA_VERSION
    assert manifest["parameters"]["freq"] == 128
    assert "pop_resample" in manifest["history"]
    assert any(record["path"].endswith("eeglab_data.fdt") for record in manifest["input_files"])
    assert all(Path(record["path"]).is_absolute() for record in manifest["input_files"] + manifest["output_files"])

    loaded = pop_loadset(str(output))
    assert loaded["srate"] == 128
    assert "pop_resample" in str(loaded.get("history"))


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
    assert payload["code"] == "OUTPUT_EXISTS"
    assert payload["path"] == str(output)
    assert output.read_text(encoding="utf-8") == "already here"


def test_rereference_channels_convert_cli_indices_to_python_indices(monkeypatch):
    captured = {}

    def fake_pop_reref(eeg, ref, **kwargs):
        captured["ref"] = ref
        captured["kwargs"] = kwargs
        return eeg, "EEG = pop_reref( EEG, [0, 1]);"

    monkeypatch.setattr(transforms, "pop_reref", fake_pop_reref)
    args = argparse.Namespace(
        method="channels",
        channels=["1", "2"],
        exclude=None,
        keep_ref=False,
        huber=None,
        refica="on",
    )

    result = transforms._rereference({"data": []}, args)

    assert captured["ref"] == [0, 1]
    assert captured["kwargs"]["refica"] == "on"
    assert "pop_reref" in result.history


def test_ica_no_deterministic_does_not_force_runica_random_reset():
    parser = transforms.build_parser()
    args = parser.parse_args(
        [
            "ica",
            str(SAMPLE_DATASET_PATH),
            "--output",
            "ica.set",
            "--no-deterministic",
        ]
    )

    assert transforms._ica_options(args) == {}
    assert transforms._ica_is_deterministic("runica", {}, args) is False


def test_ica_default_deterministic_forces_runica_random_reset():
    parser = transforms.build_parser()
    args = parser.parse_args(["ica", str(SAMPLE_DATASET_PATH), "--output", "ica.set"])

    assert transforms._ica_options(args)["rndreset"] == "off"
