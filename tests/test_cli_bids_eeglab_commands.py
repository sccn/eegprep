from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from eegprep.functions.popfunc.eeg_emptyset import eeg_emptyset
from eegprep.functions.popfunc.pop_saveset import pop_saveset


def test_bids_export_validate_import_roundtrip(tmp_path, capsys):
    from eegprep.cli.commands import bids as bids_cli

    input_set = tmp_path / "input.set"
    bids_root = tmp_path / "bids"
    imported_set = tmp_path / "imported.set"
    pop_saveset(_eeg(), input_set)

    export_payload = bids_cli.export_dataset(
        input_set,
        bids_root,
        subject="01",
        task="rest",
    )
    assert export_payload["status"] == "ok"
    assert export_payload["command"] == "bids export"
    assert Path(export_payload["bids_root"]) == bids_root
    assert export_payload["history"].startswith("LASTCOM = pop_exportbids")

    validation_payload = bids_cli.validate_dataset(bids_root)
    assert validation_payload["status"] == "ok"
    assert validation_payload["errors"] == []
    assert not (bids_root / "eegprep_bids_validation.json").exists()

    import_payload = bids_cli.import_dataset(
        bids_root,
        output=imported_set,
        subject="01",
        task="rest",
    )
    assert import_payload["status"] == "ok"
    assert Path(import_payload["output"]) == imported_set
    assert import_payload["dataset"]["nbchan"] == 2
    assert imported_set.exists()

    exit_code = bids_cli.main(["validate", str(bids_root), "--json"])
    captured = capsys.readouterr()
    assert exit_code == 0
    stdout_payload = json.loads(captured.out)
    assert stdout_payload["status"] == "ok"
    assert captured.err == ""


def test_bids_validate_missing_path_returns_structured_error(tmp_path, capsys):
    from eegprep.cli.commands import bids as bids_cli

    exit_code = bids_cli.main(["validate", str(tmp_path / "missing"), "--json"])
    captured = capsys.readouterr()

    assert exit_code == 2
    payload = json.loads(captured.out)
    assert payload["status"] == "error"
    assert payload["error"]["code"] == "INPUT_FILE_NOT_FOUND"
    assert payload["error"]["path"] == str(tmp_path / "missing")
    assert captured.err == ""


def test_bids_import_refuses_existing_manifest_without_overwrite(tmp_path):
    from eegprep.cli.commands import bids as bids_cli

    input_set = tmp_path / "input.set"
    bids_root = tmp_path / "bids"
    imported_set = tmp_path / "imported.set"
    manifest = tmp_path / "manifest.json"
    manifest.write_text("existing", encoding="utf-8")
    pop_saveset(_eeg(), input_set)
    assert bids_cli.export_dataset(input_set, bids_root, subject="01", task="rest")["status"] == "ok"

    payload = bids_cli.import_dataset(bids_root, output=imported_set, manifest=manifest)

    assert payload["status"] == "error"
    assert payload["code"] == "OUTPUT_EXISTS"
    assert manifest.read_text(encoding="utf-8") == "existing"
    assert not imported_set.exists()


def test_eeglab_history_maps_supported_and_unsupported_commands(tmp_path):
    from eegprep.cli.commands import eeglab as eeglab_cli

    set_file = tmp_path / "history.set"
    eeg = _eeg()
    eeg["history"] = "\n".join(
        [
            "EEG = pop_loadset('input.set');",
            "EEG = pop_resample( EEG, 128);",
            "EEG = pop_unknown(EEG);",
        ]
    )
    pop_saveset(eeg, set_file)

    payload = eeglab_cli.history(set_file)

    assert payload["status"] == "ok"
    assert [operation["eeglab_command"] for operation in payload["operations"]] == [
        "pop_loadset",
        "pop_resample",
        "pop_unknown",
    ]
    assert payload["operations"][1]["operation"] == "resample"
    assert payload["operations"][1]["confidence"] >= 0.8
    assert payload["operations"][2]["supported"] is False
    assert payload["operations"][2]["unsupported"]["code"] == "COMMAND_NOT_IMPLEMENTED"


def test_eeglab_compare_reports_structured_differences(tmp_path):
    from eegprep.cli.commands import eeglab as eeglab_cli

    left = tmp_path / "left.set"
    right = tmp_path / "right.set"
    eeg_left = _eeg()
    eeg_right = _eeg()
    eeg_right["srate"] = 128.0
    eeg_right["data"] = eeg_right["data"].copy()
    eeg_right["data"][0, 0] += 1.25
    pop_saveset(eeg_left, left)
    pop_saveset(eeg_right, right)

    payload = eeglab_cli.compare(left, right)

    assert payload["status"] == "ok"
    assert payload["equivalent"] is False
    differences_by_path = {difference["path"]: difference for difference in payload["differences"]}
    assert differences_by_path["srate"]["code"] == "VALUE_MISMATCH"
    assert differences_by_path["data"]["code"] == "DATA_VALUE_MISMATCH"
    assert payload["data"]["max_abs_diff"] == 1.25


def test_eeglab_compare_reports_nan_placement_differences(tmp_path):
    from eegprep.cli.commands import eeglab as eeglab_cli

    left = tmp_path / "left_nan.set"
    right = tmp_path / "right_nan.set"
    eeg_left = _eeg()
    eeg_right = _eeg()
    eeg_left["data"] = eeg_left["data"].copy()
    eeg_right["data"] = eeg_right["data"].copy()
    eeg_left["data"][0, 0] = np.nan
    eeg_right["data"][0, 1] = np.nan
    pop_saveset(eeg_left, left)
    pop_saveset(eeg_right, right)

    payload = eeglab_cli.compare(left, right)

    assert payload["equivalent"] is False
    assert any(difference["code"] == "DATA_FINITE_MASK_MISMATCH" for difference in payload["differences"])


def test_eeglab_convert_script_reports_best_effort_conversion(tmp_path):
    from eegprep.cli.commands import eeglab as eeglab_cli

    script = tmp_path / "pipeline.m"
    output = tmp_path / "pipeline.yaml"
    script.write_text(
        "\n".join(
            [
                "EEG = pop_loadset('input.set');",
                "EEG = pop_resample(EEG, 128);",
                "topoplot(EEG.data(:, 1), EEG.chanlocs);",
            ]
        ),
        encoding="utf-8",
    )

    payload = eeglab_cli.convert_script(script, output=output)

    assert payload["status"] == "ok"
    assert payload["target"] == "eegprep-yaml"
    assert payload["converted_steps"][1]["name"] == "resample"
    assert payload["unsupported_commands"][0]["command"] == "topoplot"
    assert payload["unsupported_commands"][0]["code"] == "COMMAND_NOT_IMPLEMENTED"
    assert "schema_version: eegprep.pipeline.v1" in output.read_text(encoding="utf-8")


def _eeg() -> dict:
    eeg = eeg_emptyset()
    eeg.update(
        {
            "setname": "cli-test",
            "nbchan": 2,
            "pnts": 4,
            "trials": 1,
            "srate": 256.0,
            "xmin": 0.0,
            "xmax": 3 / 256.0,
            "times": np.arange(4, dtype=float) / 256.0,
            "data": np.array([[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]], dtype=float),
            "chanlocs": [
                {
                    "labels": "Cz",
                    "theta": 0.0,
                    "radius": 0.0,
                    "X": 0.0,
                    "Y": 0.0,
                    "Z": 1.0,
                    "sph_theta": 0.0,
                    "sph_phi": 90.0,
                    "sph_radius": 1.0,
                    "type": "EEG",
                    "urchan": 0,
                    "ref": "",
                },
                {
                    "labels": "Pz",
                    "theta": 180.0,
                    "radius": 0.5,
                    "X": 0.0,
                    "Y": -1.0,
                    "Z": 0.0,
                    "sph_theta": 180.0,
                    "sph_phi": 0.0,
                    "sph_radius": 1.0,
                    "type": "EEG",
                    "urchan": 1,
                    "ref": "",
                },
            ],
            "event": [
                {"type": "stim", "latency": 1.0, "duration": 0.0, "urevent": 0},
                {"type": "resp", "latency": 3.0, "duration": 0.0, "urevent": 1},
            ],
            "urevent": [
                {"type": "stim", "latency": 1.0, "duration": 0.0},
                {"type": "resp", "latency": 3.0, "duration": 0.0},
            ],
        }
    )
    return eeg
