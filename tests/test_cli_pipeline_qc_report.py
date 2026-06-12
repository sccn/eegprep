import json
from pathlib import Path

import numpy as np
import yaml

from eegprep.cli.commands import transforms as transforms_cli
from eegprep.cli.commands.pipeline import (
    _channel_indices,
    plan_pipeline_config,
    run_pipeline_config,
    validate_pipeline_config,
)
from eegprep.cli.commands.qc import compute_qc_metrics, qc_dataset, qc_report_dataset
from eegprep.cli.commands.report import report_dataset
from eegprep.functions.popfunc.pop_saveset import pop_saveset
from tests.fixtures import create_test_eeg


SAMPLE_SET = Path(__file__).resolve().parents[1] / "sample_data" / "eeglab_data.set"


def test_pipeline_validate_plan_and_dry_run_do_not_write(tmp_path):
    config_path = _write_pipeline_config(
        tmp_path,
        steps=[{"name": "resample", "freq": 128}, {"name": "qc"}],
    )
    output_dir = tmp_path / "out"

    validation = validate_pipeline_config(config_path)
    assert validation["status"] == "ok"
    assert validation["validation"]["normalized_steps"][0]["name"] == "resample"

    plan = plan_pipeline_config(config_path)
    assert plan["status"] == "ok"
    assert output_dir.exists() is False
    assert plan["plan"]["mutates_input"] is False
    assert {item["type"] for item in plan["plan"]["output_files"]} == {"eeglab_set", "json", "manifest"}

    dry_run = run_pipeline_config(config_path, dry_run=True)
    assert dry_run["status"] == "ok"
    assert dry_run["dry_run"] is True
    assert output_dir.exists() is False


def test_pipeline_run_writes_qc_report_and_manifest(tmp_path):
    config_path = _write_pipeline_config(
        tmp_path,
        steps=[{"name": "qc"}, {"name": "report", "format": "html"}],
    )

    result = run_pipeline_config(config_path)

    assert result["status"] == "ok"
    assert (tmp_path / "out" / "qc.json").is_file()
    assert (tmp_path / "out" / "report.html").is_file()
    manifest_path = tmp_path / "out" / "eegprep_manifest.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "eegprep.manifest.v1"
    assert manifest["command"] == "pipeline run"
    assert {item["type"] for item in manifest["output_files"]} == {"json", "html_report"}
    assert isinstance(result["qc"]["recommendations"], list)


def test_pipeline_run_resample_writes_dataset_manifest_and_history(tmp_path):
    config_path = _write_pipeline_config(
        tmp_path,
        steps=[{"name": "resample", "freq": 128}],
    )

    result = run_pipeline_config(config_path)

    assert result["status"] == "ok"
    assert (tmp_path / "out" / "eeglab_data_eegprep.set").is_file()
    manifest = json.loads((tmp_path / "out" / "eegprep_manifest.json").read_text(encoding="utf-8"))
    assert "pop_resample" in manifest["history"]
    assert any(item["type"] == "eeglab_set" for item in manifest["output_files"])
    assert any("pop_resample" in item for item in result["history"])


def test_pipeline_refuses_existing_outputs_without_overwrite(tmp_path):
    config_path = _write_pipeline_config(
        tmp_path,
        steps=[{"name": "qc"}],
    )
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    existing_qc = output_dir / "qc.json"
    existing_qc.write_text("existing", encoding="utf-8")

    result = run_pipeline_config(config_path)

    assert result["status"] == "error"
    assert result["code"] == "OUTPUT_EXISTS"
    assert result["error"]["path"] == str(existing_qc)
    assert existing_qc.read_text(encoding="utf-8") == "existing"


def test_pipeline_clean_after_epoch_uses_pop_clean_continuous_data_guard(tmp_path):
    input_path = tmp_path / "epoched.set"
    eeg = create_test_eeg(n_channels=2, n_samples=12, srate=100)
    eeg["data"] = np.stack([eeg["data"], eeg["data"]], axis=2)
    eeg["pnts"] = 12
    eeg["trials"] = 2
    pop_saveset(eeg, str(input_path))
    config_path = _write_pipeline_config(
        tmp_path,
        input_path=input_path,
        steps=[{"name": "clean", "method": "asr"}],
    )

    result = run_pipeline_config(config_path)

    assert result["status"] == "error"
    assert result["code"] == "PIPELINE_STEP_FAILED"
    assert "Input data must be continuous" in result["message"]


def test_pipeline_clean_uses_direct_cli_defaults(monkeypatch, tmp_path):
    calls = []

    def fake_pop_clean_rawdata(eeg, **kwargs):
        calls.append(kwargs)
        return eeg, "EEG = pop_clean_rawdata(EEG, 'BurstCriterion', 20);"

    monkeypatch.setattr(transforms_cli, "pop_clean_rawdata", fake_pop_clean_rawdata)
    config_path = _write_pipeline_config(
        tmp_path,
        steps=[{"name": "clean", "method": "asr"}],
    )

    result = run_pipeline_config(config_path)

    assert result["status"] == "ok"
    assert calls == [
        {
            "FlatlineCriterion": "off",
            "ChannelCriterion": "off",
            "LineNoiseCriterion": "off",
            "Highpass": "off",
            "BurstCriterion": 20.0,
            "BurstRejection": False,
            "WindowCriterion": "off",
            "Distance": "Euclidean",
            "gui": False,
            "return_com": True,
        }
    ]


def test_pipeline_clean_rejects_scalar_highpass(tmp_path):
    config_path = _write_pipeline_config(
        tmp_path,
        steps=[{"name": "clean", "method": "asr", "highpass": 0.5}],
    )

    result = validate_pipeline_config(config_path)

    assert result["status"] == "error"
    assert result["error"]["code"] == "CONFIG_SCHEMA_ERROR"
    assert result["error"]["details"]["errors"][0]["path"] == "steps[0].highpass"


def test_pipeline_filter_history_uses_modern_eegfiltnew(monkeypatch, tmp_path):
    calls = []

    def fake_pop_eegfiltnew(eeg, **kwargs):
        calls.append(kwargs)
        return eeg, "EEG = pop_eegfiltnew(EEG, 'locutoff', 1);"

    monkeypatch.setattr(transforms_cli, "pop_eegfiltnew", fake_pop_eegfiltnew)
    config_path = _write_pipeline_config(
        tmp_path,
        steps=[
            {
                "name": "filter",
                "highpass": 1.0,
                "lowpass": 40.0,
                "order": 128,
                "minphase": True,
                "usefftfilt": True,
            }
        ],
    )

    result = run_pipeline_config(config_path)

    assert result["status"] == "ok"
    assert calls == [
        {
            "locutoff": 1.0,
            "hicutoff": 40.0,
            "filtorder": 128,
            "plotfreqz": False,
            "minphase": True,
            "usefftfilt": True,
            "gui": False,
            "return_com": True,
        }
    ]
    assert "pop_eegfiltnew" in result["history"][0]


def test_pipeline_filter_rejects_negative_notch_lower_edge(tmp_path):
    config_path = _write_pipeline_config(
        tmp_path,
        steps=[{"name": "filter", "notch": 1, "notch_width": 4}],
    )

    result = run_pipeline_config(config_path)

    assert result["status"] == "error"
    assert result["code"] == "CONFIG_SCHEMA_ERROR"
    assert "notch minus half notch_width must be positive" in result["message"]
    assert not (tmp_path / "out").exists()


def test_pipeline_invalid_config_returns_structured_error(tmp_path):
    config_path = _write_pipeline_config(
        tmp_path,
        steps=[{"name": "does_not_exist"}],
    )

    result = validate_pipeline_config(config_path)

    assert result["status"] == "error"
    assert result["error"]["code"] == "CONFIG_SCHEMA_ERROR"
    assert result["error"]["details"]["errors"][0]["path"] == "steps[0].name"


def test_qc_metrics_include_agent_recommendation_codes_for_bad_events():
    eeg = {
        "data": np.zeros((2, 10)),
        "nbchan": 2,
        "pnts": 10,
        "trials": 1,
        "srate": 100,
        "xmin": 0,
        "xmax": 0.09,
        "chanlocs": [
            {"labels": "Fz", "X": 0.0, "Y": 1.0, "Z": 0.0},
            {"labels": "Cz", "X": 0.0, "Y": 0.0, "Z": 1.0},
        ],
        "event": [{"type": "target", "latency": 0}],
    }

    metrics = compute_qc_metrics(eeg)

    codes = {item["code"] for item in metrics["recommendations"]}
    assert "INVALID_EVENT_LATENCY" in codes
    assert "FLAT_CHANNELS_DETECTED" in codes
    assert metrics["events"]["invalid_latency_indices"] == [1]
    assert metrics["data_quality"]["flat_channel_indices"] == [1, 2]
    assert metrics["events"]["index_base"] == 1


def test_qc_metrics_use_one_based_channel_indices():
    eeg = {
        "data": np.array([[1, 2, 3], [0, 0, 0]], dtype=float),
        "nbchan": 3,
        "pnts": 3,
        "trials": 1,
        "srate": 100,
        "chanlocs": [{"labels": "Fz", "X": 0.0, "Y": 1.0, "Z": 0.0}],
        "event": [],
    }

    metrics = compute_qc_metrics(eeg)

    assert metrics["channels"]["missing_location_indices"] == [2, 3]
    assert metrics["data_quality"]["flat_channel_indices"] == [2]
    assert metrics["channels"]["index_base"] == 1


def test_pipeline_channel_indices_are_eeglab_facing_one_based():
    assert _channel_indices([1, "2", "Cz"]) == [0, 1, "Cz"]


def test_qc_dataset_missing_file_returns_structured_error(tmp_path):
    result = qc_dataset(tmp_path / "missing.set")

    assert result["status"] == "error"
    assert result["error"]["code"] == "INPUT_FILE_NOT_FOUND"
    assert result["error"]["path"].endswith("missing.set")


def test_report_and_qc_report_write_html_and_manifests(tmp_path):
    report_result = report_dataset(
        SAMPLE_SET,
        tmp_path / "dataset_report.html",
        manifest_path=tmp_path / "dataset_report.manifest.json",
    )
    qc_result = qc_report_dataset(
        SAMPLE_SET,
        tmp_path / "qc_report.html",
        manifest_path=tmp_path / "qc_report.manifest.json",
    )

    assert report_result["status"] == "ok"
    assert qc_result["status"] == "ok"
    assert "EEGPrep Report" in (tmp_path / "dataset_report.html").read_text(encoding="utf-8")
    assert "EEGPrep QC Report" in (tmp_path / "qc_report.html").read_text(encoding="utf-8")
    assert json.loads((tmp_path / "dataset_report.manifest.json").read_text(encoding="utf-8"))["command"] == "report"
    assert json.loads((tmp_path / "qc_report.manifest.json").read_text(encoding="utf-8"))["command"] == "qc report"


def _write_pipeline_config(tmp_path, *, steps, input_path=SAMPLE_SET):
    config_path = tmp_path / "pipeline.yaml"
    config = {
        "schema_version": "eegprep.pipeline.v1",
        "input": {"path": str(input_path), "format": "eeglab"},
        "output": {"directory": str(tmp_path / "out")},
        "steps": steps,
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path
