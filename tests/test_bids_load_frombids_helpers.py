from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyedflib

EDF_SRATE = 100
EDF_SECONDS = 10
# annotation onsets in seconds -> expected 1-based latencies 1, 101, 1000 (last sample)
EDF_ANNOTATIONS = [(0.0, "A"), (1.0, "B"), (9.99, "C")]


def _write_edf_bids_dataset(root: Path) -> Path:
    """Write a minimal EDF+ BIDS dataset with annotations and an events.tsv sidecar."""
    eeg_dir = root / "sub-01" / "eeg"
    eeg_dir.mkdir(parents=True)
    (root / "dataset_description.json").write_text(json.dumps({"Name": "edf", "BIDSVersion": "1.8.0"}))
    edf_path = eeg_dir / "sub-01_task-x_eeg.edf"
    t = np.arange(EDF_SRATE * EDF_SECONDS) / EDF_SRATE
    signals = [50 * np.sin(2 * np.pi * 10 * t), 50 * np.cos(2 * np.pi * 10 * t)]
    writer = pyedflib.EdfWriter(str(edf_path), len(signals), file_type=pyedflib.FILETYPE_EDFPLUS)
    writer.setSignalHeaders(
        [
            {
                "label": f"C{index}",
                "dimension": "uV",
                "sample_frequency": EDF_SRATE,
                "physical_min": -100,
                "physical_max": 100,
                "digital_min": -32768,
                "digital_max": 32767,
                "transducer": "",
                "prefilter": "",
            }
            for index in range(len(signals))
        ]
    )
    writer.writeSamples(signals)
    for onset, text in EDF_ANNOTATIONS:
        writer.writeAnnotation(onset, 0.5, text)
    writer.close()
    (eeg_dir / "sub-01_task-x_eeg.json").write_text(
        json.dumps({"TaskName": "x", "SamplingFrequency": EDF_SRATE, "EEGReference": "Cz", "PowerLineFrequency": 50})
    )
    (eeg_dir / "sub-01_task-x_channels.tsv").write_text("name\ttype\tunits\nC0\tEEG\tuV\nC1\tEEG\tuV\n")
    # A and B duplicate EDF annotations; D exists only in events.tsv
    (eeg_dir / "sub-01_task-x_events.tsv").write_text(
        "onset\tduration\ttrial_type\n0.0\t0.5\tA\n1.0\t0.5\tB\n5.0\t0.5\tD\n"
    )
    return edf_path


def test_raw_edf_loader_returns_one_based_event_latencies(tmp_path: Path) -> None:
    from eegprep.plugins.EEG_BIDS.raw import load_raw_eeg_file

    edf_path = _write_edf_bids_dataset(tmp_path)
    eeg, srate, times_sec, _report = load_raw_eeg_file(
        str(edf_path),
        dtype=np.float64,
        numeric_null=np.array([]),
        warning=lambda msg: None,
        verbose=False,
    )

    assert srate == EDF_SRATE
    assert len(times_sec) == EDF_SRATE * EDF_SECONDS
    assert [event["type"] for event in eeg["event"]] == ["A", "B", "C"]
    assert [int(event["latency"]) for event in eeg["event"]] == [1, 101, 1000]


def test_pop_load_frombids_merge_deduplicates_raw_and_tsv_events(tmp_path: Path) -> None:
    from eegprep.functions.popfunc.pop_load_frombids import pop_load_frombids

    edf_path = _write_edf_bids_dataset(tmp_path)

    merged = pop_load_frombids(str(edf_path), bidsevent="merge", verbose=False)
    assert [(event["type"], int(event["latency"])) for event in merged["event"]] == [
        ("A", 1),
        ("B", 101),
        ("D", 501),
        ("C", 1000),
    ]

    appended = pop_load_frombids(str(edf_path), bidsevent="append", verbose=False)
    assert [(event["type"], int(event["latency"])) for event in appended["event"]] == [
        ("A", 1),
        ("A", 1),
        ("B", 101),
        ("B", 101),
        ("D", 501),
        ("C", 1000),
    ]


def test_raw_set_loader_returns_eeg_and_timing_metadata() -> None:
    from eegprep.plugins.EEG_BIDS.raw import load_raw_eeg_file

    dataset = Path(__file__).resolve().parents[1] / "sample_data" / "eeglab_data.set"
    warnings: list[str] = []

    eeg, srate, times_sec, report = load_raw_eeg_file(
        str(dataset),
        dtype=np.float64,
        numeric_null=np.array([]),
        warning=warnings.append,
        verbose=False,
    )

    assert report["ImporterUsed"] == "pop_loadset"
    assert warnings == []
    assert srate == eeg["srate"]
    assert eeg["data"].dtype == np.float64
    np.testing.assert_allclose(times_sec, np.asarray(eeg["times"], dtype=float) / 1000.0)


def test_montage_inference_uses_packaged_montage_resources() -> None:
    from eegprep.plugins.EEG_BIDS.montage import apply_montage_inference

    numeric_null = np.array([])
    eeg = {
        "chanlocs": [
            {
                "labels": "'Fp1'",
                "X": numeric_null,
                "Y": numeric_null,
                "Z": numeric_null,
                "sph_radius": numeric_null,
                "sph_theta": numeric_null,
                "sph_phi": numeric_null,
                "theta": numeric_null,
                "radius": numeric_null,
            },
            {
                "labels": "Fpz",
                "X": numeric_null,
                "Y": numeric_null,
                "Z": numeric_null,
                "sph_radius": numeric_null,
                "sph_theta": numeric_null,
                "sph_phi": numeric_null,
                "theta": numeric_null,
                "radius": numeric_null,
            },
            {
                "labels": "Fp2",
                "X": numeric_null,
                "Y": numeric_null,
                "Z": numeric_null,
                "sph_radius": numeric_null,
                "sph_theta": numeric_null,
                "sph_phi": numeric_null,
                "theta": numeric_null,
                "radius": numeric_null,
            },
        ],
        "chaninfo": {"nosedir": "+Y"},
        "etc": {},
    }
    report: dict[str, object] = {}
    warnings: list[str] = []
    errors: list[str] = []

    apply_montage_inference(
        eeg,
        "standard-10-5-342ch.locs",
        numeric_null=numeric_null,
        report=report,
        warning=warnings.append,
        error=errors.append,
    )

    assert warnings == []
    assert errors == []
    assert eeg["chaninfo"]["nosedir"] == "+X"
    assert eeg["etc"]["labelscheme"] == "10-20"
    assert report["ChanlocsFrom"] == "standard-10-5-342ch.locs"
    for chanloc in eeg["chanlocs"]:
        assert np.isfinite([chanloc["X"], chanloc["Y"], chanloc["Z"]]).all()
