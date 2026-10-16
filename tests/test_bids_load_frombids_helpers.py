from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyedflib
import pytest

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


def test_pop_load_frombids_duration_is_duration_times_srate(tmp_path: Path) -> None:
    from eegprep.functions.popfunc.pop_load_frombids import pop_load_frombids

    eeg_dir = tmp_path / "sub-01" / "eeg"
    eeg_dir.mkdir(parents=True)
    (tmp_path / "dataset_description.json").write_text(json.dumps({"Name": "duration", "BIDSVersion": "1.8.0"}))
    edf_path = eeg_dir / "sub-01_task-x_eeg.edf"
    samples = np.zeros(EDF_SRATE * EDF_SECONDS)
    writer = pyedflib.EdfWriter(str(edf_path), 1, file_type=pyedflib.FILETYPE_EDF)
    writer.setSignalHeaders(
        [
            {
                "label": "Cz",
                "dimension": "uV",
                "sample_frequency": EDF_SRATE,
                "physical_min": -100,
                "physical_max": 100,
                "digital_min": -32768,
                "digital_max": 32767,
                "transducer": "",
                "prefilter": "",
            }
        ]
    )
    writer.writeSamples([samples])
    writer.close()
    (eeg_dir / "sub-01_task-x_eeg.json").write_text(
        json.dumps({"TaskName": "x", "SamplingFrequency": EDF_SRATE, "EEGReference": "Cz", "PowerLineFrequency": 50})
    )
    (eeg_dir / "sub-01_task-x_channels.tsv").write_text("name\ttype\tunits\nCz\tEEG\tuV\n")
    # 0 stays 0, 1.5 samples stays 1.5, n/a stays missing. 0.5 s is an exact 50 samples.
    (eeg_dir / "sub-01_task-x_events.tsv").write_text(
        "onset\tduration\ttrial_type\n0.0\t0\tzero\n0.015\t0.015\tfrac\n1.0\tn/a\tmissing\n2.0\t0.5\thalf\n"
    )

    imported = pop_load_frombids(str(edf_path), verbose=False)
    by_type = {str(event["type"]): event["duration"] for event in imported["event"]}
    expected = EDF_SRATE * np.array([0.0, 0.015, np.nan, 0.5])

    assert list(by_type) == ["zero", "frac", "missing", "half"]
    np.testing.assert_allclose(
        [by_type["zero"], by_type["frac"], by_type["missing"], by_type["half"]],
        expected,
        equal_nan=True,
    )


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


def _write_brainvision_bids(root: Path, *, orientation: str) -> Path:
    """Write a two-channel BrainVision BIDS recording with a dataset reference."""
    eeg_dir = root / "sub-01" / "eeg"
    eeg_dir.mkdir(parents=True)
    (root / "dataset_description.json").write_text(json.dumps({"Name": "brainvision", "BIDSVersion": "1.8.0"}))
    stem = "sub-01_task-rest_eeg"
    raw = np.array([[1, 2, 3, 4], [10, 20, 30, 40]], dtype=np.int16)
    on_disk = raw.T if orientation == "MULTIPLEXED" else raw
    data_name = f"{stem}.dat"
    on_disk.tofile(eeg_dir / data_name)
    marker_name = f"{stem}.vmrk"
    (eeg_dir / f"{stem}.vhdr").write_text(
        "Brain Vision Data Exchange Header File Version 1.0\n"
        "\n[Common Infos]\n"
        f"DataFile={data_name}\n"
        f"MarkerFile={marker_name}\n"
        "DataFormat=BINARY\n"
        "DataType=TIMEDOMAIN\n"
        f"DataOrientation={orientation}\n"
        "NumberOfChannels=2\n"
        "DataPoints=4\n"
        "SamplingInterval=4000\n"
        "\n[Binary Infos]\n"
        "BinaryFormat=INT_16\n"
        "UseBigEndianOrder=NO\n"
        "\n[Channel Infos]\n"
        "Ch1=Cz,A1,1,µV\n"
        "Ch2=Pz,A2,0.5,µV\n",
        encoding="utf-8",
    )
    (eeg_dir / marker_name).write_text(
        "Brain Vision Data Exchange Marker File, Version 1.0\n"
        "\n[Common Infos]\n"
        f"DataFile={data_name}\n"
        "\n[Marker Infos]\n"
        "Mk1=Stimulus,S  1,2,1,1\n",
        encoding="utf-8",
    )
    (eeg_dir / f"{stem}.json").write_text(
        json.dumps({"TaskName": "rest", "SamplingFrequency": 250, "EEGReference": "linked mastoids"})
    )
    return eeg_dir / f"{stem}.vhdr"


@pytest.mark.parametrize("orientation", ["VECTORIZED", "MULTIPLEXED"])
def test_pop_load_frombids_reads_brainvision_with_pop_loadbv(tmp_path: Path, orientation: str) -> None:
    from eegprep.functions.popfunc.pop_load_frombids import pop_load_frombids

    header = _write_brainvision_bids(tmp_path, orientation=orientation)

    eeg, report = pop_load_frombids(
        str(header),
        bidsevent=False,
        infer_locations=False,
        dtype=np.float64,
        return_report=True,
        verbose=False,
    )

    expected = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 10.0, 15.0, 20.0]])
    np.testing.assert_allclose(eeg["data"], expected)
    assert eeg["data"].shape == (2, 4)
    assert eeg["srate"] == 250
    assert eeg["nbchan"] == 2
    assert eeg["pnts"] == 4
    assert eeg["trials"] == 1
    assert [channel["labels"] for channel in eeg["chanlocs"]] == ["Cz", "Pz"]
    assert [channel["ref"] for channel in eeg["chanlocs"]] == ["A1", "A2"]
    assert [(event["type"], int(event["latency"])) for event in eeg["event"]] == [("S  1", 2)]
    assert eeg["ref"] == "common"
    assert eeg["etc"]["BIDS"]["EEGReference"] == "linked mastoids"
    assert report["ImporterUsed"] == "pop_loadbv"


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
