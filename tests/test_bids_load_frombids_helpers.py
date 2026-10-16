from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pyedflib
import pytest
import scipy.io

from tests.eeg_options_snapshot import eeg_options_snapshot, restore_eeg_options
from tests.eeglab_tests.backend import skip_without_matlab_plugin

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
    eeg, srate, times_sec, _report = load_raw_eeg_file(str(edf_path), dtype=np.float64)

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


def _signal_headers(
    labels: list[str],
    *,
    srate: int,
    digital_min: int,
    digital_max: int,
) -> list[dict[str, object]]:
    return [
        {
            "label": label,
            "dimension": "uV",
            "sample_frequency": srate,
            "physical_min": -100,
            "physical_max": 100,
            "digital_min": digital_min,
            "digital_max": digital_max,
            "transducer": "",
            "prefilter": "",
        }
        for label in labels
    ]


def _write_edf_family(
    path: Path,
    signals: list[np.ndarray],
    *,
    labels: list[str],
    srate: int,
    file_type: int,
    digital_min: int,
    digital_max: int,
    annotations: list[tuple[float, str]] | None = None,
) -> None:
    writer = pyedflib.EdfWriter(str(path), len(signals), file_type=file_type)
    writer.setSignalHeaders(_signal_headers(labels, srate=srate, digital_min=digital_min, digital_max=digital_max))
    writer.writeSamples(signals)
    for onset, text in annotations or []:
        writer.writeAnnotation(onset, 0.5, text)
    writer.close()


def _reject_patient_byte(path: Path) -> None:
    payload = bytearray(path.read_bytes())
    payload[8] = 0x02
    path.write_bytes(payload)


def test_pop_load_frombids_reads_noncompliant_edf_with_pop_biosig(tmp_path: Path) -> None:
    from eegprep.functions.popfunc.pop_biosig import pop_biosig
    from eegprep.functions.popfunc.pop_load_frombids import pop_load_frombids

    eeg_dir = tmp_path / "sub-01" / "eeg"
    eeg_dir.mkdir(parents=True)
    (tmp_path / "dataset_description.json").write_text(json.dumps({"Name": "edf", "BIDSVersion": "1.8.0"}))
    edf_path = eeg_dir / "sub-01_task-x_eeg.edf"
    samples = np.linspace(-20, 20, EDF_SRATE * 2)
    _write_edf_family(
        edf_path,
        [samples, -samples],
        labels=["C0", "C1"],
        srate=EDF_SRATE,
        file_type=pyedflib.FILETYPE_EDFPLUS,
        digital_min=-32768,
        digital_max=32767,
        annotations=[(1.0, "mark")],
    )
    _reject_patient_byte(edf_path)
    with pytest.raises(OSError):
        pyedflib.EdfReader(str(edf_path))
    (eeg_dir / "sub-01_task-x_eeg.json").write_text(
        json.dumps({"TaskName": "x", "SamplingFrequency": EDF_SRATE, "EEGReference": "Cz"})
    )

    imported, report = pop_load_frombids(
        str(edf_path),
        bidsevent=False,
        infer_locations=False,
        dtype=np.float64,
        return_report=True,
        verbose=False,
    )
    direct = pop_biosig(str(edf_path))

    assert report["ImporterUsed"] == "pop_biosig"
    assert imported["nbchan"] == 2
    assert imported["pnts"] == EDF_SRATE * 2
    assert imported["srate"] == EDF_SRATE
    assert [channel["labels"] for channel in imported["chanlocs"]] == ["C0", "C1"]
    np.testing.assert_array_equal(imported["data"], np.asarray(direct["data"], dtype=np.float64))
    assert [event["type"] for event in imported["event"]] == [event["type"] for event in direct["event"]]
    assert [int(event["latency"]) for event in imported["event"]] == [
        int(event["latency"]) for event in direct["event"]
    ]
    assert imported["ref"] == "common"
    assert imported["etc"]["BIDS"]["EEGReference"] == "Cz"


def test_pop_load_frombids_reads_bdf_with_pop_biosig(tmp_path: Path) -> None:
    from eegprep.functions.popfunc.pop_biosig import pop_biosig
    from eegprep.functions.popfunc.pop_load_frombids import pop_load_frombids

    eeg_dir = tmp_path / "sub-01" / "eeg"
    eeg_dir.mkdir(parents=True)
    (tmp_path / "dataset_description.json").write_text(json.dumps({"Name": "bdf", "BIDSVersion": "1.8.0"}))
    bdf_path = eeg_dir / "sub-01_task-x_eeg.bdf"
    samples = np.linspace(-5, 5, EDF_SRATE)
    _write_edf_family(
        bdf_path,
        [samples],
        labels=["Cz"],
        srate=EDF_SRATE,
        file_type=pyedflib.FILETYPE_BDFPLUS,
        digital_min=-8_388_608,
        digital_max=8_388_607,
    )
    (eeg_dir / "sub-01_task-x_eeg.json").write_text(
        json.dumps({"TaskName": "x", "SamplingFrequency": EDF_SRATE, "EEGReference": "average"})
    )

    imported, report = pop_load_frombids(
        str(bdf_path),
        bidsevent=False,
        infer_locations=False,
        dtype=np.float64,
        return_report=True,
        verbose=False,
    )
    direct = pop_biosig(str(bdf_path))

    assert report["ImporterUsed"] == "pop_biosig"
    assert imported["data"].shape == (1, EDF_SRATE)
    np.testing.assert_array_equal(imported["data"], np.asarray(direct["data"], dtype=np.float64))
    assert [channel["labels"] for channel in imported["chanlocs"]] == ["Cz"]


def test_pop_load_frombids_decodes_latin1_and_bom_sidecars(tmp_path: Path) -> None:
    from eegprep.functions.popfunc.pop_load_frombids import pop_load_frombids
    from eegprep.plugins.EEG_BIDS.sidecar import read_bids_table

    eeg_dir = tmp_path / "sub-01" / "eeg"
    eeg_dir.mkdir(parents=True)
    (tmp_path / "dataset_description.json").write_text(json.dumps({"Name": "encoding", "BIDSVersion": "1.8.0"}))
    edf_path = eeg_dir / "sub-01_task-x_eeg.edf"
    _write_edf_family(
        edf_path,
        [np.zeros(EDF_SRATE)],
        labels=["Cz"],
        srate=EDF_SRATE,
        file_type=pyedflib.FILETYPE_EDF,
        digital_min=-32768,
        digital_max=32767,
    )
    (eeg_dir / "sub-01_task-x_eeg.json").write_bytes(
        b"\xef\xbb\xbf"
        + json.dumps(
            {"TaskName": "x", "SamplingFrequency": EDF_SRATE, "EEGReference": "Cz", "PowerLineFrequency": 50}
        ).encode()
    )
    (eeg_dir / "sub-01_task-x_channels.tsv").write_bytes(b"name\ttype\tunits\nCz\tEOG\t\xb5V\n")
    (eeg_dir / "sub-01_task-x_electrodes.tsv").write_bytes(b"name\tx\ty\tz\tunits\nCz\t1\t2\t3\t\xb5V\n")
    (eeg_dir / "sub-01_task-x_coordsystem.json").write_bytes(
        b"\xef\xbb\xbf" + json.dumps({"EEGCoordinateSystem": "EEGLAB", "EEGCoordinateUnits": "mm"}).encode()
    )
    (eeg_dir / "sub-01_task-x_events.tsv").write_bytes(
        b"\xef\xbb\xbfonset\tduration\ttrial_type\n0\t0\t" + "µ".encode() + b"\n"
    )

    channels = read_bids_table(eeg_dir / "sub-01_task-x_channels.tsv")
    electrodes = read_bids_table(eeg_dir / "sub-01_task-x_electrodes.tsv")
    assert channels["units"].iloc[0] == "µV"
    assert electrodes["units"].iloc[0] == "µV"

    imported, report = pop_load_frombids(
        str(edf_path),
        infer_locations=False,
        dtype=np.float64,
        return_report=True,
        verbose=False,
    )

    assert imported["chanlocs"][0]["type"] == "EOG"
    assert imported["event"][0]["type"] == "µ"
    assert imported["etc"]["BIDS"]["TaskName"] == "x"
    assert imported["etc"]["BIDS"]["EEGReference"] == "Cz"
    assert imported["chanlocs"][0]["X"] == 1
    assert imported["chanlocs"][0]["Y"] == 2
    assert imported["chanlocs"][0]["Z"] == 3
    assert report["ElectrodesFrom"].endswith("sub-01_task-x_electrodes.tsv")


def test_bids_tsv_decoding_differs_from_eeglab(tmp_path: Path) -> None:
    """Latin-1 TSV bytes stay code points, and a UTF-8 BOM is removed.

    MATLAB R2025b ``loadtxt`` with ``DefaultCharacterSet`` UTF-8 maps ``0xB5``
    and ``0x80`` to U+FFFD and leaves a BOM as U+FEFF on the first header. A
    Windows-1252 session maps ``0x80`` to €.
    """
    from eegprep.plugins.EEG_BIDS.sidecar import decode_bids_text, read_bids_table

    micro = decode_bids_text(b"name\tunits\nCz\t\xb5V\n", suffix=".tsv")
    assert micro.splitlines()[1] == "Cz\tµV"
    assert "\ufffd" not in micro

    control = tmp_path / "channels.tsv"
    control.write_bytes(b"name\ttype\nCz\t\x80\n")
    decoded = decode_bids_text(control.read_bytes(), suffix=".tsv")
    assert decoded.splitlines()[1].split("\t")[1] == "\u0080"
    assert "\ufffd" not in decoded
    assert "\u20ac" not in decoded
    assert read_bids_table(control)["type"].iloc[0] == "\u0080"

    bom = tmp_path / "events.tsv"
    bom.write_bytes(b"\xef\xbb\xbfonset\tduration\ttrial_type\n0\t0\tmark\n")
    table = read_bids_table(bom)
    assert list(table.columns) == ["onset", "duration", "trial_type"]
    assert not str(table.columns[0]).startswith("\ufeff")


def test_bids_json_decoding_differs_from_eeglab(tmp_path: Path) -> None:
    """Latin-1 JSON is rejected, and a UTF-8 BOM is accepted.

    MATLAB ``bids_loadfile`` reads the Latin-1 bytes for café as ``caf`` plus
    U+FFFD, and errors when the file starts with a UTF-8 BOM.
    """
    from eegprep.plugins.EEG_BIDS.sidecar import decode_bids_text, read_bids_json

    with pytest.raises(UnicodeDecodeError):
        decode_bids_text(b'{"TaskName": "caf\xe9"}', suffix=".json")

    path = tmp_path / "eeg.json"
    path.write_bytes(b'\xef\xbb\xbf{"TaskName": "x"}')
    assert read_bids_json(path)["TaskName"] == "x"
    assert not decode_bids_text(path.read_bytes(), suffix=".json").startswith("\ufeff")


def test_bids_json_decoding_leaves_stdlib_json_load() -> None:
    import json as json_module

    import bids.layout.index as index_mod
    import bids.layout.validation as validation_mod

    from eegprep.plugins.EEG_BIDS.sidecar import bids_json_decoding

    original = json_module.load
    index_json = index_mod.json
    validation_json = validation_mod.json
    with bids_json_decoding():
        assert json_module.load is original
        assert index_mod.json is not index_json
        assert validation_mod.json is not validation_json
        assert index_mod.json.load is not original
    assert json_module.load is original
    assert index_mod.json is index_json
    assert validation_mod.json is validation_json


def test_latin1_bids_json_is_rejected(tmp_path: Path) -> None:
    import json as json_module

    from eegprep.functions.popfunc.pop_load_frombids import pop_load_frombids

    eeg_dir = tmp_path / "sub-01" / "eeg"
    eeg_dir.mkdir(parents=True)
    (tmp_path / "dataset_description.json").write_text(json.dumps({"Name": "json", "BIDSVersion": "1.8.0"}))
    edf_path = eeg_dir / "sub-01_task-x_eeg.edf"
    _write_edf_family(
        edf_path,
        [np.zeros(EDF_SRATE)],
        labels=["Cz"],
        srate=EDF_SRATE,
        file_type=pyedflib.FILETYPE_EDF,
        digital_min=-32768,
        digital_max=32767,
    )
    (eeg_dir / "sub-01_task-x_eeg.json").write_bytes(b'{"TaskName": "caf\xe9"}')
    original_load = json_module.load
    with pytest.raises((UnicodeDecodeError, OSError)):
        pop_load_frombids(str(edf_path), bidsevent=False, infer_locations=False, verbose=False)
    assert json_module.load is original_load


_DS003194_FILES = (
    "dataset_description.json",
    "participants.tsv",
    "sub-NEPO01/ses-Nepopre/eeg/sub-NEPO01_ses-Nepopre_task-taskprenepo_eeg.edf",
    "sub-NEPO01/ses-Nepopre/eeg/sub-NEPO01_ses-Nepopre_task-taskprenepo_eeg.json",
    "sub-NEPO01/ses-Nepopre/eeg/sub-NEPO01_ses-Nepopre_task-taskprenepo_channels.tsv",
    "sub-NEPO01/ses-Nepopre/eeg/sub-NEPO01_ses-Nepopre_task-taskprenepo_events.tsv",
)


def _ds003194_root() -> Path:
    override = os.environ.get("EEGPREP_OPENNEURO_ROOT")
    if override:
        root = Path(override)
        if (root / "dataset_description.json").is_file():
            return root
        return root / "ds003194"
    return Path(__file__).resolve().parents[1] / "openneuro" / "ds003194"


def _saved_strings(value: object) -> list[str]:
    array = np.asarray(value)
    if array.dtype.kind in "SU":
        return [str(item) for item in array.ravel()]
    strings = []
    for item in np.asarray(value, dtype=object).ravel():
        if isinstance(item, np.ndarray):
            if item.dtype.kind in "SU":
                strings.append("".join(item.astype(str).ravel()))
            elif item.size == 0:
                strings.append("")
            else:
                strings.append(str(item.reshape(-1)[0]))
        elif isinstance(item, bytes):
            strings.append(item.decode())
        else:
            strings.append(str(item))
    return strings


def _event_token(value: object) -> str:
    if isinstance(value, bytes):
        value = value.decode()
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return ""
        value = value.reshape(-1)[0]
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if np.isnan(number):
            return "nan"
        if number.is_integer():
            return str(int(number))
        return str(number)
    return str(value).strip()


def test_openneuro_edf_import_matches_matlab_pop_importbids(eeglab_matlab_engine, tmp_path: Path) -> None:
    """Compare one OpenNeuro EDF import with MATLAB pop_importbids.

    ``pop_importbids`` writes the dataset with ``pop_saveset``, which stores
    data as single. Those samples match ``float32`` of the Python import.
    """
    from eegprep.functions.popfunc.pop_load_frombids import pop_load_frombids

    skip_without_matlab_plugin(eeglab_matlab_engine, "pop_importbids", "EEG-BIDS")
    skip_without_matlab_plugin(eeglab_matlab_engine, "sopen", "Biosig")
    root = _ds003194_root()
    missing = [name for name in _DS003194_FILES if not (root / name).is_file()]
    if missing:
        listed = "\n".join(f"https://s3.amazonaws.com/openneuro.org/ds003194/{name}" for name in _DS003194_FILES)
        pytest.fail(
            "ds003194 files are not available under "
            f"{root} (missing {missing}). Fetch them from s3://openneuro.org/ds003194/ "
            "or set EEGPREP_OPENNEURO_ROOT to the dataset root:\n" + listed
        )

    edf_path = root / _DS003194_FILES[2]
    python_eeg = pop_load_frombids(
        str(edf_path),
        eventtype="value",
        bidsevent="replace",
        bidschanloc=True,
        infer_locations=False,
        dtype=np.float64,
        verbose=False,
    )
    assert int(python_eeg["nbchan"]) == 20
    assert int(python_eeg["pnts"]) == 162400
    assert float(python_eeg["srate"]) == 200
    assert len(python_eeg["event"]) == 572

    output_dir = tmp_path / "derivatives"
    output_dir.mkdir()
    mat_path = tmp_path / "matlab_import.mat"
    bids = str(root).replace("\\", "/").replace("'", "''")
    out = str(output_dir).replace("\\", "/").replace("'", "''")
    mat = str(mat_path).replace("\\", "/").replace("'", "''")
    options_path, original = eeg_options_snapshot(eeglab_matlab_engine)
    eeglab_matlab_engine.eval("pop_editoptions('option_single', 0);", nargout=0)
    try:
        eeglab_matlab_engine.eval(
            "clear ALLEEG STUDY EEG; "
            f"[~, ALLEEG] = pop_importbids('{bids}', "
            f"'outputdir', '{out}', "
            "'subjects', {'sub-NEPO01'}, "
            "'sessions', {'ses-Nepopre'}, "
            "'eventtype', 'value', "
            "'bidsevent', 'on', "
            "'bidschanloc', 'on'); "
            "if numel(ALLEEG) ~= 1, error('expected one dataset, found %d', numel(ALLEEG)); end; "
            "EEG = ALLEEG(1); "
            "nChan = double(EEG.nbchan); "
            "chan_labels = cell(1, nChan); "
            "for iChan = 1:nChan, chan_labels{iChan} = char(EEG.chanlocs(iChan).labels); end; "
            "n = numel(EEG.event); "
            "event_type = cell(1, n); "
            "event_latency = nan(1, n); "
            "event_duration = nan(1, n); "
            "for iEvent = 1:n, "
            "event_type{iEvent} = char(string(EEG.event(iEvent).type)); "
            "event_latency(iEvent) = double(EEG.event(iEvent).latency); "
            "event_duration(iEvent) = double(EEG.event(iEvent).duration); "
            "end; "
            "nbchan = double(EEG.nbchan); "
            "pnts = double(EEG.pnts); "
            "srate = double(EEG.srate); "
            "data = double(EEG.data); "
            f"save('{mat}', 'chan_labels', 'event_type', 'event_latency', 'event_duration', "
            "'nbchan', 'pnts', 'srate', 'data', '-v7'); "
            "clear ALLEEG STUDY EEG data chan_labels event_type event_latency event_duration;",
            nargout=0,
        )
    finally:
        restore_eeg_options(options_path, original)
        eeglab_matlab_engine.eval("clear functions", nargout=0)

    saved = scipy.io.loadmat(mat_path, squeeze_me=True, chars_as_strings=True)
    assert int(python_eeg["nbchan"]) == int(saved["nbchan"])
    assert int(python_eeg["pnts"]) == int(saved["pnts"])
    assert float(python_eeg["srate"]) == float(saved["srate"])
    python_data = np.asarray(python_eeg["data"])
    matlab_data = np.asarray(saved["data"])
    assert matlab_data.shape == python_data.shape
    assert np.array_equal(matlab_data, python_data.astype(np.float32))

    python_labels = [str(channel["labels"]).strip() for channel in python_eeg["chanlocs"]]
    assert python_labels == [label.strip() for label in _saved_strings(saved["chan_labels"])]

    matlab_types = _saved_strings(saved["event_type"])
    matlab_latency = np.asarray(saved["event_latency"], dtype=float).ravel()
    matlab_duration = np.asarray(saved["event_duration"], dtype=float).ravel()
    assert len(python_eeg["event"]) == len(matlab_types) == matlab_latency.size == matlab_duration.size
    python_latency = np.array([float(event["latency"]) for event in python_eeg["event"]])
    python_duration = np.array([float(event["duration"]) for event in python_eeg["event"]])
    assert [_event_token(event["type"]) for event in python_eeg["event"]] == [
        _event_token(token) for token in matlab_types
    ]
    np.testing.assert_allclose(python_latency, matlab_latency, rtol=0, atol=0)
    np.testing.assert_allclose(python_duration, matlab_duration, rtol=0, atol=0, equal_nan=True)
