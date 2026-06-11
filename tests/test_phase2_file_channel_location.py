from __future__ import annotations

import ast
from importlib.resources import files
from pathlib import Path

import numpy as np
import pytest

from eegprep.functions.adminfunc.console import EEGPrepConsoleWorkspace, _console_python_command
from eegprep.functions.guifunc.session import EEGPrepSession
from eegprep import (
    chancenter,
    convertlocs,
    floatread,
    floatwrite,
    pop_chancenter,
    pop_chancoresp,
    pop_loadbci,
    pop_readlocs,
    pop_snapread,
    pop_writelocs,
    readlocs,
    readegilocs,
    snapread,
    writelocs,
)


def _eeg() -> dict:
    return {
        "setname": "demo",
        "filename": "",
        "filepath": "",
        "data": np.zeros((4, 20), dtype=float),
        "nbchan": 4,
        "pnts": 20,
        "trials": 1,
        "srate": 100.0,
        "xmin": 0.0,
        "xmax": 0.19,
        "times": np.arange(20, dtype=float),
        "chanlocs": [
            {"labels": "Nz", "X": 0.0, "Y": 1.0, "Z": 0.0},
            {"labels": "LPA", "X": -1.0, "Y": 0.0, "Z": 0.0},
            {"labels": "RPA", "X": 1.0, "Y": 0.0, "Z": 0.0},
            {"labels": "Cz", "X": 0.0, "Y": 0.0, "Z": 1.0},
        ],
        "urchanlocs": [],
        "chaninfo": {},
        "event": [],
        "urevent": [],
        "epoch": [],
        "icaweights": np.array([]),
        "icasphere": np.array([]),
        "icawinv": np.array([]),
        "icaact": np.array([]),
        "icachansind": np.array([], dtype=int),
        "history": "",
    }


def _assert_parseable(command: str) -> None:
    ast.parse(_console_python_command(command))


def test_readlocs_reads_packaged_mat_backed_montage() -> None:
    montage = files("eegprep").joinpath("resources", "montages", "standard-10-5-342ch.locs")

    locs = readlocs(montage)

    assert len(locs) == 342
    assert [loc["labels"] for loc in locs[:3]] == ["LPA", "RPA", "Nz"]
    assert locs[0]["type"] == "FID"
    assert {"X", "Y", "Z", "theta", "radius", "sph_theta", "sph_phi"} <= set(locs[3])


def test_readlocs_and_writelocs_round_trip_locs_and_ced(tmp_path: Path) -> None:
    locs = [
        {"labels": "Fz", "theta": 0.0, "radius": 0.25},
        {"labels": "Cz", "theta": 0.0, "radius": 0.0},
    ]
    loc_file = tmp_path / "demo.locs"
    ced_file = tmp_path / "demo.ced"

    writelocs(locs, loc_file)
    loaded, read_command = pop_readlocs(loc_file, return_com=True)
    write_command = pop_writelocs(loaded, ced_file, return_com=True)
    reloaded = readlocs(ced_file)

    assert [loc["labels"] for loc in loaded] == ["Fz", "Cz"]
    assert loaded[0]["X"] == pytest.approx(np.sqrt(0.5))
    assert [loc["labels"] for loc in reloaded] == ["Fz", "Cz"]
    assert "return_com" not in write_command
    _assert_parseable(read_command)
    _assert_parseable(write_command)


def test_readlocs_custom_format_reorders_and_applies_one_based_readchans(tmp_path: Path) -> None:
    loc_file = tmp_path / "custom.sfp"
    loc_file.write_text("2 Fp2 0.5 1 0\n1 Fp1 -0.5 1 0\n3 Cz 0 0 1\n", encoding="utf-8")

    locs = readlocs(loc_file, "filetype", "custom", "format", ["channum", "labels", "X", "Y", "Z"])
    selected = readlocs(
        loc_file,
        "filetype",
        "custom",
        "format",
        ["channum", "labels", "X", "Y", "Z"],
        "readchans",
        [2],
    )
    selected_array = readlocs(
        loc_file,
        "filetype",
        "custom",
        "format",
        ["channum", "labels", "X", "Y", "Z"],
        "readchans",
        np.asarray([2]),
    )

    assert [loc["labels"] for loc in locs] == ["Fp1", "Fp2", "Cz"]
    assert [loc["labels"] for loc in selected] == ["Fp2"]
    assert [loc["labels"] for loc in selected_array] == ["Fp2"]


def test_writelocs_accepts_numpy_elecind_selection(tmp_path: Path) -> None:
    loc_file = tmp_path / "selected.locs"
    locs = [{"labels": "Fz", "theta": 0.0, "radius": 0.5}, {"labels": "Cz", "theta": 0.0, "radius": 0.0}]

    writelocs(locs, loc_file, "elecind", np.asarray([2]))
    loaded = readlocs(loc_file)

    assert [loc["labels"] for loc in loaded] == ["Cz"]


def test_readlocs_rejects_malformed_non_chanedit_rows(tmp_path: Path) -> None:
    loc_file = tmp_path / "bad.loc"
    loc_file.write_text("1 0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="fewer columns"):
        readlocs(loc_file)


def test_convertlocs_and_chancenter_match_expected_geometry() -> None:
    locs = convertlocs([{"labels": "Cz", "X": 0.0, "Y": 0.0, "Z": 1.0}], "cart2all")
    x, y, z, center, optimized = chancenter([2.0], [0.0], [0.0], [1.0, 0.0, 0.0])

    assert locs[0]["radius"] == pytest.approx(0.0)
    assert locs[0]["sph_phi"] == pytest.approx(90.0)
    assert (x, y, z) == (pytest.approx([1.0]), pytest.approx([0.0]), pytest.approx([0.0]))
    assert center.tolist() == [1.0, 0.0, 0.0]
    assert optimized is False


def test_convertlocs_besa_spherical_matches_eeglab_angle_convention() -> None:
    lateral = convertlocs([{"labels": "Right", "sph_theta": 90.0, "sph_phi": 0.0}], "sph2sphbesa")[0]
    anterior = convertlocs([{"labels": "Front", "sph_theta": 0.0, "sph_phi": 0.0}], "sph2sphbesa")[0]
    oblique = convertlocs([{"labels": "Oblique", "sph_theta": -45.0, "sph_phi": 30.0}], "sph2sphbesa")[0]

    assert lateral["sph_theta_besa"] == pytest.approx(-90.0)
    assert lateral["sph_phi_besa"] == pytest.approx(0.0)
    assert anterior["sph_theta_besa"] == pytest.approx(90.0)
    assert anterior["sph_phi_besa"] == pytest.approx(90.0)
    assert oblique["sph_theta_besa"] == pytest.approx(60.0)
    assert oblique["sph_phi_besa"] == pytest.approx(45.0)

    round_trip = convertlocs([oblique], "sphbesa2sph")[0]
    assert round_trip["sph_theta"] == pytest.approx(-45.0)
    assert round_trip["sph_phi"] == pytest.approx(30.0)


def test_pop_chancenter_uses_one_based_omit_indices_and_console_return_shape() -> None:
    eeg = _eeg()

    out, command = pop_chancenter(eeg, [0.0, 0.0, 0.0], [4], return_com=True)

    assert out["chanlocs"][3]["X"] == 0.0
    assert out["chanlocs"][3]["Z"] == 1.0
    assert command == "EEG = pop_chancenter(EEG, [0 0 0], [4]);"
    _assert_parseable(command)


@pytest.mark.gui
def test_pop_chancenter_gui_cancel_path_returns_original_without_history() -> None:
    eeg = _eeg()

    out, command = pop_chancenter(eeg, gui=True, return_com=True)

    assert out is eeg
    assert command == ""


def test_console_pop_chancenter_updates_session_history_and_current_dataset() -> None:
    session = EEGPrepSession()
    session.store_current(_eeg(), new=True)
    workspace = EEGPrepConsoleWorkspace(session)

    result = workspace.namespace["pop_chancenter"](session.EEG, [0.0, 0.0, 0.0], [4])

    assert result.updated is True
    assert session.CURRENTSET == [1]
    assert session.EEG["chanlocs"][3]["labels"] == "Cz"
    assert session.LASTCOM == "EEG = pop_chancenter(EEG, [0 0 0], [4]);"
    assert session.ALLCOM[-1] == session.LASTCOM


def test_pop_chancoresp_autoselects_all_channels_and_fiducials() -> None:
    left = [{"labels": "Nz"}, {"labels": "Cz"}, {"labels": "LPA"}, {"labels": "RPA"}]
    right = [{"labels": "cz"}, {"labels": "rpa"}, {"labels": "lpa"}, {"labels": "nasion"}]
    template = [{"labels": "FidT10"}, {"labels": "FidT9"}, {"labels": "FidNz"}]

    all_left, all_right, command = pop_chancoresp(left, right, "autoselect", "all", return_com=True)
    fid_left, fid_right = pop_chancoresp(left, right, "autoselect", "fiducials")
    template_left, template_right = pop_chancoresp(left, template, "autoselect", "fiducials")

    assert all_left == [2, 3, 4]
    assert all_right == [1, 3, 2]
    assert fid_left == [1, 3, 4]
    assert fid_right == [4, 3, 2]
    assert template_left == [1, 3, 4]
    assert template_right == [3, 2, 1]
    _assert_parseable(command)


def test_floatread_floatwrite_round_trip_with_inferred_dimension(tmp_path: Path) -> None:
    data = np.arange(12, dtype=float).reshape(3, 4)
    filename = tmp_path / "data.fdt"

    floatwrite(data, filename, "ieee-le")
    loaded = floatread(filename, [3, np.inf], "ieee-le")

    assert np.array_equal(loaded, data)


def test_pop_loadbci_imports_ascii_file(tmp_path: Path) -> None:
    bci_file = tmp_path / "demo.bci"
    bci_file.write_text("Ch1 Ch2 State\n1 2 0\n3 4 1\n", encoding="utf-8")

    eeg, command = pop_loadbci(bci_file, 256, return_com=True)

    assert eeg["data"].shape == (3, 2)
    assert [chan["labels"] for chan in eeg["chanlocs"]] == ["Ch1", "Ch2", "State"]
    assert command.endswith(", 256);")
    _assert_parseable(command)


def test_snapread_and_pop_snapread_import_binary_file(tmp_path: Path) -> None:
    snap_file = tmp_path / "demo.SMA"
    data = np.asarray(
        [
            [0.0, 0.0, 3.0, 3.0, 0.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [5.0, 4.0, 3.0, 2.0, 1.0],
        ],
        dtype="<f4",
    )
    header = b'"NCHAN%"=3\n"NUM.POINTS"=5\n"ACT.FREQ"=100\n"TR"\n2026-06-05\n'
    snap_file.write_bytes(header + b"\xaa" + np.ravel(data, order="F").tobytes())

    raw_data, params, events, _header = snapread(snap_file)
    eeg, command = pop_snapread(snap_file, 2.0, return_com=True)

    assert raw_data.shape == (2, 5)
    assert params.tolist() == [2.0, 5.0, 100.0]
    assert np.flatnonzero(events).tolist() == [2]
    assert eeg["data"][0, 0] == pytest.approx(2.0)
    assert eeg["event"][0]["latency"] == 3.0
    _assert_parseable(command)


def test_readegilocs_uses_packaged_egi_montage() -> None:
    eeg = {"nbchan": 129, "chanlocs": [], "chaninfo": {}}

    out = readegilocs(eeg)

    assert len(out["chanlocs"]) == 129
    assert len(out["chaninfo"]["nodatchans"]) == 3
    assert out["chanlocs"][0]["labels"] == "E1"
