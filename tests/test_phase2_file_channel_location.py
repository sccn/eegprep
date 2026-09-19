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
from eegprep.functions.sigprocfunc.readlocs import readelp, readeetraklocs
from tests.eeglab_tests import eeglab_test


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


@eeglab_test(
    "unittesting_sigprocfunc/chancenter/sigprocfunc_chancenter_wrapperTest.m",
    "test_pass_2d",
)
@eeglab_test(
    "unittesting_sigprocfunc/chancenter/sigprocfunc_chancenter_wrapperTest.m",
    "test_pass_scalar",
)
def test_chancenter_explicit_center_matches_upstream_translation() -> None:
    x, y, z, center, optimized = chancenter([1, 1, 0, 0], [1, 0, 0, 1], [0, 0, 0, 0], [1, 1, 0])

    np.testing.assert_allclose(x, [0, 0, -1, -1])
    np.testing.assert_allclose(y, [0, -1, -1, 0])
    np.testing.assert_allclose(z, np.zeros(4))
    np.testing.assert_allclose(center, [1, 1, 0])
    assert optimized is False

    scalar = chancenter(1, 1, 0, [1, 1, 0])
    np.testing.assert_allclose(scalar[0], [0])
    np.testing.assert_allclose(scalar[1], [0])
    np.testing.assert_allclose(scalar[2], [0])


@eeglab_test(
    "unittesting_sigprocfunc/chancenter/sigprocfunc_chancenter_wrapperTest.m",
    "test_pass_all_negative",
)
@eeglab_test(
    "unittesting_sigprocfunc/chancenter/sigprocfunc_chancenter_wrapperTest.m",
    "test_pass_zero_radius",
)
def test_chancenter_negative_center_and_zero_radius_match_upstream() -> None:
    x, y, z, center, _ = chancenter([-1, 2, -1, 2], [1, 1, -2, -2], [0, -1, 0, 1], [3, -2, 0.5])
    np.testing.assert_allclose(x, [-4, -1, -4, -1])
    np.testing.assert_allclose(y, [3, 3, 0, 0])
    np.testing.assert_allclose(z, [-0.5, -1.5, -0.5, 0.5])
    np.testing.assert_allclose(center, [3, -2, 0.5])

    zero = chancenter(np.ones(4), np.ones(4), np.ones(4), [1, 1, 1])
    for coordinates in zero[:3]:
        np.testing.assert_allclose(coordinates, np.zeros(4))


@eeglab_test(
    "unittesting_sigprocfunc/chancenter/sigprocfunc_chancenter_wrapperTest.m",
    "test_pass_optimize",
)
def test_chancenter_automatic_sphere_fit_matches_symmetric_upstream_case() -> None:
    x, y, z, center, optimized = chancenter(
        [1, 1, 0, 0, 0.5, 0.5],
        [1, 0, 0, 1, 0.5, 0.5],
        [0, 0, 0, 0, 1, -1],
        [],
    )

    np.testing.assert_allclose(x, [0.5, 0.5, -0.5, -0.5, 0, 0], atol=1e-5)
    np.testing.assert_allclose(y, [0.5, -0.5, -0.5, 0.5, 0, 0], atol=1e-5)
    np.testing.assert_allclose(z, [0, 0, 0, 0, 1, -1], atol=1e-5)
    np.testing.assert_allclose(center, [0.5, 0.5, 0], atol=1e-5)
    assert optimized is True


@eeglab_test(
    "unittesting_sigprocfunc/convertlocs/sigprocfunc_convertlocs_wrapperTest.m",
    "test_pass_cart2all",
)
def test_convertlocs_cart2all_matches_upstream_complete_coordinate_fields() -> None:
    root = np.sqrt(2) / 2
    locs = [
        {"labels": "a", "X": -root, "Y": root, "Z": 0, "type": "EEG"},
        {"labels": "b", "X": 1, "Y": 0, "Z": 0, "type": "EEG"},
        {"labels": "c", "X": 0, "Y": -1, "Z": 0, "type": "EEG"},
        {"labels": "d", "X": root, "Y": -root, "Z": 0, "type": "EEG"},
    ]

    converted = convertlocs(locs, "cart2all")

    np.testing.assert_allclose([loc["theta"] for loc in converted], [-135, 0, 90, 45])
    np.testing.assert_allclose([loc["radius"] for loc in converted], np.full(4, 0.5))
    np.testing.assert_allclose([loc["sph_theta"] for loc in converted], [135, 0, -90, -45])
    np.testing.assert_allclose([loc["sph_phi"] for loc in converted], np.zeros(4))
    np.testing.assert_allclose([loc["sph_radius"] for loc in converted], np.ones(4))
    np.testing.assert_allclose([loc["sph_theta_besa"] for loc in converted], [-90, 90, 90, 90])
    np.testing.assert_allclose([loc["sph_phi_besa"] for loc in converted], [45, 90, 0, 45])


@eeglab_test(
    "unittesting_sigprocfunc/floatwrite/sigprocfunc_floatwrite_wrapperTest.m",
    "test_pass_general",
)
@eeglab_test(
    "unittesting_sigprocfunc/floatwrite/sigprocfunc_floatwrite_wrapperTest.m",
    "test_pass_native",
)
def test_floatwrite_matches_upstream_four_byte_column_major_encoding(tmp_path: Path) -> None:
    data = np.asarray([[1.23, 0.12], [4.56, 3.45], [7.89, 6.78]])
    big_endian = tmp_path / "big.fdt"
    native = tmp_path / "native.fdt"

    floatwrite(data, big_endian, "ieee-be")
    floatwrite(data, native)

    expected_big = np.ravel(data.astype(">f4"), order="F").tobytes()
    expected_native = np.ravel(data.astype("f4"), order="F").tobytes()
    assert big_endian.read_bytes() == expected_big
    assert native.read_bytes() == expected_native


@eeglab_test(
    "unittesting_sigprocfunc/floatread/sigprocfunc_floatread_wrapperTest.m",
    "test_pass_general",
)
@eeglab_test(
    "unittesting_sigprocfunc/floatread/sigprocfunc_floatread_wrapperTest.m",
    "test_pass_no_size",
)
@eeglab_test(
    "unittesting_sigprocfunc/floatread/sigprocfunc_floatread_wrapperTest.m",
    "test_pass_offset",
)
def test_floatread_matches_upstream_shape_inference_and_cell_offset(tmp_path: Path) -> None:
    general = np.asarray([[1.23, 0.12], [4.56, 3.45], [7.89, 6.78]])
    general_path = tmp_path / "general.fdt"
    general_path.write_bytes(np.ravel(general.astype(">f4"), order="F").tobytes())

    offset_data = np.asarray([[1.23, 7.89, 3.45], [4.56, 0.12, 6.78]])
    offset_path = tmp_path / "offset.fdt"
    offset_path.write_bytes(np.ravel(offset_data.astype(">f4"), order="F").tobytes())

    np.testing.assert_allclose(floatread(general_path, [3, 2], "ieee-be"), general)
    np.testing.assert_allclose(floatread(offset_path, [2, np.inf], "ieee-be"), offset_data)
    np.testing.assert_allclose(
        floatread(offset_path, [2, 2], "ieee-be", ([2, 3], [1, 2])),
        offset_data[:, 1:],
    )


@eeglab_test(
    "unittesting_sigprocfunc/floatread/sigprocfunc_floatread_wrapperTest.m",
    "test_pass_square",
)
def test_floatread_square_shape_matches_upstream(tmp_path: Path) -> None:
    data = np.asarray([[1.23, 0.12, 9.01], [4.56, 3.45, 2.34], [7.89, 6.78, 5.67]])
    path = tmp_path / "square.fdt"
    path.write_bytes(np.ravel(data.astype(">f4"), order="F").tobytes())

    np.testing.assert_allclose(floatread(path, "square", "ieee-be"), data, rtol=1e-6)


@eeglab_test(
    "unittesting_sigprocfunc/floatread/sigprocfunc_floatread_wrapperTest.m",
    "test_pass_nan_inf",
)
def test_floatread_preserves_upstream_nonfinite_values(tmp_path: Path) -> None:
    data = np.asarray([[np.nan, -np.inf], [np.nan, np.inf], [np.nan, 0], [np.nan, 0]])
    path = tmp_path / "nonfinite.fdt"
    path.write_bytes(np.ravel(data.astype(">f4"), order="F").tobytes())

    np.testing.assert_allclose(floatread(path, [4, 2], "ieee-be"), data, equal_nan=True)


@eeglab_test(
    "unittesting_sigprocfunc/readeetraklocs/sigprocfunc_readeetraklocs_wrapperTest.m",
    "test_pass_general",
)
def test_readeetraklocs_matches_upstream_labels_and_coordinates(tmp_path: Path) -> None:
    path = tmp_path / "test.elc"
    path.write_text(
        "NumberPositions 4\nUnitPosition mm\nPositions\n"
        "1 -2 3\n0.1 2.5 -4\n3 -4 -8.5\n-11 0 19\n"
        "Labels\nNr1 Nr2 Nr3 Ch4\n",
        encoding="utf-8",
    )

    locs = readeetraklocs(path)

    assert [loc["labels"] for loc in locs] == ["Nr1", "Nr2", "Nr3", "Ch4"]
    np.testing.assert_allclose(
        [[loc[axis] for axis in "XYZ"] for loc in locs], [[1, -2, 3], [0.1, 2.5, -4], [3, -4, -8.5], [-11, 0, 19]]
    )


@eeglab_test(
    "unittesting_sigprocfunc/readeetraklocs/sigprocfunc_readeetraklocs_wrapperTest.m",
    "test_pass_labels_positions_exchanged",
)
def test_readeetraklocs_accepts_upstream_exchanged_section_order(tmp_path: Path) -> None:
    """Strengthen the upstream script, whose intended call is currently empty."""
    path = tmp_path / "exchanged.elc"
    path.write_text(
        "NumberPositions 4\nUnitPosition mm\nLabels\nNr1 Nr2 Nr3 Ch4\n"
        "Positions\n1 -2 3\n0.1 2.5 -4\n3 -4 -8.5\n-11 0 19\n",
        encoding="utf-8",
    )

    locs = readeetraklocs(path)

    assert [loc["labels"] for loc in locs] == ["Nr1", "Nr2", "Nr3", "Ch4"]
    np.testing.assert_allclose(
        [[loc[axis] for axis in "XYZ"] for loc in locs], [[1, -2, 3], [0.1, 2.5, -4], [3, -4, -8.5], [-11, 0, 19]]
    )


@eeglab_test(
    "unittesting_sigprocfunc/readelp/sigprocfunc_readelp_wrapperTest.m",
    "test_pass_general",
)
def test_readelp_matches_upstream_fiducials_labels_and_coordinates(tmp_path: Path) -> None:
    path = tmp_path / "test.elp"
    path.write_text(
        "%F 0.1011 0.0000 0.0000\n%F -0.0135 0.0731 0.0000\n%F 0.0135 -0.0731 0.0000\n"
        "%N REF\n-0.0092 -0.0779 -0.0036\n%N FP1\n0.1091 0.0102 0.0583\n"
        "%N FPZ\n0.1176 -0.0184 0.0595\n%N FP2\n0.1179 -0.0470 0.0565\n",
        encoding="utf-8",
    )

    locs = readelp(path)

    assert [loc["labels"] for loc in locs] == ["Nz", "LPA", "RPA", "REF", "FP1", "FPZ", "FP2"]
    assert [loc["type"] for loc in locs] == ["FID", "FID", "FID", "EEG", "EEG", "EEG", "EEG"]
    np.testing.assert_allclose([loc["X"] for loc in locs], [0.1011, -0.0135, 0.0135, -0.0092, 0.1091, 0.1176, 0.1179])
    np.testing.assert_allclose([loc["Y"] for loc in locs], [0, 0.0731, -0.0731, -0.0779, 0.0102, -0.0184, -0.0470])
    np.testing.assert_allclose([loc["Z"] for loc in locs], [0, 0, 0, -0.0036, 0.0583, 0.0595, 0.0565])


@eeglab_test(
    "unittesting_sigprocfunc/readlocs/sigprocfunc_readlocs_wrapperTest.m",
    "test_pass_bugzilla_339",
)
def test_readlocs_custom_besa_columns_accept_upstream_regression_values(tmp_path: Path) -> None:
    path = tmp_path / "bugzilla_339.txt"
    path.write_text(
        "1 7.3 120.4 163.5\n2 7.6 107.1 165.1\n3 8.1 98.3 170\n4 8.7 103 -170.4\n",
        encoding="utf-8",
    )

    locs = readlocs(
        path,
        "filetype",
        "custom",
        "format",
        ["channum", "sph_radius", "sph_theta_besa", "sph_phi_besa"],
    )

    assert len(locs) == 4
    np.testing.assert_allclose([loc["sph_radius"] for loc in locs], [7.3, 7.6, 8.1, 8.7])
    assert all(np.isfinite([loc["X"], loc["Y"], loc["Z"]]).all() for loc in locs)


@eeglab_test(
    "unittesting_sigprocfunc/readlocs/sigprocfunc_readlocs_wrapperTest.m",
    "test_pass_bugzilla_72",
)
def test_readlocs_ced_preserves_upstream_quoted_channel_type(tmp_path: Path) -> None:
    path = tmp_path / "bugzilla_72.ced"
    path.write_text(
        "Number labels theta radius X Y Z sph_theta sph_phi sph_radius type\n"
        "1 A2 0 0.04 0.0219 0 0.174 0 82.8 0.175 'meg'\n"
        "2 A3 0 0.0794 0.044 0 0.172 0 75.7 0.178 'meg'\n",
        encoding="utf-8",
    )

    locs = readlocs(path)

    assert [loc["labels"] for loc in locs] == ["A2", "A3"]
    assert [loc["type"] for loc in locs] == ["meg", "meg"]


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


def _chancenter_suite_locations(last_x=0.0) -> list[dict]:
    return [
        {"labels": "", "X": 0.0, "Y": 0.0, "Z": 0.0, "theta": 0.0, "radius": 0.0},
        {"labels": "", "X": 1.0, "Y": 0.0, "Z": 0.0, "theta": 0.0, "radius": 0.0},
        {"labels": "", "X": last_x, "Y": 1.0, "Z": 1.0, "theta": 0.0, "radius": 0.0},
    ]


@eeglab_test("unittesting_popfunc/pop_chancenter/popfunc_pop_chancenter_wrapperTest.m", "test_pass_empty_center")
def test_pop_chancenter_current_suite_empty_center():
    locations = _chancenter_suite_locations()
    locations[0].update({"Y": 1.95})
    locations[1].update({"X": 2.0})
    locations[2].update({"Y": 0.0, "Z": 2.0})

    centered = pop_chancenter(locations, [])

    np.testing.assert_allclose(
        [[location[axis] for axis in ("X", "Y", "Z")] for location in centered],
        [[0.0, 1.95, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 2.0]],
        atol=0.11,
    )


@eeglab_test("unittesting_popfunc/pop_chancenter/popfunc_pop_chancenter_wrapperTest.m", "test_pass_no_omitchans")
def test_pop_chancenter_current_suite_known_center():
    centered = pop_chancenter(_chancenter_suite_locations(), [1, -1, 0])

    np.testing.assert_allclose(
        [[location[axis] for axis in ("X", "Y", "Z")] for location in centered],
        [[-1, 1, 0], [0, 1, 0], [-1, 2, 1]],
    )


@eeglab_test("unittesting_popfunc/pop_chancenter/popfunc_pop_chancenter_wrapperTest.m", "test_pass_with_omitchans")
def test_pop_chancenter_current_suite_omits_one_based_channels():
    centered = pop_chancenter(_chancenter_suite_locations(last_x=1.0), [1, -1, 0], [1])

    np.testing.assert_allclose(
        [[location[axis] for axis in ("X", "Y", "Z")] for location in centered],
        [[0, 0, 0], [0, 1, 0], [0, 2, 1]],
    )


@eeglab_test("unittesting_popfunc/pop_chancenter/popfunc_pop_chancenter_wrapperTest.m", "test_test_pop_chancenter")
def test_pop_chancenter_current_suite_center_and_omit_smoke_cases():
    locations = _eeg()["chanlocs"]
    cases = [
        ([], None),
        ([0, 0, 0], None),
        ([1, 1, 1], None),
        ([-1, 0, 1], None),
        ([100000, -1000000, 100], None),
        ([1, 1, 1], [1]),
        ([1, 1, 1], [1, 2, 3, 4]),
        ([1, 1, 1], [0]),
        ([1, 1, 1], [1, 2, 3, 4, 5]),
    ]

    for center, omitted in cases:
        output = pop_chancenter(locations, center, omitted)
        assert len(output) == len(locations)
        assert all(np.isfinite(location[axis]) for location in output for axis in ("X", "Y", "Z"))


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


@eeglab_test(
    "unittesting_sigprocfunc/readegilocs/sigprocfunc_readegilocs_wrapperTest.m",
    "test_test_readegilocs",
)
def test_readegilocs_uses_packaged_egi_montages_for_upstream_channel_counts() -> None:
    for channel_count in (32, 33, 64, 65, 128, 129, 256, 257):
        eeg = {"nbchan": channel_count, "chanlocs": [], "chaninfo": {}}

        out = readegilocs(eeg)

        assert len(out["chanlocs"]) == channel_count
        expected_nondata = {256: 1, 257: 0}.get(
            channel_count,
            4 if channel_count in {32, 64, 128} else 3,
        )
        assert len(out["chaninfo"]["nodatchans"]) == expected_nondata
        assert out["chanlocs"][0]["labels"] == "E1"
