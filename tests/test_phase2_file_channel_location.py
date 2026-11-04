from __future__ import annotations

import ast
from importlib.resources import files
from pathlib import Path

import numpy as np
import pytest

from eegprep.functions.adminfunc.console import EEGPrepConsoleWorkspace, _console_python_command
from eegprep.functions.guifunc.session import EEGPrepSession
from eegprep import (
    convertlocs,
    pop_chancenter,
    pop_chancoresp,
    pop_loadbci,
    pop_readlocs,
    pop_writelocs,
    readlocs,
    readegilocs,
    writelocs,
)
from eegprep.functions.sigprocfunc.readlocs import readeetraklocs
from tests.eeglab_tests import assert_matlab_equal, assert_matlab_near, eeglab_test
from tests.eeglab_tests.assertions import assert_matlab_struct_near, matlab_field_concat


_CHANCENTER_WRAPPER = "unittesting_sigprocfunc/chancenter/sigprocfunc_chancenter_wrapperTest.m"
_FLOATREAD_WRAPPER = "unittesting_sigprocfunc/floatread/sigprocfunc_floatread_wrapperTest.m"
_FLOATWRITE_WRAPPER = "unittesting_sigprocfunc/floatwrite/sigprocfunc_floatwrite_wrapperTest.m"
_POP_CHANCENTER_WRAPPER = "unittesting_popfunc/pop_chancenter/popfunc_pop_chancenter_wrapperTest.m"


def _double_row(values):
    return np.array([values], dtype=float)


def _location_struct(rows):
    return np.array([[tuple(row.values()) for row in rows]], dtype=[(field, object) for field in rows[0]])


def test_struct_near_assertion_matches_source(eeglab_backend, eeglab_matlab_engine, eeglab_suite_root):
    cells = np.empty((1, 1), dtype=object)
    cells[0, 0] = np.array([[1.0]])
    nan_cells = np.empty((1, 1), dtype=object)
    nan_cells[0, 0] = np.array([[np.nan]])
    row = _location_struct([{"x": 1.0, "labels": "a"}, {"x": 2.0, "labels": "b"}])
    cases = [
        (row, row.copy(), 5),
        (row, row.T, 5),  # near.m compares concatenated fields, not struct shape.
        (row, _location_struct([{"x": 1.0, "labels": "a"}]), 5),
        ({"x": np.array([[1.0]])}, {"x": np.array([[1.00005]])}, 5),
        ({"x": np.array([[1.0]])}, {"x": np.array([[1.001]])}, 5),
        ({"x": np.array([[1.0]])}, {"y": np.array([[1.0]])}, 5),
        ({"x": np.array([[1.0]])}, {"x": np.array([[1.0]])}, 1),
        ({"x": {"y": np.array([[1.0]])}}, {"x": {"y": np.array([[1.0]])}}, 2),
        ({"x": {"y": np.array([[1.0]])}}, {"x": {"y": np.array([[1.0]])}}, 3),
        ({}, {}, 5),
        (np.empty((0, 1)), np.empty((0, 2)), 5),
        (np.array([[np.nan, np.inf, -np.inf]]), np.array([[np.nan, np.inf, -np.inf]]), 5),
        (cells, cells.copy(), 5),
        (nan_cells, nan_cells.copy(), 5),
    ]
    previous = eeglab_matlab_engine.path()
    eeglab_matlab_engine.addpath(str(eeglab_suite_root / "unittesting_common/helpfunc"), nargout=0)
    try:
        for first, second, depth in cases:
            near = eeglab_backend("near", first, second, float(depth)).item()
            if near:
                assert_matlab_struct_near(first, second, depth)
            else:
                with pytest.raises(AssertionError):
                    assert_matlab_struct_near(first, second, depth)
    finally:
        eeglab_matlab_engine.path(previous, nargout=0)


@eeglab_test("unittesting_popfunc/pop_readlocs/popfunc_pop_readlocs_wrapperTest.m", "test_test_pop_readlocs")
def test_reference_pop_readlocs_four_original_files(eeglab_backend, eeglab_suite_root):
    for filename in (
        "sample_data/eeglab_chan32.locs",
        "sample_locs/GSN64v2_0.sfp",
        "sample_locs/Standard-10-10-Cap33.ced",
        "sample_locs/Standard-10-20-Cap25.locs",
    ):
        eeglab_backend("pop_readlocs", str(eeglab_suite_root / "eeglab" / filename))


@eeglab_test("unittesting_popfunc/pop_writelocs/popfunc_pop_writelocs_wrapperTest.m", "test_test_pop_writelocs")
def test_reference_pop_writelocs_recorded_locations(eeglab_backend, eeglab_suite_root, eeglab_working_directory):
    eeg = eeglab_backend("pop_loadset", str(eeglab_suite_root / "eeglab/sample_data/eeglab_data_epochs_ica.set"))
    eeglab_backend("pop_writelocs", eeg["chanlocs"], "testwirtelocs.locs", nargout=0)
    (eeglab_working_directory / "testwirtelocs.locs").unlink(missing_ok=True)


def _reference_chancenter(eeglab_backend, coordinates, center, expected):
    result = eeglab_backend("chancenter", *[_double_row(row) for row in coordinates], center, nargout=4)
    for actual, correct in zip(result, expected, strict=True):
        assert_matlab_near(actual, _double_row(correct))


@eeglab_test(_CHANCENTER_WRAPPER, "test_pass_2d")
def test_reference_chancenter_2d(eeglab_backend):
    _reference_chancenter(
        eeglab_backend,
        [[1, 1, 0, 0], [1, 0, 0, 1], [0, 0, 0, 0]],
        _double_row([1, 1, 0]),
        [[0, 0, -1, -1], [0, -1, -1, 0], [0, 0, 0, 0], [1, 1, 0]],
    )


@eeglab_test(_CHANCENTER_WRAPPER, "test_pass_all_negative")
def test_reference_chancenter_all_negative(eeglab_backend):
    _reference_chancenter(
        eeglab_backend,
        [[-1, 2, -1, 2], [1, 1, -2, -2], [0, -1, 0, 1]],
        _double_row([3, -2, 0.5]),
        [[-4, -1, -4, -1], [3, 3, 0, 0], [-0.5, -1.5, -0.5, 0.5], [3, -2, 0.5]],
    )


@eeglab_test(_CHANCENTER_WRAPPER, "test_pass_optimize")
def test_reference_chancenter_optimize(eeglab_backend):
    _reference_chancenter(
        eeglab_backend,
        [[1, 1, 0, 0, 0.5, 0.5], [1, 0, 0, 1, 0.5, 0.5], [0, 0, 0, 0, 1, -1]],
        np.empty((0, 0)),
        [[0.5, 0.5, -0.5, -0.5, 0, 0], [0.5, -0.5, -0.5, 0.5, 0, 0], [0, 0, 0, 0, 1, -1], [0.5, 0.5, 0]],
    )


@eeglab_test(_CHANCENTER_WRAPPER, "test_pass_scalar")
def test_reference_chancenter_scalar(eeglab_backend):
    _reference_chancenter(eeglab_backend, [[1], [1], [0]], _double_row([1, 1, 0]), [[0], [0], [0], [1, 1, 0]])


@eeglab_test(_CHANCENTER_WRAPPER, "test_pass_zero_radius")
def test_reference_chancenter_zero_radius(eeglab_backend):
    _reference_chancenter(eeglab_backend, [[1, 1, 1, 1]] * 3, _double_row([1, 1, 1]), [[0, 0, 0, 0]] * 3 + [[1, 1, 1]])


@eeglab_test("unittesting_sigprocfunc/convertlocs/sigprocfunc_convertlocs_wrapperTest.m", "test_pass_cart2all")
def test_reference_convertlocs_cart2all(eeglab_backend):
    root = np.sqrt(2) / 2
    xyz = [(-root, root, 0), (1, 0, 0), (0, -1, 0), (root, -root, 0)]
    original = _location_struct(
        [
            {"labels": label, "X": float(x), "Y": float(y), "Z": float(z), "type": "EEG"}
            for label, (x, y, z) in zip("abcd", xyz, strict=True)
        ]
    )
    correct = _location_struct(
        [
            {
                "labels": label,
                "theta": float(theta),
                "radius": 0.5,
                "X": float(x),
                "Y": float(y),
                "Z": float(z),
                "sph_theta": float(sph_theta),
                "sph_phi": 0.0,
                "sph_radius": 1.0,
                "sph_theta_besa": float(besa_theta),
                "sph_phi_besa": float(besa_phi),
                "type": "EEG",
            }
            for label, (x, y, z), theta, sph_theta, besa_theta, besa_phi in zip(
                "abcd", xyz, [-135, 0, 90, 45], [135, 0, -90, -45], [-90, 90, 90, 90], [45, 90, 0, 45], strict=True
            )
        ]
    )
    assert_matlab_struct_near(correct, eeglab_backend("convertlocs", original, "cart2all"))


def _reference_floatread(eeglab_backend, eeglab_suite_root, name, size, correct, *offset):
    result = eeglab_backend(
        "floatread",
        str(eeglab_suite_root / "unittesting_sigprocfunc/floatread" / f"{name}.fdt"),
        size,
        "ieee-be",
        *offset,
    )
    assert_matlab_near(result, np.array(correct, dtype=float))


@eeglab_test(_FLOATREAD_WRAPPER, "test_pass_general")
def test_reference_floatread_general(eeglab_backend, eeglab_suite_root):
    _reference_floatread(
        eeglab_backend,
        eeglab_suite_root,
        "pass_general",
        _double_row([3, 2]),
        [[1.23, 0.12], [4.56, 3.45], [7.89, 6.78]],
    )


@eeglab_test(_FLOATREAD_WRAPPER, "test_pass_no_size")
def test_reference_floatread_no_size(eeglab_backend, eeglab_suite_root):
    _reference_floatread(
        eeglab_backend,
        eeglab_suite_root,
        "pass_no_size",
        _double_row([2, np.inf]),
        [[1.23, 7.89, 3.45], [4.56, 0.12, 6.78]],
    )


@eeglab_test(_FLOATREAD_WRAPPER, "test_pass_offset")
def test_reference_floatread_offset(eeglab_backend, eeglab_suite_root):
    offset = np.empty((1, 2), dtype=object)
    offset[0, 0], offset[0, 1] = _double_row([2, 3]), _double_row([1, 2])
    _reference_floatread(
        eeglab_backend, eeglab_suite_root, "pass_offset", _double_row([2, 2]), [[7.89, 3.45], [0.12, 6.78]], offset
    )


@eeglab_test(_FLOATREAD_WRAPPER, "test_pass_square")
def test_reference_floatread_square(eeglab_backend, eeglab_suite_root):
    _reference_floatread(
        eeglab_backend,
        eeglab_suite_root,
        "pass_square",
        "square",
        [[1.23, 0.12, 9.01], [4.56, 3.45, 2.34], [7.89, 6.78, 5.67]],
    )


@eeglab_test(_FLOATREAD_WRAPPER, "test_pass_nan_inf")
def test_reference_floatread_nan_inf(eeglab_backend, eeglab_suite_root):
    _reference_floatread(
        eeglab_backend,
        eeglab_suite_root,
        "pass_nan_inf",
        _double_row([4, 2]),
        [[np.nan, -np.inf], [np.nan, np.inf], [np.nan, 0], [np.nan, 0]],
    )


def _reference_written_bytes(eeglab_backend, directory, filename, *format):
    path = directory / filename
    path.unlink(missing_ok=True)
    eeglab_backend("floatwrite", np.array([[1.23, 0.12], [4.56, 3.45], [7.89, 6.78]]), filename, *format, nargout=0)
    with path.open("rb") as stream:
        inbytes = stream.read(24)
    existed = path.is_file()
    path.unlink(missing_ok=True)
    correct = "3F 9D 70 A3 40 91 EB 85 40 FC 7A E1 3D F5 C2 8F 40 5C CC CC 40 D8 F5 C2".split()
    result = [f"{value:02X}" for value in inbytes]
    assert_matlab_near(np.array([[len(inbytes)]]), np.array([[24.0]]))
    assert_matlab_near(np.array([[1.0]]), np.array([[float(existed)]]))
    return correct, result


@eeglab_test(_FLOATWRITE_WRAPPER, "test_pass_general")
def test_reference_floatwrite_general(eeglab_backend, eeglab_working_directory):
    correct, result = _reference_written_bytes(eeglab_backend, eeglab_working_directory, "pass_general.fdt", "ieee-be")
    for index in range(3, 24, 4):
        correct[index] = result[index] = "00"
    assert correct == result


@eeglab_test(_FLOATWRITE_WRAPPER, "test_pass_native")
def test_reference_floatwrite_native(eeglab_backend, eeglab_working_directory):
    correct, result = _reference_written_bytes(eeglab_backend, eeglab_working_directory, "pass_native.fdt")
    assert sum(first != second for first, second in zip(sorted(correct), sorted(result), strict=True)) <= 4


@eeglab_test("unittesting_sigprocfunc/readeetraklocs/sigprocfunc_readeetraklocs_wrapperTest.m", "test_pass_general")
def test_reference_readeetraklocs_original_file(eeglab_backend, eeglab_suite_root):
    locs = eeglab_backend("readeetraklocs", str(eeglab_suite_root / "unittesting_sigprocfunc/readeetraklocs/test.elc"))
    assert max(locs.shape) == 4
    for index, (label, xyz) in enumerate(
        zip(["Nr1", "Nr2", "Nr3", "Ch4"], [[1, -2, 3], [0.1, 2.5, -4], [3, -4, -8.5], [-11, 0, 19]], strict=True)
    ):
        location = locs.ravel(order="F")[index]
        assert location["labels"] == label
        for field, value in zip("XYZ", xyz, strict=True):
            assert_matlab_near(np.array([[value]], dtype=float), location[field])


@eeglab_test("unittesting_sigprocfunc/readelp/sigprocfunc_readelp_wrapperTest.m", "test_pass_general")
def test_reference_readelp_five_outputs(eeglab_backend, eeglab_suite_root):
    eloc, names, x, y, z = eeglab_backend(
        "readelp", str(eeglab_suite_root / "unittesting_sigprocfunc/readelp/test.elp"), nargout=5
    )
    assert_matlab_equal(eloc["labels"].reshape((1, -1), order="F"), names)
    for field, actual, correct in zip(
        "XYZ",
        [x, y, z],
        [
            [0.1011, -0.0135, 0.0135, -0.0092, 0.1091, 0.1176, 0.1179],
            [0, 0.0731, -0.0731, -0.0779, 0.0102, -0.0184, -0.0470],
            [0, 0, 0, -0.0036, 0.0583, 0.0595, 0.0565],
        ],
        strict=True,
    ):
        assert_matlab_equal(matlab_field_concat(eloc, field), actual)
        assert_matlab_equal(actual, _double_row(correct))
    assert_matlab_equal(names, np.array([["Nz", "LPA", "RPA", "REF", "FP1", "FPZ", "FP2"]], dtype=object))
    assert_matlab_equal(
        eloc["type"].reshape((1, -1), order="F"),
        np.array([["FID", "FID", "FID", "EEG", "EEG", "EEG", "EEG"]], dtype=object),
    )


@eeglab_test("unittesting_sigprocfunc/readlocs/sigprocfunc_readlocs_wrapperTest.m", "test_pass_bugzilla_339")
def test_reference_readlocs_bugzilla_339(eeglab_backend, eeglab_suite_root):
    eeglab_backend(
        "readlocs",
        str(eeglab_suite_root / "unittesting_sigprocfunc/readlocs/bugzilla_339.txt"),
        "filetype",
        "custom",
        "format",
        np.array([["channum", "sph_radius", "sph_theta_besa", "sph_phi_besa"]], dtype=object),
    )


@eeglab_test("unittesting_sigprocfunc/readlocs/sigprocfunc_readlocs_wrapperTest.m", "test_pass_bugzilla_72")
def test_reference_readlocs_bugzilla_72(eeglab_backend, eeglab_suite_root):
    eeglab_backend("readlocs", str(eeglab_suite_root / "unittesting_sigprocfunc/readlocs/bugzilla_72.ced"))


@eeglab_test("unittesting_sigprocfunc/readegilocs/sigprocfunc_readegilocs_wrapperTest.m", "test_test_readegilocs")
def test_reference_readegilocs_all_original_channel_counts(eeglab_backend):
    eeg = eeglab_backend("eeg_emptyset")
    for count in (32, 33, 64, 65, 128, 129, 256, 257):
        eeg["nbchan"] = float(count)
        eeg = eeglab_backend("readegilocs", eeg)
        assert_matlab_near(np.array([[count]], dtype=float), np.array([[max(eeg["chanlocs"].shape)]], dtype=float))


def _reference_center_locs(xyz):
    return _location_struct(
        [{"X": float(x), "Y": float(y), "Z": float(z), "theta": 0.0, "radius": 0.0, "labels": ""} for x, y, z in xyz]
    )


@eeglab_test(_POP_CHANCENTER_WRAPPER, "test_pass_empty_center")
def test_reference_pop_chancenter_empty_center(eeglab_backend):
    locs = _reference_center_locs([[0, 1.95, 0], [2, 0, 0], [0, 0, 2]])
    _, center, _ = eeglab_backend("pop_chancenter", locs, np.empty((0, 0)), nargout=3)
    assert abs(center.ravel(order="F")[0]) < 0.11
    assert abs(center.ravel(order="F")[1]) < 0.11
    assert abs(center.ravel(order="F")[2]) < 0.11


def _reference_pop_chancenter_translation(eeglab_backend, xyz, start, *omitted):
    locs = _reference_center_locs(xyz)
    center = _double_row([1, -1, 0])
    result, newcenter, _ = eeglab_backend("pop_chancenter", locs, center, *omitted, nargout=3)
    for index in range(start, 3):
        for field, shift in zip("XYZ", center[0], strict=True):
            locs[field][0, index] -= shift
    correct = eeglab_backend("convertlocs", locs, "cart2all")
    assert_matlab_struct_near(result, correct)
    assert_matlab_near(newcenter, center)


@eeglab_test(_POP_CHANCENTER_WRAPPER, "test_pass_no_omitchans")
def test_reference_pop_chancenter_no_omitchans(eeglab_backend):
    _reference_pop_chancenter_translation(eeglab_backend, [[0, 0, 0], [1, 0, 0], [0, 1, 1]], 0)


@eeglab_test(_POP_CHANCENTER_WRAPPER, "test_pass_with_omitchans")
def test_reference_pop_chancenter_with_omitchans(eeglab_backend):
    _reference_pop_chancenter_translation(eeglab_backend, [[0, 0, 0], [1, 0, 0], [1, 1, 1]], 1, 1.0)


@eeglab_test(_POP_CHANCENTER_WRAPPER, "test_test_pop_chancenter")
def test_reference_pop_chancenter_nine_recorded_location_calls(eeglab_backend, eeglab_suite_root):
    eeg = eeglab_backend("pop_loadset", str(eeglab_suite_root / "eeglab/sample_data/eeglab_data.set"))
    eeg["chanlocs"] = eeglab_backend(
        "pop_chanedit",
        eeg["chanlocs"],
        "load",
        np.array([[str(eeglab_suite_root / "eeglab/sample_data/eeglab_chan32.locs"), "filetype", ""]], dtype=object),
        "shrink",
        -0.1,
    )
    for center in (
        np.empty((0, 0)),
        _double_row([0, 0, 0]),
        _double_row([1, 1, 1]),
        _double_row([-1, 0, 1]),
        _double_row([100000, -1000000, 100]),
    ):
        eeglab_backend("pop_chancenter", eeg["chanlocs"], center, nargout=2)
    count = int(np.asarray(eeg["nbchan"]).item())
    for omitted in (
        _double_row([1]),
        _double_row(range(1, count + 1)),
        _double_row([0]),
        _double_row(range(1, count + 2)),
    ):
        eeglab_backend("pop_chancenter", eeg["chanlocs"], _double_row([1, 1, 1]), omitted, nargout=2)


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
    assert template_right == [3, 1, 2]
    _assert_parseable(command)


def test_pop_loadbci_imports_ascii_file(tmp_path: Path) -> None:
    bci_file = tmp_path / "demo.bci"
    bci_file.write_text("Ch1 Ch2 State\n1 2 0\n3 4 1\n", encoding="utf-8")

    eeg, command = pop_loadbci(bci_file, 256, return_com=True)

    assert eeg["data"].shape == (3, 2)
    assert [chan["labels"] for chan in eeg["chanlocs"]] == ["Ch1", "Ch2", "State"]
    assert command.endswith(", 256);")
    _assert_parseable(command)


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
