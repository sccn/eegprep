"""Current ``eeglab_tests`` ports for EEGLAB's disk-backed ``mmo`` class."""

from __future__ import annotations

from copy import copy, deepcopy
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import pascal

from eegprep import MemmapData, mmo
from eegprep.functions.popfunc.eeg_eegrej import eeg_eegrej
from eegprep.functions.popfunc.eeg_emptyset import eeg_emptyset
from eegprep.functions.popfunc.pop_epoch import pop_epoch
from eegprep.functions.popfunc.pop_resample import pop_resample
from eegprep.functions.popfunc.pop_reref import pop_reref
from eegprep.functions.popfunc.pop_rmbase import pop_rmbase
from eegprep.functions.popfunc.pop_select import pop_select
from eegprep.plugins.firfilt.pop_firws import pop_firws
from tests.eeglab_tests import eeglab_test


UPSTREAM = "unittesting_adminfunc/mmo/adminfunc_mmo_wrapperTest.m"


@pytest.fixture
def mmo_backend(request, eeglab_backend, eeglab_suite_root):
    if request.config.getoption("--eeglab-backend") != "matlab":
        yield eeglab_backend
        return
    engine = request.getfixturevalue("eeglab_matlab_engine")
    original_path = engine.path()
    # transposeindices is a test-suite helper, not in the EEGLAB checkout.
    engine.addpath(str(eeglab_suite_root / "unittesting_adminfunc/mmo"), nargout=0)
    try:
        yield eeglab_backend
    finally:
        engine.path(original_path, nargout=0)


def _original_mmo_values(kind, transposed=False):
    if kind == "row":
        return np.arange(1.0, 11.0)[None, :]
    if kind == "column":
        return np.arange(1.0, 11.0)[:, None]
    values = pascal(8).astype(float)
    if transposed:
        values[:, 0] = 2
    if kind == "cube":
        values = np.stack((values, values * 2), axis=2)
    return values


def _python_mmo(values, directory, transposed):
    physical = np.moveaxis(values, 0, -1) if transposed else values
    filename = directory / "testfile.fdt"
    physical.astype(np.float32).ravel(order="F").tofile(filename)
    return mmo(filename, values.shape, True, transposed, True)


def _matlab_array_shape(values):
    # MATLAB does not retain trailing singleton dimensions beyond two.
    shape = list(values.shape)
    while len(shape) > 2 and shape[-1] == 1:
        shape.pop()
    return values.reshape(shape)


_MMO_DELETIONS = [
    ("cube", ":,:,1", (8, 8, 2), 2, [0]),
    ("cube", ":,:,2", (8, 8, 2), 2, [1]),
    ("cube", ":,2", (8, 16), 1, [1]),
    ("cube", ":,12", (8, 16), 1, [11]),
    ("cube", "4:5,:", (8, 16), 0, [3, 4]),
    ("cube", "4:5,:,:", (8, 8, 2), 0, [3, 4]),
    ("cube", "4:5", (1, 128), 1, [3, 4]),
    ("cube", "73:83", (1, 128), 1, list(range(72, 83))),
    ("matrix", "4:5,:,:", (8, 8), 0, [3, 4]),
    ("matrix", "4:5,:", (8, 8), 0, [3, 4]),
    ("matrix", ":,4:5,:,:", (8, 8), 1, [3, 4]),
    ("matrix", ":,4:5", (8, 8), 1, [3, 4]),
    ("matrix", "4", (1, 64), 1, [3]),
    ("matrix", "63", (1, 64), 1, [62]),
    ("row", "[4 7]", (1, 10), 1, [3, 6]),
    ("column", "[4 7]", (10, 1), 0, [3, 6]),
]


def _check_original_mmo_deletion(request, backend, directory, case, transposed):
    kind, subscripts, view_shape, axis, indices = case
    values = _original_mmo_values(kind, transposed)
    expected = np.delete(values.reshape(view_shape, order="F"), indices, axis=axis)
    if request.config.getoption("--eeglab-backend") == "matlab":
        actual = backend("eegprep_test_mmo_assignment", values, subscripts, np.empty((0, 0)), transposed)
    else:
        mapped = _python_mmo(values, directory, transposed)
        if "," not in subscripts:
            mapped.delete(indices)
        elif view_shape == values.shape:
            mapped.delete(indices, axis=axis)
        else:
            raise NotImplementedError("EEGPrep has no public collapsed-axis mapped deletion operation")
        actual = np.asarray(mapped)
    np.testing.assert_array_equal(_matlab_array_shape(actual), _matlab_array_shape(expected))


@pytest.mark.parametrize("case", _MMO_DELETIONS, ids=[f"case-{i}" for i in range(1, 17)])
@eeglab_test(UPSTREAM, "test_checkmmo3")
def test_upstream_mmo_original_deletions(request, mmo_backend, eeglab_working_directory, case):
    _check_original_mmo_deletion(request, mmo_backend, eeglab_working_directory, case, False)


@pytest.mark.parametrize("case", [_MMO_DELETIONS[i] for i in (0, 1, 5, 8, 9, 11)])
@eeglab_test(UPSTREAM, "test_checkmmo3_transposed")
def test_upstream_mmo_original_transposed_deletions(request, mmo_backend, eeglab_working_directory, case):
    _check_original_mmo_deletion(request, mmo_backend, eeglab_working_directory, case, True)


_ALL = slice(None)
_MMO_ASSIGNMENTS = [
    ("cube", "9,:", (9, 8, 2), (8, _ALL, _ALL)),
    ("cube", ":,9", (8, 8, 2), (_ALL, 0, 1)),
    ("cube", "9,9", (9, 8, 2), (8, 0, 1)),
    ("cube", "9:12,:,:", (12, 8, 2), (slice(8, 12), _ALL, _ALL)),
    ("cube", ":,9,:", (8, 9, 2), (_ALL, 8, _ALL)),
    ("cube", ":,:,3", (8, 8, 3), (_ALL, _ALL, 2)),
    ("cube", ":,9:10,3", (8, 10, 3), (_ALL, slice(8, 10), 2)),
    ("cube", "9,9:10,3", (9, 10, 3), (8, slice(8, 10), 2)),
    ("matrix", "9,9:10", (9, 10), (8, slice(8, 10))),
    ("matrix", "9,:", (9, 8), (8, _ALL)),
    ("matrix", ":,9", (8, 9), (_ALL, 8)),
    ("row", "11", (1, 11), (0, 10)),
    ("column", "11", (11, 1), (10, 0)),
]


def _check_original_mmo_assignment(request, backend, directory, case, transposed):
    kind, subscripts, shape, key = case
    values = _original_mmo_values(kind, transposed)
    # The eighth transposed source case grows its INPUT to nine columns first.
    if transposed and kind == "matrix" and subscripts == "9,:":
        values = np.column_stack((pascal(8), np.full(8, 2.0)))
        shape = (9, 9)
    expected = np.zeros(shape)
    expected[tuple(slice(0, size) for size in values.shape)] = values
    expected[key] = 1
    if request.config.getoption("--eeglab-backend") == "matlab":
        actual = backend("eegprep_test_mmo_assignment", values, subscripts, 1.0, transposed)
    else:
        mapped = _python_mmo(values, directory, transposed)
        # Do not resize first: implicit indexed growth is the source contract.
        mapped[key] = 1.0
        actual = np.asarray(mapped)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("case", _MMO_ASSIGNMENTS, ids=[f"case-{i}" for i in range(3, 16)])
@eeglab_test(UPSTREAM, "test_checkmmo4")
def test_upstream_mmo_original_assignments(request, mmo_backend, eeglab_working_directory, case):
    _check_original_mmo_assignment(request, mmo_backend, eeglab_working_directory, case, False)


@pytest.mark.parametrize("case", [_MMO_ASSIGNMENTS[i] for i in (0, 3, 4, 5, 6, 7, 8, 9, 10)])
@eeglab_test(UPSTREAM, "test_checkmmo4_transposed")
def test_upstream_mmo_original_transposed_assignments(request, mmo_backend, eeglab_working_directory, case):
    _check_original_mmo_assignment(request, mmo_backend, eeglab_working_directory, case, True)


@eeglab_test(UPSTREAM, "test_checkmmo")
def test_upstream_mmo_original_workspace_copy_counts(request, mmo_backend, eeglab_working_directory):
    if request.config.getoption("--eeglab-backend") != "matlab":
        pytest.fail("EEGPrep does not expose the source's caller-workspace copy-count observable")
    counts = mmo_backend("eegprep_test_mmo_copies", np.arange(1.0, 11.0)[:, None])
    expected = np.array([[1.0, 2.0, 2.0, 2.0, 0.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]])
    # Source case 5 requires only that the nested aliases are not counted as one.
    assert counts[0, 4] != 1
    np.testing.assert_array_equal(np.delete(counts, 4, axis=1), np.delete(expected, 4, axis=1))


@eeglab_test(UPSTREAM, "test_checkmmo2")
def test_upstream_mmo_original_copy_on_write_diagnostics(request, mmo_backend, eeglab_working_directory):
    if request.config.getoption("--eeglab-backend") != "matlab":
        pytest.fail("EEGPrep does not emit the source's copy-on-write debug diagnostics")
    values = np.arange(1.0, 11.0)[None, :]
    mmo_backend("eegprep_test_mmo_copies", values)
    messages = mmo_backend("eegprep_test_mmo_copywrites", values)
    for message, unique in zip(messages.flat, (False, True, False, False, False, True, True), strict=True):
        assert (message[0] == "u") == unique


@eeglab_test(UPSTREAM, "test_checkmmo4")
def test_upstream_mmo_original_returned_workspace(request, mmo_backend, eeglab_working_directory):
    values = _original_mmo_values("cube")
    if request.config.getoption("--eeglab-backend") == "matlab":
        actual = mmo_backend("eegprep_test_mmo_returned", values)
        np.testing.assert_array_equal(actual, values)
    else:
        pytest.fail("EEGPrep does not expose the source's helper-returned workspace debug observable")


@eeglab_test(UPSTREAM, "test_checkmmo_sub5")
def test_upstream_mmo_original_helper_write(request, mmo_backend, eeglab_working_directory):
    if request.config.getoption("--eeglab-backend") == "matlab":
        mmo_backend("checkmmo_sub5", nargout=0)
    else:
        mapped = _python_mmo(np.arange(1.0, 11.0)[None, :], eeglab_working_directory, False)
        mapped[0, 3] = 5.0


@eeglab_test(UPSTREAM, "test_checkmmo_sub6")
def test_upstream_mmo_original_helper_construction(request, mmo_backend, eeglab_working_directory):
    if request.config.getoption("--eeglab-backend") == "matlab":
        mmo_backend("checkmmo_sub6", nargout=0)
    else:
        _python_mmo(pascal(8).astype(float), eeglab_working_directory, False)


def _data(shape: tuple[int, ...]) -> np.ndarray:
    values = np.arange(1, int(np.prod(shape)) + 1, dtype=np.float32)
    return values.reshape(shape, order="F")


def _mapping(
    tmp_path: Path,
    name: str,
    data: np.ndarray,
    *,
    transposed: bool = False,
) -> MemmapData:
    suffix = ".dat" if transposed else ".fdt"
    return MemmapData.from_array(data, filename=tmp_path / f"{name}{suffix}", transposed=transposed)


def _eeg(data: np.ndarray | MemmapData, *, epoched: bool = False) -> dict:
    array = np.asarray(data)
    srate = 100.0
    output = eeg_emptyset()
    trials = int(array.shape[2]) if array.ndim == 3 else 1
    pnts = int(array.shape[1])
    output.update(
        {
            "setname": "mapped epochs" if epoched else "mapped continuous",
            "data": data,
            "nbchan": int(array.shape[0]),
            "pnts": pnts,
            "trials": trials,
            "srate": srate,
            "xmin": -0.2 if epoched else 0.0,
            "xmax": (-0.2 if epoched else 0.0) + (pnts - 1) / srate,
            "times": ((-0.2 if epoched else 0.0) + np.arange(pnts) / srate) * 1000,
            "chanlocs": [
                {
                    "labels": f"Ch{index + 1}",
                    "type": "EEG",
                    "X": float(np.cos(index * np.pi / 2)),
                    "Y": float(np.sin(index * np.pi / 2)),
                    "Z": 0.0,
                }
                for index in range(array.shape[0])
            ],
            "event": [],
            "urevent": [],
            "epoch": [],
            "icaweights": np.array([]),
            "icasphere": np.array([]),
            "icawinv": np.array([]),
            "icaact": np.array([]),
            "icachansind": np.array([], dtype=int),
            "saved": "no",
        }
    )
    if epoched:
        for trial in range(trials):
            event = {
                "type": "square",
                "latency": float(trial * pnts + 21),
                "epoch": trial + 1,
                "urevent": trial,
            }
            output["event"].append(event)
            output["urevent"].append({"type": "square", "latency": float(trial * pnts + 21)})
            output["epoch"].append({})
    else:
        output["event"] = [
            {"type": "square", "latency": latency, "urevent": index}
            for index, latency in enumerate((151.0, 401.0, 651.0))
        ]
        output["urevent"] = [{"type": event["type"], "latency": event["latency"]} for event in output["event"]]
    return output


def _continuous_data() -> np.ndarray:
    samples = np.arange(800, dtype=np.float64) / 100.0
    return np.vstack(
        [
            np.sin(2 * np.pi * (4 + channel) * samples) + 0.2 * np.cos(2 * np.pi * (channel + 1) * samples) + channel
            for channel in range(4)
        ]
    ).astype(np.float32)


def _epoched_data() -> np.ndarray:
    return _continuous_data()[:, :360].reshape((4, 120, 3), order="F")


def _run_workflow(name: str, eeg: dict) -> dict:
    if name.endswith("_after_rejection"):
        eeg = eeg_eegrej(eeg, [[260, 300]])
        name = name.removesuffix("_after_rejection")
    if name == "continuous_rejection":
        return eeg_eegrej(eeg, [[260, 300]])
    if name == "continuous_epoch":
        output, _indices = pop_epoch(eeg, ["square"], [-0.1, 0.2], gui=False)
        return output
    if name in {"continuous_baseline", "epoched_baseline"}:
        return pop_rmbase(eeg, [], np.arange(1, 11), gui=False)
    if name in {"continuous_filter", "epoched_filter"}:
        return pop_firws(
            eeg,
            ftype="highpass",
            fcutoff=3,
            wtype="blackman",
            forder=20,
            gui=False,
        )
    if name in {"continuous_rereference", "epoched_rereference"}:
        return pop_reref(eeg, [], gui=False)
    if name == "continuous_select":
        return pop_select(eeg, channel=[0, 2], point=[20, 200], gui=False)
    if name == "epoched_select":
        return pop_select(eeg, channel=[0, 2], point=[10, 90], trial=[1, 3], gui=False)
    if name in {"continuous_resample", "epoched_resample"}:
        return pop_resample(eeg, 50, engine="poly", gui=False)
    raise AssertionError(f"unknown workflow: {name}")


def test_check_eeglab_mmo_preprocessing_preserves_mapping_and_values(tmp_path: Path):
    workflows = (
        "continuous_rejection",
        "continuous_epoch",
        "continuous_epoch_after_rejection",
        "continuous_baseline",
        "continuous_baseline_after_rejection",
        "continuous_filter",
        "continuous_filter_after_rejection",
        "continuous_rereference",
        "continuous_select",
        "continuous_resample",
        "continuous_resample_after_rejection",
        "epoched_baseline",
        "epoched_filter",
        "epoched_rereference",
        "epoched_select",
        "epoched_resample",
    )
    for workflow in workflows:
        epoched = workflow.startswith("epoched")
        source = _epoched_data() if epoched else _continuous_data()
        mapped = _mapping(tmp_path, workflow, source)
        mapped_result = _run_workflow(workflow, _eeg(mapped, epoched=epoched))
        in_memory_result = _run_workflow(workflow, _eeg(source.copy(), epoched=epoched))

        assert isinstance(mapped_result["data"], MemmapData), workflow
        assert mapped_result["data"].path != mapped.path, workflow
        assert mapped_result["data"].shape == np.asarray(in_memory_result["data"]).shape, workflow
        np.testing.assert_allclose(
            np.asarray(mapped_result["data"]),
            np.asarray(in_memory_result["data"]),
            rtol=2e-5,
            atol=2e-6,
            err_msg=workflow,
        )
        np.testing.assert_array_equal(np.asarray(mapped), source, err_msg=f"source changed in {workflow}")


def test_checkmmo_constructs_validated_normal_and_empty_mappings(tmp_path: Path):
    values = _data((1, 10))
    path = tmp_path / "values.fdt"
    np.ravel(values, order="F").tofile(path)

    mapped = mmo(path, values.shape, writable=True, debug=True)
    assert mapped.dataFile == str(path)
    assert mapped.dimensions == values.shape
    assert mapped.writable
    assert mapped.debug
    assert mapped.type == "mmo"
    np.testing.assert_array_equal(np.asarray(mapped), values)
    readonly = mmo(path, values.shape, writable=False)
    with pytest.raises(ValueError, match="read-only"):
        readonly[0, 0] = 0

    blank = mmo(None, (2, 3))
    assert blank.shape == (2, 3)
    np.testing.assert_array_equal(np.asarray(blank), np.zeros((2, 3), dtype=np.float32))

    missing = tmp_path / "missing.fdt"
    with pytest.raises(FileNotFoundError, match="not found"):
        mmo(missing, (2, 3))
    empty = tmp_path / "empty.fdt"
    empty.touch()
    with pytest.raises(ValueError, match="empty"):
        mmo(empty, (2, 3))
    short = tmp_path / "short.fdt"
    np.ones(3, dtype=np.float32).tofile(short)
    with pytest.raises(ValueError, match="expected 24"):
        mmo(short, (2, 3))


def test_checkmmo2_copies_detach_only_when_written(tmp_path: Path):
    values = _data((2, 5))
    original = _mapping(tmp_path, "copy-on-write", values)
    shallow = copy(original)
    nested = deepcopy({"data": original})["data"]
    assert shallow.path == original.path
    assert nested.path == original.path

    shallow[0, 3] = -4
    nested[1, 4] = -9

    assert shallow.path != original.path
    assert nested.path != original.path
    assert shallow.path != nested.path
    assert original[0, 3] == values[0, 3]
    assert original[1, 4] == values[1, 4]
    assert shallow[0, 3] == -4
    assert nested[1, 4] == -9


def _assert_deletion(
    tmp_path: Path,
    name: str,
    values: np.ndarray,
    indices: object,
    *,
    axis: int | None,
    transposed: bool,
) -> None:
    mapped = _mapping(tmp_path, name, values, transposed=transposed)
    mapped.delete(indices, axis=axis)
    if axis is None:
        expected = np.delete(values.reshape(-1, order="F"), indices)
        expected = expected.reshape((-1, 1) if values.ndim == 2 and values.shape[1] == 1 else (1, -1))
    else:
        expected = np.delete(values, indices, axis=axis)
    assert mapped.shape == expected.shape
    np.testing.assert_array_equal(np.asarray(mapped), expected)


def test_checkmmo3_deletes_axes_and_column_major_linear_indices(tmp_path: Path):
    values3 = _data((8, 8, 2))
    _assert_deletion(tmp_path, "trial-first", values3, 0, axis=2, transposed=False)
    _assert_deletion(tmp_path, "trial-last", values3, 1, axis=2, transposed=False)
    _assert_deletion(tmp_path, "columns", values3, [1, 3], axis=1, transposed=False)
    _assert_deletion(tmp_path, "rows", values3, [3, 4], axis=0, transposed=False)
    _assert_deletion(tmp_path, "linear-3d", values3, np.arange(72, 83), axis=None, transposed=False)

    values2 = _data((8, 8))
    _assert_deletion(tmp_path, "rows-2d", values2, [3, 4], axis=0, transposed=False)
    _assert_deletion(tmp_path, "columns-2d", values2, [3, 4], axis=1, transposed=False)
    _assert_deletion(tmp_path, "linear-2d", values2, 3, axis=None, transposed=False)
    _assert_deletion(tmp_path, "row-vector", _data((1, 10)), [3, 6], axis=None, transposed=False)
    _assert_deletion(tmp_path, "column-vector", _data((10, 1)), [3, 6], axis=None, transposed=False)


def test_checkmmo3_transposed_deletes_logical_axes(tmp_path: Path):
    values3 = _data((8, 8, 2))
    _assert_deletion(tmp_path, "transposed-trial", values3, 0, axis=2, transposed=True)
    _assert_deletion(tmp_path, "transposed-rows", values3, [3, 4], axis=0, transposed=True)
    values2 = _data((8, 8))
    _assert_deletion(tmp_path, "transposed-rows-2d", values2, [3, 4], axis=0, transposed=True)
    _assert_deletion(tmp_path, "transposed-columns-2d", values2, [3, 4], axis=1, transposed=True)


def _assert_growth(
    tmp_path: Path,
    name: str,
    values: np.ndarray,
    shape: tuple[int, ...],
    key: object,
    assigned: object,
    *,
    transposed: bool,
) -> None:
    mapped = _mapping(tmp_path, name, values, transposed=transposed)
    mapped.resize(shape)
    expected = np.zeros(shape, dtype=np.float32)
    overlap = tuple(slice(0, value) for value in values.shape)
    expected[overlap] = values
    mapped[key] = assigned
    expected[key] = assigned
    assert mapped.shape == shape
    np.testing.assert_array_equal(np.asarray(mapped), expected)


def test_checkmmo4_grows_normal_mappings_with_zero_fill(tmp_path: Path):
    values3 = _data((8, 8, 2))
    _assert_growth(tmp_path, "grow-row", values3, (9, 8, 2), (8, slice(None), slice(None)), 1, transposed=False)
    _assert_growth(tmp_path, "grow-column", values3, (8, 9, 2), (slice(None), 8, slice(None)), 1, transposed=False)
    _assert_growth(tmp_path, "grow-trial", values3, (8, 8, 3), (slice(None), slice(None), 2), 1, transposed=False)
    _assert_growth(
        tmp_path,
        "grow-all",
        values3,
        (9, 10, 3),
        (8, slice(8, 10), 2),
        1,
        transposed=False,
    )
    values2 = _data((8, 8))
    _assert_growth(tmp_path, "grow-2d", values2, (9, 10), (8, slice(8, 10)), 1, transposed=False)
    _assert_growth(tmp_path, "grow-row-vector", _data((1, 10)), (1, 11), (0, 10), 1, transposed=False)
    _assert_growth(tmp_path, "grow-column-vector", _data((10, 1)), (11, 1), (10, 0), 1, transposed=False)


def test_checkmmo4_transposed_grows_logical_axes_with_zero_fill(tmp_path: Path):
    values3 = _data((8, 8, 2))
    _assert_growth(
        tmp_path, "transposed-grow-row", values3, (9, 8, 2), (8, slice(None), slice(None)), 1, transposed=True
    )
    _assert_growth(
        tmp_path, "transposed-grow-column", values3, (8, 9, 2), (slice(None), 8, slice(None)), 1, transposed=True
    )
    _assert_growth(
        tmp_path, "transposed-grow-trial", values3, (8, 8, 3), (slice(None), slice(None), 2), 1, transposed=True
    )
    _assert_growth(
        tmp_path,
        "transposed-grow-all",
        values3,
        (9, 10, 3),
        (8, slice(8, 10), 2),
        1,
        transposed=True,
    )


def _copy_and_write(mapped: MemmapData, key: object, value: float) -> MemmapData:
    local = copy(mapped)
    local[key] = value
    return local


def test_checkmmo_sub1_nested_function_write_detaches_argument(tmp_path: Path):
    values = _data((1, 10))
    original = _mapping(tmp_path, "sub1", values)
    changed = _copy_and_write(original, (0, 3), 5)
    assert original[0, 3] == values[0, 3]
    assert changed[0, 3] == 5


def test_checkmmo_sub2_function_argument_remains_isolated_after_return(tmp_path: Path):
    values = _data((1, 10))
    original = _mapping(tmp_path, "sub2", values)

    def mutate_argument(argument: MemmapData) -> MemmapData:
        result = copy(argument)
        result[0, 2] = 2
        return result

    changed = mutate_argument(original)
    np.testing.assert_array_equal(np.asarray(original), values)
    assert changed[0, 2] == 2


def test_checkmmo_sub3_deepcopy_inside_mapping_preserves_value_semantics(tmp_path: Path):
    values = _data((1, 10))
    namespace = {"test": _mapping(tmp_path, "sub3", values)}
    copied_namespace = deepcopy(namespace)
    copied_namespace["test"][0, 4] = -1
    np.testing.assert_array_equal(np.asarray(namespace["test"]), values)
    assert copied_namespace["test"][0, 4] == -1


def test_checkmmo_sub4_nested_container_copy_preserves_value_semantics(tmp_path: Path):
    values = _data((1, 10))
    original = _mapping(tmp_path, "sub4", values)
    copied_container = deepcopy([{"mapping": original}])
    copied_container[0]["mapping"][0, 5] = -2
    np.testing.assert_array_equal(np.asarray(original), values)
    assert copied_container[0]["mapping"][0, 5] == -2


def _create_mapping(tmp_path: Path, name: str, values: np.ndarray) -> MemmapData:
    return _mapping(tmp_path, name, values)


def test_checkmmo_sub5_helper_created_mapping_writes_through_to_disk(tmp_path: Path):
    values = _data((1, 10))
    mapped = _create_mapping(tmp_path, "sub5", values)
    path = mapped.path
    mapped[0, 3] = 5
    mapped.flush()
    reopened = MemmapData(path, values.shape)
    assert reopened[0, 3] == 5


def test_checkmmo_sub6_helper_returns_mapping_and_original_values(tmp_path: Path):
    values = _data((8, 8))

    def create_pair() -> tuple[MemmapData, np.ndarray]:
        array = values.copy()
        return _create_mapping(tmp_path, "sub6", array), array

    mapped, original = create_pair()
    np.testing.assert_array_equal(np.asarray(mapped), original)


def test_checkmmo_sub7_helper_maps_caller_supplied_multidimensional_data(tmp_path: Path):
    values = _data((8, 8, 2))
    mapped = _create_mapping(tmp_path, "sub7", values)
    assert mapped.ndim == 3
    assert mapped.size == values.size
    np.testing.assert_array_equal(np.asarray(mapped), values)


def test_checkmmo_sub8_unique_returned_mapping_mutates_without_file_replacement(tmp_path: Path):
    values = _data((8, 8))
    mapped = _create_mapping(tmp_path, "sub8", values)
    path = mapped.path
    mapped[5, 0] = 5
    mapped.flush()
    assert mapped.path == path
    assert MemmapData(path, values.shape)[5, 0] == 5


def test_transposeindices_exposes_logical_2d_and_3d_indices(tmp_path: Path):
    values2 = _data((3, 5))
    physical2 = values2.transpose(1, 0)
    path2 = tmp_path / "physical2.dat"
    np.ravel(physical2, order="F").tofile(path2)
    mapped2 = mmo(path2, values2.shape, transposed=True)
    np.testing.assert_array_equal(mapped2[1:, [0, 3]], values2[1:, [0, 3]])
    mapped2[2, 4] = -12
    physical2_reloaded = np.fromfile(path2, dtype=np.float32).reshape(physical2.shape, order="F")
    assert physical2_reloaded[4, 2] == -12

    values3 = _data((3, 5, 2))
    physical3 = values3.transpose(1, 2, 0)
    path3 = tmp_path / "physical3.dat"
    np.ravel(physical3, order="F").tofile(path3)
    mapped3 = mmo(path3, values3.shape, transposed=True)
    np.testing.assert_array_equal(mapped3[:, 1:4, 1], values3[:, 1:4, 1])
    mapped3[2, 4, 1] = -21
    physical3_reloaded = np.fromfile(path3, dtype=np.float32).reshape(physical3.shape, order="F")
    assert physical3_reloaded[4, 1, 2] == -21
