"""Execution and transport contracts for MATLAB-validated Python tests."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.io import savemat
from scipy.sparse import coo_matrix, issparse

from tests.eeglab_tests import assert_matlab_equal
from tests.eeglab_tests.backend import _load_matlab_variable, _save_matlab_arguments, call_matlab, call_python


pytest_plugins = ("pytester",)


def _forced_sidecar_call(
    engine, directory, name, *args, inline_limit_bytes=1, input_limit_bytes=None, part_limit_bytes=2**28
):
    arguments = np.empty((1, len(args)), dtype=object)
    for index, value in enumerate(args):
        arguments[0, index] = value
    input_file, output_file = directory / "input.mat", directory / "output.mat"
    if input_limit_bytes is None:
        savemat(input_file, {"arguments": arguments}, long_field_names=True)
    else:
        _save_matlab_arguments(input_file, arguments, input_limit_bytes, part_limit_bytes)
    engine.eegprep_test_call(str(input_file), str(output_file), name, 1.0, float(inline_limit_bytes), nargout=0)
    return _load_matlab_variable(output_file, "outputs")[0, 0]


@pytest.mark.gui
def test_erpimage_transport_excludes_uncompared_graphics_objects(eeglab_matlab_engine):
    data = np.array(
        [[21, 24, 25, 28, 31, 37], [22, 25, 26, 29, 32, 38], [23, 26, 27, 30, 33, 39], [24, 27, 28, 31, 34, 40]],
        dtype=float,
    )
    try:
        outputs = call_matlab(
            eeglab_matlab_engine,
            "eegprep_test_erpimage_outputs",
            "call",
            4.0,
            data,
            np.empty((0, 0)),
            np.empty((0, 0)),
            "testcase",
            1.0,
            1.0,
            "erp",
            "cbar",
            "noxlabel",
        )
        # Current MATLAB returns cells, not the numeric handle vector expected
        # by the old test. Transport must retain that genuine mismatch.
        compared_axes = outputs[0, 4]
        assert compared_axes.dtype == object
        assert compared_axes.shape == (1, 2)
        assert all(np.isnan(value).all() for value in compared_axes.flat)
        np.testing.assert_array_equal(outputs[0, 0], data)
        np.testing.assert_allclose(outputs[0, 5], data.mean(axis=1)[None, :])
    finally:
        eeglab_matlab_engine.close("all", "force", nargout=0)


def test_python_backend_resolves_functions_at_call_time():
    data = np.array([[1.0, 2.0, 3.0], [4.0, 8.0, 12.0]])
    np.testing.assert_array_equal(call_python("rmbase", data), [[-1, 0, 1], [-4, 0, 4]])
    with pytest.raises(AttributeError, match="eegprep_test_transport"):
        call_python("eegprep_test_transport", "fixture")


def test_python_backend_selects_requested_outputs():
    arguments = ([0.0, 0.5], [], 100.0, [0.0, 1.0])
    np.testing.assert_array_equal(call_python("eeg_lat2point", *arguments), [1.0, 51.0])
    points, flag = call_python("eeg_lat2point", *arguments, nargout=2)
    np.testing.assert_array_equal(points, [1.0, 51.0])
    assert not flag
    with pytest.raises(AssertionError, match="3 outputs"):
        call_python("eeg_lat2point", *arguments, nargout=3)


def _isolated_suite(pytester, source):
    root = str(Path(__file__).resolve().parents[1])
    pytester.makeconftest(f'import sys\nsys.path.insert(0, {root!r})\npytest_plugins = ("tests.conftest",)')
    pytester.makepyfile(source)


def test_contracts_require_explicit_backend_selection(pytester):
    _isolated_suite(pytester, "def test_contract(eeglab_backend):\n    raise AssertionError('must not run')")
    result = pytester.runpytest_subprocess()
    result.assert_outcomes(deselected=1)


def test_matlab_mode_rejects_unconverted_provenance_test(pytester):
    _isolated_suite(
        pytester,
        """
from tests.eeglab_tests import eeglab_test
@eeglab_test('test_reference.m', 'test_direct')
def test_direct():
    raise AssertionError('Python must not execute in the MATLAB lane')
""",
    )
    result = pytester.runpytest_subprocess("--eeglab-backend=matlab")
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines(["*still call Python directly*"])


def test_contract_marker_selects_backend_tests_only(pytester):
    _isolated_suite(
        pytester,
        """
import numpy as np
from tests.eeglab_tests import eeglab_test
def test_contract(eeglab_backend):
    np.testing.assert_array_equal(eeglab_backend('rmbase', np.array([[1., 2., 3.]])), [[-1., 0., 1.]])
@eeglab_test('test_reference.m', 'test_direct')
def test_direct():
    raise AssertionError('direct Python must not be selected as a backend contract')
def test_supplement():
    raise AssertionError('unrelated Python regression must not run')
""",
    )
    result = pytester.runpytest_subprocess("--eeglab-backend=python", "-m", "eeglab_contract")
    result.assert_outcomes(passed=1, deselected=2)
    result = pytester.runpytest_subprocess("--eeglab-backend=matlab", "-m", "eeglab_contract", "--collect-only")
    assert result.ret == pytest.ExitCode.OK
    result.stdout.fnmatch_lines(["*1/3 tests collected (2 deselected)*"])


def test_required_matlab_mode_fails_without_reference(pytester, tmp_path):
    _isolated_suite(pytester, "def test_contract(eeglab_backend):\n    eeglab_backend('eeglab_missing_function')")
    result = pytester.runpytest_subprocess("--eeglab-backend=matlab", f"--eeglab-root={tmp_path / 'absent'}")
    result.assert_outcomes(errors=1)
    result.stdout.fnmatch_lines(["*MATLAB contracts require --eeglab-root*"])


@pytest.mark.parametrize("kind", ["missing", "file"])
def test_support_path_rejects_non_directories(pytester, tmp_path, kind):
    directory = tmp_path / kind
    if kind == "file":
        directory.write_text("not a directory", encoding="utf-8")
    _isolated_suite(pytester, "def test_never_runs():\n    raise AssertionError('must not run')")
    result = pytester.runpytest_subprocess(f"--eeglab-support-path={directory}")
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines(["*--eeglab-support-path requires an existing directory*"])


def test_support_paths_resolve_relative_directories_in_option_order(pytester):
    first, second = pytester.path / "first support", pytester.path / "second support"
    first.mkdir()
    second.mkdir()
    _isolated_suite(
        pytester,
        f"""
from pathlib import Path
def test_paths(request):
    assert request.config.getoption('--eeglab-support-path') == [Path({str(first)!r}), Path({str(second)!r})]
""",
    )
    result = pytester.runpytest_subprocess(
        "--eeglab-support-path=first support", "--eeglab-support-path=second support"
    )
    result.assert_outcomes(passed=1)


def test_reference_datasets_require_an_explicit_checkout(pytester, monkeypatch):
    monkeypatch.delenv("EEGPREP_EEGLAB_ROOT", raising=False)
    _isolated_suite(pytester, "def test_source_data(eeglab_suite_root):\n    raise AssertionError('must not run')")
    result = pytester.runpytest_subprocess()
    result.assert_outcomes(errors=1)
    result.stdout.fnmatch_lines(["*Reference datasets require --eeglab-suite-root or --eeglab-root*"])


def test_reference_datasets_reject_missing_checkout(pytester, tmp_path):
    _isolated_suite(pytester, "def test_source_data(eeglab_suite_root):\n    raise AssertionError('must not run')")
    result = pytester.runpytest_subprocess(f"--eeglab-suite-root={tmp_path / 'absent'}")
    result.assert_outcomes(errors=1)
    result.stdout.fnmatch_lines(["*EEGLAB test checkout does not exist*"])


def test_matlab_gate_honors_explicit_test_selection(pytester):
    _isolated_suite(
        pytester,
        """
from tests.eeglab_tests import eeglab_test
@eeglab_test('test_reference.m', 'test_direct')
def test_direct():
    raise AssertionError('must not run')
def test_converted(eeglab_backend):
    raise AssertionError('collect only')
""",
    )
    result = pytester.runpytest_subprocess("--eeglab-backend=matlab", "-k", "converted", "--collect-only")
    assert result.ret == pytest.ExitCode.OK
    result.stdout.fnmatch_lines(["*1/2 tests collected (1 deselected)*"])


def test_python_mode_collects_then_fails_missing_capability(pytester):
    _isolated_suite(pytester, "def test_contract(eeglab_backend):\n    eeglab_backend('eeglab_missing_function')")
    result = pytester.runpytest_subprocess("--eeglab-backend=python")
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*AttributeError*eeglab_missing_function*"])


def test_python_options_are_restored_after_each_contract(pytester):
    _isolated_suite(
        pytester,
        """
from eegprep.functions.adminfunc.eeg_options import EEG_OPTIONS
original = EEG_OPTIONS.copy()
def test_changes(eeglab_backend, eeglab_options_directory):
    value = 1 - original['option_storedisk']
    eeglab_backend('pop_editoptions', option_storedisk=value)
    assert EEG_OPTIONS['option_storedisk'] == value
def test_restored():
    assert EEG_OPTIONS == original
""",
    )
    result = pytester.runpytest_subprocess("--eeglab-backend=python")
    result.assert_outcomes(passed=2)


@pytest.mark.parametrize(
    ("dtype", "matlab_class"),
    [
        (np.float64, "double"),
        (np.float32, "single"),
        (np.int16, "int16"),
        (np.uint32, "uint32"),
        (np.complex64, "single"),
        (np.complex128, "double"),
        (np.bool_, "logical"),
    ],
)
@pytest.mark.parametrize("shape", [(1, 3), (3, 1), (2, 3), (1, 2, 3), (0, 3), (0, 0)])
@pytest.mark.parametrize("sidecars", [False, "output", "both"])
def test_matlab_transport_preserves_dtype_and_dimensions(
    eeglab_matlab_engine, tmp_path, dtype, matlab_class, shape, sidecars
):
    expected = np.arange(np.prod(shape)).reshape(shape).astype(dtype)
    if np.issubdtype(dtype, np.complexfloating):
        expected += (expected + 1) * 1j
    arguments = ("echo", expected, matlab_class, np.array([shape], dtype=float))
    if sidecars:
        actual = _forced_sidecar_call(
            eeglab_matlab_engine,
            tmp_path,
            "eegprep_test_transport",
            *arguments,
            input_limit_bytes=1 if sidecars == "both" else None,
            part_limit_bytes=16,
        )
    else:
        actual = call_matlab(eeglab_matlab_engine, "eegprep_test_transport", *arguments)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)


def test_matlab_generated_struct_cells_logicals_and_complex(eeglab_matlab_engine):
    value = call_matlab(eeglab_matlab_engine, "eegprep_test_transport", "fixture")
    np.testing.assert_array_equal(value["numeric"], [[1, 3, 5], [2, 4, 6]])
    assert value["numeric"].dtype == np.float32
    np.testing.assert_array_equal(value["complex"], [[1 + 3j, 2 - 4j]])
    assert value["complex"].dtype == np.complex64
    np.testing.assert_array_equal(value["logical"], [[True, False]])
    assert value["logical"].dtype == np.bool_
    assert value["empty"].shape == (0, 3)
    assert value["empty"].dtype == np.uint16
    assert value["nested"]["label"] == "nested"
    np.testing.assert_array_equal(value["nested"]["column"], [[7], [8]])
    assert value["nested"]["column"].dtype == np.int16
    assert value["cells"].shape == (2, 2)
    assert value["cells"][0, 0].dtype == np.int8
    assert value["cells"][0, 0].item() == 1
    assert value["cells"][0, 1] == "two"
    assert value["cells"][1, 0].shape == (0, 0)
    assert value["cells"][1, 1]["last"].dtype == np.uint32
    assert value["cells"][1, 1]["last"].item() == 4
    returned = call_matlab(eeglab_matlab_engine, "eegprep_test_transport", "nested", value)
    assert returned["structs"].dtype.names == ("index",)
    assert returned["cell_structs"].dtype == object


def test_matlab_array_sidecars_preserve_nested_cells_and_struct_arrays(eeglab_matlab_engine, tmp_path):
    expected = call_matlab(eeglab_matlab_engine, "eegprep_test_transport", "fixture")
    actual = _forced_sidecar_call(eeglab_matlab_engine, tmp_path, "eegprep_test_transport", "fixture")
    assert len(list(tmp_path.glob("*.mat"))) > 2
    assert_matlab_equal(expected, actual)
    for name in ("numeric", "complex", "logical", "empty"):
        assert actual[name].dtype == expected[name].dtype
    returned = call_matlab(eeglab_matlab_engine, "eegprep_test_transport", "nested", actual)
    assert_matlab_equal(expected, returned)


def test_matlab_sidecars_cover_many_leaves_below_individual_limit(eeglab_matlab_engine, tmp_path):
    expected = np.empty((1, 10), dtype=object)
    for index in range(expected.size):
        expected[0, index] = np.full((1, 1024), index, dtype=np.float32)
    actual = _forced_sidecar_call(eeglab_matlab_engine, tmp_path, "deal", expected, inline_limit_bytes=8192)
    assert len(list(tmp_path.glob("*.mat"))) == 12
    for original, returned in zip(expected.flat, actual.flat, strict=True):
        np.testing.assert_array_equal(returned, original, strict=True)


def test_matlab_input_sidecars_preserve_nested_structs_cells_and_sparse(eeglab_matlab_engine, tmp_path):
    expected = call_matlab(eeglab_matlab_engine, "eegprep_test_transport", "fixture")
    actual = _forced_sidecar_call(
        eeglab_matlab_engine,
        tmp_path,
        "eegprep_test_transport",
        "nested",
        expected,
        input_limit_bytes=1,
        part_limit_bytes=8,
    )
    assert list(tmp_path.glob("input-array-*.mat"))
    assert_matlab_equal(expected, actual)
    sparse = coo_matrix(np.eye(3, dtype=np.float64))
    # A sparse value stays sparse alongside spilled numerical leaves.
    actual_sparse = _forced_sidecar_call(
        eeglab_matlab_engine,
        tmp_path,
        "eegprep_test_transport",
        "echo",
        sparse,
        "double",
        np.array([[3.0, 3.0]]),
        input_limit_bytes=1,
    )
    assert issparse(actual_sparse)
    np.testing.assert_array_equal(actual_sparse.toarray(), sparse.toarray(), strict=True)


def test_matlab_input_sidecars_cover_small_leaves_and_strided_fortran_order(eeglab_matlab_engine, tmp_path):
    expected = np.empty((2, 5), dtype=object)
    for index in np.ndindex(expected.shape):
        expected[index] = (np.arange(512, dtype=np.int16).reshape(16, 32) + sum(index))[::-2, ::3]
    actual = _forced_sidecar_call(
        eeglab_matlab_engine, tmp_path, "deal", expected, input_limit_bytes=2048, part_limit_bytes=32
    )
    assert len(list(tmp_path.glob("input-array-*.mat"))) > expected.size
    for original, returned in zip(expected.flat, actual.flat, strict=True):
        np.testing.assert_array_equal(returned, original, strict=True)


@pytest.mark.slow
@pytest.mark.parametrize("single_array", [False, True])
def test_matlab_inputs_over_four_gib_remain_complete(eeglab_matlab_engine, single_array):
    # Both the nested arguments container and one individual matrix can exceed
    # SciPy MAT5's 32-bit byte count. Do not allocate duplicate Python leaves.
    count = 2**32 + 3 if single_array else 2**28 + 3
    data = np.ones((1, count), dtype=np.uint8)
    data[0, 0], data[0, -1] = 7, 9
    if single_array:
        payload = data
    else:
        payload = np.empty((1, 17), dtype=object)
        payload.fill(data)
    engine = eeglab_matlab_engine
    try:
        call_matlab(engine, "assignin", "base", "transport_large_input", payload, nargout=0)
        if not single_array:
            engine.eval("assert(isequal(size(transport_large_input),[1 17]));", nargout=0)
        engine.eval(
            "for transport_index=1:" + ("1" if single_array else "17") + "; "
            "transport_value=transport_large_input" + ("" if single_array else "{transport_index}") + "; "
            f"assert(isequal(size(transport_value),[1 {count}])); "
            "assert(isa(transport_value,'uint8')); "
            "assert(transport_value(1)==7 && transport_value(end)==9); "
            "for transport_offset=2:2^24:numel(transport_value)-1; "
            "assert(all(transport_value(transport_offset:min(transport_offset+2^24-1,end-1))==1)); end; end;",
            nargout=0,
        )
    finally:
        engine.eval("clear transport_large_input transport_value transport_index transport_offset;", nargout=0)


@pytest.mark.slow
def test_matlab_aggregate_outputs_over_two_gib_remain_complete(eeglab_matlab_engine):
    engine = eeglab_matlab_engine
    # One shared native allocation, but a >2GiB output cell: this reproduces
    # the container limit hit by the original full 18-subject LIMO workflow.
    count = 2**28 + 3
    engine.eval(f"transport_large_payload = repmat({{ones(1,{count},'uint8')}},1,9);", nargout=0)
    try:
        actual = call_matlab(engine, "evalin", "base", "transport_large_payload")
        call_matlab(engine, "assignin", "base", "transport_large_received", actual, nargout=0)
        engine.eval("assert(isequaln(transport_large_received, transport_large_payload));", nargout=0)
    finally:
        engine.eval("clear transport_large_payload transport_large_received;", nargout=0)
    assert actual.shape == (1, 9)
    for value in actual.flat:
        assert value.shape == (1, count)
        assert value.dtype == np.uint8
        assert np.all(value == 1)


def test_matlab_unassigned_cells_keep_zero_by_zero_shape(eeglab_matlab_engine):
    value = call_matlab(eeglab_matlab_engine, "eegprep_test_transport", "unassigned")
    assert value[1, 0].shape == (0, 0)
    assert value[1, 1].shape == (1, 0)
    assert value[0, 1]["first"][0, 1].shape == (0, 0)
    assert value[0, 1]["second"][0, 0].shape == (0, 0)


@pytest.mark.parametrize("sidecars", [False, True])
def test_matlab_table_roundtrip_preserves_values_classes_and_metadata(eeglab_matlab_engine, tmp_path, sidecars):
    engine = eeglab_matlab_engine
    engine.eval(
        "transport_expected = table(uint16([2;1]), single([NaN;3]), "
        "categorical({'second';'first'}), datetime(2020,1,[2;1]), "
        "'VariableNames', {'count','signal','condition','date'}, 'RowNames', {'row2','row1'}); "
        "transport_expected.Properties.Description = 'Original trial information'; "
        "transport_expected.Properties.VariableUnits = {'count','uV','',''}; "
        "transport_expected.Properties.VariableDescriptions = {'Trials','Voltage','Class','Recording date'}; "
        "transport_expected.Properties.DimensionNames = {'Trial','Measurement'}; "
        "transport_expected.Properties.UserData = struct('id', uint32(19), 'label', 'source'); "
        "transport_expected = addprop(transport_expected, 'Provenance', 'table'); "
        "transport_expected.Properties.CustomProperties.Provenance = 'pinned source';",
        nargout=0,
    )
    if sidecars:
        table = _forced_sidecar_call(engine, tmp_path, "evalin", "base", "transport_expected")
    else:
        table = call_matlab(engine, "evalin", "base", "transport_expected")
    assert set(table) == {"eegprep_test_table_mat_v1"}
    assert table["eegprep_test_table_mat_v1"].dtype == np.uint8
    if sidecars:
        returned = _forced_sidecar_call(engine, tmp_path, "sortrows", table, "count", input_limit_bytes=1)
    else:
        returned = call_matlab(engine, "sortrows", table, "count")
    returned = call_matlab(engine, "head", returned, 2.0)
    call_matlab(engine, "assignin", "base", "transport_actual", returned, nargout=0)
    engine.eval(
        "transport_expected = sortrows(transport_expected, 'count'); "
        "assert(isa(transport_actual, 'table')); "
        "assert(isequaln(transport_actual, transport_expected)); "
        "assert(isa(transport_actual.count, 'uint16')); "
        "assert(isa(transport_actual.signal, 'single')); "
        "assert(iscategorical(transport_actual.condition)); "
        "assert(isdatetime(transport_actual.date)); "
        "assert(isequaln(transport_actual.Properties, transport_expected.Properties)); "
        "clear transport_actual transport_expected;",
        nargout=0,
    )


def test_matlab_fieldtrip_chain_preserves_original_trialinfo_table(eeglab_matlab_engine, eeglab_suite_root):
    engine = eeglab_matlab_engine
    dataset = eeglab_suite_root / "eeglab" / "sample_data" / "eeglab_data_epochs_ica.set"
    eeg = call_matlab(engine, "pop_loadset", str(dataset))
    data = call_matlab(engine, "eeglab2fieldtrip", eeg, "preprocessing", "none")
    table = data["trialinfo"]
    assert set(table) == {"eegprep_test_table_mat_v1"}
    call_matlab(engine, "assignin", "base", "transport_expected_trialinfo", table, nargout=0)
    call_matlab(engine, "assignin", "base", "transport_original_data", data, nargout=0)
    # The optional Fileio plugin also ships ft_defaults. Select the full
    # FieldTrip distribution that owns ft_preprocessing, then restore the path.
    original_path = engine.path()
    try:
        fieldtrip = Path(engine.which("ft_preprocessing")).parent
        engine.addpath(str(fieldtrip), "-begin", nargout=0)
        engine.clear("ft_defaults", nargout=0)
        engine.ft_defaults(nargout=0)
        data = call_matlab(engine, "ft_preprocessing", {"reref": "yes", "refchannel": "all"}, data)
        call_matlab(engine, "assignin", "base", "transport_actual_data", data, nargout=0)
        engine.eval(
            "transport_expected_data = ft_preprocessing(struct('reref','yes','refchannel','all'), "
            "transport_original_data); "
            "assert(isequaln(transport_actual_data.trial, transport_expected_data.trial)); "
            "assert(isequal(cellfun(@class, transport_actual_data.trial, 'UniformOutput', false), "
            "cellfun(@class, transport_expected_data.trial, 'UniformOutput', false))); "
            "clear transport_actual_data transport_expected_data transport_original_data;",
            nargout=0,
        )
    finally:
        engine.path(original_path, nargout=0)
    call_matlab(engine, "assignin", "base", "transport_actual_trialinfo", data["trialinfo"], nargout=0)
    engine.eval(
        "assert(istable(transport_actual_trialinfo)); "
        "assert(isequaln(transport_actual_trialinfo, transport_expected_trialinfo)); "
        "assert(isequaln(transport_actual_trialinfo.Properties, transport_expected_trialinfo.Properties)); "
        "clear transport_actual_trialinfo transport_expected_trialinfo;",
        nargout=0,
    )
    assert data["trial"].size == int(eeg["trials"].item())
    for trial in data["trial"].flat:
        assert trial.shape == eeg["data"].shape[:2]


def test_matlab_reference_figures_stay_off_desktop(eeglab_matlab_engine):
    assert call_matlab(eeglab_matlab_engine, "eegprep_test_transport", "figure_visibility") == "off"


@pytest.mark.gui
def test_matlab_tutorial_figures_retain_their_plot_target_between_calls(eeglab_matlab_engine, eeglab_working_directory):
    engine = eeglab_matlab_engine
    visibility = call_matlab(engine, "eegprep_test_transport", "figure_visibility")
    try:
        engine.set(0.0, "DefaultFigureVisible", "on", nargout=0)
        for index in range(4):
            figure = call_matlab(engine, "eegprep_test_tutorial_figure")
            call_matlab(engine, "plot", np.array([[0.0, 1.0]]), np.array([[index, index + 1.0]]), nargout=0)
            axes = call_matlab(engine, "eegprep_test_gui_handle", "gca")
            parent = call_matlab(engine, "eegprep_test_gui_handle", "get", axes, "Parent")
            np.testing.assert_array_equal(parent, figure)
    finally:
        engine.close("all", "force", nargout=0)
        engine.set(0.0, "DefaultFigureVisible", visibility, nargout=0)


def test_matlab_bids_metadata_loaders(eeglab_matlab_engine, eeglab_suite_root):
    """Exercise native JSONio and BIDS setup on the original 18-subject metadata."""
    directory = eeglab_suite_root / "ds002718"
    description = directory / "dataset_description.json"
    expected = json.loads(description.read_text(encoding="utf-8"))
    # Call JSONio directly: newer bids_loadfile versions may use jsondecode.
    parsed = call_matlab(eeglab_matlab_engine, "jsonread", str(description))
    assert parsed["Name"] == expected["Name"]
    loaded = call_matlab(eeglab_matlab_engine, "bids_loadfile", str(description))
    assert loaded["Name"] == expected["Name"]
    participants = directory / "participants.json"
    assert call_matlab(eeglab_matlab_engine, "bids_loadfile", str(participants)) == json.loads(
        participants.read_text(encoding="utf-8")
    )
    table = directory / "participants.tsv"
    loaded = call_matlab(eeglab_matlab_engine, "bids_loadfile", str(table))
    assert loaded.shape == (19, 3)
    np.testing.assert_array_equal(loaded[:, 0], [line.split("\t")[0] for line in table.read_text().splitlines()])


@pytest.mark.parametrize("trials", [1, 3])
def test_matlab_transport_eeg_without_dataset_serializers(eeglab_matlab_engine, trials):
    shape = (2, 5) if trials == 1 else (2, 5, trials)
    events = np.array([("start", 1.0), ("end", float(5 * trials))], dtype=[("type", object), ("latency", object)])
    eeg = {
        "data": np.arange(1, np.prod(shape) + 1, dtype=np.float32).reshape(shape, order="F"),
        "nbchan": 2.0,
        "pnts": 5.0,
        "trials": float(trials),
        "srate": 100.0,
        "xmin": 0.0,
        "xmax": 0.04,
        "times": np.arange(5, dtype=float)[None, :] * 10,
        "event": events,
        "urevent": events.copy(),
        "chanlocs": np.array([("Cz",), ("Pz",)], dtype=[("labels", object)]),
        "icaweights": np.empty((0, 0)),
        "etc": {"subject": "transport", "indices": np.array([[1, 5]], dtype=np.int32)},
    }
    actual = call_matlab(eeglab_matlab_engine, "eegprep_test_transport", "eeg", eeg)
    # A workflow passes returned EEG structs to the next MATLAB call unchanged.
    actual = call_matlab(eeglab_matlab_engine, "eegprep_test_transport", "eeg", actual)
    assert set(actual) == set(eeg)
    assert actual["data"].shape == shape
    assert actual["data"].dtype == np.float32
    np.testing.assert_array_equal(actual["data"], eeg["data"])
    np.testing.assert_array_equal(actual["times"], eeg["times"])
    assert actual["event"].shape == (1, 2)
    assert actual["event"][0, 0]["type"] == "start"
    assert actual["event"][0, 1]["latency"].item() == 5 * trials
    assert actual["urevent"][0, 0]["latency"].item() == 1
    assert actual["icaweights"].shape == (0, 0)
    np.testing.assert_array_equal(actual["etc"]["indices"], [[1, 5]])
    assert actual["etc"]["indices"].dtype == np.int32


def test_matlab_transport_equal_shape_arguments_and_multiple_outputs(eeglab_matlab_engine):
    first = np.array([[1, 2], [3, 4]], dtype=np.int16)
    second = np.array([[5, 6], [7, 8]], dtype=np.float32)
    outputs = call_matlab(eeglab_matlab_engine, "eegprep_test_transport", "pair", first, second, nargout=2)
    assert isinstance(outputs, tuple)
    assert len(outputs) == 2
    for actual, expected in zip(outputs, (first, second), strict=True):
        assert isinstance(actual, np.ndarray)
        assert actual.shape == expected.shape
        assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual, expected)


def test_matlab_transport_paths_and_zero_outputs(eeglab_matlab_engine, tmp_path):
    path = tmp_path / "researcher's data.txt"
    path.write_text("EEGLAB reference path", encoding="utf-8")
    assert call_matlab(eeglab_matlab_engine, "eegprep_test_transport", "path", str(path)) == "EEGLAB reference path"
    assert call_matlab(eeglab_matlab_engine, "eegprep_test_transport", "no_output", nargout=0) is None


def test_matlab_function_errors_are_not_skipped_or_retried_on_python(eeglab_matlab_engine, eeglab_backend):
    # This helper exists only in MATLAB, so a successful call cannot come from Python.
    assert eeglab_matlab_engine.which("eegprep_test_transport")
    assert eeglab_backend("eegprep_test_transport", "fixture")["numeric"].shape == (2, 3)
    matlab_engine: Any = importlib.import_module("matlab.engine")
    with pytest.raises(matlab_engine.MatlabExecutionError, match="Unknown mode"):
        eeglab_backend("eegprep_test_transport", "invalid")


def test_backend_dispatch_executes_real_reference_function(eeglab_backend):
    data = np.array([[1.0, 2.0, 3.0], [4.0, 8.0, 12.0]])
    np.testing.assert_array_equal(eeglab_backend("rmbase", data), [[-1, 0, 1], [-4, 0, 4]])


def test_workflow_directory_contains_relative_outputs(eeglab_working_directory):
    assert Path.cwd() == eeglab_working_directory
    Path("workflow.txt").write_text("isolated original workflow", encoding="utf-8")
    assert (eeglab_working_directory / "workflow.txt").read_text(encoding="utf-8") == "isolated original workflow"


def test_matlab_workflow_directory_matches_python(eeglab_working_directory, eeglab_matlab_engine):
    Path("relative.txt").write_text("original relative path", encoding="utf-8")
    assert Path(eeglab_matlab_engine.pwd()) == eeglab_working_directory
    assert call_matlab(eeglab_matlab_engine, "fileread", "relative.txt") == "original relative path"


def test_matlab_options_file_is_scratch_only(eeglab_matlab_engine, eeglab_backend, eeglab_options_directory):
    home_options = Path.home() / "eeg_options.m"
    original_home = home_options.read_bytes() if home_options.exists() else None
    eeglab_backend("pop_editoptions", option_storedisk=1.0, nargout=0)
    options_file = eeglab_options_directory / "eeg_options.m"
    assert options_file.is_file()
    assert "option_storedisk = 1" in options_file.read_text(encoding="utf-8")
    eeglab_matlab_engine.eval("eeglab_options;", nargout=0)
    assert eeglab_matlab_engine.workspace["option_storedisk"] == 1
    assert eeglab_matlab_engine.workspace["EEGOPTION_PATH"] == str(eeglab_options_directory)
    assert (home_options.read_bytes() if home_options.exists() else None) == original_home
