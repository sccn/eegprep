"""Harness tests for differential fuzzing. MATLAB is not required."""

from __future__ import annotations

from datetime import date
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from hypothesis import strategies as st

from eegprep.functions.adminfunc.eeglabcompat import MatlabWrapper
from eegprep.functions.popfunc.pop_loadset import pop_loadset
from eegprep.plugins.clean_rawdata.clean_asr import clean_asr
from tests.fuzz.harness import (
    EEGSpec,
    FuzzCase,
    MatlabBackend,
    Outcome,
    PythonBackend,
    SideResult,
    Tolerance,
    compare_data,
    data_sha256,
    default_sanity,
    dispatch_matlab,
    dispatch_python,
    draw_cases,
    evaluate_case,
    freeze_params,
    judge,
    load_tolerances,
    materialize_eeg,
    nightly_seed,
    resume_eeg_options,
    run_repro,
    run_session,
    side_from_error,
    suspend_option_single,
    thaw_params,
    tolerance_for,
)
from tests.fuzz.issues import (
    FUNCTION_ROWS,
    DASHBOARD_MARKER,
    GithubIssueTracker,
    MemoryIssueTracker,
    OpenIssue,
    SIGNATURE_MARKER,
    publish_findings,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _spec(n_channels: int = 1) -> EEGSpec:
    return EEGSpec(tuple(range(n_channels)), 0, 8, 1, -1, 0, 0)


def _reref_case(n_channels: int = 1, *, sanity: bool = False) -> FuzzCase:
    return FuzzCase(
        "pop_reref",
        _spec(n_channels),
        freeze_params({"ref": (), "keepref": "off", "exclude": ()}),
        sanity=sanity,
    )


def _ones(case: FuzzCase) -> dict:
    n_channels = len(case.spec.channels)
    return {"data": np.ones((n_channels, 8), dtype=float), "srate": 128.0}


class _Backend:
    def __init__(self, fn):
        self.fn = fn

    def run(self, case, eeg):
        try:
            data = np.asarray(self.fn(case, eeg), dtype=float)
            return SideResult(True, data, float(eeg["srate"]), None, None)
        except Exception as exc:
            return side_from_error(exc)


def _identity(case, eeg):
    del case
    return np.array(eeg["data"], dtype=float, copy=True)


def _diverge_over_two_channels(case, eeg):
    data = np.array(eeg["data"], dtype=float, copy=True)
    if len(case.spec.channels) > 2:
        data += 5
    return data


def _counts():
    return {name: {"examples": 4, "divergences": 0} for name in FUNCTION_ROWS}


def _finding(function: str, max_abs: float) -> Outcome:
    case = _reref_case()
    case = FuzzCase(function, case.spec, case.params)
    outcome = Outcome(case, True, True, "numeric", max_abs, max_abs, "detail")
    outcome.repro_command = f"uv run python -m tests.fuzz reproduce repros/{outcome.signature}.json"
    outcome.confirmed = True
    return outcome


def test_nightly_seed_is_utc_date():
    assert nightly_seed(date(2026, 10, 12)) == 20261012


def test_tolerance_file_is_marked_human_owned():
    text = (REPO_ROOT / "tests" / "fuzz" / "tolerances.toml").read_text()
    agents = (REPO_ROOT / "AGENTS.md").read_text()
    assert "Tolerances are changed only by reviewed human pull requests." in text
    assert "Agents must not loosen" in text
    assert "tests/fuzz/tolerances.toml" in agents
    assert "Agents must not loosen" in agents


def test_compare_respects_tolerance_shape_and_nan_mask():
    tolerance = Tolerance(1e-6, 1e-6)
    oracle = np.array([[1.0, 2.0]])
    matched = compare_data(oracle + 1e-9, oracle, 128.0, 128.0, tolerance)
    assert matched[0] is False

    diverged, error_class, max_abs, _max_rel, _detail = compare_data(oracle + 1e-2, oracle, 128.0, 128.0, tolerance)
    assert diverged is True
    assert error_class == "numeric"
    assert max_abs == pytest.approx(1e-2)

    diverged, error_class, *_rest = compare_data(np.zeros((1, 3)), np.zeros((1, 4)), 1.0, 1.0, tolerance)
    assert diverged is True
    assert error_class == "shape"

    nan_oracle = np.array([[1.0, np.nan]])
    nan_python = np.array([[1.0, 1.0]])
    diverged, error_class, *_rest = compare_data(nan_python, nan_oracle, 1.0, 1.0, tolerance)
    assert error_class == "nan-mask"

    both = judge(
        side_from_error(ValueError("no")),
        side_from_error(ValueError("no")),
        tolerance,
        _reref_case(),
        "abc",
    )
    assert both.diverged is False
    assert both.both_ok is False

    message = "Filter order too low. Minimum required filter order is 212."
    same_text = judge(
        side_from_error(ValueError(message)),
        side_from_error(RuntimeError(message)),
        tolerance,
        _reref_case(),
        "abc",
    )
    assert same_text.diverged is False
    assert same_text.error_class == "both-error"

    different_text = judge(
        side_from_error(ValueError("ff must be non-decreasing.")),
        side_from_error(RuntimeError("Index in position 1 is invalid.")),
        tolerance,
        _reref_case(),
        "abc",
    )
    assert different_text.diverged is True

    yulewalk = judge(
        SideResult(True, np.ones((1, 4)), 32.0, None, None),
        side_from_error(
            RuntimeError(
                "The yulewalk() function was not found and there is no pre-computed spectral filter "
                "for your sampling rate."
            )
        ),
        tolerance,
        _reref_case(),
        "abc",
    )
    assert yulewalk.diverged is False
    assert yulewalk.error_class == "oracle-infrastructure"

    index_error = judge(
        SideResult(True, np.ones((1, 4)), 32.0, None, None),
        side_from_error(RuntimeError("Index in position 1 is invalid.")),
        tolerance,
        _reref_case(),
        "abc",
    )
    assert index_error.diverged is True


def _matrix_reref(data: np.ndarray, ref: int) -> np.ndarray:
    """EEGLAB reref.m's ``refmatrix * data``, which spreads NaNs through explicit zeros."""
    refmatrix = np.eye(data.shape[0])
    refmatrix[:, ref] -= 1.0
    with np.errstate(invalid="ignore"):
        out = refmatrix @ np.asarray(data, dtype=np.float64)
    return np.delete(out, ref, axis=0)


def test_explicit_reref_zero_times_nan_is_not_a_divergence():
    case = FuzzCase(
        "pop_reref",
        EEGSpec(
            channels=tuple(range(11, 28)),
            start=512,
            length=1399,
            decimate=1,
            nan_channel=0,
            nan_start=1396,
            nan_length=3,
        ),
        freeze_params({"ref": (3,), "keepref": "off", "exclude": ()}),
    )
    eeg = materialize_eeg(case)
    python_data = np.asarray(dispatch_python(case, eeg)["data"])
    matrix = _matrix_reref(eeg["data"], 3)
    diverged, error_class, count, *_rest = compare_data(python_data, matrix, 128.0, 128.0, Tolerance(1e-6, 1e-6))
    assert diverged is True
    assert error_class == "nan-mask"
    assert count == 45
    # The .set file stores MATLAB's double refmatrix product as single. That
    # residual is larger than the pop_reref absolute tolerance.
    stored = matrix.astype(np.float32)
    finite = np.isfinite(matrix)
    assert np.max(np.abs(matrix[finite] - stored.astype(np.float64)[finite])) > 1e-6

    outcome = evaluate_case(
        case,
        lambda _case: eeg,
        PythonBackend(),
        _Backend(lambda _case, dataset: _matrix_reref(dataset["data"], 3).astype(np.float32)),
        load_tolerances(),
    )
    assert outcome.diverged is False
    assert outcome.error_class == "match"


def test_explicit_reref_numeric_error_still_diverges_when_nans_match():
    case = FuzzCase(
        "pop_reref",
        EEGSpec(channels=(0, 1, 2, 3), start=0, length=8, decimate=1, nan_channel=0, nan_start=1, nan_length=2),
        freeze_params({"ref": (3,), "keepref": "off", "exclude": ()}),
    )
    eeg = materialize_eeg(case)

    def shifted_oracle(_case, dataset):
        out = _matrix_reref(dataset["data"], 3)
        out[0, 0] += 1.0
        return out

    outcome = evaluate_case(case, lambda _case: eeg, PythonBackend(), _Backend(shifted_oracle), load_tolerances())
    assert outcome.diverged is True


def _asr_case(*, length: int, start: int) -> FuzzCase:
    return FuzzCase(
        "clean_asr",
        EEGSpec(tuple(range(11)), start, length, 1, -1, 0, 0),
        freeze_params({"entry": "clean_asr", "cutoff": 5.0, "ref_maxbadchannels": 0.075, "window_len": None}),
    )


def _asr_all_data(dataset):
    return clean_asr(dataset, cutoff=5.0, ref_maxbadchannels="off")["data"]


def test_clean_asr_short_fit_crash_is_not_a_divergence():
    """EEGLAB's 3-window calibration throws and clean_asr keeps every sample."""
    case = _asr_case(length=256, start=8)
    eeg = materialize_eeg(case)
    windowed = np.asarray(dispatch_python(case, eeg)["data"])
    fallback = np.asarray(_asr_all_data(eeg))
    diverged, error_class, max_abs, *_rest = compare_data(windowed, fallback, 128.0, 128.0, Tolerance(0.0, 1e-6))
    assert diverged is True
    assert error_class == "numeric"
    assert max_abs > 50

    outcome = evaluate_case(
        case,
        lambda _case: eeg,
        PythonBackend(),
        _Backend(lambda _case, dataset: np.asarray(_asr_all_data(dataset), dtype=np.float32)),
        load_tolerances(),
    )
    assert outcome.diverged is False
    assert outcome.error_class == "match"


def test_clean_asr_short_recording_numeric_error_still_diverges():
    case = _asr_case(length=256, start=8)
    eeg = materialize_eeg(case)

    def zeros(_case, dataset):
        return np.zeros_like(np.asarray(_asr_all_data(dataset)))

    outcome = evaluate_case(case, lambda _case: eeg, PythonBackend(), _Backend(zeros), load_tolerances())
    assert outcome.diverged is True


def test_very_short_reference_off_index_error_is_not_a_divergence():
    case = FuzzCase(
        "clean_asr",
        EEGSpec(tuple(range(24)), 90, 62, 2, -1, 0, 0),
        freeze_params({"entry": "clean_asr", "cutoff": 5.0, "ref_maxbadchannels": "off", "window_len": None}),
    )
    eeg = materialize_eeg(case)

    def index_error(_case, _dataset):
        raise RuntimeError("Index in position 1 is invalid. Array indices must be positive integers or logical values.")

    outcome = evaluate_case(case, lambda _case: eeg, PythonBackend(), _Backend(index_error), load_tolerances())
    assert outcome.diverged is False
    assert outcome.error_class == "match"


def test_low_rate_clean_rawdata_returning_the_input_is_not_a_divergence():
    case = FuzzCase(
        "clean_asr",
        EEGSpec((0,), 0, 256, 4, -1, 0, 0),
        freeze_params({"entry": "clean_rawdata", "cutoff": 5.0, "ref_maxbadchannels": "off", "window_len": None}),
    )
    eeg = materialize_eeg(case)
    original = np.asarray(eeg["data"], dtype=np.float64)

    def unchanged(_case, dataset):
        return np.asarray(dataset["data"], dtype=np.float32)

    outcome = evaluate_case(case, lambda _case: eeg, PythonBackend(), _Backend(unchanged), load_tolerances())
    assert outcome.diverged is False
    assert outcome.error_class == "match"
    assert np.array_equal(np.asarray(dispatch_python(case, eeg)["data"]), original)


def test_long_reference_off_index_error_still_diverges():
    case = FuzzCase(
        "clean_asr",
        EEGSpec(tuple(range(11)), 0, 1024, 1, -1, 0, 0),
        freeze_params({"entry": "clean_asr", "cutoff": 5.0, "ref_maxbadchannels": "off", "window_len": None}),
    )
    eeg = materialize_eeg(case)

    def index_error(_case, _dataset):
        raise RuntimeError("Index in position 1 is invalid. Array indices must be positive integers or logical values.")

    outcome = evaluate_case(case, lambda _case: eeg, PythonBackend(), _Backend(index_error), load_tolerances())
    assert outcome.diverged is True


def test_low_rate_clean_rawdata_error_still_diverges():
    case = FuzzCase(
        "clean_asr",
        EEGSpec((0,), 0, 256, 4, -1, 0, 0),
        freeze_params({"entry": "clean_rawdata", "cutoff": 5.0, "ref_maxbadchannels": "off", "window_len": None}),
    )
    eeg = materialize_eeg(case)

    class _Raise:
        def run(self, _case, _eeg):
            return side_from_error(ValueError("ff must be non-decreasing."))

    def unchanged(_case, dataset):
        return np.asarray(dataset["data"], dtype=np.float32)

    outcome = evaluate_case(case, lambda _case: eeg, _Raise(), _Backend(unchanged), load_tolerances())
    assert outcome.diverged is True


def test_clean_asr_long_recording_all_data_oracle_still_diverges():
    case = _asr_case(length=1024, start=0)
    eeg = materialize_eeg(case)
    outcome = evaluate_case(
        case,
        lambda _case: eeg,
        PythonBackend(),
        _Backend(lambda _case, dataset: _asr_all_data(dataset)),
        load_tolerances(),
    )
    assert outcome.diverged is True


def test_materialize_crops_decimates_and_injects_nans():
    case = FuzzCase(
        "pop_reref",
        EEGSpec(channels=(1, 3), start=4, length=5, decimate=2, nan_channel=0, nan_start=1, nan_length=2),
        freeze_params({"ref": (), "keepref": "off", "exclude": ()}),
    )
    eeg = materialize_eeg(case)
    source = np.asarray(pop_loadset(str(REPO_ROOT / "sample_data" / "eeglab_data.set"))["data"])
    expected = np.array(source[np.ix_([1, 3], np.arange(4, 14, 2))], dtype=float)
    expected[0, 1:3] = np.nan
    assert eeg["data"].shape == (2, 5)
    assert eeg["srate"] == 64.0
    assert eeg["nbchan"] == 2
    assert eeg["pnts"] == 5
    assert eeg["event"].size == 0
    np.testing.assert_allclose(eeg["data"], expected, equal_nan=True)
    assert len(data_sha256(eeg["data"])) == 64


def _nightly_case(function: str, spec: dict, params: dict) -> FuzzCase:
    return FuzzCase(
        function,
        EEGSpec(
            channels=tuple(spec["channels"]),
            start=spec["start"],
            length=spec["length"],
            decimate=spec["decimate"],
            nan_channel=spec["nan_channel"],
            nan_start=spec["nan_start"],
            nan_length=spec["nan_length"],
        ),
        freeze_params(params),
    )


def _channels(count: int) -> list[int]:
    return list(range(count))


_REREF_2E5 = _nightly_case(
    "pop_reref",
    {
        "channels": _channels(23),
        "start": 2797,
        "length": 256,
        "decimate": 1,
        "nan_channel": -1,
        "nan_start": 0,
        "nan_length": 0,
    },
    {"exclude": [], "keepref": "off", "ref": []},
)
_REREF_1E5 = _nightly_case(
    "pop_reref",
    {
        "channels": _channels(20),
        "start": 769,
        "length": 256,
        "decimate": 1,
        "nan_channel": -1,
        "nan_start": 0,
        "nan_length": 0,
    },
    {"exclude": [], "keepref": "off", "ref": []},
)
_MINPHASE_NAN = _nightly_case(
    "pop_eegfiltnew",
    {
        "channels": _channels(25),
        "start": 9325,
        "length": 421,
        "decimate": 1,
        "nan_channel": 24,
        "nan_start": 42,
        "nan_length": 4,
    },
    {
        "filtorder": 60,
        "hicutoff": None,
        "locutoff": 7,
        "minphase": True,
        "revfilt": False,
        "usefftfilt": True,
    },
)
_ONE_WINDOW_ASR = _nightly_case(
    "clean_asr",
    {
        "channels": _channels(27),
        "start": 2797,
        "length": 71,
        "decimate": 1,
        "nan_channel": -1,
        "nan_start": 0,
        "nan_length": 0,
    },
    {"cutoff": 5.0, "entry": "clean_asr", "ref_maxbadchannels": 0.2, "window_len": None},
)
_REFERENCE_DATA_ERROR = (
    "No enough reference data, call the function from the command line and "
    "try changing the 'BurstCriterionRefTolerances' parameters"
)


@pytest.mark.parametrize("case", [_REREF_2E5, _REREF_1E5])
def test_average_reref_survey_compares_the_float64_product(case):
    """The filed 1e-5 gap is the float32 product. The survey compares float64."""
    eeg = materialize_eeg(case)
    assert eeg["data"].dtype == np.float64
    wide = np.asarray(dispatch_python(case, eeg)["data"], dtype=np.float64)
    narrow = np.asarray(
        dispatch_python(case, {**eeg, "data": np.asarray(eeg["data"], dtype=np.float32)})["data"],
        dtype=np.float64,
    )
    tolerance = load_tolerances()["pop_reref"]
    diverged, _, max_abs, _, _ = compare_data(narrow, wide, 128.0, 128.0, tolerance)
    assert diverged is True
    assert 1e-6 < max_abs < 1e-3
    assert compare_data(wide, wide, 128.0, 128.0, tolerance)[0] is False


def test_minphase_fftfilt_nan_stays_in_one_block():
    """Order-60 causal fftfilt uses a 256-point block, so 196 of 421 samples are NaN."""
    eeg = materialize_eeg(_MINPHASE_NAN)
    out = np.asarray(dispatch_python(_MINPHASE_NAN, eeg)["data"])
    bad = ~np.isfinite(out)
    assert int(bad.sum()) == 196
    assert int(bad[24].sum()) == 196
    assert int(bad[:24].sum()) == 0
    assert int(np.flatnonzero(bad[24])[-1]) == 195
    params = thaw_params(_MINPHASE_NAN.params)
    params["usefftfilt"] = False
    time_case = FuzzCase(_MINPHASE_NAN.function, _MINPHASE_NAN.spec, freeze_params(params))
    time = np.asarray(dispatch_python(time_case, eeg)["data"])
    np.testing.assert_allclose(out[0], time[0], atol=1e-8, rtol=1e-8)


def test_one_rms_window_is_the_same_failure_as_eeglab():
    eeg = materialize_eeg(_ONE_WINDOW_ASR)

    def oracle_error(_case, _dataset):
        raise RuntimeError(_REFERENCE_DATA_ERROR)

    outcome = evaluate_case(
        _ONE_WINDOW_ASR,
        lambda _case: eeg,
        PythonBackend(),
        _Backend(oracle_error),
        load_tolerances(),
    )
    assert outcome.diverged is False
    assert outcome.error_class == "both-error"
    assert "Fewer than 2 calibration windows" in outcome.detail
    assert "got 1" in outcome.detail
    assert "71 samples" in outcome.detail
    assert "window length 64" in outcome.detail
    assert "No enough reference data" in outcome.detail


class _OptionsFileEngine:
    def __init__(self, path: Path):
        self.path = path

    def eval(self, statement, nargout=0):
        if statement in {"eeglab_options;", "clear functions"}:
            return None
        if statement == "char(option_file)":
            return str(self.path)
        if statement.startswith("pop_editoptions"):
            self.path.write_bytes(b"% EEGLAB full options\noption_single = 0;\n")
            return None
        raise AssertionError(statement)


def test_matlab_backend_restores_eeg_options_bytes(tmp_path):
    path = tmp_path / "eeg_options.m"
    original = b"% local note\noption_single = 1;\n"
    path.write_bytes(original)
    raw = _OptionsFileEngine(path)
    backend = MatlabBackend()
    backend._engine = SimpleNamespace(engine=raw)
    backend._options = suspend_option_single(raw)
    assert path.read_bytes() != original
    backend.close()
    assert path.read_bytes() == original
    backend.close()
    assert path.read_bytes() == original


def test_one_rms_window_message_does_not_hide_a_different_error():
    outcome = judge(
        side_from_error(
            ValueError(
                "ASR calibration failed: Fewer than 2 calibration windows: got 1 from 71 samples with window length 64."
            )
        ),
        side_from_error(RuntimeError("Index in position 1 is invalid.")),
        Tolerance(0.0, 1e-6),
        _ONE_WINDOW_ASR,
        "abc",
    )
    assert outcome.diverged is True


_ONE_CHANNEL_256 = {
    "channels": [0],
    "start": 0,
    "length": 256,
    "decimate": 1,
    "nan_channel": -1,
    "nan_start": 0,
    "nan_length": 0,
}
_FILT_SHAPE_20261016 = _nightly_case(
    "pop_eegfiltnew",
    _ONE_CHANNEL_256,
    {
        "filtorder": 40,
        "hicutoff": 5,
        "locutoff": None,
        "minphase": False,
        "revfilt": False,
        "usefftfilt": False,
    },
)
_REREF_SHAPE_20261016 = _nightly_case(
    "pop_reref",
    _ONE_CHANNEL_256,
    {"exclude": [], "keepref": "off", "ref": []},
)
_RESAMPLE_SHAPE_20261016 = _nightly_case("pop_resample", _ONE_CHANNEL_256, {"freq": 64})
_ASR_SHAPE_20261016 = _nightly_case(
    "clean_asr",
    _ONE_CHANNEL_256,
    {"cutoff": 5.0, "entry": "clean_asr", "ref_maxbadchannels": "off", "window_len": None},
)
_SHAPE_INPUT_SHA256 = "c91ef4bd21a589939fe3a8ddf16c39dda3d20e3fa1d63350302abbcc74e20f57"


def _isolate_eeg_options(engine, directory: Path) -> str:
    """Point EEGLAB's option file at ``directory`` via ``EEGOPTION_PATH``."""
    directory.mkdir()
    original_path = engine.path()
    original_icadefs = str(engine.which("icadefs")).replace("'", "''")
    options_path = str(directory).replace("'", "''")
    (directory / "icadefs.m").write_text(
        f"run('{original_icadefs}');\nEEGOPTION_PATH = '{options_path}';\n",
        encoding="utf-8",
    )
    engine.addpath(str(directory), "-begin", nargout=0)
    engine.eval("clear icadefs eeg_options; icadefs;", nargout=0)
    return original_path


def test_replay_20261016_fuzz_mismatches(eeglab_matlab_engine, tmp_path):
    """Replay the 20261016 shape signatures with no eeg_options.m, as CI does.

    Run with --eeglab-backend=matlab. pop_editoptions then writes a fresh options
    file from eeg_optionsbackup.m, which saves data inside the .set.
    """
    home = Path.home() / "eeg_options.m"
    home_bytes = home.read_bytes() if home.is_file() else None
    options_dir = tmp_path / "eeg_options"
    original_path = _isolate_eeg_options(eeglab_matlab_engine, options_dir)
    engine = MatlabWrapper(eeglab_matlab_engine)
    try:
        eeglab_matlab_engine.eval("clear functions; eeglab_options;", nargout=0)
        option_file = Path(str(eeglab_matlab_engine.eval("char(option_file)", nargout=1)).strip()).expanduser()
        print(f"option_file={option_file}")
        assert option_file.resolve().is_relative_to(options_dir.resolve())
        assert not option_file.is_file()
        snapshot = suspend_option_single(eeglab_matlab_engine)
        try:
            eeglab_matlab_engine.eval("eeglab_options;", nargout=0)
            savetwo = int(eeglab_matlab_engine.eval("option_savetwofiles", nargout=1))
            print(f"option_savetwofiles={savetwo}")
            assert savetwo == 0
            tolerances = load_tolerances()
            for name, case in (
                ("pop_eegfiltnew", _FILT_SHAPE_20261016),
                ("pop_reref", _REREF_SHAPE_20261016),
                ("pop_resample", _RESAMPLE_SHAPE_20261016),
                ("clean_asr", _ASR_SHAPE_20261016),
            ):
                eeg = materialize_eeg(case)
                digest = data_sha256(np.asarray(eeg["data"]))
                assert digest == _SHAPE_INPUT_SHA256
                python_eeg = dispatch_python(case, copy.deepcopy(eeg))
                oracle_eeg = dispatch_matlab(engine, case, copy.deepcopy(eeg))
                python = np.asarray(python_eeg["data"], dtype=np.float64)
                oracle = np.asarray(oracle_eeg["data"], dtype=np.float64)
                diverged, error_class, max_abs, max_rel, detail = compare_data(
                    python,
                    oracle,
                    float(python_eeg["srate"]),
                    float(oracle_eeg["srate"]),
                    tolerance_for(case.function, tolerances),
                )
                print(
                    f"{name} diverged={diverged} class={error_class} abs={max_abs} rel={max_rel} "
                    f"python_shape={python.shape} oracle_shape={oracle.shape} {detail}"
                )
                assert diverged is False, detail
        finally:
            resume_eeg_options(eeglab_matlab_engine, snapshot)
    finally:
        eeglab_matlab_engine.path(original_path, nargout=0)
        eeglab_matlab_engine.eval("clear icadefs eeg_options; eeglab_options;", nargout=0)
        assert (home.read_bytes() if home.is_file() else None) == home_bytes


def test_replay_20261015_fuzz_mismatches(eeglab_matlab_engine):
    """Replay the 20261015 filings against EEGLAB. Run with --eeglab-backend=matlab -s."""
    engine = MatlabWrapper(eeglab_matlab_engine)
    eeglab_matlab_engine.eval("eeglab_options;", nargout=0)
    before = eeglab_matlab_engine.eval("option_single", nargout=1)
    print(f"option_single={before}")
    snapshot = suspend_option_single(eeglab_matlab_engine)
    eeglab_matlab_engine.eval("eeglab_options;", nargout=0)
    option_single = eeglab_matlab_engine.eval("option_single", nargout=1)
    print(f"option_single={option_single}")
    assert int(option_single) == 0
    tolerances = load_tolerances()
    path, original = snapshot

    try:
        _replay_cases(engine, tolerances)
    finally:
        resume_eeg_options(eeglab_matlab_engine, snapshot)
    if path is None or original is None:
        pytest.fail(f"eeglab_options did not load an options file to restore (path={path})")
    print(f"eeg_options path={path} bytes={len(original)}")
    assert path.read_bytes() == original


def _replay_cases(engine, tolerances):
    for name, case in (
        ("pop_reref 2e-5", _REREF_2E5),
        ("pop_reref 1e-5", _REREF_1E5),
        ("pop_eegfiltnew", _MINPHASE_NAN),
    ):
        eeg = materialize_eeg(case)
        python = np.asarray(dispatch_python(case, copy.deepcopy(eeg))["data"], dtype=np.float64)
        oracle = np.asarray(dispatch_matlab(engine, case, copy.deepcopy(eeg))["data"], dtype=np.float64)
        diverged, error_class, max_abs, max_rel, detail = compare_data(
            python,
            oracle,
            float(eeg["srate"]),
            float(eeg["srate"]),
            tolerance_for(case.function, tolerances),
        )
        python_bad = int(np.count_nonzero(~np.isfinite(python)))
        oracle_bad = int(np.count_nonzero(~np.isfinite(oracle)))
        print(
            f"{name} diverged={diverged} class={error_class} abs={max_abs} rel={max_rel} "
            f"python_nonfinite={python_bad} oracle_nonfinite={oracle_bad} {detail}"
        )
        assert diverged is False, detail

    eeg = materialize_eeg(_ONE_WINDOW_ASR)
    python_message = ""
    oracle_message = ""
    try:
        dispatch_python(_ONE_WINDOW_ASR, copy.deepcopy(eeg))
    except Exception as exc:
        python_message = str(exc).splitlines()[0]
    try:
        dispatch_matlab(engine, _ONE_WINDOW_ASR, copy.deepcopy(eeg))
    except Exception as exc:
        oracle_message = str(exc).splitlines()[0]
    print(f"clean_asr python={python_message}")
    print(f"clean_asr oracle={oracle_message}")
    assert "Fewer than 2 calibration windows" in python_message
    assert "got 1" in python_message
    assert "71 samples" in python_message
    assert "window length 64" in python_message
    assert "No enough reference data" in oracle_message


def test_reref_sanity_signal_matches_across_float32_and_float64():
    case = next(item for item in default_sanity() if item.function == "pop_reref")
    data = np.asarray(materialize_eeg(case)["data"], dtype=np.float64)
    ref64 = data - data.mean(axis=0, keepdims=True)
    narrow = np.asarray(data, dtype=np.float32)
    ref32 = (narrow - narrow.mean(axis=0, keepdims=True)).astype(np.float64)
    assert np.max(np.abs(ref64 - ref32)) <= 1e-6


def test_same_seed_draws_the_same_cases():
    first = draw_cases("pop_reref", seed_value=20261012, examples=4)
    second = draw_cases("pop_reref", seed_value=20261012, examples=4)
    other = draw_cases("pop_reref", seed_value=20261013, examples=4)
    assert first == second
    assert first != other
    assert all(case.function == "pop_reref" for case in first)


def test_shrink_reduces_channel_count_to_the_failure_boundary(tmp_path):
    strategy = st.integers(1, 8).map(lambda n: _reref_case(n))
    result = run_session(
        seed_value=7,
        output_dir=tmp_path,
        examples=20,
        shrink_budget=40,
        max_shrinks=2,
        python_backend=_Backend(_identity),
        oracle_backend=_Backend(_diverge_over_two_channels),
        strategies={"pop_reref": strategy},
        sanity=[_reref_case(1, sanity=True)],
        materialize=_ones,
    )
    assert result.harness_ok
    assert result.functions["pop_reref"]["divergences"] >= 1
    assert result.divergences
    assert len(result.divergences[0].case.spec.channels) == 3
    repro = json.loads(next((tmp_path / "repros").glob("*.json")).read_text())
    assert repro["seed"] == 7
    assert repro["command"].startswith("uv run python -m tests.fuzz reproduce ")


def test_sanity_failure_does_not_file_divergence_issues(tmp_path):
    tracker = MemoryIssueTracker()

    def explode(case, eeg):
        del case, eeg
        raise RuntimeError("bridge down")

    result = run_session(
        seed_value=20261012,
        output_dir=tmp_path,
        examples=3,
        python_backend=_Backend(_identity),
        oracle_backend=_Backend(explode),
        strategies={"pop_reref": st.just(_reref_case(2))},
        sanity=[_reref_case(1, sanity=True)],
        materialize=_ones,
        tracker=tracker,
        run_url="https://github.com/sccn/eegprep/actions/runs/1",
    )
    assert result.harness_ok is False
    assert result.created == []
    assert result.divergences == []
    assert all(not issue.title.startswith("[fuzz]") for issue in tracker.issues)
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["harness_ok"] is False
    assert metrics["seed"] == 20261012
    assert DASHBOARD_MARKER in tracker.issues[-1].body


def test_publish_dedupes_and_caps_new_issues():
    existing = _finding("pop_reref", 0.02)
    tracker = MemoryIssueTracker(
        [OpenIssue(4, "old", SIGNATURE_MARKER.format(signature=existing.signature) + "\n", ["fuzz"])]
    )
    findings = [
        existing,
        _finding("pop_eegfiltnew", 0.2),
        _finding("pop_resample", 2.0),
        _finding("clean_asr", 20.0),
        _finding("pop_reref", 3.0),
    ]
    # The fifth new signature shares a function with the existing one but not a magnitude.
    functions = _counts()
    for finding in findings:
        functions[finding.report]["divergences"] += 1
    result = publish_findings(
        seed=20261012,
        functions=functions,
        findings=findings,
        tracker=tracker,
        cap=3,
        run_url="https://github.com/sccn/eegprep/actions/runs/9",
        harness_ok=True,
    )
    assert result.commented == [4]
    assert len(result.created) == 3
    assert len(result.unfiled) == 1
    created = [issue for issue in tracker.issues if issue.number in result.created]
    assert all(issue.labels == ["fuzz", "fuzz-unconfirmed"] for issue in created)
    assert all("fuzz-signature:" in issue.body for issue in created)
    assert all("uv run python -m tests.fuzz reproduce" in issue.body for issue in created)
    assert "https://github.com/sccn/eegprep/actions/runs/9#artifacts" in created[0].body
    dashboard = next(issue for issue in tracker.issues if DASHBOARD_MARKER in issue.body)
    assert "pop_reref=" in dashboard.body
    again = publish_findings(
        seed=20261013,
        functions=functions,
        findings=[],
        tracker=tracker,
        cap=3,
        run_url="",
        harness_ok=True,
    )
    assert again.created == []
    body = next(issue for issue in tracker.issues if DASHBOARD_MARKER in issue.body).body
    assert "- 20261012:" in body
    assert "- 20261013:" in body


def test_github_tracker_sends_signature_and_labels():
    calls = []

    def runner(args):
        calls.append(list(args))
        if "list" in args:
            return "[]"
        if "create" in args:
            return "https://github.com/sccn/eegprep/issues/15\n"
        if "view" in args:
            return json.dumps({"id": "NODE"})
        return ""

    tracker = GithubIssueTracker("sccn/eegprep", "https://github.com/sccn/eegprep/actions/runs/3", runner)
    finding = _finding("pop_resample", 0.2)
    publish_findings(
        seed=20261012,
        functions=_counts(),
        findings=[finding],
        tracker=tracker,
        cap=3,
        run_url="https://github.com/sccn/eegprep/actions/runs/3",
        harness_ok=True,
    )
    bodies = []
    for args in calls:
        if "--body-file" in args:
            bodies.append(Path(args[args.index("--body-file") + 1]).read_text())
    assert any(f"fuzz-signature: {finding.signature}" in body for body in bodies)
    create = next(
        args for args in calls if "create" in args and "--title" in args and "[fuzz]" in args[args.index("--title") + 1]
    )
    assert create.count("--label") == 2
    assert "fuzz-unconfirmed" in create


def test_reproduce_reruns_and_rejects_a_changed_input_hash(tmp_path):
    python = _Backend(_identity)
    oracle = _Backend(_diverge_over_two_channels)
    result = run_session(
        seed_value=3,
        output_dir=tmp_path,
        examples=1,
        shrink_budget=2,
        python_backend=python,
        oracle_backend=oracle,
        strategies={"pop_reref": st.just(_reref_case(4))},
        sanity=[_reref_case(1, sanity=True)],
        materialize=_ones,
    )
    assert result.harness_ok
    repro = next((tmp_path / "repros").glob("*.json"))
    outcome = run_repro(repro, python_backend=python, oracle_backend=oracle, materialize=_ones)
    assert outcome.diverged is True
    payload = json.loads(repro.read_text())
    payload["input_sha256"] = "0" * 64
    repro.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="input hash"):
        run_repro(repro, python_backend=python, oracle_backend=oracle, materialize=_ones)


def test_cli_fake_match_on_sample_data_passes(tmp_path):
    from tests.fuzz.__main__ import main

    code = main(
        [
            "run",
            "--oracle",
            "fake",
            "--fake-mode",
            "match",
            "--examples",
            "1",
            "--shrink-budget",
            "1",
            "--output",
            str(tmp_path),
            "--seed",
            "20261012",
        ]
    )
    assert code == 0
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["harness_ok"] is True
    assert metrics["seed"] == 20261012
    assert metrics["functions"]["pop_reref"]["divergences"] == 0
    assert metrics["functions"]["clean_asr"]["examples"] + metrics["functions"]["pop_clean_rawdata"]["examples"] == 1


def test_workflow_schedules_the_matlab_batch_job():
    text = (REPO_ROOT / ".github" / "workflows" / "fuzz-nightly.yml").read_text()
    assert 'cron: "23 10 * * *"' in text
    assert "timeout-minutes: 60" in text
    assert "issues: write" in text
    assert "MLM_LICENSE_TOKEN" in text
    assert "matlab-batch" in text
    assert "eegprep_batch_engine_serve" in text
