"""MATLAB parity and properties of the NumPy ERP-image engine.

The committed fixture ``tests/matlab/erpimage_engine_fixtures.mat`` holds
EEGLAB ``erpimage`` outputs from MATLAB R2025b for every scenario below, so CI
compares without MATLAB; ``TestErpimageMatlabParity`` regenerates it live when
MATLAB is available.
"""

from __future__ import annotations

import ast
from functools import cache
import logging
from pathlib import Path
import subprocess
import sys
import warnings

from hypothesis import given, settings, strategies as st
import numpy as np
import pytest
from scipy import signal

from eegprep.functions.adminfunc.eeglabcompat import _resolve_eeglab_root, get_eeglab
from eegprep.functions.miscfunc.gauss import gauss
from eegprep.functions.popfunc.pop_loadset import pop_loadset
from eegprep.functions.sigprocfunc import erpimage_engine as engine
from eegprep.functions.sigprocfunc.erpimage import butterworth_filter
from eegprep.functions.sigprocfunc.movav import movav
from eegprep.functions.sigprocfunc.nan_mean import nan_mean
from eegprep.functions.sigprocfunc.rmbase import rmbase
from eegprep.functions.timefreqfunc.timewarp import timewarp
from tests.eeglab_tests import load_matlab_test_fixture
from tests.fixtures import matlab_engine_available

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "tests" / "matlab" / "erpimage_engine_fixtures.mat"
SAMPLE_SET = REPO_ROOT / "sample_data" / "eeglab_data_epochs_ica.set"
RTOL = 1e-12
OUTPUTS = (
    "outdata",
    "outvar",
    "outtrials",
    "limits",
    "erp",
    "amps",
    "cohers",
    "cohsig",
    "ampsig",
    "outamps",
    "phsangls",
    "phsamp",
    "erpsig",
)
# Outputs in data units are compared within RTOL of the input data's largest magnitude.
DATA_DOMAIN = {"outdata", "erp", "erpsig"}
NON_SCENARIO_KEYS = {"datasets", "seed", "matlab_version"}
ENGINE_IMPORTS = {"__future__", "ast", "dataclasses", "logging", "typing", "numpy"}
SRATE = 128.0


def _load(path: Path) -> dict:
    return load_matlab_test_fixture(path)


FIXTURE_DATA = _load(FIXTURE)
SCENARIOS = sorted(key for key in FIXTURE_DATA if key not in NON_SCENARIO_KEYS)


def _text(value: np.ndarray) -> str:
    return str(value.ravel()[0]) if value.size else ""


def _matlab_value(value: np.ndarray):
    """MATLAB cell/char/numeric fixture value as an EEGLAB option token."""
    if value.dtype.kind in "US":
        return _text(value)
    if value.dtype == object:
        return tuple(_matlab_value(item) for item in value.ravel())
    return np.asarray(value, dtype=float)


def _field(fixture: dict, name: str, field: str) -> np.ndarray:
    value = fixture[name][0, 0][field]
    if value.dtype.kind in "US" and _text(value).startswith("same:"):
        other, other_field = _text(value)[5:].split(".")
        return _field(fixture, other, other_field)
    return value


@cache
def _sample_data() -> np.ndarray:
    return np.asarray(pop_loadset(str(SAMPLE_SET))["data"], dtype=float)


def _sample_slice(fixture: dict) -> np.ndarray:
    datasets = fixture["datasets"][0, 0]
    channel = int(datasets["real_channel"].ravel()[0]) - 1
    samples = datasets["real_samples"].ravel().astype(int) - 1
    epochs = datasets["real_epochs"].ravel().astype(int) - 1
    return _sample_data()[channel][np.ix_(samples, epochs)]


def _noise(frames: int = 64, trials: int = 12, seed: int = 11) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian noise trials and their latencies (-100 ms onward at 128 Hz)."""
    return np.random.default_rng(seed).normal(size=(frames, trials)), -100 + np.arange(frames) * 1000 / SRATE


def _scenario_inputs(fixture: dict, name: str) -> dict:
    record = fixture[name][0, 0]
    datasets = fixture["datasets"][0, 0]
    data_id = _text(record["data_id"])
    data = {
        "synth": lambda: datasets["synth"],
        "synth_vector": lambda: datasets["synth"].reshape(1, -1, order="F"),
        "synth_nan": lambda: datasets["synth_nan"],
        "real": lambda: _sample_slice(fixture),
    }[data_id]()
    options = record["options"]
    tokens = list(_matlab_value(options)) if options.size else []
    return {
        "data": np.asarray(data, dtype=float),
        "sortvar": record["sortvar"],
        "times": None if bool(record["times_omitted"].ravel()[0]) else record["times"],
        "avewidth": float(record["avewidth"].ravel()[0]),
        "decimate": float(record["decimate"].ravel()[0]),
        "options": engine.parse_erpimage_options(tokens),
        "seed": int(fixture["seed"].ravel()[0]),
    }


def _compute(fixture: dict, name: str, *, plotted: bool) -> engine.ErpimageComputation:
    inputs = _scenario_inputs(fixture, name)
    return engine.compute_erpimage(
        inputs["data"],
        inputs["sortvar"],
        inputs["times"],
        inputs["avewidth"],
        inputs["decimate"],
        inputs["options"],
        rng=np.random.RandomState(inputs["seed"]),
        plotted=plotted,
        trial_filter=butterworth_filter,
    )


def _expected_outvar(value: np.ndarray):
    if value.dtype == object:
        return (
            np.asarray(value.ravel()[0], dtype=float).ravel(),
            tuple(np.ravel(item) for item in value.ravel()[1].ravel()),
        )
    return value


def _assert_close(label: str, expected, actual, scale: float) -> None:
    if isinstance(expected, tuple):
        assert isinstance(actual, tuple), label
        _assert_close(label, expected[0], actual[0], scale)
        assert len(expected[1]) == len(actual[1]), label
        for expected_item, actual_item in zip(expected[1], actual[1], strict=True):
            _assert_close(label, expected_item, actual_item, scale)
        return
    expected = np.asarray(expected)
    if expected.size == 0:
        assert actual is None, f"{label}: expected no output, got {actual!r}"
        return
    assert actual is not None, f"{label}: missing output"
    expected = expected.astype(complex if np.iscomplexobj(expected) else float).ravel()
    actual = np.asarray(actual).ravel()
    assert expected.size == actual.size, f"{label}: {actual.size} values, expected {expected.size}"
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected), err_msg=label)
    finite = ~np.isnan(expected)
    tolerance = RTOL * max(float(np.max(np.abs(expected[finite]), initial=0.0)), scale)
    np.testing.assert_allclose(actual[finite], expected[finite], rtol=0, atol=tolerance, err_msg=label)


def _assert_scenario(fixture: dict, name: str) -> None:
    record = fixture[name][0, 0]
    inputs = _scenario_inputs(fixture, name)
    data_scale = float(np.nanmax(np.abs(inputs["data"])))
    for mode, prefix, fields in (("noshow", "", OUTPUTS), ("plot", "plot_", ("limits", "erpsig"))):
        if prefix + "limits" not in record.dtype.names:
            continue
        result = _compute(fixture, name, plotted=mode == "plot")
        for field in fields:
            # erpalpha thresholds use unbiased signs (sccn/eeglab#976); test_erpalpha_* compares
            # them with MATLAB given MATLAB's own sign draws.
            if name == "erpalpha" and (field == "erpsig" or (mode == "plot" and field == "limits")):
                continue
            expected = _field(fixture, name, prefix + field)
            if field == "outvar":
                expected = _expected_outvar(expected)
            scale = data_scale if field in DATA_DOMAIN else 0.0
            _assert_close(f"{name} {mode} {field}", expected, getattr(result, field), scale)
        if mode == "noshow":
            sortvar = np.asarray(inputs["sortvar"], dtype=float).ravel()
            trials = inputs["data"].shape[1] if not sortvar.size else sortvar.size
            kept = np.arange(trials) if not sortvar.size else np.flatnonzero(~np.isnan(sortvar))
            expected_order = kept[np.asarray(_field(fixture, name, "sortidx"), dtype=int).ravel() - 1]
            np.testing.assert_array_equal(result.sortidx, expected_order, err_msg=f"{name} sortidx")


@pytest.mark.parity
@pytest.mark.parametrize("name", SCENARIOS)
def test_engine_matches_matlab_fixture(name: str) -> None:
    _assert_scenario(FIXTURE_DATA, name)


def test_fixture_stays_small() -> None:
    assert FIXTURE.stat().st_size < 200_000


def test_erpalpha_threshold_matches_matlab_given_its_sign_draws() -> None:
    record = FIXTURE_DATA["erpalpha"][0, 0]
    result = _compute(FIXTURE_DATA, "erpalpha", plotted=False)
    signs = np.asarray(record["matlab_signs"], dtype=float)
    erp, threshold = engine._permutation_threshold(result.erp_data, 0.05, signs)
    _assert_close("erpalpha erp", record["erp"], erp, float(np.max(np.abs(result.erp_data))))
    _assert_close("erpalpha erpsig", record["erpsig"], threshold, float(np.max(np.abs(result.erp_data))))
    plotted = _field(FIXTURE_DATA, "erpalpha", "plot_limits").ravel()
    erpsig = np.vstack([threshold, -threshold])
    assert engine._max_erp_limit(erp[np.newaxis, :], None, erpsig) == plotted[3]
    assert engine._min_erp_limit(erp[np.newaxis, :], None, erpsig) == plotted[2]


def test_erpalpha_band_matches_the_unbiased_sign_flip_null() -> None:
    # On constant data the permutation statistic is |mean(signs)|, whose 90th percentile is
    # about 1.645/sqrt(40) = 0.26 for unbiased signs; EEGLAB's biased draw puts it near 0.5.
    options = engine.parse_erpimage_options(["erp", "erpalpha", 0.05])
    result = engine.compute_erpimage(np.ones((8, 40)), None, None, 1, 1, options, rng=3, plotted=False)
    assert np.all((result.erpsig > 0.15) & (result.erpsig < 0.35))


@pytest.mark.parametrize(("alpha", "count"), [(0.1, 500), (0.05, 500), (0.001, 3000)])
def test_permutation_count_follows_eeglab(alpha, count) -> None:
    assert engine._permutation_count(alpha) == count


def test_erpalpha_emits_no_runtime_warnings() -> None:
    data, _times = _noise(30, 24)
    options = engine.parse_erpimage_options(["erp", "erpalpha", 0.05])
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        engine.compute_erpimage(data, None, None, 1, 1, options, rng=0, plotted=False)


def test_erpalpha_signs_are_unbiased_unlike_eeglab() -> None:
    eeglab_signs = np.asarray(FIXTURE_DATA["erpalpha"][0, 0]["matlab_signs"], dtype=float)
    # EEGLAB's sign(randn - 0.5) is +1 with probability 0.31 (sccn/eeglab#976).
    assert np.mean(eeglab_signs == 1) < 0.35
    signs = engine._permutation_signs(np.random.default_rng(976), 2000, 50)
    assert set(np.unique(signs)) <= {-1.0, 1.0}
    assert abs(np.mean(signs == 1) - 0.5) < 0.01


def test_erpalpha_is_reproducible_with_a_seeded_rng() -> None:
    data = FIXTURE_DATA["datasets"][0, 0]["synth"]
    options = engine.parse_erpimage_options(["erp", "erpalpha", 0.05])
    first, second, other = (
        engine.compute_erpimage(data, None, None, 1, 1, options, rng=np.random.default_rng(seed), plotted=False)
        for seed in (5, 5, 6)
    )
    seeded = engine.compute_erpimage(data, None, None, 1, 1, options, rng=5, plotted=False)
    assert first.erpsig is not None
    np.testing.assert_array_equal(first.erpsig, second.erpsig)
    np.testing.assert_array_equal(first.erpsig, seeded.erpsig)
    assert not np.array_equal(first.erpsig, other.erpsig)
    assert first.erpsig.shape == (data.shape[0],)
    plotted = engine.compute_erpimage(data, None, None, 1, 1, options, rng=np.random.default_rng(5))
    np.testing.assert_array_equal(plotted.erpsig, np.vstack([first.erpsig, -first.erpsig]))
    with pytest.raises(TypeError, match="rng must be"):
        engine.compute_erpimage(data, None, None, 1, 1, options, rng="seed")


def test_engine_imports_only_numpy_and_the_standard_library() -> None:
    path = Path(engine.__file__)
    imported = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add(str(node.module).split(".")[0])
    assert imported <= ENGINE_IMPORTS
    # Load the file on its own, as the browser build includes it, and run each kind of computation.
    code = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('erpimage_engine', {str(path)!r})\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "sys.modules['erpimage_engine'] = module\n"
        "spec.loader.exec_module(module)\n"
        "import numpy as np\n"
        "data = np.random.default_rng(0).normal(size=(40, 10))\n"
        "for tokens in (\n"
        "    ['erp', 'phasesort', [100, 0, 10], 'coher', [10, 10, 0.05]],\n"
        "    ['erp', 'erpalpha', 0.05, 'ampsort', [100, 0, 10, 20], 'renorm', '2*x+1'],\n"
        "    ['plotamps', 'coher', [10, 10, 0.05]],\n"
        "    ['erp', 'timewarp', (np.full((10, 1), 100.0),)],\n"
        "):\n"
        "    module.compute_erpimage(data, None, np.arange(40) * 8.0, 3, 1, module.parse_erpimage_options(tokens), rng=0)\n"
        "module.welch_psd(data.ravel(), 40, 125.0)\n"
        "heavy = {'matplotlib', 'scipy', 'mne', 'eegprep'}\n"
        "print(sorted(name for name in sys.modules if name.split('.')[0] in heavy))\n"
    )
    completed = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "[]"


@settings(max_examples=60, deadline=None)
@given(
    trials=st.integers(4, 40),
    rows=st.integers(1, 3),
    width_fraction=st.floats(0.0, 1.0),
    advance=st.sampled_from([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 7.0]),
    gaussian=st.booleans(),
    seed=st.integers(0, 2**16),
)
def test_moving_average_matches_the_general_movav_port(trials, rows, width_fraction, advance, gaussian, seed) -> None:
    values = np.random.default_rng(seed).normal(size=(rows, trials))
    if gaussian:
        sdev = 1 + width_fraction * max((trials - 1) / 6 - 1, 0)
        width, weights = engine._smoothing_window(sdev, "gaussian", trials)
    else:
        width, weights = engine._smoothing_window(float(1 + round(width_fraction * (trials - 1))), "boxcar", trials)
    expected, expected_x = movav(values, np.arange(1, trials + 1), width, advance, None, None, weights)
    actual, actual_x = engine._moving_average(values, width, advance, weights)
    np.testing.assert_array_equal(actual_x, expected_x)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=RTOL * np.max(np.abs(values)))


@settings(max_examples=40, deadline=None)
@given(frames=st.integers(64, 300), trials=st.integers(1, 12), seed=st.integers(0, 2**16))
def test_welch_psd_matches_scipy(frames, trials, seed) -> None:
    samples = np.random.default_rng(seed).normal(size=frames * trials)
    psd, freqs = engine.welch_psd(samples, frames, 250.0)
    nfft = max(1024, 2 ** int(np.ceil(np.log2(frames))))
    expected_freqs, expected = signal.welch(
        samples,
        250.0,
        window=signal.windows.hamming(frames, sym=True),
        nperseg=frames,
        noverlap=0,
        nfft=nfft,
        detrend=False,
    )
    np.testing.assert_allclose(freqs, expected_freqs, rtol=1e-12)
    np.testing.assert_allclose(psd, expected, rtol=1e-10, atol=0)


@settings(max_examples=30, deadline=None)
@given(frames=st.integers(40, 160), trials=st.integers(2, 20), freq=st.floats(6.0, 30.0), seed=st.integers(0, 2**16))
def test_wavelet_transform_matches_a_direct_per_frame_projection(frames, trials, freq, seed) -> None:
    values = np.random.default_rng(seed).normal(size=(frames, trials))
    srate, cycles = 128.0, 3.0
    coswin, sinwin = engine._wavelet(srate, freq, cycles)
    length = coswin.size
    if length > frames:
        return
    half = length // 2
    extended = np.vstack([values[half::-1], values, values[np.arange(frames - 1, frames - 1 - (length - half), -1)]])
    expected = np.empty((frames, trials), dtype=complex)
    for frame in range(frames):
        epoch = extended[frame : frame + length] - extended[frame : frame + length].mean(axis=0)
        expected[frame] = np.dot(coswin, epoch) + 1j * np.dot(sinwin, epoch)
    actual = engine._wavelet_transform(extended, frames, coswin, sinwin)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12 * np.max(np.abs(expected)))


def test_sortidx_indexes_the_original_trials_after_nan_removal(caplog) -> None:
    data = np.arange(60.0).reshape(10, 6)
    with caplog.at_level(logging.WARNING, logger=engine.logger.name):
        result = engine.compute_erpimage(data, [3.0, np.nan, 1.0, 2.0, 1.0, np.nan], None, 1, 1, plotted=False)
    np.testing.assert_array_equal(result.sortidx, [2, 4, 3, 0])  # stable: equal sortvar keeps input order
    np.testing.assert_array_equal(result.outdata, data[:, [2, 4, 3, 0]])
    assert "Removing 2 trials with NaN sortvar values (input trials [1, 5], 0-based)" in caplog.text


def test_noplot_hides_only_the_sortvar_trace() -> None:
    data = FIXTURE_DATA["datasets"][0, 0]["synth"]
    sortvar = np.arange(data.shape[1], dtype=float)[::-1]
    shown = engine.compute_erpimage(data, sortvar, None, 3, 1, engine.parse_erpimage_options(["erp"]))
    hidden = engine.compute_erpimage(data, sortvar, None, 3, 1, engine.parse_erpimage_options(["erp", "noplot", "on"]))
    assert shown.show_sortvar and not hidden.show_sortvar
    np.testing.assert_array_equal(shown.outdata, hidden.outdata)
    np.testing.assert_array_equal(shown.limits, hidden.limits)


def test_signif_with_coher_applies_the_given_thresholds() -> None:
    # MATLAB R2025b errors here (vector operand to ||); the help describes these thresholds.
    data = FIXTURE_DATA["datasets"][0, 0]["synth"]
    times = -125 + np.arange(data.shape[0]) * 1000 / 128
    options = engine.parse_erpimage_options(["coher", [10, 10, 0.05], "signif", [-2, 2, 0.4]])
    result = engine.compute_erpimage(data, None, times, 3, 1, options, rng=np.random.default_rng(0), plotted=False)
    assert result.cohsig == 0.4
    np.testing.assert_allclose(result.ampsig, np.array([-2.0, 2.0]) - result.baseamp)


@pytest.mark.timeout(30)
def test_zero_data_keeps_eeglab_limits_without_hanging() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)  # no overflow in the limit rounding loops
        result = engine.compute_erpimage(np.zeros((20, 6)), None, None, 1, 1, engine.parse_erpimage_options(["erp"]))
        zero = engine._amp_coh_limits(np.zeros(3), np.zeros(3), np.array([-3.0, 5.0]), True, *[np.nan] * 4)
    assert np.isnan(result.limits[2]) and np.isnan(result.limits[3])
    assert result.clim == (0.0, 0.0)
    # Amplitude limits fall back to 0 without the 'ampsig' comparison; the coherence limit stays NaN.
    assert zero[:2] == (0.0, 0.0) and np.isnan(zero[2]) and zero[3] == 0.0


def test_timewarp_with_sorting_warps_each_trial_by_its_own_events() -> None:
    data = FIXTURE_DATA["datasets"][0, 0]["synth"]
    times = -125 + np.arange(data.shape[0]) * 1000 / 128
    events = 20 + np.arange(data.shape[1])[:, np.newaxis] * np.array([[1.0, 4.0]]) + np.array([[0.0, 60.0]])
    sortvar = np.arange(data.shape[1], dtype=float)[::-1]
    sorted_result = engine.compute_erpimage(
        data, sortvar, times, 1, 1, engine.parse_erpimage_options(["timewarp", (events,)]), plotted=False
    )
    in_order = engine.compute_erpimage(
        data[:, ::-1],
        sortvar[::-1],
        times,
        1,
        1,
        engine.parse_erpimage_options(["timewarp", (events[::-1],)]),
        plotted=False,
    )
    np.testing.assert_allclose(sorted_result.outdata, in_order.outdata)


@pytest.mark.parametrize(
    ("formula", "expected"),
    [
        ("yes", None),
        ("2*x+1", lambda x: 2 * x + 1),
        ("(X-2)/4", lambda x: (x - 2) / 4),
        ("x.^2", lambda x: x**2),
        ("-x+3", lambda x: -x + 3),
        ("x^2", lambda x: x**2),
        ("x./4", lambda x: x / 4),
    ],
)
def test_renorm_formulas(formula, expected) -> None:
    sortvar = np.array([10.0, 30.0, 20.0, 40.0])
    times = np.linspace(-100, 100, 5)
    result = engine._renorm(sortvar, formula, times)
    if expected is None:
        np.testing.assert_allclose(result, (sortvar - 10) / 30 * 100 - 100 + 80)
    else:
        np.testing.assert_allclose(result, expected(sortvar))


@pytest.mark.parametrize(
    ("tokens", "kwargs", "message"),
    [
        (["bogus"], {}, "unknown arg bogus"),
        ([], {"bogus": 1}, "unknown option"),
        ([1.0], {}, "unknown arg 1"),
        (["title"], {}, "needs a value"),
        (["allcohers", 1], {}, "UNIMPLEMENTED"),
        (["erp", "5"], {}, "unknown arg 5"),
        ([], {"erp": "sometimes"}, "'erp' must be"),
        ([], {"erp": 2.5}, "'erp' must be"),
        (["caxis", [1, 2, 3]], {}, "'caxis' takes 1 or 2 values"),
        (["caxis", "erp"], {}, "'caxis' must be numeric, got 'erp'"),
        ([], {"caxis": [5, -5]}, "increasing"),
        ([], {"caxis": -1.0}, "fraction must be positive"),
        ([], {"caxis": np.nan}, "finite"),
        (["coher", [1, 2, 3, 4]], {}, "'coher' takes 1 or 2 or 3 values"),
        (["coher", [10, 10, 0.5]], {}, "alpha value"),
        ([], {"coher": [np.nan]}, "finite"),
        ([], {"srate": [100, 200]}, "'srate' takes 1 value, got 2"),
        ([], {"cycles": 0}, "'cycles' must be positive"),
        ([], {"cycles": np.nan}, "finite"),
        ([], {"noshow": 0}, "'noshow' must be True/False"),
        ([], {"plotmode": "bogus"}, "'plotmode' must be"),
        ([], {"align": np.nan}, "latency in ms or Inf"),
        ([], {"sortvarpercent": 1.5}, "fractions from 0 to 1"),
        ([], {"sortvarpercent": -0.2}, "fractions from 0 to 1"),
        ([], {"baseline": [0]}, "'baseline' takes 2 values"),
        ([], {"baseline": []}, "'baseline' needs a value"),
        ([], {"baseline": [np.nan, 0]}, "finite"),
        (["phasesort", [100, 0, 10, 12, 200]], {}, "topphase"),
        (["limits", [5, 1]], {}, "time limits"),
        (["limits", [0, 1, 5, 1]], {}, "erp limits"),
        (["phasesort", [0, 120, 10]], {}, "phasesort"),
        (["phasesort", [0, 10]], {}, "at least"),
        (["ampsort", [0, 10, 12, 8]], {}, "increasing"),
        (["valsort", [10, 0]], {}, "increasing"),
        (["valsort", [0, 10, 0]], {}, "direction"),
        (["erpalpha", 0.5], {}, "out of bounds"),
        (["avg_type", "triangle"], {}, "avg_type"),
        (["replace_ties", "maybe"], {}, "replace_ties"),
        (["sortvar_limits", [1, 2, 3]], {}, "two element"),
        (["spec", [1]], {}, "'spec' takes 2 values"),
        (["specaxis", "semilog"], {}, "lin"),
        (["signif", [1, 2]], {}, "3 values"),
        (["topo", [1]], {}, "length 2 or 3"),
        ([], {"cbar": "maybe"}, "True/False"),
    ],
)
def test_invalid_options_raise(tokens, kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        engine.parse_erpimage_options(tokens, **kwargs)


def test_option_grammar_follows_eeglab() -> None:
    options = engine.parse_erpimage_options(
        [
            "erp",
            2,
            "cbar",
            "noxlabel",
            False,
            "nosort",
            "off",
            "colorbar",
            "on",
            "NoShow",
            "on",
            "erp_grid",
            "Specaxis",
            "lin",
            "coher",
            10,
        ]
    )
    assert options.erp == 2 and options.cbar and not options.nosort and options.noshow and options.erp_grid
    assert not options.noxlabel
    assert options.specaxis == "linear"
    # 'coher' turns the ERP on, as in EEGLAB, unless a later 'erp' turns it off.
    assert engine.parse_erpimage_options(["coher", 10]).erp == 1
    assert engine.parse_erpimage_options(["coher", 10, "erp", "off"]).erp == 0
    keywords = engine.parse_erpimage_options(erp=3, cbar=True, noshow=False, replace_ties=True, avg_type="Gaussian")
    assert keywords.erp == 3 and keywords.cbar and not keywords.noshow and keywords.replace_ties
    assert keywords.avg_type == "gaussian"


@pytest.mark.parametrize(
    ("args", "kwargs", "message"),
    [
        ((np.zeros((10, 1)),), {}, "times vector"),
        ((np.zeros((10, 4)), [1.0]), {}, "too few trials"),
        ((np.zeros((1, 10)), [1.0, 2.0, 3.0], np.arange(3.0)), {}, "doesn't divide"),
        ((np.zeros((10, 4)), [1.0, 2.0, 3.0]), {}, r"data are \(10, 4\) but sortvar has 3 values"),
        ((np.zeros((4, 10)), np.arange(4.0), np.arange(10.0)), {}, r"data are \(4, 10\) but sortvar has 4 values"),
        ((np.zeros((10, 4)), np.zeros((4, 2))), {}, "sortvar must be a vector"),
        ((np.zeros((10, 4)), [np.nan] * 4), {}, "all 4 sortvar values are NaN"),
        ((np.zeros((10, 4)), [np.nan, np.nan, np.nan, 1.0]), {}, "only one trial"),
        ((np.full((10, 4), np.inf),), {}, "40 infinite values"),
        ((np.zeros((2, 3, 4, 5)), None, np.arange(3.0)), {}, r"\(frames, trials\)"),
        ((np.zeros((10, 3)), None, None, 2), {}, "smoothing needs at least 4 trials, got 3"),
        ((np.zeros((10, 6)), None, None, [5, 7]), {}, "avewidth must be a single number"),
        ((np.zeros((10, 6)), None, None, "5"), {}, "avewidth must be a single number"),
        ((np.zeros((10, 6)), None, None, np.inf), {}, "avewidth must be finite"),
        ((np.zeros((10, 6)), None, None, np.nan), {"avg_type": "gaussian"}, "avewidth must be finite"),
        ((np.zeros((10, 6)), None, None, 0.5), {"avg_type": "gaussian"}, "cannot be < 1"),
        ((np.zeros((10, 6)), None, None, 1, np.nan), {}, "decimate must be finite"),
        ((np.zeros((10, 6)), None, None, 1, -7), {}, "use a value from -6 to 6"),
        ((np.zeros((10, 6)), None, np.zeros(10)), {}, "strictly increasing"),
        ((np.zeros((10, 6)), None, np.arange(10.0)[::-1]), {}, "strictly increasing"),
        ((np.zeros((10, 6)), None, [0.0, 10.0, -5.0]), {}, "srate > 0"),
        ((np.zeros((1, 6)), np.arange(6.0), [0.0]), {}, "at least 2 frames"),
        ((np.zeros((10, 6)), None, None, 2.5), {}, "integer"),
        ((np.zeros((10, 6)), None, None, 0.5), {}, "cannot be < 1"),
        ((np.zeros((10, 6)), None, [0.0, 20.0, 100.0]), {}, "gives 20 frames"),
        ((np.zeros((10, 6)), None, np.arange(9.0)), {}, "differs from frames"),
        ((np.zeros((40, 6)), None, np.arange(40.0)), {"phasesort": [500, 0, 10]}, "out of bound"),
        ((np.zeros((40, 6)), None, np.arange(40.0)), {"phasesort": [10, 0, 1]}, "too low"),
        ((np.zeros((40, 6)), None, np.arange(40.0)), {"coher": [600]}, "out of range"),
        ((np.zeros((40, 6)), None, np.arange(40.0)), {"plotamps": True, "coher": 10}, "require coher"),
        ((np.zeros((40, 6)), None, np.arange(40.0)), {"align": 500}, "align time"),
        ((np.zeros((40, 6)), None, np.arange(40.0)), {"filt": [0, 30]}, "trial filter"),
        ((np.zeros((40, 6)), None, np.arange(40.0)), {"filt": [0, 900]}, "sampling rate"),
        ((np.zeros((40, 6)), None, np.arange(40.0)), {"baseline": [5, 1]}, "baseline"),
        ((np.zeros((40, 6)), None, np.arange(40.0) * 4), {"baseline": [10.5, 11.5]}, "contains no samples"),
        ((np.zeros((40, 6)), None, np.arange(40.0)), {"limits": [100, 200]}, "do not overlap the data"),
        ((np.zeros((40, 6)), np.ones(6), np.arange(40.0)), {"renorm": "yes"}, "not all equal"),
        ((np.zeros((40, 6)), None, np.arange(40.0)), {"phasesort": [10, 0, 600]}, "less than Nyquist"),
        ((np.zeros((40, 6)), None, np.arange(40.0)), {"ampsort": [10, 0, 600]}, "less than Nyquist"),
        ((np.zeros((40, 6)), None, np.arange(40.0)), {"coher": [100, 100, 0.01]}, "'coher' amplitude is zero"),
        ((np.zeros((40, 6)), None, np.arange(40.0)), {"vert": np.zeros((3, 2))}, "vert arg matrix"),
        ((np.zeros((40, 6)), None, np.arange(40.0)), {"auxvar": np.zeros((3, 2))}, "auxvar size"),
        ((np.zeros((40, 6)), None, np.arange(40.0)), {"renorm": "x+x"}, "renormalizing"),
        ((np.zeros((40, 6)), None, np.arange(40.0)), {"renorm": "abs(x)"}, "renormalizing"),
        ((np.zeros((40, 6)), None, np.arange(40.0)), {"ampsort": [np.inf, 0, 100]}, "sortwin"),
        (
            (np.zeros((40, 6)), None, np.arange(40.0)),
            {"timewarp": (np.full((6, 1), 10.0),), "phasesort": [10, 0, 100]},
            "timewarp",
        ),
        ((np.zeros((40, 6)), None, np.arange(40.0)), {"timewarp": (np.array([[10.0, 5.0]] * 6),)}, "ascending"),
        ((np.zeros((40, 6)), None, np.arange(40.0)), {"timewarp": (np.full((5, 1), 10.0),)}, "one row per trial"),
        (
            (np.zeros((40, 6)), None, np.arange(40.0)),
            {"timewarp": (np.array([[10.0], [np.nan], [10.0], [10.0], [10.0], [10.0]]),)},
            "trial 2 has",
        ),
        ((np.zeros((40, 6)), None, np.arange(40.0)), {"timewarp": (np.full((6, 1), 90.0),)}, "lie in the epoch"),
        (
            (np.zeros((40, 6)), None, np.arange(40.0)),
            {"timewarp": (np.full((6, 2), [10.0, 20.0]), [15.0])},
            "one reference latency per event column",
        ),
        (
            (np.zeros((40, 6)), None, np.arange(40.0)),
            {"timewarp": (np.full((6, 1), 10.0), [90.0])},
            "reference latencies .* must lie in the epoch",
        ),
        (
            (np.zeros((40, 6)), None, np.arange(40.0)),
            {"timewarp": (np.full((6, 2), [10.0, 20.0]), [25.0, 15.0])},
            "reference latencies must be in ascending order",
        ),
        (
            (np.zeros((40, 6)), None, np.arange(40.0)),
            {"timewarp": (np.full((6, 1), 10.0),), "align": 10},
            "not compatible",
        ),
    ],
)
def test_invalid_inputs_raise(args, kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        engine.compute_erpimage(*args, options=engine.parse_erpimage_options(**kwargs), plotted=False)


def test_colon_matches_matlab_examples() -> None:
    np.testing.assert_array_equal(engine._colon(1, 1, 4.5), [1, 2, 3, 4])
    np.testing.assert_array_equal(engine._colon(0, 2, 7), [0, 2, 4, 6])
    np.testing.assert_allclose(engine._colon(-2, 4 / 6, 2), np.linspace(-2, 2, 7), rtol=0, atol=1e-15)
    assert engine._colon(0, 0.1, 0.3).size == 4
    assert engine._colon(2, 1, 1).size == 0
    np.testing.assert_array_equal(engine._colon(-2, np.inf, 2), [-2])


def test_order_of_magnitude() -> None:
    assert engine.order_of_magnitude(345) == 100
    assert engine.order_of_magnitude(-0.042) == pytest.approx(0.01)
    assert engine.order_of_magnitude(0) == 1


@pytest.mark.matlab
class TestErpimageMatlabParity:
    """Regenerate the fixture with live MATLAB and compare every scenario again."""

    @pytest.fixture(scope="class")
    def regenerated(self, tmp_path_factory):
        if not matlab_engine_available():
            pytest.skip("MATLAB engine not available or skipped")
        try:
            matlab = get_eeglab("MAT", auto_file_roundtrip=False)
        except Exception as exc:  # MATLAB start-up failures are environment problems, not test failures
            pytest.skip(f"MATLAB not available: {exc}")
        output = tmp_path_factory.mktemp("erpimage") / "erpimage_engine_fixtures.mat"
        matlab.erpimage_engine_fixtures(str(_resolve_eeglab_root()), str(SAMPLE_SET), str(output), nargout=0)
        return _load(output)

    @pytest.mark.parametrize("name", SCENARIOS)
    def test_parity_live(self, regenerated, name: str) -> None:
        _assert_scenario(regenerated, name)


@settings(max_examples=60, deadline=None)
@given(
    trials=st.integers(4, 30),
    width=st.floats(0.2, 30.0),
    advance=st.floats(0.3, 6.0),
    nan_fraction=st.floats(0.0, 0.3),
    seed=st.integers(0, 2**16),
)
def test_unweighted_irregular_windows_match_the_general_movav_port(trials, width, advance, nan_fraction, seed) -> None:
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(2, trials))
    values[rng.random(values.shape) < nan_fraction] = np.nan
    width = min(width, float(trials))
    expected, expected_x = movav(values, np.arange(1, trials + 1), width, advance, None, None, None)
    actual, actual_x = engine._moving_average(values, width, advance, np.ones(1))
    steps = np.full(actual_x.size, advance)
    steps[0] = 1 + width / 2
    np.testing.assert_array_equal(actual_x, np.cumsum(steps))  # EEGLAB adds the advance repeatedly
    assert actual_x.size == expected_x.size
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12, equal_nan=True)


@settings(max_examples=40, deadline=None)
@given(trials=st.integers(4, 30), width=st.integers(2, 6), seed=st.integers(0, 2**16))
def test_weighted_windows_with_missing_values_match_the_general_movav_port(trials, width, seed) -> None:
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(3, trials))
    values[rng.random(values.shape) < 0.25] = np.nan
    values[2] = np.nan  # an all-missing row stays missing
    width = min(width, trials)
    weights = np.full(width, 1.0 / width)
    expected, _x = movav(values, np.arange(1, trials + 1), width, 1.0, None, None, weights)
    actual, _x = engine._moving_average(values, width, 1.0, weights)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12, equal_nan=True)


def test_auxvar_follows_alignment_ties_and_orientation() -> None:
    data = np.arange(400.0).reshape(40, 10) % 9
    sortvar = np.array([5.0, 3.0, 5.0, 1.0, 3.0, 2.0, 4.0, 6.0, 7.0, 8.0])
    auxvar = np.column_stack([np.arange(10.0), np.arange(10.0) * 2])  # (trials, k) is transposed like EEGLAB
    options = engine.parse_erpimage_options(["auxvar", auxvar, "replace_ties", "on"])
    result = engine.compute_erpimage(data, sortvar, np.arange(40.0), 1, 1, options, plotted=False)
    ties = result.sortidx[np.isin(sortvar[result.sortidx], [3.0])]
    np.testing.assert_allclose(result.auxvar[0, np.isin(result.sortidx, ties)], [2.5, 2.5])
    aligned = engine.compute_erpimage(
        data, sortvar, np.arange(40.0), 1, 1, engine.parse_erpimage_options(auxvar=auxvar, align=5), plotted=False
    )
    shifts = np.sign(5 - sortvar[aligned.sortidx]) * np.floor(np.abs(5 - sortvar[aligned.sortidx]) + 0.5)
    np.testing.assert_allclose(aligned.auxvar[0], auxvar[aligned.sortidx, 0] + shifts)


def test_nan_data_is_replaced_by_zeros_and_logged(caplog) -> None:
    data, times = _noise()
    data[[3, 5], 2] = np.nan
    with caplog.at_level(logging.WARNING, logger=engine.logger.name):
        result = engine.compute_erpimage(data, None, times, 1, 1, plotted=False)
    np.testing.assert_array_equal(result.outdata[[3, 5], 2], [0.0, 0.0])
    assert "Replaced 2 NaN values in data with 0s." in caplog.text


def test_plotamps_rejects_trials_without_amplitude() -> None:
    data, times = _noise()
    data[:, 4] = np.nan
    options = engine.parse_erpimage_options(plotamps=True, coher=[10, 20, 0.01])
    with pytest.raises(ValueError, match="1 of 12 trials have zero wavelet amplitude"):
        engine.compute_erpimage(data, None, times, 1, 1, options, rng=0)


def test_percentile_erps_with_fewer_trials_than_groups_give_missing_rows() -> None:
    data = np.arange(30.0).reshape(10, 3)
    erp = engine.compute_erpimage(data, None, np.arange(10.0), 1, 1, engine.parse_erpimage_options(erp=4)).erp
    np.testing.assert_allclose(erp[:3], data.T)
    assert np.all(np.isnan(erp[3]))


def _late_epoch(options: list) -> engine.ErpimageComputation:
    """Eight noise trials whose epoch starts at 100 ms, so no sample precedes time zero."""
    data = np.random.default_rng(4).normal(size=(64, 8))
    times = np.arange(64.0) * 1000 / SRATE + 100
    return engine.compute_erpimage(
        data, np.arange(8.0), times, 1, 0, engine.parse_erpimage_options(options), rng=np.random.RandomState(0)
    )


@pytest.mark.parametrize(
    "options",
    [["coher", [10, 10, 0.05], "baselinedb", [-1000, -900]], ["plotamps", "coher", [10, 10, 0.05]]],
)
def test_amplitude_baseline_falls_back_to_the_first_quarter_of_the_epoch(options) -> None:
    result = _late_epoch(options)
    assert abs(float(np.mean(result.amps[:16]))) < 1e-12


@pytest.mark.parametrize("options", [["coher", [10, 10, 0.05]], ["plotamps", "coher", [10, 10, 0.05]]])
def test_a_given_baseline_amplitude_is_subtracted(options) -> None:
    computed = _late_epoch(options)
    given = _late_epoch([*options, "limits", [np.nan] * 8 + [1.0]])
    np.testing.assert_allclose(given.amps, computed.amps + computed.baseamp - 1.0, atol=1e-12)
    assert given.limits[8] == given.baseamp == 1.0


def test_timewarp_reference_lines_replace_vert_unless_auxvar_is_given() -> None:
    warp = (np.full((8, 1), 120.0), [150.0], ["r--"])
    result = _late_epoch(["timewarp", warp])
    # 150 ms rounds to frame round(1 + 50 * 0.128) = 7, drawn at its latency.
    np.testing.assert_allclose(result.verttimes, [100 + 6 * 1000 / SRATE])
    assert result.vert_colors == ("r--",)
    assert _late_epoch(["timewarp", warp, "auxvar", np.zeros((1, 8))]).verttimes is None


def test_sortvar_limits_give_the_trial_range_they_span() -> None:
    assert _late_epoch(["sortvar_limits", [-100, 1000]]).trial_limits == (1.0, 8.0)
    assert _late_epoch(["sortvar_limits", [2, 5]]).trial_limits == (3.0, 6.0)


def test_amplitude_sorting_orders_trials_by_wavelet_amplitude() -> None:
    result = _late_epoch(["ampsort", [200, 0, 10, 10]])
    center = int(np.argmin(np.abs(np.arange(64.0) * 1000 / SRATE + 100 - 200)))
    assert np.all(np.diff(result.outamps[center, result.sortidx]) >= 0)


def test_phase_sorting_rejects_the_weakest_trials_and_clamps_the_band(caplog) -> None:
    with caplog.at_level(logging.INFO, logger=engine.logger.name):
        result = _late_epoch(["phasesort", [200, 25, 10, 900]])
    assert result.sortidx.size == 6
    assert "Reducing max 'phasesort' frequency to Nyquist rate (64 Hz)" in caplog.text
    assert "Retaining 6 epochs (75 percent) with largest power" in caplog.text


@pytest.mark.parametrize("sort", [["phasesort", [100, 0, 10]], ["ampsort", [100, 0, 10]]])
def test_plotamps_images_the_sorting_frequency(sort) -> None:
    # erpimage.m replaces the 'coher' frequency by the sorting frequency before 'plotamps'.
    data, times = _noise(trials=20)
    results = [
        engine.compute_erpimage(
            data,
            None,
            times,
            1,
            1,
            engine.parse_erpimage_options([*sort, "plotamps", "coher", [freq, freq, 0.05]]),
            rng=0,
        )
        for freq in (10, 14)
    ]
    np.testing.assert_array_equal(results[0].outdata, results[1].outdata)
    assert results[0].coherfreq == results[1].coherfreq == 10.0


def test_decimate_above_the_trial_count_is_clamped(caplog) -> None:
    data, _times = _noise(30, 24)
    sortvar = np.random.default_rng(0).permutation(24).astype(float)
    with caplog.at_level(logging.WARNING, logger=engine.logger.name):
        clamped = engine.compute_erpimage(data, sortvar, None, 3, 100, plotted=False)
    at_count = engine.compute_erpimage(data, sortvar, None, 3, 24, plotted=False)
    np.testing.assert_array_equal(clamped.outdata, at_count.outdata)
    np.testing.assert_array_equal(clamped.outtrials, at_count.outtrials)
    assert "Setting variable decfactor to max 24." in caplog.text


def test_srate_option_overrides_the_rate_implied_by_times() -> None:
    data = np.zeros((40, 6))
    data[10] = 1.0
    options = engine.parse_erpimage_options(align=140.0, srate=100.0)
    result = engine.compute_erpimage(data, np.full(6, 100.0), np.arange(40) * 4.0, 1, 1, options, plotted=False)
    # The shift is round((140 - 100) * 100 / 1000) = 4 frames at the given 100 Hz, not 10 at 250 Hz.
    assert np.all(np.argmax(result.outdata, axis=0) == 14)


def test_per_trial_vertical_latencies_follow_the_trial_sort() -> None:
    data, _times = _noise(40, 8)
    sortvar = np.array([5.0, 3.0, 8.0, 1.0, 7.0, 2.0, 6.0, 4.0])
    vert = np.column_stack([np.arange(8.0) * 10 + 20, np.arange(8.0) * 10 + 25])
    options = engine.parse_erpimage_options(vert=vert)
    result = engine.compute_erpimage(data, sortvar, np.arange(40.0) * 5, 1, 1, options, plotted=False)
    np.testing.assert_array_equal(result.verttimes, vert[result.sortidx])


def test_timewarp_moves_each_trials_events_to_the_rounded_reference_frames() -> None:
    frames, trials = 64, 5
    times = -125 + np.arange(frames) * 1000 / SRATE
    events = np.array([[20.0, 40.0]] * trials) + np.arange(trials)[:, np.newaxis] * np.array([[2.0, 3.0]])
    data = np.zeros((frames, trials))
    for trial in range(trials):
        for event in events[trial]:
            data[int(np.round(1 + (event + 125) * SRATE / 1000)) - 1, trial] = 1.0
    options = engine.parse_erpimage_options(timewarp=(events, [153.9, 300.0]))
    result = engine.compute_erpimage(data, None, times, 1, 1, options, plotted=False)
    # 1 + (153.9 + 125) * 0.128 = 36.7 rounds to frame 37 (0-based 36).
    assert np.all(np.argmax(result.erp_data, axis=0) == 36)


def test_baseline_window_past_the_epoch_edges_uses_the_whole_trial() -> None:
    data, times = _noise()
    data += np.arange(12.0)
    options = engine.parse_erpimage_options(baseline=[-500.0, 5000.0])
    result = engine.compute_erpimage(data, None, times, 1, 1, options, plotted=False)
    np.testing.assert_allclose(result.outdata, data - data.mean(axis=0), atol=1e-12)


def test_three_frames_with_three_latencies_are_a_time_vector() -> None:
    result = engine.compute_erpimage(np.arange(18.0).reshape(3, 6), None, [0.0, 10.0, 20.0], 1, 1, plotted=False)
    np.testing.assert_array_equal(result.times, [0.0, 10.0, 20.0])
    assert result.srate == 100.0


def test_two_element_caxis_sets_the_color_limits_of_both_images() -> None:
    data, times = _noise()
    for options in ({}, {"plotamps": True, "coher": [10, 10, 0.05]}):
        result = engine.compute_erpimage(
            data, None, times, options=engine.parse_erpimage_options(caxis=[-3, 3], **options), rng=0
        )
        assert result.clim == (-3.0, 3.0)


@pytest.mark.parametrize(
    ("args", "kwargs", "message"),
    [
        ((20,), {}, "Setting variable avewidth to max 12."),
        ((3,), {"avg_type": "gaussian"}, "Gaussian avewidth 3 is too big for 12 trials"),
        ((1, 50), {}, "Setting variable decfactor to max 12."),
        ((1, 1), {"phasesort": [100, -25, 10]}, "Retaining 9 epochs (75 percent) with smallest power"),
        ((1, 1), {"ampsort": [100, 25, 10]}, "Retaining 9 epochs (75 percent) with largest power"),
        ((1, 1), {"ampsort": [100, 0, 10, 100]}, "Reducing max 'ampsort' frequency to Nyquist rate (64 Hz)"),
        ((1, 1), {"limits": [-500, 1000]}, "Adjusting mintime limit to first data value -100"),
        ((1, 1), {"limits": [-500, 1000]}, "Adjusting maxtime limit to last data value 392.188"),
        ((1, 1), {"phasesort": [100, 0, 10], "nosort": True}, "'phasesort' takes precedence over 'nosort'"),
        ((1, 1), {"sortwin": [0, 100]}, "'sortwin' applies only to 'ampsort' with an Inf center"),
        ((1, 1), {"showwin": True}, "'showwin' has no sorting window"),
        ((1, 1), {"valsort": [0, 100], "sortvar_limits": [1, 5]}, "no effect when sorting by value"),
        ((1, 1), {"ampsort": [100, 0, 10], "img_trialax_label": "RT"}, "no effect when sorting by amp"),
        ((1, 1), {"erp": 3, "erpalpha": 0.05}, "Only plotting one ERP (not 3 percentile ERPs)"),
        ((1, 1), {"sortvarpercent": [0.5]}, "'sortvarpercent' needs smoothing or decimation"),
    ],
)
def test_eeglab_notes_are_logged(caplog, args, kwargs, message) -> None:
    data, times = _noise()
    with caplog.at_level(logging.INFO, logger=engine.logger.name):
        engine.compute_erpimage(
            data, np.arange(12.0), times, *args, options=engine.parse_erpimage_options(**kwargs), rng=0, plotted=False
        )
    assert message in caplog.text


def test_too_large_alignment_shifts_leave_empty_trials_and_are_logged(caplog) -> None:
    data, times = _noise()
    sortvar = np.r_[5000.0, np.full(11, 100.0)]
    with caplog.at_level(logging.WARNING, logger=engine.logger.name):
        result = engine.compute_erpimage(
            data, sortvar, times, 1, 1, engine.parse_erpimage_options(align=100.0), plotted=False
        )
    np.testing.assert_array_equal(result.outdata[:, result.sortidx == 0], 0.0)
    assert "No aligned data for input trials [0] (0-based): shifts of [-627] frames are too large." in caplog.text


def test_amplitude_rejection_must_leave_two_trials() -> None:
    data, times = _noise()
    for option in ({"phasesort": [100, 90, 10]}, {"ampsort": [100, 90, 10]}):
        with pytest.raises(ValueError, match="amplitude rejection leaves 1 of 12 trials"):
            engine.compute_erpimage(data, None, times, options=engine.parse_erpimage_options(**option), plotted=False)


def test_decimation_reuses_the_rejection_fraction_like_eeglab() -> None:
    assert engine._adjust_decimation(10, 40, 3, 0.25) == pytest.approx(30 / 10)
    assert engine._adjust_decimation(10, 40, 3, -0.25) == 0
    assert engine._adjust_decimation(-10, 40, 4, None) == pytest.approx((40 - 4) / 10)
    assert engine._adjust_decimation(3, 40, 4, None) == 3


def test_amplitude_and_coherence_limits_follow_eeglab_rules() -> None:
    amps = np.array([0.0, 0.4, -0.2])
    maxamp, minamp, maxcoh, mincoh = engine._amp_coh_limits(
        amps, np.array([0.2, 0.95]), np.array([-3.0, 5.0]), True, np.nan, np.nan, np.nan, np.nan
    )
    assert maxamp == pytest.approx(5.05) and minamp == pytest.approx(-3.03)
    assert maxcoh == 1.0 and mincoh == 0.0


def test_helper_edge_cases_follow_eeglab() -> None:
    assert engine.find_corresponding_point(50.0, np.array([1.0, 2.0]), np.array([1.0, 2.0])) is None
    assert engine.find_corresponding_point(2.0, np.array([1.0, 2.0, 2.0]), np.array([1.0, 2.0, 3.0])) == 2.5
    np.testing.assert_array_equal(engine._seqsum(np.zeros((0, 3)), axis=0), np.zeros(3))
    np.testing.assert_allclose(engine._gauss(1, 2.0), [np.exp(-4.0)])  # gauss.m: -sds:Inf:sds is -sds
    with pytest.raises(ValueError, match="empty"):
        engine._gauss(0, 2.0)
    with pytest.raises(ValueError, match="too short"):
        engine._phasecoher(np.zeros((10, 3)), 100.0, 1.0, 3.0)
    with pytest.raises(ValueError, match="alpha out of bounds"):
        engine._phasecoher(np.zeros((400, 3)), 100.0, 10.0, 3.0, 0.5, np.random.default_rng(0))
    amps = engine._phasecoher(np.random.default_rng(0).normal(size=(64, 4)), 128.0, 10.0, 3.0, 0.0)
    assert amps[2] is None and amps[3] is None
    with pytest.raises(ValueError, match="no spectral estimate"):
        engine._peak_frequency(np.ones((64, 2)), 64, 128.0, 10.01, 10.02)
    with pytest.raises(ValueError, match="no spectral estimate"):
        engine._ampsort_frequencies(np.ones((64, 2)), 64, 128.0, np.array([0, 0, 10.01, 10.02]))
    with pytest.raises(ValueError, match="renormalizing"):
        engine._renorm(np.arange(3.0), "x+", np.arange(3.0))


@pytest.mark.parametrize(
    ("tokens", "message"),
    [
        (["limits", np.zeros(10)], "sized"),
        (["limits", [0, 1, np.nan, np.nan, 5, 1]], "amp limits"),
        (["limits", [0, 1, np.nan, np.nan, np.nan, np.nan, 1, 0]], "coh limits"),
        (["phasesort", [0, 0, 10, 12, 0, 1]], "Too many"),
        (["phasesort", [0, 0, -10]], "negative frequency"),
        (["phasesort", [0, 0, 10, -1]], "negative argument"),
        (["phasesort", [0, 0, 12, 10]], "increasing"),
        (["ampsort", [0, 0, 10, 12, 1]], "Too many"),
        (["ampsort", [0, 0]], "at least"),
        (["ampsort", [0, 0, 10, -12]], "negative argument"),
        (["ampsort", [0, 200, 10]], "percentile"),
        (["sortwin", [0, 1, 2]], "Too many"),
        (["sortwin", [2, 1]], "increasing"),
        (["valsort", [0, 1, 1, 1]], "Too many"),
        (["baselinedb", [0, 1, 2]], "2 argument"),
        (["erp", 7], "unknown arg 2"),
        (["align", "later"], "numeric"),
    ],
)
def test_more_invalid_options_raise(tokens, message) -> None:
    with pytest.raises(ValueError, match=message):
        engine.parse_erpimage_options(tokens)


def test_keyword_flags_accept_eeglab_strings_and_erp_strings() -> None:
    options = engine.parse_erpimage_options(["erp", "2"], cbar="on", nosort="off", erp_grid=True)
    assert options.erp == 2 and options.cbar and not options.nosort and options.erp_grid
    assert engine.parse_erpimage_options(erp="off").erp == 0
    assert engine.parse_erpimage_options(erp=np.array([3.0])).erp == 3
    with pytest.raises(ValueError, match="'erp' must be"):
        engine.parse_erpimage_options(erp=7)
    auxvar = engine.parse_erpimage_options(auxvar=(np.zeros((2, 5)), ["r", "b--"]))
    assert auxvar.auxcolors == ("r", "b--")
    # A list is (matrix, colors) only when its second item names line styles.
    rows = engine.parse_erpimage_options(auxvar=[[1, 2, 3], [4, 5, 6]])
    assert rows.auxvar.shape == (2, 3) and rows.auxcolors is None
    assert engine.parse_erpimage_options(auxvar=[[1, 2], [3, 4], [5, 6]]).auxvar.shape == (3, 2)


@pytest.mark.parametrize(
    ("args", "kwargs", "message"),
    [
        ((np.zeros((40, 6)), None, np.arange(40.0)), {"ampsort": [500, 0, 100]}, "amplitude sorting"),
        ((np.zeros((40, 6)), None, np.arange(40.0)), {"valsort": [0, 500]}, "value sorting"),
        ((np.zeros((40, 6)), None, np.arange(40.0)), {"coher": [10]}, "too low"),
        ((np.zeros((40, 6)), None, np.arange(40.0)), {"filt": [0, 30, 40]}, "two element"),
        ((np.zeros((40, 6)), None, np.arange(40.0)), {"filt": [0, 500]}, "first element must be greater"),
        ((np.zeros((40, 6)), None, np.arange(40.0)), {"filt": [20, 10]}, "greater than first"),
        ((np.zeros((40, 6)), None, np.arange(40.0)), {"filt": [-10, 20]}, "BOTH"),
        ((np.zeros((40, 6)), None, np.arange(40.0)), {"filt": [-10, -600]}, "stop band"),
        ((np.zeros((40, 6)), None, np.arange(40.0)), {"baseline": [-50, -20]}, "epoch start"),
        ((np.zeros((40, 6)), None, np.arange(40.0)), {"baseline": [50, 60]}, "epoch end"),
    ],
)
def test_more_invalid_inputs_raise(args, kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        engine.compute_erpimage(*args, options=engine.parse_erpimage_options(**kwargs), plotted=False)


def test_ampsort_windows_of_different_lengths_raise() -> None:
    data = np.random.default_rng(1).normal(size=(64, 6))
    options = engine.parse_erpimage_options(ampsort=[np.inf, 0, 10], sortwin=[0, 400])
    with pytest.raises(ValueError, match="different lengths"):
        engine.compute_erpimage(data, None, np.arange(64.0) * 1000 / 128, 1, 1, options, plotted=False)


def test_squeezed_three_dimensional_input_and_boxcar_zero_width() -> None:
    data = np.random.default_rng(2).normal(size=(1, 20, 6))
    result = engine.compute_erpimage(data, None, np.arange(20.0), 0, 0, plotted=False)
    np.testing.assert_array_equal(result.outdata, data[0])
    gaussian = engine.compute_erpimage(
        data, None, np.arange(20.0), 0, 0, engine.parse_erpimage_options(avg_type="gaussian"), plotted=False
    )
    np.testing.assert_array_equal(gaussian.outdata, data[0])


def test_wavelet_transform_ignores_trial_offsets_and_spans_trial_blocks() -> None:
    rng = np.random.default_rng(7)
    frames, trials = 48, engine.WAVELET_TRIAL_BLOCK + 44
    values = rng.normal(size=(frames + 40, trials))
    coswin, sinwin = engine._wavelet(128.0, 10.0, 3.0)
    length = coswin.size
    direct = np.empty((frames, trials), dtype=complex)
    for frame in range(frames):
        epoch = values[frame : frame + length] - values[frame : frame + length].mean(axis=0)
        direct[frame] = np.dot(coswin, epoch) + 1j * np.dot(sinwin, epoch)
    offset = engine._wavelet_transform(values + 1e6, frames, coswin, sinwin)
    np.testing.assert_allclose(offset, direct, rtol=0, atol=1e-9 * np.max(np.abs(direct)))


def test_weighted_windows_wider_than_their_weights_match_the_general_movav_port() -> None:
    values = np.random.default_rng(8).normal(size=(2, 12))
    weights = np.array([0.2, 0.5, 0.3])
    expected, _x = movav(values, np.arange(1, 13), 3.5, 1.0, None, None, weights)
    actual, _x = engine._moving_average(values, 3.5, 1.0, weights)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)


def test_a_row_of_concatenated_trials_is_read_in_column_major_order() -> None:
    data, times = _noise(64, 6)
    flat = engine.compute_erpimage(data.reshape(1, -1, order="F"), np.arange(6.0), times, 1, 1, plotted=False)
    np.testing.assert_array_equal(flat.outdata, data)


def test_ampsort_frequency_needs_three_cycles_in_the_epoch() -> None:
    data, times = _noise()
    with pytest.raises(ValueError, match="amplitude-sorting freq"):
        engine.compute_erpimage(data, None, times, options=engine.parse_erpimage_options(ampsort=[100, 0, 1]))


def test_timewarp_line_styles_cycle_like_eeglab() -> None:
    _data, times = _noise()
    _marks, _ref, styles = engine._timewarp_marks(
        (np.full((6, 3), [50.0, 100.0, 150.0]), None, ("r", "g")), times, SRATE, 64
    )
    assert styles == ("", "r", "g", "r", "")


def test_private_helpers_match_the_package_versions() -> None:
    rng = np.random.default_rng(12)
    values = rng.normal(size=(30, 7))
    values[rng.random(values.shape) < 0.2] = np.nan
    for axis in (0, 1):
        np.testing.assert_allclose(engine._nan_mean(values, axis), nan_mean(values, axis), rtol=1e-13, equal_nan=True)
    for length, sds in ((7, 2.0), (40, 2.0), (64, 1.5)):
        np.testing.assert_allclose(engine._gauss(length, sds), gauss(length, sds), rtol=1e-13)
    data, times = _noise()
    window = (times >= -50) & (times <= 20)
    expected = rmbase(data[np.newaxis], data.shape[0], np.flatnonzero(window) + 1)[0]
    np.testing.assert_allclose(engine._remove_baseline(data, np.array([-50.0, 20.0]), times), expected, atol=1e-13)
    old = np.array([1.0, 10.0, 27.0, 50.0, 64.0])
    new = np.array([1.0, 14.0, 22.0, 47.0, 64.0])
    left, weight_left, weight_right = engine._timewarp_weights(old, new)
    matrix = np.zeros((64, 64))
    rows = np.arange(64)
    np.add.at(matrix, (rows, left), weight_left)
    np.add.at(matrix, (rows, np.minimum(left + 1, 63)), weight_right)
    np.testing.assert_allclose(matrix, timewarp(old, new), atol=1e-12)
