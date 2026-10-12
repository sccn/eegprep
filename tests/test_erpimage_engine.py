"""MATLAB parity and properties of the NumPy ERP-image engine.

The committed fixture ``tests/matlab/erpimage_engine_fixtures.mat`` holds
EEGLAB ``erpimage`` outputs from MATLAB R2025b for every scenario below, so CI
compares without MATLAB; ``TestErpimageMatlabParity`` regenerates it live when
MATLAB is available.
"""

from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys

from hypothesis import given, settings, strategies as st
import numpy as np
import pytest
from scipy import signal

from eegprep.functions.adminfunc.eeglabcompat import _resolve_eeglab_root, get_eeglab
from eegprep.functions.popfunc.pop_loadset import pop_loadset
from eegprep.functions.sigprocfunc import erpimage_engine as engine
from eegprep.functions.sigprocfunc.erpimage import butterworth_filter
from eegprep.functions.sigprocfunc.movav import movav
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
# Data-unit outputs are compared on the input scale: an ERP after 'rmerp' is rounding noise.
DATA_DOMAIN = {"outdata", "erp", "erpsig"}
NON_SCENARIO_KEYS = {"datasets", "seed", "matlab_version"}
ENGINE_IMPORTS = {"__future__", "ast", "dataclasses", "typing", "numpy"}


def _load(path: Path) -> dict:
    return load_matlab_test_fixture(path)


FIXTURE_DATA = _load(FIXTURE)
SCENARIOS = sorted(key for key in FIXTURE_DATA if key not in NON_SCENARIO_KEYS)
_SAMPLE_CACHE: dict[str, np.ndarray] = {}


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


def _sample_slice(fixture: dict) -> np.ndarray:
    if "real" not in _SAMPLE_CACHE:
        datasets = fixture["datasets"][0, 0]
        channel = int(datasets["real_channel"].ravel()[0]) - 1
        samples = datasets["real_samples"].ravel().astype(int) - 1
        epochs = datasets["real_epochs"].ravel().astype(int) - 1
        eeg = pop_loadset(str(SAMPLE_SET))
        _SAMPLE_CACHE["real"] = np.asarray(eeg["data"], dtype=float)[channel][np.ix_(samples, epochs)]
    return _SAMPLE_CACHE["real"]


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
    for mode, prefix, fields in (("noplot", "", OUTPUTS), ("plot", "plot_", ("limits", "erpsig"))):
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
        if mode == "noplot":
            sortvar = np.asarray(inputs["sortvar"], dtype=float).ravel()
            trials = inputs["data"].shape[1] if not sortvar.size else sortvar.size
            kept = np.arange(trials) if not sortvar.size else np.flatnonzero(~np.isnan(sortvar))
            expected_order = kept[np.asarray(_field(fixture, name, "sortidx"), dtype=int).ravel() - 1]
            np.testing.assert_array_equal(result.sortidx, expected_order, err_msg=f"{name} sortidx")


@pytest.mark.parity
@pytest.mark.parametrize("name", SCENARIOS)
def test_engine_matches_matlab_fixture(name: str) -> None:
    _assert_scenario(FIXTURE_DATA, name)


def test_fixture_was_recorded_on_matlab_r2025b_and_stays_small() -> None:
    assert "R2025b" in _text(FIXTURE_DATA["matlab_version"])
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
    assert first.erpsig is not None
    np.testing.assert_array_equal(first.erpsig, second.erpsig)
    assert not np.array_equal(first.erpsig, other.erpsig)
    assert first.erpsig.shape == (data.shape[0],)
    plotted = engine.compute_erpimage(data, None, None, 1, 1, options, rng=np.random.default_rng(5))
    np.testing.assert_array_equal(plotted.erpsig, np.vstack([first.erpsig, -first.erpsig]))


def test_engine_imports_only_numpy_and_the_standard_library() -> None:
    path = Path(engine.__file__)
    imported = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add(str(node.module).split(".")[0])
    assert imported <= ENGINE_IMPORTS
    # Load the file on its own, as the browser build includes it, and run a computation.
    code = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('erpimage_engine', {str(path)!r})\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "sys.modules['erpimage_engine'] = module\n"
        "spec.loader.exec_module(module)\n"
        "import numpy as np\n"
        "data = np.arange(400.0).reshape(40, 10) % 7\n"
        "options = module.parse_erpimage_options(['erp', 'phasesort', [100, 0, 10], 'coher', [10, 10, 0.05]])\n"
        "module.compute_erpimage(data, None, np.arange(40) * 8.0, 3, 1, options, rng=np.random.default_rng(0))\n"
        "heavy = {'matplotlib', 'scipy', 'mne', 'eegprep'}\n"
        "print(sorted(name for name in sys.modules if name.split('.')[0] in heavy))\n"
    )
    completed = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, check=True)
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
        expected[frame] = coswin @ epoch + 1j * (sinwin @ epoch)
    actual = engine._wavelet_transform(extended, frames, coswin, sinwin)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12 * np.max(np.abs(expected)))


def test_sortidx_indexes_the_original_trials_after_nan_removal() -> None:
    data = np.arange(60.0).reshape(10, 6)
    result = engine.compute_erpimage(data, [3.0, np.nan, 1.0, 2.0, 1.0, np.nan], None, 1, 1, plotted=False)
    np.testing.assert_array_equal(result.sortidx, [2, 4, 3, 0])  # stable: equal sortvar keeps input order
    np.testing.assert_array_equal(result.outdata, data[:, [2, 4, 3, 0]])


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


def test_zero_data_keeps_eeglab_limits_without_hanging() -> None:
    result = engine.compute_erpimage(np.zeros((20, 6)), None, None, 1, 1, engine.parse_erpimage_options(["erp"]))
    assert np.isnan(result.limits[2]) and np.isnan(result.limits[3])
    assert result.clim == (0.0, 0.0)


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
    [("yes", None), ("2*x+1", lambda x: 2 * x + 1), ("(X-2)/4", lambda x: (x - 2) / 4), ("x.^2", lambda x: x**2)],
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
        (["allcohers", 1], {}, "deprecated"),
        (["erp", "5"], {}, "unknown arg 5"),
        ([], {"erp": "sometimes"}, "'erp' must be"),
        (["caxis", [1, 2, 3]], {}, "caxis arg"),
        (["coher", [1, 2, 3, 4]], {}, "coher arg"),
        (["coher", [10, 10, 0.5]], {}, "alpha value"),
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
        (["spec", [1]], {}, "length 2"),
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
        ((np.zeros((10, 4)), [1.0, 2.0, 3.0]), {}, "doesn't divide"),
        ((np.zeros((10, 4)), [np.nan, np.nan, np.nan, 1.0]), {}, "Too few trials"),
        ((np.zeros((10, 3)), None, None, 2), {}, "too short"),
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


@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
def test_nan_after_smoothing_zeroes_the_whole_row_column_block() -> None:
    data = np.ones((6, 8))
    data[2, 3], data[2, 4] = np.inf, -np.inf  # inf - inf turns into NaN inside the moving average
    data[4, 6] = np.inf
    data[4, 7] = -np.inf
    result = engine.compute_erpimage(data, None, np.arange(6.0), 2, 1, plotted=False)
    rows, columns = np.nonzero(result.outdata == 0)
    assert set(rows) == {2, 4} and set(columns) >= {3, 6}


def test_percentile_erps_with_fewer_trials_than_groups_give_missing_rows() -> None:
    data = np.arange(30.0).reshape(10, 3)
    erp = engine.compute_erpimage(data, None, np.arange(10.0), 1, 1, engine.parse_erpimage_options(erp=4)).erp
    np.testing.assert_allclose(erp[:3], data.T)
    assert np.all(np.isnan(erp[3]))


@pytest.mark.parametrize(
    ("options", "check"),
    [
        (["timewarp", (np.full((8, 1), 120.0), [150.0], ["r--"])], lambda r: r.vert_colors == ("r--",)),
        (["timewarp", (np.full((8, 1), 120.0),), "auxvar", np.zeros((1, 8))], lambda r: r.verttimes is None),
        (["avg_type", "gaussian"], lambda r: r.outdata.shape[1] == 8),
        (["ampsort", [200, 0, 10, 10]], lambda r: r.sort_mode == "amp"),
        (["phasesort", [200, 25, 10, 900]], lambda r: r.sort_mode == "phase"),
        (["phasesort", [200, 25, 10], "erp"], lambda r: r.erp.shape == (1, 64)),
        (["coher", [10, 10, 0.05], "limits", [np.nan] * 8 + [1.0]], lambda r: r.limits[8] == 1.0),
        (["plotamps", "coher", [10, 10, 0.05], "limits", [np.nan] * 8 + [2.0]], lambda r: r.limits[8] == 2.0),
        (["coher", [10, 10, 0.05], "baselinedb", [-1000, -900]], lambda r: np.isfinite(r.limits[8])),
        (["sortvar_limits", [-100, 1000]], lambda r: r.trial_limits == (1.0, 8.0)),
        (["renorm", "-x+3"], lambda r: np.isfinite(r.outvar).all()),
    ],
)
def test_engine_option_branches(options, check) -> None:
    rng = np.random.default_rng(4)
    data = rng.normal(size=(64, 8))
    times = np.arange(64.0) * 1000 / 128 + 100
    avewidth = 0 if options[:2] == ["avg_type", "gaussian"] else 1
    result = engine.compute_erpimage(
        data, np.arange(8.0), times, avewidth, 0, engine.parse_erpimage_options(options), rng=np.random.RandomState(0)
    )
    assert check(result)


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
    zero = engine._amp_coh_limits(np.zeros(3), np.zeros(3), None, False, np.nan, np.nan, np.nan, np.nan)
    assert zero[:2] == (0.0, 0.0)


def test_small_helpers_cover_degenerate_inputs() -> None:
    assert engine.find_corresponding_point(50.0, np.array([1.0, 2.0]), np.array([1.0, 2.0])) is None
    assert engine.find_corresponding_point(2.0, np.array([1.0, 2.0, 2.0]), np.array([1.0, 2.0, 3.0])) == 2.5
    np.testing.assert_array_equal(engine._seqsum(np.zeros((0, 3)), axis=0), np.zeros(3))
    psd, freqs = engine.welch_psd(np.ones(4), 1, 100.0)
    assert psd.shape == freqs.shape
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
        (["auxvar", ([1, 2], [3, 4], [5, 6])], "length-2 cell"),
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


@pytest.mark.parametrize(
    ("args", "kwargs", "message"),
    [
        ((np.zeros((40, 6)), None, np.arange(40.0)[::-1]), {}, "srate"),
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
        direct[frame] = coswin @ epoch + 1j * (sinwin @ epoch)
    offset = engine._wavelet_transform(values + 1e6, frames, coswin, sinwin)
    np.testing.assert_allclose(offset, direct, rtol=0, atol=1e-9 * np.max(np.abs(direct)))


def test_weighted_windows_wider_than_their_weights_match_the_general_movav_port() -> None:
    values = np.random.default_rng(8).normal(size=(2, 12))
    weights = np.array([0.2, 0.5, 0.3])
    expected, _x = movav(values, np.arange(1, 13), 3.5, 1.0, None, None, weights)
    actual, _x = engine._moving_average(values, 3.5, 1.0, weights)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)


def test_remaining_input_forms_and_checks() -> None:
    data = np.random.default_rng(9).normal(size=(64, 6))
    times = np.arange(64.0) * 1000 / 128
    flat = engine.compute_erpimage(data.reshape(-1, order="F"), np.arange(6.0), times, 1, 1, plotted=False)
    np.testing.assert_array_equal(flat.outdata, data)
    with pytest.raises(ValueError, match=r"\(frames, trials\)"):
        engine.compute_erpimage(np.zeros((2, 3, 4, 5)), None, np.arange(3.0), plotted=False)
    with pytest.raises(ValueError, match="topphase"):
        engine.compute_erpimage(
            data, None, times, options=engine.parse_erpimage_options(phasesort=[100, 0, 10, 12, 200])
        )
    with pytest.raises(ValueError, match="amplitude-sorting freq"):
        engine.compute_erpimage(data, None, times, options=engine.parse_erpimage_options(ampsort=[100, 0, 1]))
    with pytest.raises(ValueError, match="cannot be < 1"):
        engine.compute_erpimage(data, None, times, 0.5, options=engine.parse_erpimage_options(avg_type="gaussian"))
    clamped = engine.compute_erpimage(
        data, None, times, options=engine.parse_erpimage_options(ampsort=[100, 0, 10, 900]), plotted=False
    )
    assert clamped.sort_mode == "amp"
    amps = engine.compute_erpimage(
        data,
        None,
        times,
        options=engine.parse_erpimage_options(plotamps=True, coher=[10, 10, 0.05], caxis=[-3, 3]),
        rng=np.random.default_rng(0),
    )
    assert amps.clim == (-3.0, 3.0)
    marks, ref, styles = engine._timewarp_marks(
        (np.full((6, 3), [50.0, 100.0, 150.0]), None, ("r", "g")), times, 128.0, 64
    )
    assert styles == ("", "r", "g", "r", "")
    np.testing.assert_allclose(engine._gauss(1, 2.0), [np.exp(-4.0)])
