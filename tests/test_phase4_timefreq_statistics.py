from __future__ import annotations

import ast
from copy import deepcopy
import importlib
import os
from pathlib import Path
from unittest import mock

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import scipy.io

from eegprep.functions.guifunc.menu_actions import MenuActionDispatcher, action_kind
from eegprep.functions.guifunc.spec import controls_by_tag
from eegprep.functions.guifunc.session import EEGPrepSession
from eegprep.functions.popfunc.pop_epoch import pop_epoch
from eegprep.functions.popfunc.pop_eventstat import event_values, pop_eventstat, pop_eventstat_dialog_spec
from eegprep.functions.popfunc.pop_crossf import pop_crossf
from eegprep.functions.popfunc.pop_loadset import pop_loadset
from eegprep.functions.popfunc.pop_newcrossf import pop_newcrossf, pop_newcrossf_dialog_spec
from eegprep.functions.popfunc.pop_newtimef import _run_gui as run_newtimef_gui
from eegprep.functions.popfunc.pop_newtimef import pop_newtimef, pop_newtimef_dialog_spec
from eegprep.functions.popfunc.pop_timef import pop_timef
from eegprep.functions.popfunc.pop_signalstat import pop_signalstat, pop_signalstat_dialog_spec
from eegprep.functions.sigprocfunc.signalstat import signalstat
from eegprep.functions.timefreqfunc.dftfilt3 import dftfilt3
from eegprep.functions.timefreqfunc.newcrossf import newcrossf
from eegprep.functions.timefreqfunc.newtimef import newtimef
from eegprep.functions.timefreqfunc.newtimefbaseln import newtimefbaseln
from eegprep.functions.timefreqfunc.newtimefpowerunit import newtimefpowerunit
from eegprep.functions.timefreqfunc.timefreq import timefreq
from tests.fixtures import SAMPLE_DATASET_PATH, create_test_eeg_with_ica


@pytest.fixture(scope="module")
def sample_eeg():
    return pop_loadset(SAMPLE_DATASET_PATH)


@pytest.fixture(scope="module")
def sample_epoch(sample_eeg):
    eeg, _command = pop_epoch(deepcopy(sample_eeg), ["square"], [-0.1, 0.2], return_com=True)
    return eeg


@pytest.fixture
def ica_epoch():
    return create_test_eeg_with_ica(n_channels=6, n_samples=96, n_trials=5, n_components=4)


def test_newtimef_synthetic_returns_deterministic_shapes():
    srate = 128
    times = np.arange(0, 1, 1 / srate)
    trials = np.stack([np.sin(2 * np.pi * 10 * times), np.sin(2 * np.pi * 10 * times + 0.2)], axis=1)

    result = newtimef(trials, trials.shape[0], [0, 1000], srate, 0, freqs=[5, 20], timesout=12, plot="off")

    assert result.ersp.shape == result.itc.shape == (result.freqs.size, result.times.size)
    assert result.tfdata.shape[2] == trials.shape[1]
    assert result.times.size <= 12
    assert np.isfinite(result.ersp).all()
    assert np.all(np.abs(result.itc) <= 1 + 1e-12)


def test_newtimef_rejects_unknown_options():
    signal = np.sin(2 * np.pi * 10 * np.arange(128) / 128)

    with pytest.raises(TypeError, match="unexpected keyword"):
        newtimef(signal, 128, [0, 1000], 128, 0, unsupported_option=1)


def test_newtimef_nonzero_cycles_use_wavelet_time_grid(sample_epoch):
    result = pop_newtimef(sample_epoch, 1, 1, [-100, 200], [3, 0.8], plot="off")

    assert result.times.size > 1
    assert result.freqs.size > 0
    assert result.tfdata.shape == (result.freqs.size, result.times.size, sample_epoch["trials"])


def test_newcrossf_identical_synthetic_signals_have_unit_phase_coherence():
    srate = 128
    times = np.arange(0, 1, 1 / srate)
    trials = np.stack([np.sin(2 * np.pi * 12 * times), np.sin(2 * np.pi * 12 * times + 0.3)], axis=1)

    result = newcrossf(trials, trials, trials.shape[0], [0, 1000], srate, 0, freqs=[8, 16], plot="off")

    assert result.coherence.shape == result.phase.shape
    assert np.nanmean(result.coherence) > 0.99
    assert np.nanmax(np.abs(result.phase)) < 1e-10


def test_newcrossf_single_trial_switches_to_cross_spectrum():
    rng = np.random.default_rng(0)
    first = rng.normal(size=512)
    second = rng.normal(size=512)

    result = newcrossf(first, second, 512, [0, 511], 256, 0, plot="off")

    assert result.coherence.shape == result.phase.shape
    assert result.allcoher.shape[2] == 1
    np.testing.assert_allclose(result.coherence, np.abs(result.allcoher[:, :, 0]))


def test_newcrossf_rejects_unknown_options():
    signal = np.sin(2 * np.pi * 10 * np.arange(128) / 128)

    with pytest.raises(TypeError, match="Unsupported newcrossf option"):
        newcrossf(signal, signal, 128, [0, 1000], 128, 0, unsupported_option=1, plot="off")


def test_pop_newtimef_channel_and_component_paths_are_replayable(sample_epoch, ica_epoch):
    result, command = pop_newtimef(sample_epoch, 1, 1, [-100, 200], [0], plot="off", return_com=True)
    component_result, component_command = pop_newtimef(ica_epoch, 0, 1, [-100, 200], [0], plot="off", return_com=True)

    assert result.ersp.ndim == 2
    assert component_result.ersp.ndim == 2
    assert "pop_newtimef(EEG, 1, 1" in command
    assert "pop_newtimef(EEG, 0, 1" in component_command
    _assert_python_command(command)
    _assert_python_command(component_command)
    namespace = {"EEG": sample_epoch, "pop_newtimef": pop_newtimef}
    replayed = eval(command, namespace)
    assert replayed.ersp.shape == result.ersp.shape


def test_pop_newcrossf_channel_and_component_paths_are_replayable(sample_epoch, ica_epoch):
    result, command = pop_newcrossf(sample_epoch, 1, 1, 2, [-100, 200], [0], plot="off", return_com=True)
    component_result, component_command = pop_newcrossf(
        ica_epoch, 0, 1, 2, [-100, 200], [0], plot="off", return_com=True
    )

    assert result.coherence.ndim == 2
    assert component_result.coherence.ndim == 2
    assert "pop_newcrossf(EEG, 1, 1, 2" in command
    assert "pop_newcrossf(EEG, 0, 1, 2" in component_command
    _assert_python_command(command)
    _assert_python_command(component_command)


def test_pop_signalstat_sample_and_component_paths(sample_eeg, ica_epoch):
    result, command = pop_signalstat(sample_eeg, 1, 1, 5, return_com=True)
    component_result, component_command = pop_signalstat(ica_epoch, 0, 1, 5, return_com=True)

    assert np.isfinite(result.mean)
    assert np.isfinite(component_result.mean)
    assert result.trimmed_indices.size > 0
    assert "pop_signalstat(EEG, 1, 1, 5)" == command
    assert "pop_signalstat(EEG, 0, 1, 5)" == component_command
    _assert_python_command(command)
    plt.close(result.figure)
    plt.close(component_result.figure)


def test_pop_eventstat_extracts_sample_event_latencies(sample_eeg):
    values = event_values(sample_eeg, "latency", type=["square"])
    result, command = pop_eventstat(sample_eeg, "latency", ["square"], [], 5, return_com=True)

    assert values.size > 0
    assert result.mean == pytest.approx(float(np.mean(values)))
    assert "pop_eventstat(EEG, 'latency', ['square'], [], 5)" == command
    _assert_python_command(command)
    plt.close(result.figure)


def test_pop_eventstat_epoched_latrange_uses_epoch_relative_latency(sample_epoch):
    all_values = event_values(sample_epoch, "latency", type=["square"])
    ranged_values = event_values(sample_epoch, "latency", type=["square"], latrange=[-100, 200])

    assert ranged_values.size == all_values.size == sample_epoch["trials"]


def test_pop_newtimef_gui_defaults_are_replayable_and_honest(sample_eeg):
    result = run_newtimef_gui(sample_eeg, typeproc=1, renderer=_DefaultDialogRenderer())

    assert result["options"]["timesout"] == 200
    assert "plotphase" not in result["options"]
    assert "plottype" not in result["options"]


def test_pop_newtimef_baseline_modes_bootstrap_and_curve_plot(sample_epoch):
    result, command = pop_newtimef(
        sample_epoch,
        1,
        1,
        [-100, 200],
        [3, 0.8],
        basenorm="on",
        trialbase="full",
        alpha=0.1,
        naccu=12,
        rng=0,
        plottype="curve",
        plot="off",
        return_com=True,
    )

    assert result.ersp.shape == result.itc.shape
    assert result.erspboot.shape == (result.freqs.size, 2)
    assert result.itcboot.shape == (result.freqs.size,)
    assert result.ersp_significant.shape == result.ersp.shape
    assert "basenorm='on'" in command
    assert "trialbase='full'" in command
    _assert_python_command(command)


def test_pop_newtimef_accepts_supplied_bootstrap_limits(sample_epoch):
    result = pop_newtimef(
        sample_epoch,
        1,
        1,
        [-100, 200],
        [0],
        alpha=0.05,
        pboot=np.asarray([[-1, 1], [-1, 1]], dtype=float),
        rboot=np.asarray([0.5, 0.5], dtype=float),
        freqs=[8, 12],
        nfreqs=2,
        plot="off",
    )

    np.testing.assert_array_equal(result.erspboot, np.asarray([[-1, 1], [-1, 1]], dtype=float))
    np.testing.assert_array_equal(result.itcboot, np.asarray([0.5, 0.5], dtype=float))


def test_timefreq_statistics_dialog_specs_match_eeglab_control_inventory(sample_eeg):
    newtimef = pop_newtimef_dialog_spec(sample_eeg, typeproc=1)
    newcrossf = pop_newcrossf_dialog_spec(sample_eeg, typeproc=1)
    signal_spec = pop_signalstat_dialog_spec(sample_eeg, typeproc=1)
    event_spec = pop_eventstat_dialog_spec(sample_eeg)

    assert newtimef.title == "Plot channel time frequency -- pop_newtimef()"
    assert controls_by_tag(newtimef)["num_button"].callback.name == "select_channels"
    assert controls_by_tag(newtimef)["baseline"].value == "0"
    assert controls_by_tag(newtimef)["calcpush"].enabled is True
    assert controls_by_tag(newtimef)["plotcurve"].enabled is True
    assert controls_by_tag(newtimef)["alpha"].enabled is True
    assert controls_by_tag(newcrossf)["coher"].value is False
    assert signal_spec.title == "Plot signal statistics -- pop_signalstat()"
    assert controls_by_tag(signal_spec)["percent"].value == "5"
    assert event_spec.title == "Plot event statistics -- pop_eventstat()"
    assert controls_by_tag(event_spec)["eventfield"].value == "latency"


def test_timefreq_statistics_menu_actions_are_implemented():
    assert action_kind("pop_newtimef:channels") == "implemented"
    assert action_kind("pop_newcrossf:components") == "implemented"
    assert action_kind("pop_signalstat:channels") == "implemented"
    assert action_kind("pop_eventstat") == "implemented"


def test_legacy_timef_and_crossf_wrappers_return_replayable_history(sample_epoch):
    timef_result, timef_command = pop_timef(sample_epoch, 1, 1, [-100, 200], [0], plot="off", return_com=True)
    crossf_result, crossf_command = pop_crossf(sample_epoch, 1, 1, 2, [-100, 200], [0], plot="off", return_com=True)

    assert timef_result.ersp.ndim == 2
    assert crossf_result.coherence.ndim == 2
    assert timef_command.startswith("pop_timef(EEG, 1, 1")
    assert crossf_command.startswith("pop_crossf(EEG, 1, 1, 2")
    _assert_python_command(timef_command)
    _assert_python_command(crossf_command)


def test_newcrossf_supported_phase4_options_are_headless():
    srate = 128
    times = np.arange(0, 1, 1 / srate)
    trials = np.stack(
        [
            np.sin(2 * np.pi * 12 * times),
            np.sin(2 * np.pi * 12 * times + 0.3),
            np.sin(2 * np.pi * 12 * times + 0.7),
        ],
        axis=1,
    )

    result = newcrossf(
        trials,
        trials,
        trials.shape[0],
        [0, 1000],
        srate,
        0,
        freqs=[8, 16],
        type="phasecoher2",
        subitc="on",
        alpha=0.1,
        naccu=12,
        rng=0,
        plot="off",
    )
    amp = newcrossf(trials, trials, trials.shape[0], [0, 1000], srate, 0, freqs=[8, 16], type="amp", plot="off")

    assert result.rboot.shape == (result.freqs.size,)
    assert result.significant.shape == result.coherence.shape
    assert np.nanmax(np.abs(result.coherence)) <= 1 + 1e-12
    assert amp.coherence.shape == amp.phase.shape


def test_newtimefpowerunit_matches_eeglab_labels():
    assert newtimefpowerunit({"scale": "log", "baseline": 0, "basenorm": "off"}) == "dB"
    assert newtimefpowerunit({"scale": "abs", "baseline": [float("nan")], "basenorm": "off"}) == "uV^{2}/Hz"
    assert newtimefpowerunit({"scale": "abs", "baseline": 0, "basenorm": "on"}) == "std."


@pytest.mark.parametrize(
    ("action", "module_path", "expected_kwargs", "command"),
    [
        (
            "pop_newtimef:channels",
            "eegprep.functions.popfunc.pop_newtimef.pop_newtimef",
            {"typeproc": 1, "return_com": True},
            "pop_newtimef(EEG, 1, 1)",
        ),
        (
            "pop_newcrossf:components",
            "eegprep.functions.popfunc.pop_newcrossf.pop_newcrossf",
            {"typeproc": 0, "return_com": True},
            "pop_newcrossf(EEG, 0, 1, 2)",
        ),
        (
            "pop_signalstat:channels",
            "eegprep.functions.popfunc.pop_signalstat.pop_signalstat",
            {"typeproc": 1, "return_com": True},
            "pop_signalstat(EEG, 1, 1, 5)",
        ),
        (
            "pop_eventstat",
            "eegprep.functions.popfunc.pop_eventstat.pop_eventstat",
            {"return_com": True},
            "pop_eventstat(EEG, 'latency', [], [], 5)",
        ),
    ],
)
def test_timefreq_statistics_menu_dispatch_records_history(sample_eeg, action, module_path, expected_kwargs, command):
    session = EEGPrepSession()
    session.store_current(sample_eeg, new=True)
    stored_eeg = session.EEG
    dispatcher = MenuActionDispatcher(session)

    with mock.patch(module_path, return_value=("figure", command)) as pop_function:
        dispatcher.dispatch(action)

    pop_function.assert_called_once()
    assert len(pop_function.call_args.args) == 1
    assert pop_function.call_args.args[0] is stored_eeg
    assert pop_function.call_args.kwargs == expected_kwargs
    assert session.EEG is stored_eeg
    assert session.ALLCOM[-1] == command


def test_signalstat_matches_numpy_for_known_vector():
    values = np.asarray([1, 2, 3, 4, 100], dtype=float)
    result = signalstat(values, plotlab=0, percent=40)

    assert result.mean == pytest.approx(np.mean(values))
    assert result.std == pytest.approx(np.std(values, ddof=1))
    assert result.median == pytest.approx(np.median(values))
    assert result.zlow == pytest.approx(1.5)
    assert result.zhigh == pytest.approx(52.0)
    assert result.trimmed_indices.tolist() == [1, 2, 3]


def test_signalstat_plots_topographic_context_when_map_is_available(ica_epoch):
    result = pop_signalstat(ica_epoch, 0, 1, 5)

    titles = [axis.get_title() for axis in result.figure.axes]
    assert "Topographic map" in titles
    plt.close(result.figure)


@pytest.mark.matlab
def test_signalstat_statistics_match_eeglab(tmp_path):
    if os.environ.get("EEGPREP_SKIP_MATLAB") == "1":
        pytest.skip("MATLAB tests disabled via EEGPREP_SKIP_MATLAB")
    try:
        matlab_engine = importlib.import_module("matlab.engine")
    except ImportError as exc:
        pytest.skip(f"MATLAB not available: {exc}")
    eeglab_root = _eeglab_reference_root()
    if eeglab_root is None:
        pytest.skip("EEGLAB reference checkout not available")

    values = np.asarray([1.0, 2.5, -3.0, 4.25, 5.5, 9.0, 12.0, 20.0])
    output = tmp_path / "signalstat.mat"
    engine = matlab_engine.start_matlab()
    try:
        engine.addpath(str(eeglab_root / "functions" / "sigprocfunc"), nargout=0)
        engine.eval(
            f"""
            data = [{_matlab_vector(values)}];
            [M,SD,sk,k,med,zlow,zhi,tM,tSD,tndx,ksh] = signalstat(data, 0, [], 10);
            save('{_matlab_string(output)}', 'M', 'SD', 'sk', 'k', 'med', 'zlow', 'zhi', 'tM', 'tSD', 'tndx', 'ksh');
            """,
            nargout=0,
        )
    finally:
        engine.quit()

    result = signalstat(values, plotlab=0, percent=10)
    matlab = scipy.io.loadmat(output, squeeze_me=True)
    np.testing.assert_allclose(result.mean, matlab["M"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result.std, matlab["SD"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result.skewness, matlab["sk"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result.kurtosis, matlab["k"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result.median, matlab["med"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result.zlow, matlab["zlow"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result.zhigh, matlab["zhi"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result.trimmed_mean, matlab["tM"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result.trimmed_std, matlab["tSD"], rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(result.trimmed_indices + 1, np.asarray(matlab["tndx"], dtype=int).ravel())


@pytest.mark.matlab
def test_timefreq_helpers_match_eeglab_deterministic_outputs(tmp_path):
    if os.environ.get("EEGPREP_SKIP_MATLAB") == "1":
        pytest.skip("MATLAB tests disabled via EEGPREP_SKIP_MATLAB")
    try:
        matlab_engine = importlib.import_module("matlab.engine")
    except ImportError as exc:
        pytest.skip(f"MATLAB not available: {exc}")
    eeglab_root = _eeglab_reference_root()
    if eeglab_root is None:
        pytest.skip("EEGLAB reference checkout not available")

    srate = 128.0
    sample_times = np.arange(128) / srate
    trials = np.stack(
        [
            np.sin(2 * np.pi * 10 * sample_times),
            np.sin(2 * np.pi * 10 * sample_times + 0.2),
        ],
        axis=1,
    )
    power = np.arange(1, 1 + 3 * 5 * 4, dtype=float).reshape(3, 5, 4) / 10.0
    time_values = np.asarray([-200, -100, 0, 100, 200], dtype=float)
    inputs = tmp_path / "timefreq_inputs.mat"
    output = tmp_path / "timefreq_outputs.mat"
    scipy.io.savemat(inputs, {"data": trials, "P": power, "time_values": time_values})

    engine = matlab_engine.start_matlab()
    try:
        engine.addpath(engine.genpath(str(eeglab_root / "functions")), nargout=0)
        engine.eval(
            f"""
            load('{_matlab_string(inputs)}');
            [wavelet,cycles,freqresol,timeresol] = dftfilt3([6 10], [3 5], {srate}, 'cycleinc', 'linear');
            wav1 = wavelet{{1}};
            wav2 = wavelet{{2}};
            [tf,freqs,times] = timefreq(data, {srate}, 'cycles', 0, 'tlimits', [0 1000], ...
                'freqs', [5 20], 'ntimesout', 12, 'padratio', 2, 'verbose', 'off');
            [PP,baseln,mbase] = newtimefbaseln(P, time_values, 'baseline', [-200 0], ...
                'basenorm', 'off', 'trialbase', 'off', 'verbose', 'off');
            save('{_matlab_string(output)}', 'wav1', 'wav2', 'cycles', 'freqresol', 'timeresol', ...
                'tf', 'freqs', 'times', 'PP', 'baseln', 'mbase');
            """,
            nargout=0,
        )
    finally:
        engine.quit()

    matlab = scipy.io.loadmat(output, squeeze_me=True)
    wavelets, py_cycles, py_freqresol, py_timeresol = dftfilt3([6, 10], [3, 5], srate, cycleinc="linear")
    decomposition = timefreq(
        trials,
        srate,
        frames=128,
        cycles=0,
        tlimits=[0, 1000],
        freqs=[5, 20],
        ntimesout=12,
        padratio=2,
        verbose="off",
    )
    py_power, py_baseln, py_mbase = newtimefbaseln(
        power,
        time_values,
        baseline=[-200, 0],
        basenorm="off",
        trialbase="off",
    )

    np.testing.assert_allclose(wavelets[0], matlab["wav1"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(wavelets[1], matlab["wav2"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(py_cycles, matlab["cycles"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(py_freqresol, matlab["freqresol"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(py_timeresol, matlab["timeresol"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(decomposition.freqs, np.asarray(matlab["freqs"]).ravel(), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(decomposition.times, np.asarray(matlab["times"]).ravel(), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(decomposition.tfdata, matlab["tf"], rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(py_power, matlab["PP"], rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(py_baseln + 1, np.asarray(matlab["baseln"], dtype=int).ravel())
    np.testing.assert_allclose(py_mbase, matlab["mbase"], rtol=1e-12, atol=1e-12)


def _assert_python_command(command: str) -> None:
    ast.parse(command, mode="eval")


def _matlab_vector(values: np.ndarray) -> str:
    return " ".join(f"{float(value):.17g}" for value in np.asarray(values, dtype=float).ravel())


def _matlab_string(path: Path) -> str:
    return str(path).replace("'", "''")


def _eeglab_reference_root() -> Path | None:
    repo_root = Path(__file__).resolve().parents[1]
    candidates = []
    if os.environ.get("EEGPREP_EEGLAB_ROOT"):
        candidates.append(Path(os.environ["EEGPREP_EEGLAB_ROOT"]))
    candidates.extend(
        [
            repo_root / "src" / "eegprep" / "eeglab",
            repo_root.parent / "eeglab",
            Path("/tmp/eeglab-timefreq-ref"),
        ]
    )
    for candidate in candidates:
        if (candidate / "functions" / "sigprocfunc" / "signalstat.m").exists():
            return candidate
    return None


class _DefaultDialogRenderer:
    def run(self, spec, initial_values=None):
        _ = initial_values
        return {control.tag: control.value for control in spec.controls if control.tag}
