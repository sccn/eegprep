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
from eegprep.functions.popfunc.pop_loadset import pop_loadset
from eegprep.functions.popfunc.pop_newcrossf import pop_newcrossf, pop_newcrossf_dialog_spec
from eegprep.functions.popfunc.pop_newtimef import pop_newtimef, pop_newtimef_dialog_spec
from eegprep.functions.popfunc.pop_signalstat import pop_signalstat, pop_signalstat_dialog_spec
from eegprep.functions.sigprocfunc.signalstat import signalstat
from eegprep.functions.timefreqfunc.newcrossf import newcrossf
from eegprep.functions.timefreqfunc.newtimef import newtimef
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


def test_newcrossf_identical_synthetic_signals_have_unit_phase_coherence():
    srate = 128
    times = np.arange(0, 1, 1 / srate)
    trials = np.stack([np.sin(2 * np.pi * 12 * times), np.sin(2 * np.pi * 12 * times + 0.3)], axis=1)

    result = newcrossf(trials, trials, trials.shape[0], [0, 1000], srate, 0, freqs=[8, 16], plot="off")

    assert result.coherence.shape == result.phase.shape
    assert np.nanmean(result.coherence) > 0.99
    assert np.nanmax(np.abs(result.phase)) < 1e-10


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


def test_timefreq_statistics_dialog_specs_match_eeglab_control_inventory(sample_eeg):
    newtimef = pop_newtimef_dialog_spec(sample_eeg, typeproc=1)
    newcrossf = pop_newcrossf_dialog_spec(sample_eeg, typeproc=1)
    signal_spec = pop_signalstat_dialog_spec(sample_eeg, typeproc=1)
    event_spec = pop_eventstat_dialog_spec(sample_eeg)

    assert newtimef.title == "Plot channel time frequency -- pop_newtimef()"
    assert controls_by_tag(newtimef)["num_button"].callback.name == "select_channels"
    assert controls_by_tag(newtimef)["baseline"].value == "0"
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


@pytest.mark.matlab
def test_signalstat_statistics_match_eeglab(tmp_path):
    if os.environ.get("EEGPREP_SKIP_MATLAB") == "1":
        pytest.skip("MATLAB tests disabled via EEGPREP_SKIP_MATLAB")
    try:
        matlab_engine = importlib.import_module("matlab.engine")
    except ImportError as exc:
        pytest.skip(f"MATLAB not available: {exc}")
    eeglab_root = Path("/tmp/eeglab-timefreq-ref")
    if not eeglab_root.exists():
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


def _assert_python_command(command: str) -> None:
    ast.parse(command, mode="eval")


def _matlab_vector(values: np.ndarray) -> str:
    return " ".join(f"{float(value):.17g}" for value in np.asarray(values, dtype=float).ravel())


def _matlab_string(path: Path) -> str:
    return str(path).replace("'", "''")
