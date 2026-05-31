from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import eegprep.functions.sigprocfunc.eegplot as eegplot_module
from eegprep.functions.popfunc.pop_eegplot import pop_eegplot
from eegprep.functions.popfunc.pop_loadset import pop_loadset
from eegprep.functions.sigprocfunc.eegplot import (
    build_eegplot_model,
    decimate_minmax,
    eegplot,
    event_latency_to_sample,
    normalize_events,
    parse_eegplot_options,
    visible_sample_bounds,
)
from tests.fixtures import create_test_eeg


SAMPLE_DATASET = Path(__file__).resolve().parents[1] / "sample_data" / "eeglab_data.set"


def test_parse_eegplot_options_accepts_eeglab_key_value_pairs() -> None:
    options = parse_eegplot_options(("srate", 100, "xgrid", "on"), {"winlength": 2})

    assert options == {"srate": 100, "xgrid": "on", "winlength": 2}


def test_parse_eegplot_options_rejects_unknown_options() -> None:
    with pytest.raises(ValueError, match="unrecognized option"):
        parse_eegplot_options((), {"bogus": 1})


def test_continuous_data_normalization_defaults_and_bounds() -> None:
    data = np.arange(20, dtype=float).reshape(2, 10)
    model = build_eegplot_model(data, srate=10, winlength=0.4, spacing=2, show=False)

    assert model.data.data.shape == (2, 10, 1)
    assert model.data.flat_data.shape == (2, 10)
    assert model.data.mode == "continuous"
    assert model.state.dispchans == 2
    assert visible_sample_bounds(model.data, model.state) == (0, 4)


def test_epoched_data_flattens_in_eeglab_trial_order_and_clamps_window() -> None:
    data = np.zeros((1, 4, 3), dtype=float)
    data[0, :, 0] = [1, 2, 3, 4]
    data[0, :, 1] = [5, 6, 7, 8]
    data[0, :, 2] = [9, 10, 11, 12]
    model = build_eegplot_model(data, srate=4, winlength=1, time=2, spacing=1, show=False)

    np.testing.assert_array_equal(model.data.flat_data[0], np.arange(1, 13))
    assert model.data.mode == "epoched"
    assert visible_sample_bounds(model.data, model.state) == (4, 8)


def test_component_mode_uses_ica_activation_labels_without_mutating_eeg() -> None:
    eeg = create_test_eeg(n_channels=2, n_samples=5, n_trials=1, srate=10)
    eeg["icaact"] = np.array([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]], dtype=float)
    data_before = np.array(eeg["data"], copy=True)
    ica_before = np.array(eeg["icaact"], copy=True)

    model = build_eegplot_model(eeg, component=True, spacing=1, show=False)

    assert model.data.mode == "component"
    assert model.data.channel_labels == ("Comp 1", "Comp 2")
    np.testing.assert_array_equal(model.data.flat_data, eeg["icaact"])
    np.testing.assert_array_equal(eeg["data"], data_before)
    np.testing.assert_array_equal(eeg["icaact"], ica_before)


def test_spectral_and_overlay_inputs_are_normalized_together() -> None:
    data = np.arange(20, dtype=float).reshape(2, 10)
    overlay = data + 100
    freqs = np.linspace(1, 10, 10)

    model = build_eegplot_model(
        data,
        data2=overlay,
        freqs=freqs,
        freqlimits=[3, 7],
        spacing=1,
        winlength=2,
        show=False,
    )

    assert model.data.mode == "spectral"
    assert model.data.data.shape == (2, 5, 1)
    assert model.state.limits == (3.0, 7.0)
    np.testing.assert_array_equal(model.data.x_values, freqs[2:7])
    np.testing.assert_array_equal(model.data.flat_data2, overlay[:, 2:7])


def test_event_latency_conversion_uses_eeglab_one_based_samples() -> None:
    model = build_eegplot_model(
        np.zeros((1, 10)), srate=10, spacing=1, events=[{"type": "a", "latency": 1}], show=False
    )

    events = normalize_events([{"type": "stim", "latency": 1}, {"type": "boundary", "latency": 10, "duration": 2}])

    assert event_latency_to_sample(1, model.data) == 0
    assert event_latency_to_sample(10, model.data) == 9
    assert events[0].type == "stim"
    assert events[1].duration == 2


def test_winrej_state_preserves_color_and_channel_mask() -> None:
    model = build_eegplot_model(
        np.zeros((3, 10)),
        srate=10,
        spacing=1,
        winrej=[[1, 5, 0.1, 0.2, 0.3, 1, 0, 1]],
        show=False,
    )

    assert len(model.state.winrej) == 1
    assert model.state.winrej[0].color == (0.1, 0.2, 0.3)
    assert model.state.winrej[0].channel_mask == (True, False, True)


def test_wincolor_sets_normalized_marking_color() -> None:
    model = build_eegplot_model(np.zeros((2, 10)), spacing=1, wincolor=(0.5, 0.2, 0.1), show=False)

    assert model.state.mark_color == (0.5, 0.2, 0.1)


def test_wincolor_rejects_out_of_range_rgb_values() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        build_eegplot_model(np.zeros((2, 10)), spacing=1, wincolor=(255, 255, 255), show=False)


def test_winrej_rejects_out_of_range_rows() -> None:
    with pytest.raises(ValueError, match="sample range"):
        build_eegplot_model(np.zeros((2, 10)), spacing=1, winrej=[[0, 11]], show=False)


def test_decimation_preserves_endpoints_and_limits_point_count() -> None:
    x = np.arange(1000, dtype=float)
    y = np.sin(x / 10)

    dec_x, dec_y = decimate_minmax(x, y, pixel_width=80)

    assert dec_x.size <= 160
    assert dec_x[0] == 0
    assert dec_x[-1] == 999
    assert dec_y.size == dec_x.size


def test_decimation_handles_all_nan_segments() -> None:
    x = np.arange(1000, dtype=float)
    y = np.full(1000, np.nan)

    dec_x, dec_y = decimate_minmax(x, y, pixel_width=20)

    assert dec_x[0] == 0
    assert dec_x[-1] == 999
    assert np.isnan(dec_y).all()


def test_eegplot_show_false_returns_model_and_does_not_import_qt() -> None:
    model = eegplot(np.zeros((2, 20)), "srate", 20, spacing=1, show=False)

    assert model.data.n_channels == 2
    assert model.state.srate == 20


def test_pop_eegplot_returns_unchanged_eeg_and_history_command() -> None:
    eeg = create_test_eeg(n_channels=2, n_samples=10, n_trials=1, srate=10)
    data_before = np.array(eeg["data"], copy=True)

    out, command = pop_eegplot(eeg, return_com=True, show=False, spacing=1)

    assert out is eeg
    np.testing.assert_array_equal(eeg["data"], data_before)
    assert command == "pop_eegplot(EEG, 1, 0, 1)"


def test_pop_eegplot_component_mode_computes_activations_once(monkeypatch: pytest.MonkeyPatch) -> None:
    eeg = create_test_eeg(n_channels=2, n_samples=10, n_trials=1, srate=10)
    calls = 0

    def fake_component_activations(dataset: dict) -> np.ndarray:
        nonlocal calls
        calls += 1
        assert dataset is eeg
        return np.ones((2, 10, 1), dtype=float)

    monkeypatch.setattr(eegplot_module, "component_activations", fake_component_activations)

    out, command = pop_eegplot(eeg, icacomp=0, return_com=True, show=False)

    assert out is eeg
    assert calls == 1
    assert command == "pop_eegplot(EEG, 0, 0, 1)"


def test_sample_data_eeglab_set_builds_non_mutating_model() -> None:
    eeg = pop_loadset(str(SAMPLE_DATASET))
    data_before = np.array(eeg["data"], copy=True)

    model = build_eegplot_model(eeg, winlength=1, show=False)

    assert model.data.n_channels == int(eeg["nbchan"])
    assert model.state.srate == float(eeg["srate"])
    np.testing.assert_array_equal(eeg["data"], data_before)
