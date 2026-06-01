from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import eegprep.functions.sigprocfunc.eegplot as eegplot_module
import eegprep.functions.popfunc.pop_eegplot as pop_eegplot_module
from eegprep.functions.popfunc.pop_eegplot import pop_eegplot
from eegprep.functions.popfunc.pop_eegplot import apply_eegplot_rejections
from eegprep.functions.popfunc.pop_eegplot import eegplot_accept_creates_dataset
from eegprep.functions.popfunc.pop_loadset import pop_loadset
from eegprep.functions.sigprocfunc.eegplot import (
    add_winrej_region,
    build_eegplot_model,
    decimate_minmax,
    eegplot,
    eegplot2event,
    eegplot2trial,
    event_latency_to_sample,
    normalize_events,
    parse_eegplot_options,
    toggle_winrej_at_sample,
    trial2eegplot,
    visible_sample_bounds,
    winrej_to_array,
)
from tests.fixtures import create_test_eeg, matlab_engine_available


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
    color_events = normalize_events(
        [
            {"type": "stim", "latency": 1},
            {"type": "resp", "latency": 2},
            {"type": "boundary", "latency": 3},
        ]
    )
    assert [event.color_index for event in color_events] == [2, 1, 0]


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


def test_trial2eegplot_converts_epoch_and_channel_marks() -> None:
    rows = trial2eegplot(
        [True, False, True],
        np.array([[True, False, False], [False, False, True]]),
        4,
        [0.1, 0.2, 0.3],
    )

    np.testing.assert_array_equal(rows[:, :5], [[0, 3, 0.1, 0.2, 0.3], [8, 11, 0.1, 0.2, 0.3]])
    np.testing.assert_array_equal(rows[:, 5:], [[1, 0], [0, 1]])


def test_eegplot2trial_filters_colors_and_handles_first_epoch_boundary() -> None:
    rows = np.array(
        [
            [1, 9, 1.0, 0.9, 0.9, 1, 0],
            [10, 19, 0.3, 0.4, 0.5, 0, 1],
            [15, 17, 1.0, 0.9, 0.9, 1, 1],
        ]
    )

    trial_marks, row_marks = eegplot2trial(rows, 10, 3, color=[[1.0, 0.9, 0.9]])
    excluded_marks, excluded_rows = eegplot2trial(rows, 10, 3, colorout=[[1.0, 0.9, 0.9]])

    np.testing.assert_array_equal(trial_marks, [True, False, False])
    np.testing.assert_array_equal(row_marks, [[True, False, False], [False, False, False]])
    np.testing.assert_array_equal(excluded_marks, [False, True, False])
    np.testing.assert_array_equal(excluded_rows, [[False, False, False], [False, True, False]])


def test_eegplot2event_converts_continuous_marks_for_eeg_eegrej() -> None:
    rows = np.array([[2.2, 5.8, 0.7, 1.0, 0.9, 1], [8, 9, 0.1, 0.2, 0.3, 1]])

    events = eegplot2event(rows, -1, colorout=[[0.1, 0.2, 0.3]])

    np.testing.assert_array_equal(events, [[-1, 1, 2, 6, 0.7, 1.0, 0.9]])


def test_winrej_add_merge_reversed_duplicate_and_boundary_regions() -> None:
    regions = add_winrej_region([], 8, 3, n_channels=2, total_samples=10, color=[0.2, 0.3, 0.4])
    regions = add_winrej_region(regions, 7, 20, n_channels=2, total_samples=10, color=[0.2, 0.3, 0.4])
    regions = add_winrej_region(regions, 3, 10, n_channels=2, total_samples=10, color=[0.2, 0.3, 0.4])

    assert len(regions) == 1
    assert (regions[0].start, regions[0].end) == (3, 10)
    assert regions[0].channel_mask == (True, True)


def test_single_click_unmarks_or_toggles_channel_specific_marks() -> None:
    regions = add_winrej_region([], 2, 6, n_channels=3, total_samples=10, channel_index=1)
    regions = toggle_winrej_at_sample(regions, 4, n_channels=3, channel_index=1)

    assert regions == []

    regions = add_winrej_region([], 2, 6, n_channels=3, total_samples=10)
    regions = toggle_winrej_at_sample(regions, 4, n_channels=3, channel_index=0)
    regions = toggle_winrej_at_sample(regions, 4, n_channels=3)

    assert regions == []


def test_epoched_drag_marks_whole_epochs_and_merges_channel_masks() -> None:
    regions = add_winrej_region([], 15, 3, n_channels=2, total_samples=30, pnts=10)
    regions = add_winrej_region(regions, 12, 18, n_channels=2, total_samples=30, pnts=10, channel_index=1)

    np.testing.assert_array_equal(winrej_to_array(regions, 2)[:, :2], [[0, 9], [10, 19]])
    np.testing.assert_array_equal(winrej_to_array(regions, 2)[:, 5:], [[1, 1], [1, 1]])


def test_conversion_helpers_handle_empty_inputs() -> None:
    assert trial2eegplot([], np.zeros((2, 0)), 10).shape == (0, 7)
    assert eegplot2event([]).shape == (0, 7)
    trial_marks, row_marks = eegplot2trial([], 10, 3)
    np.testing.assert_array_equal(trial_marks, [False, False, False])
    assert row_marks.shape == (0, 3)


@pytest.mark.matlab
@pytest.mark.skipif(not matlab_engine_available(), reason="MATLAB engine not available or skipped")
def test_eegplot_conversion_helpers_match_matlab() -> None:
    from eegprep.functions.adminfunc.eeglabcompat import get_eeglab

    eeglab = get_eeglab("MAT")
    rej = np.array([1, 0, 1], dtype=float)
    rej_e = np.array([[1, 0, 0], [0, 0, 1]], dtype=float)
    color = np.array([0.1, 0.2, 0.3], dtype=float)
    rows = trial2eegplot(rej, rej_e, 4, color)

    matlab_rows = np.asarray(eeglab.trial2eegplot(rej, rej_e, 4, color), dtype=float)
    matlab_events = np.asarray(eeglab.eegplot2event(rows, -1), dtype=float)
    matlab_trial, matlab_elec = eeglab.eegplot2trial(rows, 4, 3, color, [], nargout=2)

    np.testing.assert_allclose(rows, matlab_rows)
    np.testing.assert_allclose(eegplot2event(rows, -1), matlab_events)
    py_trial, py_elec = eegplot2trial(rows, 4, 3, color, None)
    np.testing.assert_array_equal(py_trial, np.asarray(matlab_trial, dtype=bool).ravel())
    np.testing.assert_array_equal(py_elec, np.asarray(matlab_elec, dtype=bool))


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


def test_apply_eegplot_rejections_removes_continuous_regions_and_inserts_boundary() -> None:
    eeg = create_test_eeg(n_channels=1, n_samples=10, n_trials=1, srate=10)
    eeg["data"] = np.arange(10, dtype=float).reshape(1, 10)
    eeg["xmax"] = 0.9
    eeg["event"] = [{"type": "stim", "latency": 8.0, "urevent": 1}]
    eeg["urevent"] = [{"type": "stim", "latency": 8.0}]
    rows = np.array([[3, 5, 0.7, 1.0, 0.9, 1]])

    out, command = apply_eegplot_rejections(eeg, rows, return_com=True)

    np.testing.assert_array_equal(out["data"], [[0, 1, 5, 6, 7, 8, 9]])
    assert out["event"][0]["type"] == "boundary"
    assert out["event"][0]["latency"] == 2.5
    assert out["event"][0]["duration"] == 3.0
    assert out["event"][1]["latency"] == 5.0
    assert command == "pop_eegplot(EEG, 1, 0, 1)"


def test_apply_eegplot_rejections_updates_continuous_mark_only_rows() -> None:
    eeg = create_test_eeg(n_channels=2, n_samples=10, n_trials=1, srate=10)
    rows = np.array([[2, 5, 0.7, 1.0, 0.9, 1, 0]])

    out, command = apply_eegplot_rejections(eeg, rows, reject=0, return_com=True)

    assert command == "pop_eegplot(EEG, 1, 0, 0)"
    np.testing.assert_array_equal(out["data"], eeg["data"])
    np.testing.assert_array_equal(out["reject"]["rejmanualwinrej"], rows)
    np.testing.assert_array_equal(out["reject"]["rejmanualcol"], [1.0, 0.9, 0.9])


def test_apply_eegplot_rejections_clears_continuous_mark_only_rows_after_rejecting() -> None:
    eeg = create_test_eeg(n_channels=1, n_samples=10, n_trials=1, srate=10)
    eeg["data"] = np.arange(10, dtype=float).reshape(1, 10)
    eeg["reject"]["rejmanualwinrej"] = np.array([[3, 5, 0.7, 1.0, 0.9, 1]])
    rows = np.array([[3, 5, 0.7, 1.0, 0.9, 1]])

    out = apply_eegplot_rejections(eeg, rows, reject=1)

    np.testing.assert_array_equal(out["data"], [[0, 1, 5, 6, 7, 8, 9]])
    assert out["reject"]["rejmanualwinrej"].shape == (0, 6)


def test_apply_eegplot_rejections_updates_or_rejects_epochs() -> None:
    eeg = create_test_eeg(n_channels=2, n_samples=4, n_trials=3, srate=10)
    eeg["data"] = np.arange(24, dtype=float).reshape(2, 4, 3)
    rows = trial2eegplot([False, True, False], [[False, True, False], [False, False, False]], 4, [1.0, 0.9, 0.9])

    marked, _command = apply_eegplot_rejections(eeg, rows, reject=0, return_com=True)
    rejected = apply_eegplot_rejections(eeg, rows, reject=1)

    np.testing.assert_array_equal(marked["reject"]["rejmanual"], [False, True, False])
    np.testing.assert_array_equal(marked["reject"]["rejmanualE"], [[False, True, False], [False, False, False]])
    assert rejected["trials"] == 2


def test_apply_eegplot_rejections_empty_winrej_clears_epoch_marks() -> None:
    eeg = create_test_eeg(n_channels=2, n_samples=4, n_trials=3, srate=10)
    eeg["reject"]["rejmanual"] = np.array([False, True, False])
    eeg["reject"]["rejmanualE"] = np.array([[False, True, False], [False, False, False]])

    out = apply_eegplot_rejections(eeg, np.zeros((0, 7)), reject=0)

    np.testing.assert_array_equal(out["reject"]["rejmanual"], [False, False, False])
    np.testing.assert_array_equal(out["reject"]["rejmanualE"], np.zeros((2, 3), dtype=bool))


def test_pop_eegplot_superpose_two_passes_method_specific_color_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    eeg = create_test_eeg(n_channels=2, n_samples=4, n_trials=3, srate=10)
    eeg["reject"]["rejmanual"] = np.array([True, False, False])
    eeg["reject"]["rejmanualE"] = np.array([[True, False, False], [False, False, False]])
    eeg["reject"]["rejthresh"] = np.array([False, True, False])
    eeg["reject"]["rejthreshE"] = np.array([[False, False, False], [False, True, False]])
    eeg["reject"]["rejthreshcol"] = np.array([0.2, 0.8, 0.4])
    eeg["reject"]["disprej"] = ["thresh"]
    captured = {}

    def fake_eegplot(_eeg, *args, **kwargs):
        del args
        captured["winrej"] = kwargs["winrej"]
        return "window"

    monkeypatch.setattr(pop_eegplot_module, "eegplot", fake_eegplot)

    pop_eegplot(eeg, superpose=2)

    rows = captured["winrej"]
    np.testing.assert_array_equal(rows[:, :5], [[4, 7, 0.2, 0.8, 0.4], [0, 3, 1.0, 0.9, 0.9]])
    np.testing.assert_array_equal(rows[:, 5:], [[0, 1], [1, 0]])


def test_pop_eegplot_loads_continuous_mark_only_rows_when_updating_or_superposing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    eeg = create_test_eeg(n_channels=2, n_samples=10, n_trials=1, srate=10)
    stored = np.array([[2, 5, 0.7, 1.0, 0.9, 1, 0]])
    eeg["reject"]["rejmanualwinrej"] = stored
    captured = []

    def fake_eegplot(_eeg, *args, **kwargs):
        del args
        captured.append(kwargs["winrej"])
        return "window"

    monkeypatch.setattr(pop_eegplot_module, "eegplot", fake_eegplot)

    pop_eegplot(eeg, reject=0)
    pop_eegplot(eeg, reject=1, superpose=1)
    pop_eegplot(eeg, reject=1, superpose=0)
    pop_eegplot(eeg, reject=1, winrej=[])

    np.testing.assert_array_equal(captured[0], stored)
    np.testing.assert_array_equal(captured[1], stored)
    np.testing.assert_array_equal(captured[2], stored)
    assert np.asarray(captured[3]).shape == (0,)


def test_pop_eegplot_component_mode_computes_activations_once(monkeypatch: pytest.MonkeyPatch) -> None:
    eeg = create_test_eeg(n_channels=2, n_samples=10, n_trials=1, srate=10)
    eeg["icaweights"] = np.eye(2)
    eeg["icasphere"] = np.eye(2)
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


def test_pop_eegplot_component_mode_requires_ica() -> None:
    eeg = create_test_eeg(n_channels=2, n_samples=10, n_trials=1, srate=10)

    with pytest.raises(ValueError, match="run ICA"):
        pop_eegplot(eeg, icacomp=0, show=False)


def test_eegplot_accept_creates_dataset_only_when_reject_removes_data() -> None:
    continuous = create_test_eeg(n_channels=1, n_samples=10, n_trials=1, srate=10)
    continuous_out = dict(continuous, pnts=7, data=np.zeros((1, 7)))
    epoched = create_test_eeg(n_channels=1, n_samples=4, n_trials=3, srate=10)
    epoched_out = dict(epoched, trials=2, data=np.zeros((1, 4, 2)))

    assert eegplot_accept_creates_dataset(continuous, continuous_out, reject=1) is True
    assert eegplot_accept_creates_dataset(epoched, epoched_out, reject=1) is True
    assert eegplot_accept_creates_dataset(continuous, continuous_out, reject=0) is False
    assert eegplot_accept_creates_dataset(continuous, continuous, reject=1) is False


def test_sample_data_eeglab_set_builds_non_mutating_model() -> None:
    eeg = pop_loadset(str(SAMPLE_DATASET))
    data_before = np.array(eeg["data"], copy=True)

    model = build_eegplot_model(eeg, winlength=1, show=False)

    assert model.data.n_channels == int(eeg["nbchan"])
    assert model.state.srate == float(eeg["srate"])
    np.testing.assert_array_equal(eeg["data"], data_before)


def test_sample_data_pop_eegplot_channel_api_flow_returns_browser_model() -> None:
    eeg = pop_loadset(str(SAMPLE_DATASET))
    data_before = np.array(eeg["data"], copy=True)

    model = pop_eegplot(eeg, superpose=1, show=False, winlength=1)

    assert model.data.mode == "continuous"
    assert model.data.n_channels == int(eeg["nbchan"])
    assert model.state.title.startswith("Scroll channel activities -- eegplot()")
    np.testing.assert_array_equal(eeg["data"], data_before)
