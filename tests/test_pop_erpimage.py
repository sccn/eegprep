"""``pop_erpimage``: argument handling, event sorting, dialog mapping, history and MATLAB parity."""

from __future__ import annotations

from copy import deepcopy
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from eegprep.functions.adminfunc.eeglabcompat import _resolve_eeglab_root, get_eeglab
from eegprep.functions.popfunc import pop_erpimage as pop_erpimage_module
from eegprep.functions.popfunc.eeg_point2lat import eeg_point2lat
from eegprep.functions.popfunc.pop_erpimage import pop_erpimage
from eegprep.functions.popfunc.plot_utils import component_map_data
from eegprep.functions.popfunc.pop_loadset import pop_loadset
from eegprep.functions.sigprocfunc import erpimage as erpimage_module
from eegprep.functions.sigprocfunc.erpimage import erpimage
from eegprep.functions.sigprocfunc.topoplot import topo_screen_coords, topoplot
from tests.eeglab_tests import load_matlab_test_fixture
from tests.fixtures import create_erpimage_epochs, matlab_engine_available

REPO_ROOT = Path(__file__).resolve().parents[1]
EPOCHS_PATH = REPO_ROOT / "sample_data" / "eeglab_data_epochs_ica.set"
FIXTURE = REPO_ROOT / "tests" / "matlab" / "pop_erpimage_fixtures.mat"
RTOL = 1e-10


pytestmark = pytest.mark.usefixtures("close_figures")


def _rt_latencies(EEG: dict) -> np.ndarray:
    """First 'rt' latency (ms) per epoch, NaN without one, from the raw event table."""
    values = np.full(EEG["trials"], np.nan)
    limits = [EEG["xmin"] * 1000, EEG["xmax"] * 1000]
    for event in EEG["event"]:
        epoch = int(event["epoch"])
        if event["type"] == "rt" and np.isnan(values[epoch - 1]):
            values[epoch - 1] = eeg_point2lat([event["latency"]], [epoch], EEG["srate"], limits, 1e-3)[0]
    return values


def _times(EEG: dict) -> np.ndarray:
    return np.linspace(EEG["xmin"] * 1000, EEG["xmax"] * 1000, EEG["pnts"])


def _activations(EEG: dict) -> np.ndarray:
    """ICA activity recomputed from the weights, (components, points, trials)."""
    data = np.asarray(EEG["data"], dtype=float)
    unmixing = np.dot(np.asarray(EEG["icaweights"]), np.asarray(EEG["icasphere"]))
    return np.einsum("kc,cpt->kpt", unmixing, data[np.asarray(EEG["icachansind"])])


# ----------------------------------------------------------------------------- command line


def test_channel_image_is_the_erpimage_of_the_channel_sorted_by_event_latency(epochs) -> None:
    result = pop_erpimage(epochs, 1, 14, None, "", 10, 1, ["rt"], None, "latency", noshow=True)
    data = np.asarray(epochs["data"][13], dtype=float)
    expected = erpimage(data, _rt_latencies(epochs), _times(epochs), "", 10, 1, "erp", noshow=True)
    np.testing.assert_array_equal(result.outdata, expected.outdata)
    np.testing.assert_array_equal(result.sortidx, expected.sortidx)
    np.testing.assert_array_equal(result.erp, expected.erp)
    assert np.all(np.diff(_rt_latencies(epochs)[result.sortidx]) >= 0)


def test_channels_are_given_by_number_or_label_and_several_are_averaged(epochs) -> None:
    by_label = pop_erpimage(epochs, 1, "Cz", noshow=True, smooth=1)
    by_number = pop_erpimage(epochs, 1, 14, noshow=True, smooth=1)
    np.testing.assert_array_equal(by_label.outdata, by_number.outdata)
    averaged = pop_erpimage(epochs, 1, [3, 7], noshow=True, smooth=1)
    expected = np.mean(np.asarray(epochs["data"], dtype=float)[[2, 6]], axis=0)
    np.testing.assert_allclose(averaged.outdata, expected, rtol=0, atol=1e-12)


def test_without_a_sorting_field_trials_keep_their_order_at_the_epoch_end(epochs) -> None:
    result = pop_erpimage(epochs, 1, 14, noshow=True, smooth=1)
    np.testing.assert_array_equal(result.sortidx, np.arange(epochs["trials"]))
    np.testing.assert_allclose(result.outvar, np.full(epochs["trials"], epochs["xmax"] * 1000))


def test_component_activity_and_its_projections(epochs) -> None:
    eeg = deepcopy(epochs)
    eeg["icaact"] = np.array([])
    activity = _activations(eeg)
    maps = np.asarray(eeg["icawinv"])
    plain = pop_erpimage(eeg, 0, 6, noshow=True, smooth=1)
    np.testing.assert_allclose(plain.outdata, activity[5], rtol=0, atol=1e-9)
    projected = pop_erpimage(eeg, 0, 6, 27, noshow=True, smooth=1)
    np.testing.assert_allclose(projected.outdata, maps[26, 5] * activity[5], rtol=0, atol=1e-9)
    summed = pop_erpimage(eeg, 0, [6, 7], "POz", noshow=True, smooth=1)
    np.testing.assert_allclose(summed.outdata, np.tensordot(maps[26, [5, 6]], activity[[5, 6]], 1), atol=1e-9)
    averaged = pop_erpimage(eeg, 0, 6, [27, 31], noshow=True, smooth=1)
    np.testing.assert_allclose(averaged.outdata, maps[[26, 30], 5].mean() * activity[5], rtol=0, atol=1e-9)


def test_channel_figure_has_erp_colorbar_scalp_map_and_label_title(epochs) -> None:
    result = pop_erpimage(epochs, 1, 27, plot="off")
    image_ax, colorbar_ax, erp_ax, _amp, _coh, scalp_ax = result.axhndls
    assert image_ax.get_title() == "POz"
    assert colorbar_ax is not None and scalp_ax is not None
    assert erp_ax.get_ylabel() == "µV"
    bare = pop_erpimage(epochs, 1, 27, erp=False, cbar=False, plotmap=False, plot="off")
    assert bare.axhndls == (bare.axhndls[0], None, None, None, None)


def test_component_figure_shows_its_map_and_projection_units(epochs) -> None:
    component = pop_erpimage(epochs, 0, 6, plot="off")
    assert component.axhndls[0].get_title() == "Comp. 6"
    assert component.axhndls[2].get_ylabel() == ""
    assert component.axhndls[5].collections or component.axhndls[5].images  # the topoplot of the map
    projected = pop_erpimage(epochs, 0, 6, 27, plot="off")
    assert projected.axhndls[0].get_title() == "Comp. 6 -> Chan. 27"
    assert projected.axhndls[2].get_ylabel() == "µV"


def test_projected_components_show_their_mean_map(epochs) -> None:
    result = pop_erpimage(epochs, 0, [6, 7], 27, plot="off")
    maps, chanlocs = component_map_data(epochs)
    expected_axes = plt.figure().add_subplot()
    topoplot(maps[:, [5, 6]].mean(axis=1), chanlocs, electrodes="off", axes=expected_axes)
    np.testing.assert_allclose(result.axhndls[5].images[0].get_array(), expected_axes.images[0].get_array())


def test_several_channels_have_no_scalp_map(epochs, caplog) -> None:
    with caplog.at_level(logging.WARNING, logger=pop_erpimage_module.logger.name):
        result = pop_erpimage(epochs, 1, [3, 7], plot="off")
    assert len(result.axhndls) == 5
    assert "the scalp map marks one channel, so it is left out for 2 channels" in caplog.text
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=pop_erpimage_module.logger.name):
        mapped = pop_erpimage(epochs, 1, [3, 4, 5], topo=(4, epochs["chanlocs"]), plot="off")
    assert len(mapped.axhndls) == 6 and caplog.text == ""


@pytest.mark.parametrize("plotmap", ["off", 0, False])
def test_plotmap_takes_on_off_words_and_numbers(plotmap) -> None:
    result, command = pop_erpimage(
        create_erpimage_epochs(), 1, 1, smooth=1, plotmap=plotmap, plot="off", return_com=True
    )
    assert len(result.axhndls) == 5 and command.endswith("plotmap=False)")
    assert len(pop_erpimage(create_erpimage_epochs(), 1, 1, smooth=1, plotmap="on", plot="off").axhndls) == 6


def test_titles_from_the_argument_or_the_title_option(epochs) -> None:
    named = pop_erpimage(epochs, 1, 14, None, "My title", plot="off", plotmap=False)
    assert named.axhndls[0].get_title() == "My title"
    option = pop_erpimage(epochs, 1, 14, None, "My title", plot="off", plotmap=False, title="Option title")
    assert option.axhndls[0].get_title() == "Option title"


def test_default_titles_list_every_channel_and_component(epochs) -> None:
    # As pop_erpimage.m builds them: [EEG.chanlocs(channel).labels] and 'Comp. ' int2str(channel).
    averaged = pop_erpimage(create_erpimage_epochs(), 1, [1, 3], smooth=1, plot="off")
    assert averaged.axhndls[0].get_title() == "AC"
    projected = pop_erpimage(epochs, 0, [6, 7], [27, 31], smooth=1, plot="off")
    assert projected.axhndls[0].get_title() == "Comp. 6  7 -> Chan. [27 31]"


def test_channel_scalp_map_marks_the_plotted_electrode() -> None:
    eeg = create_erpimage_epochs()
    for channel, theta in ((1, -40.0), (3, 40.0)):
        result = pop_erpimage(eeg, 1, channel, smooth=1, plot="off")
        marker = result.axhndls[5].collections[-1].get_offsets()
        np.testing.assert_allclose(marker, [topo_screen_coords(theta, 0.3)])


def test_nosort_keeps_the_trial_order_and_draws_the_values() -> None:
    eeg = create_erpimage_epochs()
    result = pop_erpimage(eeg, 1, 1, sortingtype="rt", sortingeventfield="rt", smooth=1, nosort=True, plot="off")
    np.testing.assert_array_equal(result.sortidx, np.arange(6))
    np.testing.assert_allclose(result.outvar, [120, 60, 180, 30, 150, 90])
    widths = [line.get_linewidth() for line in result.axhndls[0].lines]
    assert erpimage_module.SORTWIDTH in widths


def test_noplot_sorts_the_trials_without_drawing_the_values() -> None:
    eeg = create_erpimage_epochs()
    result = pop_erpimage(eeg, 1, 1, sortingtype="rt", sortingeventfield="rt", smooth=1, noplot=True, plot="off")
    np.testing.assert_array_equal(result.sortidx, [3, 1, 5, 0, 4, 2])
    widths = [line.get_linewidth() for line in result.axhndls[0].lines]
    assert erpimage_module.SORTWIDTH not in widths


def test_text_event_values_sort_alphabetically_like_eeglab() -> None:
    eeg = create_erpimage_epochs()
    result = pop_erpimage(eeg, 1, 1, sortingtype="rt", sortingeventfield="code", smooth=1, noshow=True)
    codes = np.array(["cab"[trial % 3] for trial in range(6)])
    assert list(codes[result.sortidx]) == sorted(codes)


def test_rescale_maps_the_event_values_into_the_epoch() -> None:
    eeg = create_erpimage_epochs()
    result = pop_erpimage(eeg, 1, 1, sortingtype="rt", sortingeventfield="rt", smooth=1, renorm="yes", noshow=True)
    span = (eeg["xmax"] - eeg["xmin"]) * 1000
    assert np.min(result.outvar) == pytest.approx(eeg["xmin"] * 1000 + 0.4 * span)
    assert np.max(result.outvar) == pytest.approx(eeg["xmin"] * 1000 + 0.9 * span)


def test_epochs_without_a_matching_event_are_left_out_and_logged(caplog) -> None:
    eeg = create_erpimage_epochs()
    eeg["event"] = [event for event in eeg["event"] if not (event["type"] == "rt" and event["epoch"] in (2, 5))]
    with caplog.at_level(logging.WARNING, logger=pop_erpimage_module.logger.name):
        result = pop_erpimage(eeg, 1, 1, sortingtype="rt", sortingeventfield="latency", smooth=1, noshow=True)
    assert sorted(result.sortidx.tolist()) == [0, 2, 3, 5]
    assert "2 of 6 epochs have no event 'latency' value for event type 'rt' (epochs [2, 5], counted from 1)" in (
        caplog.text
    )


def test_several_latencies_in_an_epoch_use_the_first_with_a_warning(caplog) -> None:
    eeg = create_erpimage_epochs()
    with caplog.at_level(logging.WARNING, logger=pop_erpimage_module.logger.name):
        result = pop_erpimage(eeg, 1, 1, sortingeventfield="latency", smooth=1, noshow=True)
    np.testing.assert_allclose(result.outvar, np.zeros(6))  # every epoch's first event is the stimulus at 0 ms
    assert "multiple event latencies found in epochs [1, 2, 3, 4, 5, 6]; the first is used." in caplog.text


def test_only_latency_sorting_warns_about_several_events(caplog) -> None:
    with caplog.at_level(logging.WARNING, logger=pop_erpimage_module.logger.name):
        pop_erpimage(create_erpimage_epochs(), 1, 1, smooth=1, sortingeventfield="rt", noshow=True)
    assert "multiple event latencies" not in caplog.text


def test_events_may_carry_the_sorting_field_on_some_events_only() -> None:
    eeg = create_erpimage_epochs()
    for event in eeg["event"]:
        if event["type"] == "stim":
            del event["rt"]
    result = pop_erpimage(eeg, 1, 1, smooth=1, sortingtype="rt", sortingeventfield="rt", noshow=True)
    np.testing.assert_array_equal(result.sortidx, [3, 1, 5, 0, 4, 2])


def test_sorting_field_ignores_surrounding_spaces() -> None:
    eeg = create_erpimage_epochs()
    spaced = pop_erpimage(eeg, 1, 1, smooth=1, sortingtype="rt", sortingeventfield=" rt ", noshow=True)
    np.testing.assert_array_equal(spaced.sortidx, [3, 1, 5, 0, 4, 2])


def test_missing_event_error_names_the_types_and_the_time_window() -> None:
    message = r"no epoch has an event 'latency' value for event type 'stim' 'rt' between 500 and 900 ms"
    with pytest.raises(ValueError, match=message):
        pop_erpimage(
            create_erpimage_epochs(),
            1,
            1,
            sortingtype=["stim", "rt"],
            sortingwin=[500, 900],
            sortingeventfield="latency",
            noshow=True,
        )


@pytest.mark.parametrize(
    ("change", "arguments", "message"),
    [
        (lambda eeg: eeg.update(event=[]), {"sortingeventfield": "latency"}, "has no events"),
        (None, {"sortingeventfield": "duration"}, "events have no 'duration' field"),
        (None, {"sortingeventfield": "latency", "sortingtype": "button"}, "no epoch has an event 'latency' value"),
        (None, {"sortingeventfield": "latency", "sortingwin": [0, 50, 100]}, r"sortingwin must be \[start end\]"),
        (None, {"sortingeventfield": "latency", "sortingwin": ["a", "b"]}, "sortingwin must be numbers"),
        (
            lambda eeg: [event.update(code={"bad": 1}) for event in eeg["event"]],
            {"sortingeventfield": "code"},
            r"event 'code' values that are not numbers or text in epochs \[1, 2, 3, 4, 5, 6\]",
        ),
    ],
)
def test_sorting_events_that_cannot_be_used_raise(change, arguments, message) -> None:
    eeg = create_erpimage_epochs()
    if change is not None:
        change(eeg)
    with pytest.raises(ValueError, match=message):
        pop_erpimage(eeg, 1, 1, noshow=True, **arguments)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"smooth": 2.7}, "boxcar avewidth must be an integer"),
        ({"smooth": [3, 5]}, "avewidth must be a single number"),
        ({"smooth": -1}, "cannot be < 1"),
        ({"decimate": "2"}, "decimate must be a single number"),
        ({"limits": [600, -200]}, "time limits out of order"),
        ({"limits": [5000, 6000]}, "do not overlap the data"),
        ({"limits": [100]}, r"limits arg must be a vector sized"),
        ({"plotmap": "yes"}, "plotmap must be True/False, 'on'/'off' or 1/0"),
        ({"bogus": 1}, "unknown option 'bogus'"),
    ],
)
def test_invalid_smoothing_limits_and_options_raise(arguments, message) -> None:
    with pytest.raises(ValueError, match=message):
        pop_erpimage(create_erpimage_epochs(), 1, 1, noshow=True, **arguments)


def test_zero_smoothing_shows_every_trial_unchanged() -> None:
    eeg = create_erpimage_epochs()
    np.testing.assert_array_equal(pop_erpimage(eeg, 1, 1, smooth=0, noshow=True).outdata, eeg["data"][0])


def test_limits_set_the_axes_without_cropping_the_data(epochs) -> None:
    result = pop_erpimage(epochs, 1, 14, limits=[-200, 600], plot="off", plotmap=False)
    assert result.axhndls[0].get_xlim() == (-200.0, 600.0)
    assert result.outdata.shape[0] == epochs["pnts"]


@pytest.mark.parametrize(
    ("change", "arguments", "message"),
    [
        (None, {"typeplot": 2}, "typeplot must be 1"),
        (None, {"typeplot": True}, "typeplot must be 1"),
        (lambda eeg: eeg.update(icaweights=np.zeros((0, 0)), icasphere=np.eye(3)), {"typeplot": 0}, "no ICA data"),
        (lambda eeg: eeg.update(icaweights=np.eye(3), icasphere=np.zeros((0, 0))), {"typeplot": 0}, "no ICA data"),
        (lambda eeg: eeg.update(trials=1), {}, "more than one trial"),
        (None, {"channel": 0}, r"channel 0 is outside 1\.\.3"),
        (None, {"channel": "Q"}, "unknown channel label 'Q'"),
        (None, {"channel": 1.5}, "whole numbers"),
        (None, {"channel": []}, "select at least one channel"),
        (None, {"channel": [None]}, "channels are given by number or label, got None"),
    ],
)
def test_invalid_selections_raise(change, arguments, message) -> None:
    eeg = create_erpimage_epochs()
    if change is not None:
        change(eeg)
    call = {"typeplot": 1, "channel": 1, **arguments}
    with pytest.raises(ValueError, match=message):
        pop_erpimage(eeg, call.pop("typeplot"), call.pop("channel"), gui=False, noshow=True)


def test_last_channel_and_component_are_accepted_and_the_next_one_is_rejected(epochs) -> None:
    eeg = create_erpimage_epochs()
    np.testing.assert_array_equal(pop_erpimage(eeg, 1, 3, smooth=0, noshow=True).outdata, eeg["data"][2])
    with pytest.raises(ValueError, match=r"channel 4 is outside 1\.\.3"):
        pop_erpimage(eeg, 1, 4, noshow=True)
    count = np.asarray(epochs["icaweights"]).shape[0]
    last = pop_erpimage(epochs, 0, count, smooth=0, noshow=True)
    np.testing.assert_allclose(last.outdata, _activations(epochs)[count - 1], rtol=1e-6, atol=1e-6)
    with pytest.raises(ValueError, match=rf"component {count + 1} is outside 1\.\.{count}"):
        pop_erpimage(epochs, 0, count + 1, noshow=True)


def test_channels_accept_ranges_brackets_and_labels_in_any_case(epochs) -> None:
    expected = np.mean(np.asarray(epochs["data"], dtype=float)[2:5], axis=0)
    for channel in ("3:5", "[3 4 5]", [3, 4, 5], "{'F3', 'fz', 'F4'}", "['F3', 'Fz', 'F4']"):
        np.testing.assert_allclose(pop_erpimage(epochs, 1, channel, smooth=1, noshow=True).outdata, expected)


def test_projection_channels_are_numbers_or_labels_and_ignored_for_channels(epochs) -> None:
    by_label = pop_erpimage(epochs, 0, 6, "poz", smooth=1, noshow=True)
    by_text = pop_erpimage(epochs, 0, 6, "27", smooth=1, noshow=True)
    np.testing.assert_array_equal(by_label.outdata, by_text.outdata)
    mixed = pop_erpimage(epochs, 0, 6, [27, "Oz"], smooth=1, noshow=True)
    np.testing.assert_array_equal(mixed.outdata, pop_erpimage(epochs, 0, 6, "27:4:31", smooth=1, noshow=True).outdata)
    with pytest.raises(ValueError, match=r"projection channel 40 is outside 1\.\.32"):
        pop_erpimage(epochs, 0, 6, 40, noshow=True)
    with pytest.raises(ValueError, match="unknown projection channel label 'Q9'"):
        pop_erpimage(epochs, 0, 6, "Q9", noshow=True)
    ignored = pop_erpimage(epochs, 1, 14, 2.5, smooth=1, noshow=True)
    np.testing.assert_array_equal(ignored.outdata, pop_erpimage(epochs, 1, 14, smooth=1, noshow=True).outdata)


def test_a_string_sortingtype_is_one_event_type_even_with_spaces() -> None:
    eeg = create_erpimage_epochs()
    for event in eeg["event"]:
        event["type"] = "S  1" if event["type"] == "rt" else "Stimulus/S 2"
    result = pop_erpimage(eeg, 1, 1, sortingtype="S  1", sortingeventfield="rt", smooth=1, noshow=True)
    np.testing.assert_array_equal(result.sortidx, [3, 1, 5, 0, 4, 2])
    listed = pop_erpimage(eeg, 1, 1, sortingtype=["S  1"], sortingeventfield="rt", smooth=1, noshow=True)
    np.testing.assert_array_equal(listed.sortidx, result.sortidx)


@pytest.mark.parametrize("requested", [[2.0], [2], ["2"], ["2.0"], np.array([2], dtype=np.uint16)])
def test_numeric_event_types_are_matched_by_value(requested) -> None:
    eeg = create_erpimage_epochs()
    for event in eeg["event"]:
        event["type"] = 2.0 if event["type"] == "rt" else 1.0
    result = pop_erpimage(eeg, 1, 1, sortingtype=requested, sortingeventfield="rt", smooth=1, noshow=True)
    np.testing.assert_array_equal(result.sortidx, [3, 1, 5, 0, 4, 2])


@pytest.mark.parametrize("plot", ["maybe", None, 1])
def test_an_invalid_plot_flag_raises_before_drawing(plot) -> None:
    before = plt.get_fignums()
    with pytest.raises(ValueError, match="plot must be 'on'/'off'"):
        pop_erpimage(create_erpimage_epochs(), 1, 1, plot=plot)
    assert plt.get_fignums() == before


def test_history_names_a_copy_of_the_dataset_locations_as_the_dataset_locations(epochs) -> None:
    _result, command = pop_erpimage(epochs, 1, 14, topo=(14, deepcopy(epochs["chanlocs"])), plot="off", return_com=True)
    assert command == "pop_erpimage(EEG, 1, 14, smooth=5, decimate=0, topo=(14, EEG['chanlocs']))"
    chaninfo = {"nosedir": "+X"}
    _result, command = pop_erpimage(epochs, 1, 14, topo=(14, epochs["chanlocs"], chaninfo), plot="off", return_com=True)
    assert command.endswith("topo=(14, EEG['chanlocs'], {'nosedir': '+X'}))")


def test_history_writes_other_locations_in_full(caplog) -> None:
    eeg = create_erpimage_epochs()
    moved = deepcopy(eeg["chanlocs"])
    moved[0]["theta"] = -60.0
    result, command = pop_erpimage(eeg, 1, 1, smooth=1, topo=(1, moved), plot="off", return_com=True)
    assert "EEG['chanlocs']" not in command and "'theta': -60" in command
    replayed = eval(command, {"EEG": eeg, "pop_erpimage": pop_erpimage})
    np.testing.assert_allclose(
        replayed.axhndls[5].collections[-1].get_offsets(), result.axhndls[5].collections[-1].get_offsets()
    )


@pytest.mark.parametrize(("field", "value"), [("icaweights", None), ("icasphere", np.array([])), ("icaweights", [])])
def test_components_need_ica_weights_and_sphere(epochs, field, value) -> None:
    eeg = deepcopy(epochs)
    eeg[field] = value
    with pytest.raises(ValueError, match="no ICA data for this set, first run ICA"):
        pop_erpimage(eeg, 0, 1, noshow=True)


def test_sorting_needs_epoch_numbers_in_the_events() -> None:
    eeg = create_erpimage_epochs()
    for event in eeg["event"]:
        del event["epoch"]
    with pytest.raises(ValueError, match="events have no 'epoch' field; epoch the data before sorting by events"):
        pop_erpimage(eeg, 1, 1, sortingeventfield="latency", noshow=True)


def test_the_scalp_map_is_left_out_without_a_location(epochs, caplog) -> None:
    eeg = create_erpimage_epochs()
    eeg["chanlocs"][1]["theta"] = np.nan
    with caplog.at_level(logging.WARNING, logger=pop_erpimage_module.logger.name):
        unlocated = pop_erpimage(eeg, 1, 2, plot="off")
    assert len(unlocated.axhndls) == 5
    assert "channel 2 has no location, so the scalp map is left out" in caplog.text
    assert len(pop_erpimage(eeg, 1, 1, plot="off").axhndls) == 6
    components = deepcopy(epochs)
    for location in components["chanlocs"]:
        location["theta"] = np.nan
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=pop_erpimage_module.logger.name):
        assert len(pop_erpimage(components, 0, 6, plot="off").axhndls) == 5
    assert "the channels have no locations, so the scalp map is left out" in caplog.text
    eeg.update(icaweights=np.eye(2), icasphere=np.eye(2), icawinv=np.eye(2), icachansind=np.array([0, 1]))
    eeg["icaact"] = np.array([])
    eeg["chanlocs"][0]["theta"] = np.nan
    with caplog.at_level(logging.WARNING, logger=pop_erpimage_module.logger.name):
        assert len(pop_erpimage(eeg, 0, 1, plot="off").axhndls) == 5
    assert "the ICA channels have no locations, so the scalp map is left out" in caplog.text


def test_datasets_without_locations_draw_no_scalp_map(caplog) -> None:
    eeg = create_erpimage_epochs()
    eeg["chanlocs"] = []
    with caplog.at_level(logging.WARNING, logger=pop_erpimage_module.logger.name):
        bare = pop_erpimage(eeg, 1, [2, 3], smooth=1, plot="off")
    assert len(bare.axhndls) == 5 and bare.axhndls[0].get_title() == "2  3" and caplog.text == ""
    eeg["chanlocs"] = [{"labels": label} for label in "ABC"]
    with caplog.at_level(logging.WARNING, logger=pop_erpimage_module.logger.name):
        assert len(pop_erpimage(eeg, 1, 2, smooth=1, plot="off").axhndls) == 5
    assert "the channels have no locations" in caplog.text


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (("phasesort", [0, 25, 10]), "'phasesort' was passed as projchan but is an erpimage option"),
        (("nosort",), "'nosort' was passed as projchan"),
        ((None, "erp", 3), "'erp' was passed as titleplot"),
        ((None, [0, 25, 10]), r"titleplot must be text, got \[0, 25, 10\]"),
    ],
)
def test_options_in_positional_places_raise(arguments, message) -> None:
    with pytest.raises(ValueError, match=message):
        pop_erpimage(create_erpimage_epochs(), 1, 1, *arguments, noshow=True)


def test_a_title_may_name_an_option_in_capitals() -> None:
    result = pop_erpimage(create_erpimage_epochs(), 1, 1, None, "ERP", smooth=1, plot="off", plotmap=False)
    assert result.axhndls[0].get_title() == "ERP"


def test_projection_titles_quote_labels_as_eeglab_does(epochs) -> None:
    assert pop_erpimage(epochs, 0, 6, "POz", plot="off").axhndls[0].get_title() == 'Comp. 6 -> Chan. "POz"'
    mixed = pop_erpimage(epochs, 0, 6, [27, "Oz"], plot="off", plotmap=False)
    assert mixed.axhndls[0].get_title() == 'Comp. 6 -> Chan. {27,"Oz"}'


def test_long_lists_of_epochs_are_shortened_in_messages(caplog) -> None:
    eeg = create_erpimage_epochs(trials=30)
    eeg["event"] = [event for event in eeg["event"] if not (event["type"] == "rt" and event["epoch"] <= 12)]
    with caplog.at_level(logging.WARNING, logger=pop_erpimage_module.logger.name):
        pop_erpimage(eeg, 1, 1, sortingtype="rt", sortingeventfield="rt", smooth=1, noshow=True)
    assert (
        "12 of 30 epochs have no event 'rt' value for event type 'rt' (epochs [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, ...]"
        in (caplog.text)
    )


def test_a_later_option_token_wins_as_in_eeglab(epochs) -> None:
    drawn = pop_erpimage(epochs, 1, 14, None, "", 5, 0, None, None, "", "coher", [10, 10, 0.05], "erp", "off", rng=1)
    assert drawn.axhndls[2] is None and drawn.erp is None
    switched = pop_erpimage(epochs, 1, 14, None, "", 5, 0, None, None, "", "erp", "off", "erp", "on", noshow=True)
    assert switched.erp is not None


def test_amplitude_image_and_erp_deviation_reach_erpimage(epochs) -> None:
    amplitudes = pop_erpimage(epochs, 1, 14, coher=[10, 10, 0.01], plotamps=True, rng=1, noshow=True)
    direct = erpimage(
        np.asarray(epochs["data"], dtype=float)[13],
        np.full(epochs["trials"], epochs["xmax"] * 1000),
        _times(epochs),
        "",
        5,
        0,
        coher=[10, 10, 0.01],
        plotamps=True,
        rng=1,
        noshow=True,
    )
    np.testing.assert_allclose(amplitudes.outdata, direct.outdata)
    bare = pop_erpimage(epochs, 1, 14, plot="off")
    deviation = pop_erpimage(epochs, 1, 14, erpstd=True, plot="off")
    assert len(deviation.axhndls[2].lines) > len(bare.axhndls[2].lines)


@pytest.mark.parametrize("arguments", [{"sortingtype": "rt"}, {"sortingwin": [0, 100]}])
def test_sorting_types_or_window_without_a_field_are_reported(caplog, arguments) -> None:
    with caplog.at_level(logging.WARNING, logger=pop_erpimage_module.logger.name):
        result = pop_erpimage(create_erpimage_epochs(), 1, 1, smooth=1, noshow=True, **arguments)
    np.testing.assert_array_equal(result.sortidx, np.arange(6))
    assert "sortingtype and sortingwin need sortingeventfield; the trials are not sorted" in caplog.text


def test_one_latency_per_epoch_needs_no_warning_and_a_blank_window_is_none(caplog) -> None:
    eeg = create_erpimage_epochs()
    with caplog.at_level(logging.WARNING, logger=pop_erpimage_module.logger.name):
        result, command = pop_erpimage(
            eeg,
            1,
            1,
            smooth=1,
            sortingtype="rt",
            sortingwin="",
            sortingeventfield="latency",
            noshow=True,
            rng=np.int64(5),
            return_com=True,
        )
    assert caplog.text == ""
    np.testing.assert_allclose(result.outvar, np.sort([120, 60, 180, 30, 150, 90]))
    assert command.endswith("sortingtype='rt', sortingeventfield='latency', noshow=True, rng=5)")


@pytest.mark.parametrize("channel", ["", [], None])
def test_an_empty_channel_opens_the_dialog(channel) -> None:
    class Renderer:
        def run(self, spec, initial_values=None):
            return None

    assert pop_erpimage(create_erpimage_epochs(), 1, channel, renderer=Renderer(), return_com=True) == (None, "")


def test_several_components_need_a_projection_channel(epochs) -> None:
    with pytest.raises(ValueError, match="select a channel to plot the sum of several component projections"):
        pop_erpimage(epochs, 0, [1, 2], noshow=True)
    with pytest.raises(ValueError, match="components are given by number, got 'first'"):
        pop_erpimage(epochs, 0, "first", noshow=True)


def test_components_are_bounded_by_the_ica_size_not_the_channels() -> None:
    eeg = create_erpimage_epochs()
    eeg.update(
        icaweights=np.eye(2, 3),
        icasphere=np.eye(3),
        icawinv=np.eye(3, 2),
        icachansind=np.arange(3),
        icaact=np.array([]),
    )
    np.testing.assert_allclose(pop_erpimage(eeg, 0, 2, smooth=0, noshow=True).outdata, eeg["data"][1])
    with pytest.raises(ValueError, match=r"component 3 is outside 1\.\.2"):
        pop_erpimage(eeg, 0, 3, noshow=True)


def test_plot_off_target_and_noshow(epochs) -> None:
    figure = plt.figure()
    target = figure.add_subfigure(figure.add_gridspec(1, 2)[0, 1])
    before = plt.get_fignums()
    drawn = pop_erpimage(epochs, 1, 14, target=target, plot="off")
    assert drawn.figure is target and plt.get_fignums() == before
    hidden = pop_erpimage(epochs, 1, 14, plot="off")
    assert hidden.figure is not None and hidden.figure.number not in plt.get_fignums()
    assert pop_erpimage(epochs, 1, 14, noshow=True).figure is None


def test_history_command_replays_the_same_image(epochs) -> None:
    result, command = pop_erpimage(
        epochs,
        1,
        "POz",
        None,
        "",
        10,
        1,
        ["rt"],
        [0, 1000],
        "latency",
        "erp",
        "phasesort",
        np.array([0, 25, 10]),
        coher=[10, 10, 0.01],
        plotmap=False,
        rng=5,
        noshow=True,
        return_com=True,
    )
    assert command == (
        "pop_erpimage(EEG, 1, 'POz', smooth=10, decimate=1, sortingtype=['rt'], sortingwin=[0, 1000], "
        "sortingeventfield='latency', erp=True, phasesort=[0, 25, 10], coher=[10, 10, 0.01], noshow=True, "
        "plotmap=False, rng=5)"
    )
    assert result.cohsig is not None and result.cohers.size
    replayed = eval(command, {"EEG": epochs, "pop_erpimage": pop_erpimage})
    np.testing.assert_array_equal(replayed.outdata, result.outdata)
    np.testing.assert_array_equal(replayed.cohsig, result.cohsig)
    np.testing.assert_array_equal(replayed.cohers, result.cohers)


def test_history_writes_an_explicit_scalp_map_with_the_dataset_locations(epochs) -> None:
    _result, command = pop_erpimage(epochs, 1, 14, topo=(14, epochs["chanlocs"]), plot="off", return_com=True)
    assert command == "pop_erpimage(EEG, 1, 14, smooth=5, decimate=0, topo=(14, EEG['chanlocs']))"
    replayed = eval(command, {"EEG": epochs, "pop_erpimage": pop_erpimage})
    assert replayed.axhndls[5] is not None


def test_history_keeps_empty_smoothing_arguments() -> None:
    _result, command = pop_erpimage(create_erpimage_epochs(), 1, 1, None, "", [], [], noshow=True, return_com=True)
    assert command == "pop_erpimage(EEG, 1, 1, smooth=[], decimate=[], noshow=True)"


def test_missing_dataset_returns_nothing() -> None:
    assert pop_erpimage(None) is None
    assert pop_erpimage(None, return_com=True) == (None, "")


# ----------------------------------------------------------------------------- MATLAB parity


def _matlab_value(value: np.ndarray):
    if value.dtype.kind in "US":
        return str(value.ravel()[0]) if value.size else ""
    if value.dtype == object:
        return [_matlab_value(item) for item in value.ravel()]
    return np.asarray(value, dtype=float)


def _parity_dataset() -> dict:
    """The sample epochs in double precision with ICA activity recomputed, as the fixture script uses them."""
    eeg = pop_loadset(str(EPOCHS_PATH))
    eeg["data"] = np.asarray(eeg["data"], dtype=float)
    eeg["icaact"] = np.array([])
    return eeg


def _assert_scenario(fixture: dict, name: str, eeg: dict) -> None:
    record = fixture[name][0, 0]
    label = _matlab_value(record["channel_label"])
    channel = label or record["channel"].ravel().astype(int).tolist()
    projchan = record["projchan"].ravel().astype(int).tolist()
    window = record["sortingwin"].ravel().tolist()
    result = pop_erpimage(
        eeg,
        int(record["typeplot"].ravel()[0]),
        channel,
        projchan or None,
        "",
        float(record["smooth"].ravel()[0]),
        float(record["decimate"].ravel()[0]),
        _matlab_value(record["sortingtype"]) or None,
        window or None,
        _matlab_value(record["field"]),
        *[_matlab_value(item) for item in record["options"].ravel()],
        "erp",
        "NoShow",
        "on",
    )
    scale = float(np.max(np.abs(result.outdata)))
    np.testing.assert_allclose(result.outvar, record["outvar"].ravel(), rtol=RTOL, err_msg=f"{name} outvar")
    np.testing.assert_allclose(result.outtrials, record["outtrials"].ravel(), rtol=RTOL, err_msg=f"{name} outtrials")
    # EEGLAB's sortidx indexes the trials left after removing those without an event value.
    kept = np.flatnonzero(~np.isnan(record["sortvar"].ravel()))
    expected_order = kept[record["sortidx"].ravel().astype(int) - 1]
    np.testing.assert_array_equal(result.sortidx, expected_order, err_msg=f"{name} sortidx")
    rows = result.outdata.sum(axis=0)
    np.testing.assert_allclose(rows, record["outdata_rows"].ravel(), rtol=0, atol=RTOL * scale * eeg["pnts"])
    weights = np.arange(1, result.outdata.shape[0] + 1)
    weighted_tolerance = RTOL * scale * eeg["pnts"] ** 2
    np.testing.assert_allclose(
        np.dot(weights, result.outdata), record["outdata_weighted"].ravel(), rtol=0, atol=weighted_tolerance
    )
    frames = record["frames"].ravel().astype(int) - 1
    np.testing.assert_allclose(result.outdata[frames], record["outdata_frames"], rtol=0, atol=RTOL * scale)
    np.testing.assert_allclose(result.erp.ravel()[frames], record["erp_frames"].ravel(), rtol=0, atol=RTOL * scale)
    np.testing.assert_allclose(result.erp.sum(), float(record["erp_sum"].ravel()[0]), atol=RTOL * scale * eeg["pnts"])
    erp_weighted = float(record["erp_weighted"].ravel()[0])
    np.testing.assert_allclose(np.dot(weights, result.erp.ravel()), erp_weighted, rtol=0, atol=weighted_tolerance)


PARITY_SCENARIOS = ["channel_rt", "label_position_renorm", "channels_window", "component", "projected"]


@pytest.mark.parity
@pytest.mark.parametrize("name", PARITY_SCENARIOS)
def test_pop_erpimage_matches_matlab_fixture(name: str) -> None:
    _assert_scenario(load_matlab_test_fixture(FIXTURE), name, _parity_dataset())


def test_pop_fixture_keeps_all_erpimage_fixtures_small() -> None:
    """The committed erpimage MATLAB fixtures (engine and pop) share a 200 KB repository budget."""
    fixtures = list(FIXTURE.parent.glob("*erpimage*_fixtures.mat"))
    assert FIXTURE in fixtures
    assert sum(path.stat().st_size for path in fixtures) < 200_000


@pytest.mark.matlab
class TestPopErpimageMatlabParity:
    """Regenerate the pop_erpimage fixture with live MATLAB and compare again."""

    @pytest.fixture(scope="class")
    def regenerated(self, tmp_path_factory):
        if not matlab_engine_available():
            pytest.skip("MATLAB engine not available or skipped")
        try:
            matlab = get_eeglab("MAT", auto_file_roundtrip=False)
        except Exception as exc:  # MATLAB start-up failures are environment problems, not test failures
            pytest.skip(f"MATLAB not available: {exc}")
        output = tmp_path_factory.mktemp("pop_erpimage") / "pop_erpimage_fixtures.mat"
        matlab.pop_erpimage_fixtures(str(_resolve_eeglab_root()), str(EPOCHS_PATH), str(output), nargout=0)
        return load_matlab_test_fixture(output)

    @pytest.mark.parametrize("name", PARITY_SCENARIOS)
    def test_parity_live(self, regenerated, name: str) -> None:
        _assert_scenario(regenerated, name, _parity_dataset())
