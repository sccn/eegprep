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
from eegprep.functions.popfunc.pop_loadset import pop_loadset
from eegprep.functions.sigprocfunc import erpimage as erpimage_module
from eegprep.functions.sigprocfunc.erpimage import erpimage
from tests.eeglab_tests import load_matlab_test_fixture
from tests.fixtures import create_erpimage_epochs, matlab_engine_available

REPO_ROOT = Path(__file__).resolve().parents[1]
EPOCHS_PATH = REPO_ROOT / "sample_data" / "eeglab_data_epochs_ica.set"
FIXTURE = REPO_ROOT / "tests" / "matlab" / "pop_erpimage_fixtures.mat"
RTOL = 1e-10


@pytest.fixture(scope="module")
def epochs() -> dict:
    return pop_loadset(str(EPOCHS_PATH))


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


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


def test_titles_from_the_argument_or_the_title_option(epochs) -> None:
    named = pop_erpimage(epochs, 1, 14, None, "My title", plot="off", plotmap=False)
    assert named.axhndls[0].get_title() == "My title"
    option = pop_erpimage(epochs, 1, 14, None, "My title", plot="off", plotmap=False, title="Option title")
    assert option.axhndls[0].get_title() == "Option title"


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
    with caplog.at_level(logging.INFO, logger=pop_erpimage_module.logger.name):
        result = pop_erpimage(eeg, 1, 1, sortingtype="rt", sortingeventfield="latency", smooth=1, noshow=True)
    assert sorted(result.sortidx.tolist()) == [0, 2, 3, 5]
    assert "2 of 6 epochs have no event 'latency' value for event type 'rt' (epochs [2, 5])" in caplog.text


def test_several_latencies_in_an_epoch_use_the_first_with_a_warning(caplog) -> None:
    eeg = create_erpimage_epochs()
    with caplog.at_level(logging.WARNING, logger=pop_erpimage_module.logger.name):
        result = pop_erpimage(eeg, 1, 1, sortingeventfield="latency", smooth=1, noshow=True)
    np.testing.assert_allclose(result.outvar, np.zeros(6))  # every epoch's first event is the stimulus at 0 ms
    assert "multiple event latencies found in epochs [1, 2, 3, 4, 5, 6]; the first is used." in caplog.text


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
        ({"plotmap": "yes"}, "plotmap must be True or False"),
        ({"bogus": 1}, "unknown option 'bogus'"),
    ],
)
def test_invalid_smoothing_limits_and_options_raise(arguments, message) -> None:
    with pytest.raises(ValueError, match=message):
        pop_erpimage(create_erpimage_epochs(), 1, 1, noshow=True, **arguments)


def test_zero_smoothing_shows_every_trial() -> None:
    assert pop_erpimage(create_erpimage_epochs(), 1, 1, smooth=0, noshow=True).outdata.shape == (32, 6)


def test_limits_set_the_axes_without_cropping_the_data(epochs) -> None:
    result = pop_erpimage(epochs, 1, 14, limits=[-200, 600], plot="off", plotmap=False)
    assert result.axhndls[0].get_xlim() == (-200.0, 600.0)
    assert result.outdata.shape[0] == epochs["pnts"]


@pytest.mark.parametrize(
    ("change", "arguments", "message"),
    [
        (None, {"typeplot": 2}, "typeplot must be 1"),
        (lambda eeg: eeg.update(icaweights=np.zeros((0, 0))), {"typeplot": 0}, "no ICA data"),
        (lambda eeg: eeg.update(trials=1), {}, "more than one trial"),
        (None, {"channel": 0}, r"channel 0 is outside 1\.\.3"),
        (None, {"channel": "Q"}, "unknown channel label 'Q'"),
        (None, {"channel": 1.5}, "whole numbers"),
        (None, {"channel": []}, "select at least one channel"),
    ],
)
def test_invalid_selections_raise(change, arguments, message) -> None:
    eeg = create_erpimage_epochs()
    if change is not None:
        change(eeg)
    call = {"typeplot": 1, "channel": 1, **arguments}
    with pytest.raises(ValueError, match=message):
        pop_erpimage(eeg, call.pop("typeplot"), call.pop("channel"), noshow=True)


def test_several_components_need_a_projection_channel(epochs) -> None:
    with pytest.raises(ValueError, match="select a channel to plot the sum of several component projections"):
        pop_erpimage(epochs, 0, [1, 2], noshow=True)
    with pytest.raises(ValueError, match="components are given by number"):
        pop_erpimage(epochs, 0, "first", noshow=True)


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
    replayed = eval(command, {"EEG": epochs, "pop_erpimage": pop_erpimage})
    np.testing.assert_array_equal(replayed.outdata, result.outdata)
    np.testing.assert_array_equal(replayed.cohsig, result.cohsig)


def test_history_writes_an_explicit_scalp_map_with_the_dataset_locations(epochs) -> None:
    _result, command = pop_erpimage(epochs, 1, 14, topo=(14, epochs["chanlocs"]), plot="off", return_com=True)
    assert command == "pop_erpimage(EEG, 1, 14, smooth=5, decimate=0, topo=(14, EEG['chanlocs']))"
    replayed = eval(command, {"EEG": epochs, "pop_erpimage": pop_erpimage})
    assert replayed.axhndls[5] is not None


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
    frames = record["frames"].ravel().astype(int) - 1
    np.testing.assert_allclose(result.outdata[frames], record["outdata_frames"], rtol=0, atol=RTOL * scale)
    np.testing.assert_allclose(result.erp.ravel()[frames], record["erp_frames"].ravel(), rtol=0, atol=RTOL * scale)
    np.testing.assert_allclose(result.erp.sum(), float(record["erp_sum"].ravel()[0]), atol=RTOL * scale * eeg["pnts"])


PARITY_SCENARIOS = ["channel_rt", "label_position_renorm", "channels_window", "component", "projected"]


@pytest.mark.parity
@pytest.mark.parametrize("name", PARITY_SCENARIOS)
def test_pop_erpimage_matches_matlab_fixture(name: str) -> None:
    _assert_scenario(load_matlab_test_fixture(FIXTURE), name, _parity_dataset())


def test_pop_fixture_keeps_all_erpimage_fixtures_small() -> None:
    sizes = FIXTURE.stat().st_size + (FIXTURE.parent / "erpimage_engine_fixtures.mat").stat().st_size
    assert sizes < 200_000


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
