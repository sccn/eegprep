"""``pop_erpimage`` dialog: EEGLAB layout, result mapping and Qt callbacks."""

from __future__ import annotations

import logging
from copy import deepcopy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from eegprep.functions.guifunc.spec import controls_by_tag
from eegprep.functions.popfunc.pop_erpimage import pop_erpimage, pop_erpimage_dialog_spec
from eegprep.functions.popfunc.pop_loadset import pop_loadset
from tests.fixtures import create_erpimage_epochs


@pytest.fixture(scope="module")
def epochs() -> dict:
    return pop_loadset(str(Path(__file__).resolve().parents[1] / "sample_data" / "eeglab_data_epochs_ica.set"))


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


class _Renderer:
    """Dialog seam: the user edits ``values`` (None cancels); keeps the spec and initial values it was shown."""

    def __init__(self, values: dict | None) -> None:
        self.values = values
        self.spec = None
        self.initial_values = None

    def run(self, spec, initial_values=None):
        self.spec, self.initial_values = spec, initial_values
        if self.values is None:
            return None
        defaults = {
            tag: control.value for tag, control in controls_by_tag(spec).items() if control.style != "pushbutton"
        }
        return {**defaults, **(initial_values or {}), **self.values}


def _command(eeg: dict, values: dict, typeplot: int = 1) -> str:
    return pop_erpimage(eeg, typeplot, renderer=_Renderer(values), noshow=True, return_com=True)[1]


def test_cancelled_dialog_returns_nothing(epochs) -> None:
    assert pop_erpimage(epochs, 1, renderer=_Renderer(None), return_com=True) == (None, "")


def test_dialog_defaults_give_eeglabs_settings(epochs) -> None:
    renderer = _Renderer({})
    result, command = pop_erpimage(epochs, 1, renderer=renderer, plot="off", return_com=True)
    assert command == "pop_erpimage(EEG, 1, 1, smooth=10, decimate=1)"
    assert result.axhndls[1] is not None and result.axhndls[2] is not None and result.axhndls[5] is not None
    tags = controls_by_tag(renderer.spec)
    assert tags["limtime"].value == "-1000 1992.1875"
    assert tags["smooth"].value == "10" and tags["decimate"].value == "1"


EVERY_FIELD_COMMAND = (
    "pop_erpimage(EEG, 1, 'POz', titleplot='Sorted', smooth=3, decimate=2, sortingtype=['rt'], "
    "sortingwin=[0, 1000], sortingeventfield='latency', erpstd='on', cbar=False, "
    "limits=[-200, 800, -5, 5, -3, 3, 0, 1, 1], caxis=[-10, 10], renorm='yes', align=float('inf'), "
    "nosort=True, noplot=True, phasesort=[100, 25, 10, 12], coher=[10, 10, 0.01], spec=[2, 40], "
    "vert=[0, 500], plotmap=False)"
)


def test_dialog_maps_every_field_like_pop_erpimage_m(epochs) -> None:
    values = {
        "chan": "POz",
        "title": "Sorted",
        "smooth": "3",
        "decimate": "2",
        "plotmap": False,
        "erp": True,
        "limerp": "-5 5",
        "limtime": "-200 800",
        "cbar": False,
        "caxis": "-10 10",
        "field": "latency",
        "type": "rt",
        "eventrange": "0 1000",
        "renorm": "yes",
        "align": "Inf",
        "nosort": True,
        "noplot": True,
        "phase": "10 12",
        "phase2": "25",
        "phase3": "100",
        "coher": "10",
        "coher2": "0.01",
        "limamp": "-3 3",
        "limcoher": "0 1",
        "spec": "2 40",
        "limbaseamp": "1",
        "vert": "0 500",
        "others": "'erpstd', 'on'",
    }
    _result, command = pop_erpimage(epochs, 1, renderer=_Renderer(values), plot="off", rng=1, return_com=True)
    assert command == EVERY_FIELD_COMMAND[:-1] + ", rng=1)"


def test_explicit_arguments_fill_the_dialog_and_come_back_unchanged(epochs) -> None:
    renderer = _Renderer({})
    _result, command = pop_erpimage(
        epochs,
        1,
        "POz",
        None,
        "Sorted",
        3,
        2,
        ["rt"],
        [0, 1000],
        "latency",
        "cbar",
        "off",
        gui=True,
        renderer=renderer,
        plot="off",
        return_com=True,
        limits=[-200, 800, -5, 5, -3, 3, 0, 1, 1],
        caxis=[-10, 10],
        renorm="yes",
        align=np.inf,
        nosort=True,
        noplot=True,
        phasesort=[100, 25, 10, 12],
        coher=[10, 10, 0.01],
        spec=[2, 40],
        vert=[0, 500],
        erpstd="on",
        plotmap=False,
    )
    assert command == EVERY_FIELD_COMMAND
    assert renderer.initial_values == {
        "plotmap": False,
        "chan": "POz",
        "title": "Sorted",
        "smooth": "3",
        "decimate": "2",
        "type": "rt",
        "eventrange": "0 1000",
        "field": "latency",
        "cbar": False,
        "limtime": "-200 800",
        "limerp": "-5 5",
        "limamp": "-3 3",
        "limcoher": "0 1",
        "limbaseamp": "1",
        "caxis": "-10 10",
        "renorm": "yes",
        "align": "inf",
        "nosort": True,
        "noplot": True,
        "phase3": "100",
        "phase2": "25",
        "phase": "10 12",
        "coher": "10 10",
        "coher2": "0.01",
        "spec": "2 40",
        "vert": "0 500",
        "others": "erpstd='on'",
    }


def test_lastcom_reopens_the_dialog_with_the_previous_command(epochs) -> None:
    renderer = _Renderer({})
    _result, command = pop_erpimage(epochs, 1, lastcom=EVERY_FIELD_COMMAND, renderer=renderer, return_com=True)
    assert command == EVERY_FIELD_COMMAND
    assert renderer.initial_values["phase"] == "10 12" and renderer.initial_values["align"] == "inf"


def test_lastcom_skips_expressions_and_replays_without_the_dialog(epochs, caplog) -> None:
    lastcom = "image = pop_erpimage(EEG, 1, 14, smooth=3, topo=(14, EEG['chanlocs']), rng=2)"
    with caplog.at_level(logging.INFO, logger="eegprep.functions.popfunc.pop_erpimage"):
        _result, command = pop_erpimage(epochs, 1, lastcom=lastcom, gui=False, plot="off", return_com=True)
    assert command == "pop_erpimage(EEG, 1, 14, smooth=3, decimate=0)"
    assert "topo of the previous command is not a literal, so it is not restored" in caplog.text


@pytest.mark.parametrize(
    ("channel", "lastcom", "message"),
    [
        (14, "pop_erpimage(EEG, 1, 3)", "lastcom holds the arguments; pass no others with it"),
        (None, "pop_erpimage(EEG, 1, 3", "lastcom is not a Python command"),
        (None, "pop_topoplot(EEG, 1, 3)", "lastcom is not a pop_erpimage command"),
    ],
)
def test_lastcom_that_cannot_be_used_raises(epochs, channel, lastcom, message) -> None:
    with pytest.raises(ValueError, match=message):
        pop_erpimage(epochs, 1, channel, lastcom=lastcom, renderer=_Renderer({}))


def test_dialog_values_replace_the_callers_options(epochs) -> None:
    renderer = _Renderer({"cbar": True, "others": ""})
    _result, command = pop_erpimage(
        epochs, 1, 14, gui=True, cbar=False, erp=2, erpstd=True, renderer=renderer, plot="off", return_com=True
    )
    assert renderer.initial_values["cbar"] is False and "erp" not in renderer.initial_values
    assert renderer.initial_values["others"] == "erpstd=True"
    assert command == "pop_erpimage(EEG, 1, 14, smooth=10, decimate=1, erp=2)"


def test_options_the_dialog_cannot_show_are_applied_as_given(epochs) -> None:
    renderer = _Renderer({})
    topo = (14, deepcopy(epochs["chanlocs"]))
    limits = [-200, 800, np.nan, 5]
    result, command = pop_erpimage(
        epochs, 1, 14, gui=True, topo=topo, limits=limits, renderer=renderer, plot="off", return_com=True
    )
    assert "others" not in renderer.initial_values and "limtime" not in renderer.initial_values
    assert command.endswith("decimate=1, topo=(14, EEG['chanlocs']), limits=[-200, 800, float('nan'), 5])")
    assert result.axhndls[5] is not None
    edited = pop_erpimage(
        epochs, 1, 14, gui=True, limits=limits, renderer=_Renderer({"limerp": "-5 5"}), noshow=True, return_com=True
    )[1]
    assert "limits=[float('nan'), float('nan'), -5, 5, " in edited and "800" not in edited


def test_unchanged_time_limits_are_not_passed_and_a_single_coher_frequency_is_kept(epochs) -> None:
    values = {"erp": False, "coher": "10 12", "phase": "", "limtime": "-1000 1992.1875"}
    _result, command = pop_erpimage(epochs, 1, renderer=_Renderer(values), noshow=True, return_com=True)
    assert command == "pop_erpimage(EEG, 1, 1, smooth=10, decimate=1, noshow=True, erp=False, coher=[10, 12])"


def test_dialog_caxis_vert_and_spec_reach_the_figure(epochs) -> None:
    values = {"caxis": "-10 10", "vert": "250 500", "spec": "2 40"}
    result = pop_erpimage(epochs, 1, renderer=_Renderer(values), plot="off")
    image_ax = result.axhndls[0]
    assert image_ax.images[0].get_clim() == (-10.0, 10.0)
    marks = {round(float(line.get_xdata()[0])) for line in image_ax.lines if np.ptp(line.get_xdata()) == 0}
    assert {250, 500} <= marks
    assert result.axhndls[6].get_xlim() == (2.0, 40.0)


def test_dialog_image_amps_is_passed_on_with_a_coherence_frequency(epochs) -> None:
    command = _command(epochs, {"plotamps": True, "coher": "10", "coher2": "0.01"})
    assert command == (
        "pop_erpimage(EEG, 1, 1, smooth=10, decimate=1, noshow=True, plotamps=True, coher=[10, 10, 0.01])"
    )
    assert "coher=[10, 12, 0.01]" in _command(epochs, {"coher": "10 12", "coher2": "0.01"})
    with pytest.raises(ValueError, match="allamps and allcohers flags require coher"):
        _command(epochs, {"plotamps": True})


@pytest.mark.parametrize("trials", [2, 5])
def test_dialog_smoothing_starts_at_zero_when_there_are_fewer_than_six_trials(trials) -> None:
    eeg = create_erpimage_epochs(trials=trials)
    assert controls_by_tag(pop_erpimage_dialog_spec(eeg))["smooth"].value == "0"
    assert _command(eeg, {}) == "pop_erpimage(EEG, 1, 1, smooth=0, decimate=1, noshow=True)"


def test_dialog_keeps_quoted_event_types_with_spaces() -> None:
    eeg = create_erpimage_epochs()
    for event in eeg["event"]:
        event["type"] = "S  1" if event["type"] == "rt" else "Stimulus/S 2"
    values = {"field": "rt", "type": "'S  1' 'Stimulus/S 2'", "smooth": "1"}
    result, command = pop_erpimage(eeg, 1, renderer=_Renderer(values), noshow=True, return_com=True)
    assert "sortingtype=['S  1', 'Stimulus/S 2']" in command
    np.testing.assert_array_equal(result.sortidx, [3, 1, 5, 0, 4, 2])
    renderer = _Renderer({})
    reopened = pop_erpimage(eeg, 1, lastcom=command, renderer=renderer, return_com=True)[1]
    assert renderer.initial_values["type"] == "'S  1' 'Stimulus/S 2'" and reopened == command


def test_dialog_lists_numeric_event_types_sorted_as_numbers() -> None:
    eeg = create_erpimage_epochs()
    for event in eeg["event"]:
        event["type"] = 10.0 if event["type"] == "rt" else 2.0
    tags = controls_by_tag(pop_erpimage_dialog_spec(eeg))
    assert tags["type_button"].callback.params["event_types"] == ["2", "10"]
    result = pop_erpimage(
        eeg, 1, renderer=_Renderer({"field": "rt", "type": "10", "smooth": "1"}), noshow=True, return_com=False
    )
    np.testing.assert_array_equal(result.sortidx, [3, 1, 5, 0, 4, 2])


def test_dialog_numbers_accept_ranges_brackets_and_cell_text(epochs) -> None:
    values = {"chan": "[3 4]", "vert": "0:250:500", "eventrange": "[0 1000]", "caxis": "{-10 10}", "type": "rt"}
    values["field"] = "latency"
    command = _command(epochs, values)
    assert command.startswith("pop_erpimage(EEG, 1, [3, 4], smooth=10, decimate=1, sortingtype=['rt'], ")
    assert "sortingwin=[0, 1000]" in command and "caxis=[-10, 10]" in command and "vert=[0, 250, 500]" in command
    assert _command(epochs, {"chan": "{'Cz','Pz'}"}).startswith("pop_erpimage(EEG, 1, ['Cz', 'Pz'], ")
    assert "caxis=[-10, 10], vert=[0, 500]" in _command(epochs, {"caxis": "[-10, 10]", "vert": "0,500"})
    assert "sortingeventfield='latency'" in _command(epochs, {"field": "'latency'", "type": "rt"})
    assert "projchan=[26, 27]" in _command(epochs, {"chan": "6", "projchan": "26:27"}, typeplot=0)


@pytest.mark.parametrize(
    ("values", "typeplot", "message"),
    [
        ({"chan": ""}, 1, "'Channel' is required"),
        ({"chan": " "}, 0, r"'Component\(s\)' is required"),
        ({"smooth": ""}, 1, "'Smoothing' is empty"),
        ({"decimate": ""}, 1, "'Downsampling' is empty"),
    ],
)
def test_blank_required_dialog_entries_raise(epochs, values, typeplot, message) -> None:
    with pytest.raises(ValueError, match=message):
        _command(epochs, values, typeplot=typeplot)


def test_dialog_fields_replace_more_options_as_in_eeglab(epochs, caplog) -> None:
    values = {"caxis": "-5 5", "others": "caxis=[-1, 1], erp=False, yerplabel='uV', erpstd=True"}
    with caplog.at_level(logging.WARNING, logger="eegprep.functions.popfunc.pop_erpimage"):
        result, command = pop_erpimage(epochs, 1, renderer=_Renderer(values), plot="off", return_com=True)
    assert command == "pop_erpimage(EEG, 1, 1, smooth=10, decimate=1, caxis=[-5, 5], erpstd=True)"
    assert result.axhndls[2].get_ylabel() == "µV"
    assert "the dialog sets the ERP axis label, so yerplabel in More options is ignored" in caplog.text
    unchecked = pop_erpimage(epochs, 1, renderer=_Renderer({"erp": False, "others": "erp=True"}), plot="off")
    assert unchecked.axhndls[2] is not None


@pytest.mark.parametrize("typed", ["''", "'S 22", "rt '' stim"])
def test_dialog_event_types_with_empty_or_unclosed_quotes_raise(epochs, typed) -> None:
    with pytest.raises(ValueError, match="'Event type\\(s\\)' has an empty or unclosed quoted type"):
        _command(epochs, {"field": "latency", "type": typed})


def test_dialog_lists_mixed_event_types_as_text() -> None:
    eeg = create_erpimage_epochs()
    for event, value in zip(eeg["event"], [1.0, "boundary", 2.0, 10.0] * 3):
        event["type"] = value
    tags = controls_by_tag(pop_erpimage_dialog_spec(eeg))
    assert tags["type_button"].callback.params["event_types"] == ["1", "10", "2", "boundary"]


def test_dialog_rescale_needs_a_sorting_field(epochs) -> None:
    _result, command = pop_erpimage(epochs, 1, renderer=_Renderer({"renorm": "yes"}), noshow=True, return_com=True)
    assert command == "pop_erpimage(EEG, 1, 1, smooth=10, decimate=1, noshow=True)"
    for renorm in ("No", ""):  # pop_erpimage.m compares with strcmpi(res.renorm, 'no'); blank keeps no rescaling
        assert "renorm" not in _command(epochs, {"field": "latency", "type": "rt", "renorm": renorm})


def test_component_dialog_maps_the_projection_and_keeps_the_scalp_map(epochs) -> None:
    renderer = _Renderer({"chan": "6", "projchan": "POz"})
    result, command = pop_erpimage(epochs, 0, renderer=renderer, plot="off", return_com=True)
    assert command == "pop_erpimage(EEG, 0, 6, projchan='POz', smooth=10, decimate=1)"
    assert result.axhndls[2].get_ylabel() == "µV" and result.axhndls[5] is not None
    tags = controls_by_tag(renderer.spec)
    assert tags["plotmap"].style == "checkbox" and tags["plotmap"].value is True
    assert tags["projchan_button"].enabled


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"limerp": "5"}, r"'ERP limits' takes 2 value\(s\)"),
        ({"smooth": "abc"}, "'Smoothing' must be numbers"),
        ({"eventrange": "0"}, r"'Event time range' takes 2 value\(s\)"),
        ({"others": "'erpstd'"}, "More options needs 'key', value pairs .*: Plot options must be key/value pairs"),
        ({"others": "'erpstd', 'on"}, "More options needs .*: Unterminated quoted plot option"),
        ({"phase2": "25"}, "'phasesort' needs at least"),
        ({"chan": "Q9"}, "unknown channel label 'Q9'"),
        ({"align": "1 2"}, r"'Align' takes 1 value\(s\)"),
        ({"limtime": "5"}, r"'Time limits \(ms\)' takes 2 value\(s\)"),
        ({"limamp": "5"}, r"'Amplitude limits \(dB\)' takes 2 value\(s\)"),
        ({"limcoher": "1"}, r"'Coher limits' takes 2 value\(s\)"),
        ({"limbaseamp": "1 2"}, r"'Baseline ampl. \(dB\)' takes 1 value\(s\)"),
        ({"phase": "10", "phase2": "1 2"}, r"'Percent low-amp. trials' takes 1 value\(s\)"),
        ({"phase": "10", "phase3": "1 2"}, r"'Window center' takes 1 value\(s\)"),
        ({"smooth": "1 2"}, r"'Smoothing' takes 1 value\(s\)"),
        ({"coher": "10", "coher2": "0.01 0.02"}, r"'Signif. level' takes 1 value\(s\)"),
    ],
)
def test_dialog_values_that_cannot_be_used_raise(epochs, values, message) -> None:
    with pytest.raises(ValueError, match=message):
        pop_erpimage(epochs, 1, renderer=_Renderer(values), noshow=True)


def test_dialog_spec_follows_eeglab_order_and_states(epochs) -> None:
    channel = pop_erpimage_dialog_spec(epochs, typeplot=1)
    component = pop_erpimage_dialog_spec(epochs, typeplot=0)
    channel_tags = [control.tag for control in channel.controls if control.tag]
    assert channel_tags[:7] == ["chan", "chan_button", "title", "smooth", "plotmap", "decimate", "erp"]
    assert channel_tags[-4:] == ["spec", "limbaseamp", "vert", "others"]
    component_tags = [control.tag for control in component.controls if control.tag]
    assert component_tags[:5] == ["chan", "projchan", "projchan_button", "title", "smooth"]
    assert "plotmap" in component_tags
    tags = controls_by_tag(channel)
    assert set(tags["field_button"].callback.params["fields"]) == {"type", "position", "latency", "urevent", "epoch"}
    assert tags["type_button"].callback.params["event_types"] == ["rt", "square"]
    no_locations = deepcopy(epochs)
    no_locations["chanlocs"] = []
    assert not controls_by_tag(pop_erpimage_dialog_spec(no_locations))["chan_button"].enabled
    assert controls_by_tag(pop_erpimage_dialog_spec(create_erpimage_epochs()))["smooth"].value == "1"


def test_qt_dialog_wires_the_event_lists_and_frequency_mirrors(qt_dialogs) -> None:
    widgets = qt_dialogs.build(pop_erpimage_dialog_spec(create_erpimage_epochs()))
    qt_dialogs.answers["listdlg2"] = ([2], 1, "")  # type, latency, epoch, code, rt: the events' field order
    widgets["field_button"].click()
    qt_dialogs.answers["chansel"] = ([1], 1, "")
    widgets["type_button"].click()
    widgets["coher"].setText("10 10")
    widgets["phase"].setText("8 12")
    widgets["phase"].editingFinished.emit()
    assert (widgets["field"].text(), widgets["type"].text(), widgets["coher"].text()) == ("latency", "rt", "8 12")
    assert qt_dialogs.messages == [("Warning", "Do not forget to select an event type in the next edit box")]
    (_name, fields), (_name, types) = qt_dialogs.calls
    assert (fields["liststring"], fields["selectionmode"]) == (["type", "latency", "epoch", "code", "rt"], "single")
    assert (types["liststring"], types["initialvalue"], types["selectionmode"]) == (["rt", "stim"], [], "multiple")


def test_channel_lists_open_without_a_preselection(epochs, qt_dialogs) -> None:
    widgets = qt_dialogs.build(pop_erpimage_dialog_spec(epochs))
    widgets["chan"].setText("3:5")
    widgets["chan_button"].click()  # cancelled
    assert widgets["chan"].text() == "3:5"
    qt_dialogs.answers["chansel"] = ([27], 1, "")
    widgets["chan_button"].click()
    component = qt_dialogs.build(pop_erpimage_dialog_spec(epochs, typeplot=0))
    component["projchan"].setText("{'Fz','Cz'}")
    component["projchan_button"].click()
    assert (widgets["chan"].text(), component["projchan"].text()) == ("POz", "POz")
    for _name, call in qt_dialogs.calls:
        assert (call["initialvalue"], call["selectionmode"], len(call["liststring"])) == ([], "single", 32)


def test_event_field_list_accepted_without_a_choice_changes_nothing(qt_dialogs) -> None:
    widgets = qt_dialogs.build(pop_erpimage_dialog_spec(create_erpimage_epochs()))
    qt_dialogs.answers["listdlg2"] = ([], 1, "")
    widgets["field_button"].click()
    assert widgets["field"].text() == ""
    assert qt_dialogs.messages == [TYPE_REMINDER]


FIELD_REMINDER = ("Warning", "Do not forget to select an epoch-sorting field in the previous edit box")
TYPE_REMINDER = ("Warning", "Do not forget to select an event type in the next edit box")


def test_coherence_frequency_is_mirrored_into_the_phase_box(qt_dialogs) -> None:
    widgets = qt_dialogs.build(pop_erpimage_dialog_spec(create_erpimage_epochs()))
    widgets["phase"].setText("8 12")
    widgets["coher"].setText("10 10")
    widgets["coher"].editingFinished.emit()
    assert widgets["phase"].text() == "10 10"


def test_event_type_list_replaces_typed_types_and_reminds_about_the_field(qt_dialogs) -> None:
    widgets = qt_dialogs.build(pop_erpimage_dialog_spec(create_erpimage_epochs()))
    widgets["type"].setText("bogus")
    widgets["type_button"].click()  # cancelled
    assert widgets["type"].text() == "bogus"
    assert qt_dialogs.messages == [FIELD_REMINDER]
    qt_dialogs.messages.clear()
    qt_dialogs.answers["chansel"] = ([1, 2], 1, "")
    widgets["field"].setText("latency")
    widgets["type_button"].click()
    assert widgets["type"].text() == "rt stim"
    assert qt_dialogs.messages == []


def test_event_lists_report_a_dataset_without_events(qt_dialogs) -> None:
    eeg = create_erpimage_epochs()
    eeg["event"] = []
    widgets = qt_dialogs.build(pop_erpimage_dialog_spec(eeg))
    widgets["field_button"].click()
    widgets["type_button"].click()
    assert qt_dialogs.messages == [("Error", "No events"), TYPE_REMINDER, ("Error", "No type field"), FIELD_REMINDER]


def test_event_field_list_cancel_keeps_the_field_and_reminds_about_the_type(qt_dialogs) -> None:
    widgets = qt_dialogs.build(pop_erpimage_dialog_spec(create_erpimage_epochs()))
    widgets["field"].setText("latency")
    widgets["field_button"].click()  # cancelled
    assert widgets["field"].text() == "latency"
    assert qt_dialogs.messages == [TYPE_REMINDER]


def test_real_dialog_widgets_round_trip_into_the_command(epochs, qt_dialogs) -> None:
    class QtWidgetsRenderer:
        """Builds the real dialog with its initial values, edits widgets and reads them as the Qt renderer does."""

        def run(self, spec, initial_values=None):
            widgets = qt_dialogs.build(spec, initial_values)
            widgets["field"].setText("rt")
            widgets["type"].setText("rt")
            widgets["nosort"].setChecked(True)
            return {tag: qt_dialogs.qt._read_widget(widget) for tag, widget in widgets.items()}

    result, command = pop_erpimage(
        create_erpimage_epochs(), 1, 2, gui=True, smooth=1, renderer=QtWidgetsRenderer(), plot="off", return_com=True
    )
    assert command == (
        "pop_erpimage(EEG, 1, 2, smooth=1, decimate=1, sortingtype=['rt'], sortingeventfield='rt', nosort=True)"
    )
    np.testing.assert_allclose(result.outvar, [120, 60, 180, 30, 150, 90])
    component = qt_dialogs.build(pop_erpimage_dialog_spec(epochs, typeplot=0))
    assert component["projchan"].isEnabled() and component["projchan_button"].isEnabled()
