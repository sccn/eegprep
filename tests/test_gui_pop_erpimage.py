"""``pop_erpimage`` dialog: EEGLAB layout, result mapping and Qt callbacks."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
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
    """Dialog seam: returns the given values (None cancels) and keeps the spec it was shown."""

    def __init__(self, values: dict | None) -> None:
        self.values = values
        self.spec = None

    def run(self, spec, initial_values=None):
        self.spec = spec
        if self.values is None:
            return None
        defaults = {
            tag: control.value for tag, control in controls_by_tag(spec).items() if control.style != "pushbutton"
        }
        return {**defaults, **self.values}


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
    assert command == (
        "pop_erpimage(EEG, 1, 'POz', titleplot='Sorted', smooth=3, decimate=2, sortingtype=['rt'], "
        "sortingwin=[0, 1000], sortingeventfield='latency', cbar=False, "
        "limits=[-200, 800, -5, 5, -3, 3, 0, 1, 1], caxis=[-10, 10], renorm='yes', align=float('inf'), "
        "nosort=True, noplot=True, phasesort=[100, 25, 10, 12], coher=[10, 10, 0.01], spec=[2, 40], "
        "vert=[0, 500], erpstd='on', plotmap=False, rng=1)"
    )


def test_unchanged_time_limits_are_not_passed_and_a_single_coher_frequency_is_kept(epochs) -> None:
    values = {"erp": False, "coher": "10 12", "phase": "", "limtime": "-1000 1992.1875"}
    _result, command = pop_erpimage(epochs, 1, renderer=_Renderer(values), noshow=True, return_com=True)
    assert command == "pop_erpimage(EEG, 1, 1, smooth=10, decimate=1, noshow=True, erp=False, coher=[10, 12])"


def test_dialog_rescale_needs_a_sorting_field(epochs) -> None:
    _result, command = pop_erpimage(epochs, 1, renderer=_Renderer({"renorm": "yes"}), noshow=True, return_com=True)
    assert command == "pop_erpimage(EEG, 1, 1, smooth=10, decimate=1, noshow=True)"


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
        ({"others": "'erpstd'"}, "More options needs 'key', value pairs"),
        ({"phase2": "25"}, "'phasesort' needs at least"),
        ({"chan": "Q9"}, "unknown channel label 'Q9'"),
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
    assert len(component.geometry) == len(channel.geometry) + 1
    assert sum(len(row) for row in component.geometry) == len(component.controls)
    assert sum(len(row) for row in channel.geometry) == len(channel.controls)
    tags = controls_by_tag(channel)
    assert set(tags["field_button"].callback.params["fields"]) == {"type", "position", "latency", "urevent", "epoch"}
    assert tags["type_button"].callback.params["event_types"] == ["square", "rt"]
    assert tags["phase"].callback.params == {"source": "phase", "target": "coher"}
    no_locations = deepcopy(epochs)
    no_locations["chanlocs"] = []
    assert not controls_by_tag(pop_erpimage_dialog_spec(no_locations))["chan_button"].enabled
    assert controls_by_tag(pop_erpimage_dialog_spec(create_erpimage_epochs()))["smooth"].value == "1"


def test_qt_dialog_wires_the_event_lists_and_frequency_mirrors(qt_dialogs) -> None:
    widgets = qt_dialogs.build(pop_erpimage_dialog_spec(create_erpimage_epochs()))
    qt_dialogs.answers["fields"] = ([2], 1, "")  # type, latency, epoch, code, rt: the events' field order
    widgets["field_button"].click()
    qt_dialogs.answers["types"] = ([2], 1, "")
    widgets["type_button"].click()
    widgets["coher"].setText("10 10")
    widgets["phase"].setText("8 12")
    widgets["phase"].editingFinished.emit()
    assert (widgets["field"].text(), widgets["type"].text(), widgets["coher"].text()) == ("latency", "rt", "8 12")
    assert qt_dialogs.messages == [("Warning", "Do not forget to select an event type in the next edit box")]
