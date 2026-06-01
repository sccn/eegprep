import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from eegprep.functions.guifunc.pophelp import pophelp_text
from eegprep.plugins.ICLabel.pop_prop_extended import pop_prop_extended, pop_prop_extended_dialog_spec
from eegprep.plugins.ICLabel.pop_viewprops import pop_viewprops
from tests.fixtures import create_test_eeg_with_ica


def _dashboard_eeg() -> dict:
    eeg = create_test_eeg_with_ica(n_channels=4, n_samples=100, srate=100.0, n_components=4, n_trials=1)
    samples = np.linspace(0.0, 1.0, 100)
    eeg["data"] = np.vstack([np.sin(2 * np.pi * (index + 1) * samples) for index in range(4)])
    eeg["icaweights"] = np.eye(4)
    eeg["icasphere"] = np.eye(4)
    eeg["icawinv"] = np.eye(4)
    eeg["icaact"] = np.vstack([np.cos(2 * np.pi * (index + 1) * samples) for index in range(4)])
    eeg["icachansind"] = np.arange(4)
    eeg["times"] = samples * 1000.0
    eeg["xmin"] = 0.0
    eeg["xmax"] = 1.0
    eeg["event"] = [{"type": "stim", "latency": 25.0, "duration": 0.0}]
    eeg["reject"] = {"gcompreject": np.zeros(4, dtype=int)}
    eeg["etc"] = {
        "ic_classification": {
            "ICLabel": {
                "classifications": np.array(
                    [
                        [0.70, 0.10, 0.10, 0.03, 0.02, 0.03, 0.02],
                        [0.02, 0.94, 0.02, 0.01, 0.00, 0.00, 0.01],
                        [0.05, 0.02, 0.91, 0.01, 0.00, 0.00, 0.01],
                        [0.80, 0.05, 0.05, 0.02, 0.02, 0.03, 0.03],
                    ]
                ),
                "classes": ["Brain", "Muscle", "Eye", "Heart", "Line Noise", "Channel Noise", "Other"],
            }
        }
    }
    return eeg


def _axis_by_title(figure, title: str):
    return next(axis for axis in figure.axes if axis.get_title() == title)


def _event_marker_labels(figure, title: str) -> list[str]:
    axis = _axis_by_title(figure, title)
    return [text.get_text() for text in axis.texts]


def test_gui_dashboard_creation_has_eeglab_labels_titles_and_activity_browser() -> None:
    eeg = _dashboard_eeg()

    figure = pop_prop_extended(eeg, 0, [1, 2], spec_opt="'freqrange', [2 40]", scroll_event=1)

    titles = [axis.get_title() for axis in figure.axes]
    assert figure.eegprep_dashboard_data.index == 1
    assert figure._suptitle.get_text() == "IC1 - pop_prop_extended()"
    assert "IC1" in titles
    assert "ICLabel" in titles
    assert "Scrolling IC1 Activity" in titles
    assert "Continuous Data" in titles
    assert "IC1 Activity Power Spectrum" in titles
    assert figure.eegprep_activity_view.state.title == "Scrolling IC1 Activity -- eegplot()"
    assert len(figure.eegprep_activity_view.state.events) == 1
    assert _event_marker_labels(figure, "Scrolling IC1 Activity") == ["stim"]
    assert set(figure.eegprep_dashboard_navigation) == {"previous", "next"}
    plt.close(figure)


def test_gui_dashboard_navigation_updates_visible_component() -> None:
    eeg = _dashboard_eeg()
    figure = pop_prop_extended(eeg, 0, [1, 2], scroll_event=1)

    figure.eegprep_dashboard_navigation["next"]()

    assert figure.eegprep_dashboard_data.index == 2
    assert figure._suptitle.get_text() == "IC2 - pop_prop_extended()"
    assert any(axis.get_title() == "Scrolling IC2 Activity" for axis in figure.axes)
    plt.close(figure)


def test_gui_dashboard_activity_browser_honors_event_display_option() -> None:
    eeg = _dashboard_eeg()

    with_events = pop_prop_extended(eeg, 0, 1, scroll_event=1)
    without_events = pop_prop_extended(eeg, 0, 1, scroll_event=0)

    assert len(with_events.eegprep_activity_view.state.events) == 1
    assert without_events.eegprep_activity_view.state.events == []
    assert _event_marker_labels(with_events, "Scrolling IC1 Activity") == ["stim"]
    assert _event_marker_labels(without_events, "Scrolling IC1 Activity") == []
    plt.close(with_events)
    plt.close(without_events)


def test_gui_dashboard_epoched_static_events_only_mark_displayed_trace() -> None:
    eeg = _dashboard_eeg()
    eeg["data"] = np.repeat(eeg["data"][:, :, np.newaxis], 2, axis=2)
    eeg["icaact"] = np.repeat(eeg["icaact"][:, :, np.newaxis], 2, axis=2)
    eeg["trials"] = 2
    eeg["event"] = [
        {"type": "first", "latency": 25.0, "duration": 0.0, "epoch": 1},
        {"type": "second", "latency": 125.0, "duration": 0.0, "epoch": 2},
    ]
    eeg["epoch"] = [
        {"event": [0], "eventtype": ["first"], "eventlatency": [0.0], "eventduration": [0.0]},
        {"event": [1], "eventtype": ["second"], "eventlatency": [0.0], "eventduration": [0.0]},
    ]

    figure = pop_prop_extended(eeg, 0, 1, scroll_event=1)

    assert _event_marker_labels(figure, "Scrolling IC1 Activity") == ["first"]
    assert [event.type for event in figure.eegprep_activity_view.state.events] == ["first", "second"]
    plt.close(figure)


def test_pop_viewprops_component_mode_opens_extended_dashboard_when_classifier_is_available() -> None:
    eeg = _dashboard_eeg()

    figures = pop_viewprops(eeg, 0, [1, 2], plot=True, show_activity=False)

    assert len(figures) == 1
    assert figures[0].eegprep_dashboard_data.index == 1
    assert figures[0].eegprep_dashboard_data.classifier.name == "ICLabel"
    figures[0].eegprep_dashboard_navigation["next"]()
    assert figures[0].eegprep_dashboard_data.index == 2
    plt.close(figures[0])


def test_pop_prop_extended_dialog_and_help_are_packaged() -> None:
    eeg = _dashboard_eeg()

    spec = pop_prop_extended_dialog_spec(eeg, 0)
    help_text, source_path = pophelp_text("pop_prop_extended")

    assert spec.title == "Component properties - pop_prop_extended()"
    assert spec.eeglab_source == "plugins/ICLabel/viewprops/pop_prop_extended.m"
    assert spec.help_text == "pophelp('pop_prop_extended')"
    assert "POP_PROP_EXTENDED" in help_text
    assert source_path == "eegprep/resources/help/pop_prop_extended.md"
