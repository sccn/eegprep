import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from eegprep.plugins.ICLabel.pop_prop_extended import pop_prop_extended
from tests.fixtures import create_test_eeg_with_ica


def _dashboard_eeg(*, include_dipfit: bool = False) -> dict:
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
    if include_dipfit:
        eeg["dipfit"] = {
            "coordformat": "MNI",
            "model": [
                {"posxyz": [0, -20, 40], "momxyz": [1, 0, 0], "rv": 0.12, "component": 1},
                {
                    "posxyz": [[25, 10, 35], [-25, 10, 35]],
                    "momxyz": [[0, 1, 0], [0, 2, 0]],
                    "rv": 0.2,
                    "component": 2,
                },
                {"posxyz": [], "momxyz": [], "rv": 1.0, "component": 3},
                {"posxyz": [], "momxyz": [], "rv": 1.0, "component": 4},
            ],
        }
    return eeg


def _axis_by_title(figure, title: str):
    return next(axis for axis in figure.axes if axis.get_title() == title)


def _event_marker_labels(figure, title: str) -> list[str]:
    axis = _axis_by_title(figure, title)
    return [text.get_text() for text in axis.texts]


def _dashed_marker_x_positions(figure, title: str) -> list[float]:
    axis = _axis_by_title(figure, title)
    return [float(line.get_xdata()[0]) for line in axis.lines if line.get_linestyle() == "--"]


def _reject_button_label(figure) -> str:
    return figure.eegprep_dashboard_rejection_buttons["status"].label.get_text()


def test_gui_dashboard_rejection_controls_commit_selected_component_flags() -> None:
    eeg = _dashboard_eeg()
    figure = pop_prop_extended(eeg, 0, [1, 2], scroll_event=1)

    assert _reject_button_label(figure) == "ACCEPT"
    figure.eegprep_dashboard_rejection["toggle"]()
    assert _reject_button_label(figure) == "REJECT"
    np.testing.assert_array_equal(eeg["reject"]["gcompreject"], [0, 0, 0, 0])

    figure.eegprep_dashboard_navigation["next"]()
    assert figure.eegprep_dashboard_data.index == 2
    assert _reject_button_label(figure) == "ACCEPT"
    figure.eegprep_dashboard_rejection["toggle"]()
    figure.eegprep_dashboard_navigation["previous"]()
    assert _reject_button_label(figure) == "REJECT"

    figure.eegprep_dashboard_rejection["ok"]()

    np.testing.assert_array_equal(eeg["reject"]["gcompreject"], [1, 1, 0, 0])
    plt.close(figure)


def test_gui_dashboard_rejection_cancel_discards_pending_flags() -> None:
    eeg = _dashboard_eeg()
    figure = pop_prop_extended(eeg, 0, 1, scroll_event=1)

    figure.eegprep_dashboard_rejection["toggle"]()
    figure.eegprep_dashboard_rejection["cancel"]()

    np.testing.assert_array_equal(eeg["reject"]["gcompreject"], [0, 0, 0, 0])
    plt.close(figure)


def test_gui_dashboard_epoched_static_events_use_flattened_event_latencies() -> None:
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

    marker_labels = _event_marker_labels(figure, "Scrolling IC1 Activity")
    assert "first" in marker_labels
    assert "second" in marker_labels
    assert "epoch 1" in marker_labels
    assert "epoch 2" in marker_labels
    assert sorted(round(position) for position in _dashed_marker_x_positions(figure, "Scrolling IC1 Activity")) == [
        25,
        125,
    ]
    assert [event.type for event in figure.eegprep_activity_view.state.events] == ["first", "second"]
    plt.close(figure)
