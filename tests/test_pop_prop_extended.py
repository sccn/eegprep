import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from eegprep.plugins.ICLabel.pop_prop_extended import (
    DEFAULT_ICLABEL_CLASSES,
    build_extended_property_data,
    classifier_names,
    resolve_classifier_data,
    selected_property_indices,
)
from eegprep.plugins.ICLabel.pop_viewprops import pop_viewprops
from tests.fixtures import create_test_eeg_with_ica


def _iclabel_eeg(*, include_classifier: bool = True) -> dict:
    rng = np.random.default_rng(8)
    eeg = create_test_eeg_with_ica(n_channels=4, n_samples=120, srate=100.0, n_components=4, n_trials=2)
    eeg["xmin"] = -0.1
    eeg["xmax"] = 1.09
    eeg["times"] = np.linspace(-100.0, 1090.0, 120)
    eeg["data"] = rng.normal(0.0, 1.0, (4, 120, 2))
    eeg["icaweights"] = np.eye(4)
    eeg["icasphere"] = np.eye(4)
    eeg["icawinv"] = np.eye(4)
    eeg["icaact"] = rng.normal(0.0, 0.5, (4, 120, 2))
    eeg["icachansind"] = np.arange(4)
    eeg["event"] = [
        {"type": "stim", "latency": 20.0, "duration": 0.0},
        {"type": "resp", "latency": 80.0, "duration": 0.0},
    ]
    eeg["reject"] = {"gcompreject": np.zeros(4, dtype=int)}
    if include_classifier:
        eeg["etc"] = {
            "ic_classification": {
                "Other": {
                    "classifications": np.full((4, 3), 1.0 / 3.0),
                    "classes": ["One", "Two", "Three"],
                },
                "ICLabel": {
                    "classifications": np.array(
                        [
                            [0.70, 0.10, 0.10, 0.03, 0.02, 0.03, 0.02],
                            [0.02, 0.94, 0.02, 0.01, 0.00, 0.00, 0.01],
                            [0.05, 0.02, 0.91, 0.01, 0.00, 0.00, 0.01],
                            [0.80, 0.05, 0.05, 0.02, 0.02, 0.03, 0.03],
                        ]
                    ),
                },
            }
        }
    return eeg


def test_classifier_data_parsing_defaults_to_iclabel_and_standard_classes() -> None:
    eeg = _iclabel_eeg()

    classifier = resolve_classifier_data(eeg, component_total=4, require=True)

    assert classifier.name == "ICLabel"
    assert classifier.classes == DEFAULT_ICLABEL_CLASSES
    np.testing.assert_allclose(classifier.probabilities[1], [0.02, 0.94, 0.02, 0.01, 0.0, 0.0, 0.01])
    assert classifier_names(eeg) == ["Other", "ICLabel"]


def test_classifier_data_rejects_component_count_mismatch() -> None:
    eeg = _iclabel_eeg()

    with pytest.raises(ValueError, match="rows for 3 ICA components"):
        resolve_classifier_data(eeg, "ICLabel", component_total=3, require=True)


def test_selected_component_indices_are_eeglab_facing_one_based() -> None:
    eeg = _iclabel_eeg()

    assert selected_property_indices(eeg, 0, "1:2") == [1, 2]
    assert selected_property_indices(eeg, 0, [], default_all=True) == [1, 2, 3, 4]
    with pytest.raises(ValueError, match="Index out of range"):
        selected_property_indices(eeg, 0, [0])


def test_dashboard_data_assembly_includes_classification_surfaces() -> None:
    eeg = _iclabel_eeg()

    dashboard = build_extended_property_data(eeg, 0, 2, spec_opt="'freqrange', [2 40]")

    assert dashboard.figure_title == "IC2 - pop_prop_extended()"
    assert dashboard.topography_title == "IC2"
    assert dashboard.activity_title == "Scrolling IC2 Activity"
    assert dashboard.classifier is not None
    assert dashboard.classifier.name == "ICLabel"
    assert dashboard.class_probabilities is not None
    assert dashboard.class_probabilities[1] == pytest.approx(0.94)
    assert dashboard.spectrum_freqs.size == dashboard.spectrum_power.size
    assert dashboard.image_data.size > 0
    assert dashboard.pvaf is None or np.isfinite(dashboard.pvaf)


def test_missing_classifier_falls_back_to_lightweight_viewprops_display() -> None:
    eeg = _iclabel_eeg(include_classifier=False)

    figures = pop_viewprops(eeg, 0, [1, 2], plot=True, show_activity=False)

    assert len(figures) == 1
    assert not hasattr(figures[0], "eegprep_dashboard_data")
    assert len(figures[0].eegprep_activity_views) == 2
    assert figures[0].eegprep_activity_views[0].state.events
    plt.close(figures[0])
