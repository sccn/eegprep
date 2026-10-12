from __future__ import annotations


import numpy as np

from eegprep.plugins.ICLabel.eeg_icalabelstat import eeg_icalabelstat
from eegprep.plugins.ICLabel.pop_icflag import ICLABEL_CLASSES


def _classified_eeg() -> dict:
    return {
        "setname": "classified",
        "data": np.zeros((4, 100)),
        "nbchan": 4,
        "pnts": 100,
        "trials": 1,
        "srate": 100.0,
        "icaweights": np.eye(4),
        "icasphere": np.eye(4),
        "icawinv": np.eye(4),
        "icachansind": np.arange(4),
        "reject": {"gcompreject": np.array([0, 1, 1, 0])},
        "etc": {
            "ic_classification": {
                "ICLabel": {
                    "classes": list(ICLABEL_CLASSES),
                    "classifications": np.array(
                        [
                            [0.70, 0.10, 0.10, 0.03, 0.02, 0.03, 0.02],
                            [0.02, 0.94, 0.02, 0.01, 0.00, 0.00, 0.01],
                            [0.05, 0.02, 0.91, 0.01, 0.00, 0.00, 0.01],
                            [0.80, 0.05, 0.05, 0.02, 0.02, 0.03, 0.03],
                        ]
                    ),
                }
            }
        },
    }


def test_eeg_icalabelstat_matches_eeglab_threshold_counts_and_prints_summary(capsys) -> None:
    stats = eeg_icalabelstat(_classified_eeg(), threshold=0.9)

    assert stats["classes"] == list(ICLABEL_CLASSES)
    assert stats["component_count"] == 4
    np.testing.assert_array_equal(stats["counts"], [0, 1, 1, 0, 0, 0, 0])
    assert stats["component_indices"][1] == [2]
    assert stats["component_indices"][2] == [3]
    np.testing.assert_array_equal(stats["rejected_counts"], [0, 1, 1, 0, 0, 0, 0])
    np.testing.assert_array_equal(stats["kept_counts"], [0, 0, 0, 0, 0, 0, 0])

    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == len(ICLABEL_CLASSES)
    assert lines[0].strip() == 'IClabel class "Brain": 0/4 components at 90% threshold'
    assert lines[1].strip() == 'IClabel class "Muscle": 1/4 components at 90% threshold'
    assert lines[2].strip() == 'IClabel class "Eye": 1/4 components at 90% threshold'


def test_eeg_icalabelstat_accepts_class_specific_thresholds_and_default_classes() -> None:
    eeg = _classified_eeg()
    eeg["etc"]["ic_classification"]["ICLabel"].pop("classes")

    stats = eeg_icalabelstat(eeg, threshold=[0.6, 0.9, 0.9, 0.5, 0.5, 0.5, 0.5], verbose=False)

    assert stats["classes"] == list(ICLABEL_CLASSES)
    np.testing.assert_array_equal(stats["counts"], [2, 1, 1, 0, 0, 0, 0])
    np.testing.assert_allclose(stats["threshold"], [0.6, 0.9, 0.9, 0.5, 0.5, 0.5, 0.5])
    np.testing.assert_array_equal(stats["dominant_counts"], [2, 1, 1, 0, 0, 0, 0])
