from __future__ import annotations

import copy

import numpy as np
import pytest

from eegprep import pop_loadset
from eegprep.functions.guifunc.pophelp import pophelp_text
from eegprep.plugins.ICLabel.eeg_icalabelstat import eeg_icalabelstat
from eegprep.plugins.ICLabel.iclabel import iclabel
from eegprep.plugins.ICLabel.pop_icflag import ICLABEL_CLASSES
from eegprep.plugins.ICLabel.pop_viewprops import pop_viewprops


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


def test_eeg_icalabelstat_rejects_missing_or_malformed_iclabel_state() -> None:
    eeg = _classified_eeg()
    eeg["etc"] = {}

    with pytest.raises(ValueError, match="No ICLabel classifications"):
        eeg_icalabelstat(eeg, verbose=False)

    malformed = _classified_eeg()
    malformed["etc"]["ic_classification"]["ICLabel"]["classes"] = ["Brain"]
    with pytest.raises(ValueError, match="ICLabel class list has 1 labels"):
        eeg_icalabelstat(malformed, verbose=False)


def test_sample_data_ica_iclabel_state_drives_statistics_and_viewprops_history() -> None:
    eeg = pop_loadset("sample_data/eeglab_data_with_ica_tmp.set")
    classifications = np.zeros((eeg["icaweights"].shape[0], len(ICLABEL_CLASSES)), dtype=float)
    classifications[:, 0] = 0.8
    classifications[0, 1] = 0.95
    classifications[1, 2] = 0.96
    eeg = copy.deepcopy(eeg)
    eeg.setdefault("etc", {})["ic_classification"] = {
        "ICLabel": {"classes": list(ICLABEL_CLASSES), "classifications": classifications}
    }

    stats = eeg_icalabelstat(eeg, threshold=0.9, verbose=False)
    _figures, command = pop_viewprops(eeg, 0, [1, 2], plot=False, return_com=True)

    assert stats["component_count"] == 32
    np.testing.assert_array_equal(stats["counts"][:3], [0, 1, 1])
    assert command == "pop_viewprops(EEG, 0, [1 2], [], [], 1, '');"


def test_python_iclabel_rejects_unbundled_alternate_networks_before_runtime_dependencies() -> None:
    eeg = _classified_eeg()

    with pytest.raises(NotImplementedError, match="standalone Python ICLabel only ships the default network"):
        iclabel(eeg, algorithm="lite", engine=None)


def test_eeg_icalabelstat_help_is_packaged() -> None:
    help_text, source_path = pophelp_text("eeg_icalabelstat")

    assert "EEG_ICALABELSTAT" in help_text
    assert source_path == "eegprep/resources/help/eeg_icalabelstat.md"


def test_eeg_icalabelstat_is_public_lazy_export() -> None:
    from eegprep import eeg_icalabelstat as exported

    assert exported is eeg_icalabelstat
