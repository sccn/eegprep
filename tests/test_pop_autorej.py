"""MATLAB parity tests for pop_autorej on sample_data/eeglab_data_epochs_ica.set.

Expected epoch numbers come from tests/matlab/pop_autorej_reference.m run with
MATLAB R2025b against EEGLAB develop aadfd19a7 (pop_autorej.m after the unreachable
15% branch was removed). Rejected epoch numbers must match exactly; EEGLAB lists
them in rejection order, EEGPrep sorted. EEGLAB's final kurtosis pass reads the
component field in channel mode and so never rejects there; the reference script
re-applies it with the channel field, and EEGPrep applies it in both modes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from eegprep.functions.popfunc.pop_autorej import pop_autorej
from eegprep.functions.popfunc.pop_loadset import pop_loadset
from tests.eeglab_tests import eeglab_test

EPOCHED_DATASET_PATH = Path(__file__).resolve().parents[1] / "sample_data" / "eeglab_data_epochs_ica.set"


@eeglab_test("unittesting_popfunc/pop_autorej/popfunc_pop_autorej_wrapperTest.m", "test_test_pop_autorej")
def test_reference_pop_autorej(eeglab_backend, eeglab_suite_root):
    eeg = eeglab_backend("pop_loadset", str(eeglab_suite_root / "eeglab/sample_data/eeglab_data_epochs_ica.set"))
    channel_count = np.asarray(eeg["nbchan"]).item()
    for options in (
        (
            "threshold",
            1000.0,
            "electrodes",
            np.arange(1.0, channel_count + 1)[None, :],
            "icacomps",
            np.empty((0, 0)),
            "startprob",
            5.0,
            "maxrej",
            5.0,
        ),
        (),
        ("threshold", 100000000.0),
        ("threshold", 0.1),
        ("threshold", 1000.0, "electrodes", np.array([[1.0]])),
        ("threshold", 1000.0, "electrodes", np.arange(1.0, np.fix(channel_count / 2) + 1)[None, :]),
    ):
        eeglab_backend("pop_autorej", eeg, *options, "eegplot", "off", "nogui", "on", nargout=2)


# Each case names the loop phases it exercises on the 80-epoch dataset.
EEGLAB_REJECTIONS = [
    # 4 of 80 flagged at 5 s.d. is not fewer than 5%: raise to 5.5, reject 4 epochs over two
    # passes, then one pruning round back at 5 s.d. rejects epoch 28; kurtosis adds 16.
    pytest.param((), [1, 2, 16, 28, 46, 70], id="default"),
    # Raise to 6 s.d., reject epoch 70, then eight pruning rounds oscillate 5.5/6 s.d.
    pytest.param(("maxrej", 2.5), [16, 70], id="maxrej2p5"),
    # Start below the default: five raises before anything is rejected.
    pytest.param(("startprob", 3), [1, 2, 16, 28, 46, 70], id="start3"),
    pytest.param(("startprob", 3, "maxrej", 2.5), [16, 70], id="start3_maxrej2p5"),
    pytest.param(("startprob", 4, "maxrej", 2.5), [16, 70], id="start4_maxrej2p5"),
    # 1.25% of 80 epochs means nothing can be rejected; eight pruning rounds then kurtosis.
    pytest.param(("maxrej", 1.25), [16], id="maxrej1p25"),
    # Two pruning rounds reach 5 s.d. and stop.
    pytest.param(("electrodes", list(range(1, 17)), "maxrej", 2.5), [16, 46, 70], id="elec1to16_maxrej2p5"),
    # Component mode: rejects at 5 s.d. without raising, kurtosis adds two epochs.
    pytest.param(("icacomps", list(range(1, 33))), [2, 9, 10, 43], id="ica_default"),
    pytest.param(("icacomps", list(range(1, 33)), "maxrej", 2.5), [9, 10], id="ica_maxrej2p5"),
    pytest.param(("icacomps", list(range(1, 11)), "maxrej", 2.5), [9], id="ica1to10_maxrej2p5"),
]


@pytest.mark.parametrize(("options", "expected"), EEGLAB_REJECTIONS)
def test_pop_autorej_rejects_the_same_epochs_as_eeglab(options, expected):
    EEG = pop_loadset(EPOCHED_DATASET_PATH)

    out, rejected = pop_autorej(EEG, *options, "nogui", "on")

    assert rejected == expected
    assert out["trials"] == EEG["trials"] - len(expected)
