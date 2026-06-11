from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

import eegprep
from eegprep.functions.popfunc.pop_loadset import pop_loadset
from eegprep.functions.studyfunc.pop_study import pop_study
from eegprep.functions.studyfunc.std_pac import std_pac
from eegprep.functions.studyfunc.std_pacplot import std_pacplot
from eegprep.functions.studyfunc.std_readdata import std_readpac
from eegprep.functions.timefreqfunc._pac_support import PAC_UNSUPPORTED_MESSAGE
from eegprep.functions.timefreqfunc.pac import pac
from eegprep.functions.timefreqfunc.pac_cont import pac_cont
from tests.fixtures import SAMPLE_DATASET_PATH, create_test_eeg


def test_standalone_pac_helpers_raise_explicit_limitation_on_sample_data():
    eeg = pop_loadset(SAMPLE_DATASET_PATH)
    signal = np.asarray(eeg["data"])[0]

    with pytest.raises(NotImplementedError, match="does not implement standalone phase-amplitude coupling"):
        pac(signal, signal, eeg["srate"])
    with pytest.raises(NotImplementedError, match="does not implement standalone phase-amplitude coupling"):
        pac_cont(signal, signal, eeg["srate"])

    assert eegprep.pac is pac
    assert eegprep.pac_cont is pac_cont


def test_study_pac_entry_points_raise_same_limitation():
    study, alleeg = _study_pair()

    with pytest.raises(NotImplementedError, match="does not implement standalone phase-amplitude coupling") as pac_exc:
        std_pac(alleeg[0])
    with pytest.raises(NotImplementedError, match="does not implement standalone phase-amplitude coupling") as plot_exc:
        std_pacplot(study, alleeg)

    assert str(pac_exc.value) == str(plot_exc.value) == PAC_UNSUPPORTED_MESSAGE
    assert eegprep.std_pac is std_pac
    assert eegprep.std_pacplot is std_pacplot


def test_std_readpac_reads_and_slices_eegprep_owned_channel_cache():
    study, alleeg = _study_pair()
    raw = np.arange(2 * 4 * 5, dtype=float).reshape(2, 4, 5)
    study["changrp"] = [
        {
            "name": "Ch1",
            "channels": ["Ch1"],
            "inds": [1],
            "pacdata": raw,
            "pacfreqs": [4.0, 8.0, 12.0, 16.0],
            "pactimes": [-100.0, 0.0, 100.0, 200.0, 300.0],
        }
    ]

    _study, pacdata, pactimes, pacfreqs = std_readpac(
        study,
        alleeg,
        channels=[1],
        timerange=[0.0, 200.0],
        freqrange=[8.0, 16.0],
    )

    np.testing.assert_allclose(pacdata[0], raw[:, 1:4, 1:4])
    np.testing.assert_allclose(pactimes, [0.0, 100.0, 200.0])
    np.testing.assert_allclose(pacfreqs, [8.0, 12.0, 16.0])


def test_std_readpac_rejects_missing_or_malformed_pac_caches():
    study, alleeg = _study_pair()

    with pytest.raises(NotImplementedError, match="PAC reading requires EEGPrep-owned pacdata caches"):
        std_readpac(study, alleeg, clusters=1)

    malformed = deepcopy(study)
    malformed["cluster"][0]["pacdata"] = np.ones((2, 3, 4))
    malformed["cluster"][0]["pacfreqs"] = [4.0, 8.0]
    malformed["cluster"][0]["pactimes"] = [0.0, 100.0, 200.0, 300.0]
    with pytest.raises(ValueError, match="PAC cache shape"):
        std_readpac(malformed, alleeg, clusters=1)

    with pytest.raises(ValueError, match="Unknown std_readpac option"):
        std_readpac(study, alleeg, clusters=1, unsupported="on")


def _study_pair():
    first = create_test_eeg(n_channels=2, n_samples=64, n_trials=3, srate=128)
    first.update({"setname": "one", "subject": "S01", "condition": "target"})
    second = deepcopy(first)
    second.update({"setname": "two", "subject": "S02", "condition": "standard"})
    second["data"] = second["data"] * 2.0
    return pop_study(None, [first, second], name="PAC cache study")
