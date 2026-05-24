from __future__ import annotations

import numpy as np

from eegprep.functions.popfunc.eeg_runica import eeg_runica
from eegprep.functions.popfunc.pop_runica import pop_runica


def _epoched_eeg(offset=0):
    data = np.zeros((2, 3, 2), dtype=np.float64)
    data[0, :, 0] = np.array([1, 2, 3]) + offset
    data[0, :, 1] = np.array([4, 5, 6]) + offset
    data[1, :, 0] = np.array([11, 12, 13]) + offset
    data[1, :, 1] = np.array([14, 15, 16]) + offset
    return {
        "data": data,
        "nbchan": 2,
        "pnts": 3,
        "trials": 2,
        "srate": 100,
        "chanlocs": [],
    }


def _eeglab_flattened(data):
    return np.concatenate([data[:, :, trial] for trial in range(data.shape[2])], axis=1)


def test_eeg_runica_flattens_and_reshapes_epoched_data_like_eeglab(monkeypatch):
    eeg = _epoched_eeg()
    captured = {}

    def fake_runica(data, **_kwargs):
        captured["data"] = data.copy()
        return np.eye(2), np.eye(2), np.zeros(2), np.zeros((2, 1)), np.ones((2, 1)), []

    monkeypatch.setattr("eegprep.functions.popfunc.eeg_runica.runica", fake_runica)

    out = eeg_runica(eeg, extended=1, maxsteps=1)

    np.testing.assert_array_equal(captured["data"], _eeglab_flattened(eeg["data"]))
    np.testing.assert_array_equal(out["icaact"], eeg["data"])
    np.testing.assert_array_equal(out["icachansind"], np.array([0, 1]))


def test_pop_runica_concatenates_epoched_datasets_in_eeglab_order(monkeypatch):
    first = _epoched_eeg()
    second = _epoched_eeg(offset=100)
    captured = {}

    def fake_eeg_runica(eeg, sortcomps="off", **_kwargs):
        captured["data"] = np.asarray(eeg["data"]).copy()
        return dict(
            eeg,
            icasphere=np.eye(2),
            icaweights=np.eye(2),
            icawinv=np.eye(2),
            icaact=np.zeros((2, int(eeg["pnts"]), int(eeg["trials"]))),
            icachansind=np.array([0, 1]),
        )

    monkeypatch.setattr("eegprep.functions.popfunc.pop_runica.eeg_runica", fake_eeg_runica)

    out, command = pop_runica([first, second], concatenate="on", return_com=True)

    expected = np.concatenate([_eeglab_flattened(first["data"]), _eeglab_flattened(second["data"])], axis=1)
    np.testing.assert_array_equal(captured["data"], expected)
    assert len(out) == 2
    assert "'concatenate', 'on'" in command
