"""Importer routing regressions for F18/F23/F119."""

import struct

import numpy as np
import pytest

from eegprep.functions.popfunc.pop_fileio import pop_fileio
from tests.test_legacy_signal_io_eeglab_tests import _write_neuroscan_eeg


@pytest.mark.parametrize('dtype', ['short', 'int32'])
def test_f119_neuroscan_eeg_loads_epochs_and_sweep_events(tmp_path, dtype):
    path = tmp_path / 'recording.EEG'
    first, second = _write_neuroscan_eeg(path, dtype=dtype)
    raw = bytearray(path.read_bytes())
    struct.pack_into('<f', raw, 505, 0.0)
    path.write_bytes(raw)
    eeg, command = pop_fileio(path, return_com=True)
    expected = np.stack([first, second], axis=2).astype(float) - np.array([100, 200])[:, None, None]
    expected *= np.array([2 / 204.8, 8 / 204.8])[:, None, None]
    np.testing.assert_allclose(eeg['data'], expected)
    assert (eeg['nbchan'], eeg['pnts'], eeg['trials'], eeg['srate']) == (2, 3, 2, 250)
    assert [loc['labels'] for loc in eeg['chanlocs']] == ['Fz', 'Cz']
    assert [event['type'] for event in eeg['event']] == ['trial', 'accept', 'correct', 'response', 'rt'] * 2
    assert [event['value'] for event in eeg['event']] == [11, 1, 1, 21, 350, 12, 0, 0, 22, 450]
    assert [event['latency'] for event in eeg['event']] == [1] * 5 + [4] * 5
    assert 'pop_fileio' in command


def test_f119_neuroscan_selection_keeps_trial_metadata(tmp_path):
    path = tmp_path / 'recording.eeg'
    _write_neuroscan_eeg(path)
    eeg = pop_fileio(path, channels=[2], trials=[2, 2])
    assert eeg['data'].shape[:2] == (1, 3)
    assert eeg['chanlocs'][0]['labels'] == 'Cz'
    assert eeg['event'][0]['value'] == 12
