"""Numerical and options regressions for F19/F20/F21/F22."""

import numpy as np
import pytest

from eegprep import pop_chanevent, pop_importdata, pop_importepoch, pop_importevent
from eegprep.functions.adminfunc.console import EEGPrepConsoleWorkspace
from eegprep.functions.guifunc.session import EEGPrepSession
from eegprep.functions.popfunc.pop_importerplab import pop_importerplab
from eegprep.functions.popfunc.pop_importpres import pop_importpres
from tests.test_pop_loadset import _singleton_eeg


@pytest.mark.parametrize('transpose', [False, True])
def test_f21_ascii_matches_matlab_import_defaults(tmp_path, transpose):
    data = np.arange(48, dtype=float).reshape(4, 12) / 7
    path = tmp_path / 'data.txt'
    np.savetxt(path, data.T if transpose else data)
    eeg, command = pop_importdata(data=path, srate=0, return_com=True)
    np.testing.assert_array_equal(eeg['data'], data.astype(np.float32))
    assert eeg['data'].dtype == np.float32
    assert eeg['srate'] == 1
    assert eeg['setname'] == ''
    assert len(eeg['chanlocs']) == 0
    assert "'srate', 1" in command


def test_f21_epoch_layout_and_imported_ica():
    data = np.arange(24).reshape(2, 12)
    eeg = pop_importdata(data=data, pnts=4, srate=100, icaweights=np.eye(2), icasphere=np.eye(2))
    np.testing.assert_array_equal(eeg['data'][:, :, 1], data[:, 4:8])
    np.testing.assert_array_equal(eeg['icaweights'], np.eye(2))


def test_f19_capitalized_event_headers_require_explicit_fields(tmp_path):
    path = tmp_path / 'events.txt'
    path.write_text('Type Latency\nstim 0.5\n')
    eeg = _singleton_eeg(np.zeros((2, 100)))
    with pytest.raises(ValueError, match=r'pop_importevent.*lowercase.*fields'):
        pop_importevent(eeg, event=path, append='no')
    out = pop_importevent(eeg, event=path, fields=['type', 'latency'], skipline=1, timeunit=1, append='no')
    assert [(event['type'], event['latency']) for event in out['event']] == [('stim', 51)]
    assert not eeg['event']


def test_f20_both_edges_keep_types_at_coincident_latencies():
    eeg = _singleton_eeg(np.array([[0.0, -3.0, -1.0, -4.0, 0.0]]))
    out = pop_chanevent(eeg, 1, edge='both', delchan='off')
    assert [event['latency'] for event in out['event']] == [1, 3, 3, 5]
    assert [event['type'] for event in out['event']] == [-3.0, -3.0, -4.0, -4.0]
    assert [event['type'] for event in out['urevent']] == [-3.0, -3.0, -4.0, -4.0]


def test_f22_epoch_import_requires_field_mapping(tmp_path):
    path = tmp_path / 'epochs.txt'
    path.write_text('response RT\nleft 0.1\nright 0.2\n')
    eeg = _singleton_eeg(np.zeros((2, 30, 2)))
    with pytest.raises(ValueError, match=r'pop_importepoch.*field.*headerlines'):
        pop_importepoch(eeg, path)
    out = pop_importepoch(eeg, path, ['response', 'RT'], headerlines=1, latencyfields=['RT'], timeunit=1)
    assert [event['type'] for event in out['event']] == ['TLE', 'RT', 'TLE', 'RT']
    assert [event['latency'] for event in out['event']] == [1, 11, 31, 51]


def test_f10_erplab_maps_event_list_and_contrast_bins(tmp_path):
    event_list = tmp_path / 'events.txt'
    event_list.write_text('11 "Left target" 1\n12 "Right target" 2\n13 "Other" 3\n')
    bins = tmp_path / 'bins.txt'
    bins.write_text('b1 = (1) label Left\nb2 = (2) label Right\nb3 = b1-b2 label Side contrast\n')
    eeg = _singleton_eeg(np.zeros((2, 100)))
    eeg['event'] = [{'type': n, 'latency': i * 10 + 1} for i, n in enumerate((11, 12, 13))]
    out, command = pop_importerplab(eeg, event_list, bins, 1, return_com=True)
    assert [event['newtype'] for event in out['event']] == ['Left target', 'Right target', 'Other']
    assert [event['Left'] for event in out['event']] == [1, 0, 0]
    assert [event['Side_contrast'] for event in out['event']] == ['Left', 'Right', 'none of these']
    assert [event['latency'] for event in out['event']] == [1, 11, 21]
    assert 'newtype' not in eeg['event'][0]
    assert str(bins) in command


def test_f10_presentation_history_retains_selected_fields(tmp_path):
    path = tmp_path / 'events.log'
    path.write_text('Presentation export\n\nEvent Type\tCode\tTime\tDuration\nPicture\tstim\t100\t20\n')
    eeg = _singleton_eeg(np.zeros((2, 100)))
    out, command = pop_importpres(
        eeg, path, typefield='Event Type', latfield='Time', durfield='Duration', return_com=True
    )
    assert out['event'][0]['type'] == 'Picture'
    assert out['event'][0]['latency'] == 2
    assert "'Event Type', 'Time', 'Duration'" in command


def test_f21_channel_metadata_history_replays():
    eeg, command = pop_importdata(
        data=np.arange(12).reshape(2, 6), chanlocs=[{'labels': 'Fz'}, {'labels': 'Cz'}], return_com=True
    )
    session = EEGPrepSession()
    session.store_current(eeg, new=True, command=command)
    workspace = EEGPrepConsoleWorkspace(session)
    workspace.execute_history_command(command)
    assert [channel['labels'] for channel in session.EEG['chanlocs']] == ['Fz', 'Cz']
    np.testing.assert_array_equal(session.EEG['data'], eeg['data'])
