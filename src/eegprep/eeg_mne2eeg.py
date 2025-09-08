from .eeg_autocorr import eeg_autocorr
from .pop_loadset import pop_loadset
import mne
import tempfile
import os
from mne.export import export_raw
import numpy as np

def _mne_events_to_eeglab_events(raw_or_epochs):
    """
    Convert MNE Annotations or events to EEGLAB event structure (list of dicts).
    """
    events = []
    sfreq = raw_or_epochs.info['sfreq']
    # Handle Annotations (Raw)
    if hasattr(raw_or_epochs, 'annotations') and raw_or_epochs.annotations is not None and len(raw_or_epochs.annotations) > 0:
        for ann in raw_or_epochs.annotations:
            latency = int(ann['onset'] * sfreq) + 1  # EEGLAB is 1-based
            events.append({'latency': latency, 'type': ann['description']})
    # Handle Epochs/events array
    elif hasattr(raw_or_epochs, 'events') and raw_or_epochs.events is not None:
        for ev in raw_or_epochs.events:
            latency = int(ev[0]) + 1  # sample index, 1-based
            # Try to get event type from event_id
            event_type = None
            if hasattr(raw_or_epochs, 'event_id'):
                for k, v in raw_or_epochs.event_id.items():
                    if v == ev[2]:
                        event_type = k
                        break
            if event_type is None:
                event_type = str(ev[2])
            events.append({'latency': latency, 'type': event_type})
    return events

# write a funtion that converts a MNE raw object to an EEGLAB set file
def eeg_mne2eeg(raw):
    # Generate a temporary file name
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file_path = temp_file.name    
    
    base, _ = os.path.splitext(temp_file_path)
    new_temp_file_path = base + ".set"

    # save the raw file as a new EEGLAB .set file using MNE EEGLAB writer
    export_raw(new_temp_file_path, raw, fmt='eeglab')

    # load the EEGLAB set file
    EEG = pop_loadset(new_temp_file_path)

    # Inject events/annotations from MNE object into EEGLAB structure
    eeglab_events = _mne_events_to_eeglab_events(raw)
    if eeglab_events:
        EEG['event'] = eeglab_events
    
    return EEG

def test_eeg_mne2eeg():
    eeglab_file_path = './eeglab_data_with_ica_tmp.set'
    eeglab_file_path = '/System/Volumes/Data/data/matlab/eeglab/sample_data/eeglab_data_epochs_ica.set'
    EEG = pop_loadset(eeglab_file_path)

    # create MNE info structure
    info = mne.create_info(ch_names=[ x['labels'] for x in EEG['chanlocs']], sfreq=EEG['srate'], ch_types='eeg')
    if EEG['trials'] > 1:
        events = np.array([[i, 0, 1] for i in range(EEG['trials'])]) # NOT CORRECT CONVERTION JUST FOR TESTING
        event_id = dict(dummy=1)
        raw = mne.EpochsArray(EEG['data'].transpose(2,0,1), info, events, tmin=0, event_id=event_id)
    else:
        raw = mne.io.RawArray(EEG['data'], info)
    
    EEG2 = eeg_mne2eeg(raw)
    
    # print the keys of the EEG dictionary
    print(EEG2.keys())

# test_eeg_mne2eeg()
