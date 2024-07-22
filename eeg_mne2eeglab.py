from eeg_autocorr import eeg_autocorr
from pop_loadset import pop_loadset
import mne
import tempfile
import os
from mne.export import export_raw
import eeglabio
import numpy as np

# write a funtion that converts a MNE raw object to an EEGLAB set file
def eeg_mne2eeglab(raw):
    # Generate a temporary file name
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file_path = temp_file.name    
    
    base, _ = os.path.splitext(temp_file_path)
    new_temp_file_path = base + ".set"

    # save the raw file as a new EEGLAB .set file using MNE EEGLAB writer
    export_raw(new_temp_file_path, raw, fmt='eeglab')

    # load the EEGLAB set file
    EEG = pop_loadset(new_temp_file_path)
    
    return EEG

def test_eeg_mne2eeglab():
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
    
    EEG2 = eeg_mne2eeglab(raw)
    
    # print the keys of the EEG dictionary
    print(EEG2.keys())

test_eeg_mne2eeglab()
