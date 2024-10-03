from eeg_autocorr import eeg_autocorr
from pop_loadset import pop_loadset
import mne
import tempfile
import os
from mne.export import export_raw
from pop_saveset import pop_saveset # in development

# write a funtion that converts a MNE raw object to an EEGLAB set file
def eeg_eeglab2mne(EEG):
    
    # Generate a temporary file name
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file_path = temp_file.name    
    
    base, _ = os.path.splitext(temp_file_path)
    new_temp_file_path = base + ".set"

    # save the raw file as a new EEGLAB .set file using MNE EEGLAB writer
    pop_saveset(EEG, new_temp_file_path)
    
    # load the EEGLAB set file
    if EEG['trials'] > 1:
        raw = mne.io.read_epochs_eeglab(new_temp_file_path)
    else:
        raw = mne.io.read_raw_eeglab(new_temp_file_path, preload=True)
    
    return raw

def test_eeg_eeglab2mne():
    eeglab_file_path = './eeglab_data_with_ica_tmp.set'
    eeglab_file_path = '/System/Volumes/Data/data/matlab/eeglab/sample_data/eeglab_data_epochs_ica.set'
    EEG = pop_loadset(eeglab_file_path)
    raw = eeg_eeglab2mne(EEG)
    
    # print the keys of the EEG dictionary
    print(raw.info)

# test_eeg_eeglab2mne()