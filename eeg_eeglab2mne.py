from eeg_autocorr import eeg_autocorr
from pop_loadset import pop_loadset
import mne
import tempfile
import os
from mne.export import export_raw
from pop_saveset import pop_saveset # in development
import eeglabio
import numpy as np

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
    raw = eeglabio.load_set(new_temp_file_path)
    
    return raw
