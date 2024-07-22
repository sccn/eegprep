# Example to export MNE epochs to EEGLAB dataset
# Events are not handled correctly in this example but it works

import mne
from mne.datasets import sample
import numpy as np
from scipy.io import savemat

# Load example data
sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = (
    sample_data_folder / "MEG" / "sample" / "sample_audvis_filt-0-40_raw.fif"
)
raw = mne.io.read_raw_fif(sample_data_raw_file)
events = mne.find_events(raw, stim_channel="STI 014")
event_dict = {
    "auditory/left": 1,
    "auditory/right": 2,
    "visual/left": 3,
    "visual/right": 4,
    "smiley": 5,
    "buttonpress": 32,
}
epochs = mne.Epochs(
    raw,
    events,
    event_id=event_dict,
    tmin=-0.2,
    tmax=0.5,
    preload=True,
)

aud_epochs = epochs["auditory"]

# export to EEGLAB dataset
data = epochs.get_data()  # Get the data from the epochs
n_epochs, n_channels, n_times = data.shape

# Create a dictionary to save as a MATLAB file
eeglab_dict = {
    'data': data,
    'times': epochs.times,
    'srate': epochs.info['sfreq'],
    'nbchan': n_channels,
    'trials': n_epochs,
    'pnts': n_times,
    'xmin': aud_epochs.tmin,
    'xmax': aud_epochs.tmax,
    'event': np.array([[event[0], 0, event[2]] for event in events])  # Create event array
}

# # Step 4: Save the EEGLAB dataset as a .mat file
# savemat('output_file.mat', eeglab_dict)

#print("EEGLAB dataset saved successfully!")