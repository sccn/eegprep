"""Export MNE epochs to an EEGLAB dataset, including epoch events."""

import mne

from eegprep import eeg_mne2eeg, pop_saveset

sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = sample_data_folder / "MEG" / "sample" / "sample_audvis_filt-0-40_raw.fif"
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
EEG = eeg_mne2eeg(aud_epochs)
pop_saveset(EEG, "auditory_epochs.set")
