import mne
from mne.preprocessing import ICA
from mne_icalabel import label_components
from eeg_mne2eeglab_epochs import eeg_mne2eeglab_epochs
from iclabel import iclabel
from pop_loadset import pop_loadset

eeglab_data_path = '/System/Volumes/Data/data/matlab/eeglab/sample_data/eeglab_data_epochs_ica.set'

if 1:
    raw = mne.io.read_epochs_eeglab(eeglab_data_path)
    raw.filter(1., 40., fir_design='firwin')

    # Select the EEG channels
    picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False, exclude='bads')

    # Fit ICA
    ica = ICA(n_components=30, random_state=97)
    ica.fit(raw, picks=picks)

    # Label components using ICALabel
    labels2 = label_components(raw, ica, method='iclabel')

    # Print the labels
    for idx, label in enumerate(labels2['labels']):
        print(f"Component {idx}: {label}")
    EEG = eeg_mne2eeglab_epochs(raw, ica)
else:
    eeglab_data_path = '/System/Volumes/Data/data/matlab/eeglab/sample_data/eeglab_data_epochs_ica.set'
    EEG = pop_loadset(eeglab_data_path)

EEG = iclabel(EEG)
labels = EEG['etc']['ic_classification']['label']
print(labels)