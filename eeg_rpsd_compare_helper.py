from eeg_rpsd import eeg_rpsd
from pop_loadset import pop_loadset

eeglab_file_path = '/System/Volumes/Data/data/matlab/eeglab/sample_data/eeglab_data_epochs_ica.set'
EEG = pop_loadset(eeglab_file_path)

# Print the loaded data
res = eeg_rpsd(EEG, 100)

# save in a matlab file
import scipy.io
scipy.io.savemat('eeg_rpsd_data.mat', {'grid': res})
print('Saved eeg_rpsd_data.mat')