from .eeg_autocorr_welch import eeg_autocorr_welch
from .pop_loadset import pop_loadset

eeglab_file_path = './eeglab_data_with_ica_tmp.set'
EEG = pop_loadset(eeglab_file_path)

# Print the loaded data
res = eeg_autocorr_welch(EEG, 100)

# save in a matlab file
import scipy.io
scipy.io.savemat('eeg_autocorr_welch_data.mat', {'grid': res})
print('Saved eeg_autocorr_welch_data.mat')