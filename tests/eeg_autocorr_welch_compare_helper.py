import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from eegprep import pop_loadset
from eegprep import eeg_autocorr_welch

eeglab_file_path = '../data/eeglab_data_with_ica_tmp.set'
EEG = pop_loadset(eeglab_file_path)

# Print the loaded data
res = eeg_autocorr_welch(EEG, 100)

# save in a matlab file
import scipy.io
scipy.io.savemat('eeg_autocorr_welch_data.mat', {'grid': res})
print('Saved eeg_autocorr_welch_data.mat')