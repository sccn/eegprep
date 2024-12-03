from eeg_autocorr_fftw import eeg_autocorr_fftw
from pop_loadset import pop_loadset
from pop_reref import pop_reref
import sys

# check if a parameter is present and if it is assign eeglab_file_path to it
if len(sys.argv) > 1:
    eeglab_file_path = sys.argv[1]
else:
    eeglab_file_path = './eeglab_data_with_ica_tmp.set'
EEG = pop_loadset(eeglab_file_path)
EEG = pop_reref(EEG, [])

# Print the loaded data
res = eeg_autocorr_fftw(EEG, 100)

# save in a matlab file
import scipy.io
scipy.io.savemat('eeg_autocorr_data.mat', {'grid': res})
print('Saved eeg_autocorr_data.mat')