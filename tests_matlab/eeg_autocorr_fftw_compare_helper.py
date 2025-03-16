import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from eegprep import pop_loadset
from eegprep import eeg_autocorr_fftw
from eegprep import pop_reref

# check if a parameter is present and if it is assign eeglab_file_path to it
if len(sys.argv) > 1:
    eeglab_file_path = sys.argv[1]
else:
    eeglab_file_path = '../data/eeglab_data_with_ica_tmp.set'
EEG = pop_loadset(eeglab_file_path)
EEG = pop_reref(EEG, [])

# Print the loaded data
res = eeg_autocorr_fftw(EEG, 100)

# save in a matlab file
import scipy.io
scipy.io.savemat('eeg_autocorr_data.mat', {'grid': res})
print('Saved eeg_autocorr_data.mat')