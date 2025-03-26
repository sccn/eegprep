import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from eegprep import eeg_rpsd
from eegprep import pop_loadset

if len(sys.argv) > 1:
    eeglab_file_path = sys.argv[1]
else:
    eeglab_file_path = '/System/Volumes/Data/data/matlab/eeglab/sample_data/eeglab_data_epochs_ica.set'
EEG = pop_loadset(eeglab_file_path)

# Print the loaded data
res = eeg_rpsd(EEG, 100)

# save in a matlab file
import scipy.io
scipy.io.savemat('eeg_rpsd_data.mat', {'grid': res})
print('Saved eeg_rpsd_data.mat')