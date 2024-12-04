from ICL_feature_extractor import ICL_feature_extractor
from pop_loadset import pop_loadset
import sys

# check if a parameter is present and if it is assign eeglab_file_path to it
if len(sys.argv) > 1:
    eeglab_file_path = sys.argv[1]
else:
    eeglab_file_path = './eeglab_data_with_ica_tmp.set'
    eeglab_file_path = '/Users/arno/Downloads/STUDYstern/S01/Memorize.set'
EEG = pop_loadset(eeglab_file_path)

# Print the loaded data
res = ICL_feature_extractor(EEG, True)

# save in a matlab file
import scipy.io
scipy.io.savemat('python_temp.mat', {'grid': res})
print('Saved python_temp.mat')