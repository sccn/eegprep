from ICL_feature_extractor import ICL_feature_extractor
from pop_loadset import pop_loadset

eeglab_file_path = './eeglab_data_with_ica_tmp.set'
EEG = pop_loadset(eeglab_file_path)

# Print the loaded data
res = ICL_feature_extractor(EEG, True)

# save in a matlab file
import scipy.io
scipy.io.savemat('python_temp.mat', {'grid': res})
print('Saved python_temp.mat')