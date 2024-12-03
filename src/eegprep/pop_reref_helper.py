from ICL_feature_extractor import ICL_feature_extractor
from pop_loadset import pop_loadset
from pop_saveset import pop_saveset
from pop_reref import pop_reref
import sys

# check if a parameter is present and if it is assign eeglab_file_path to it
if len(sys.argv) > 2:
    eeglab_file_path_in  = sys.argv[1]
    eeglab_file_path_out = sys.argv[2]
else:
    eeglab_file_path_in  = './eeglab_data_with_ica_tmp.set'
    eeglab_file_path_out = './eeglab_data_with_ica_tmp_averef.set'
    
EEG = pop_loadset(eeglab_file_path_in)

# Print the loaded data
EEG = pop_reref(EEG, [])

# save dataset
pop_saveset(EEG, eeglab_file_path_out)

