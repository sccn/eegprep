from pop_reref import pop_reref
from pop_loadset import pop_loadset
import sys
import os
import scipy.io

# Check if a parameter was provided
if len(sys.argv) > 1:
    # The first element in sys.argv is the script name, so we access the second one for the argument
    file_name = sys.argv[1]
    print(f"The EEG file is: {file_name}")
    print(f"Output will be in python_temp.mat (result field)")
else:
    file_name = './eeglab_data_with_ica_tmp.set'
    # raise("Provide the name of a file containing EEG (full path) as the first parameter.")
    
# check if file name exists on disk
if not os.path.isfile(file_name):
    raise(f"File {file_name} does not exist.")

EEG = pop_loadset(file_name)
EEG = pop_reref(EEG, None)
EEG['chanlocs'] = []
EEG['urchanlocs'] = []
EEG['event'] = []
EEG['urevent'] = []
EEG['epoch'] = []

# save in a matlab file
scipy.io.savemat('python_temp.mat', EEG)
print('Saved python_temp.mat')