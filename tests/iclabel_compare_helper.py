import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from eegprep import iclabel
from eegprep import pop_loadset
from eegprep import pop_saveset

# check if there are 2 arguments otherwise issue an error
if len(sys.argv) != 3:
    raise ValueError('Please provide the input and output file paths as command line arguments')

# get first command line argument
eeglab_file_path_in  = sys.argv[1]
eeglab_file_path_out = sys.argv[2]

EEG = pop_loadset(eeglab_file_path_in)
# EEG['data']   = EEG['data'].reshape(EEG['data'].shape[0], -1)
# EEG['icaact'] = EEG['icaact'].reshape(EEG['icaact'].shape[0], -1)
# EEG['nbchan'] = EEG['data'].shape[0]
# EEG['pnts']   = EEG['data'].shape[1]
# EEG['trials'] = 1

# Print the loaded data
# print('Data shape:', EEG['data'].shape)
EEG = iclabel(EEG, 'default')

# save dataset
pop_saveset(EEG, eeglab_file_path_out)
