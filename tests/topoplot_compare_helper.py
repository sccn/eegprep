# get relative path
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from eegprep import topoplot
from eegprep import pop_loadset
import sys

print('Topoplot compare helper')
    
# need 2 arguments or error
if len(sys.argv) != 3:
    print(f'len(sys.argv) = {len(sys.argv)}')
    print('Usage: need 2 arguments python topoplot_compare_helper.py <eeglab_file_path> <comp_num>')
    sys.exit(1)

# Load the data
eeglab_file_path = sys.argv[1]
comp_num = int(sys.argv[2])

# '/System/Volumes/Data/data/matlab/eeglab/sample_data/eeglab_data_epochs_ica.set'
EEG = pop_loadset(eeglab_file_path)

# Print the loaded data
res = topoplot(EEG['icawinv'].transpose()[comp_num], EEG['chanlocs'], noplot='on')

grid = res[1]

# save in a matlab file
import scipy.io
scipy.io.savemat('topoplot_data.mat', {'grid': grid})
print('Saved topoplot_data.mat')
    