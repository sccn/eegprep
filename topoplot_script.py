from topoplot import topoplot
from pop_loadset import pop_loadset

eeglab_file_path = '/System/Volumes/Data/data/matlab/eeglab/sample_data/eeglab_data_epochs_ica.set'
EEG = pop_loadset(eeglab_file_path)

# Print the loaded data
res = topoplot(EEG['icawinv'].transpose()[0], EEG['chanlocs'], noplot='on')

grid = res[1]

# save in a matlab file
import scipy.io
scipy.io.savemat('topoplot.mat', {'grid': grid})
print('Saved topoplot.mat')