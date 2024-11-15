pyenv('Version', '/Users/arno/miniconda3/envs/p39env/bin/python');
system('/Users/arno/miniconda3/envs/p311env/bin/python iclabel_net.py');
labels_py4 = load('-mat','output4.mat');
labels_py4 = reshape(mean(reshape(labels_py4.output', [], 4), 2), 7, [])';
delete('output4.mat');

% call EEGLAB function
if ~exist('pop_loadset')
    addpath('~/eeglab');
end
eeglabpath = which('eeglab.m');
eeglabpath = eeglabpath(1:end-length('eeglab.m'));
EEG = pop_loadset(fullfile(eeglabpath, 'sample_data', 'eeglab_data_epochs_ica.set'));
EEG = pop_iclabel(EEG, 'default');

labels_mat = EEG.etc.ic_classification.ICLabel.classifications;

compare_variables(labels_py4, labels_mat);

