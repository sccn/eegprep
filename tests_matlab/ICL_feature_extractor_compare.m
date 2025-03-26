clear

% this script compares the MATLAB and Python version of the function
pythonFunc = '../.venv/bin/python';
pyenv('Version', pythonFunc);
dataset = '/System/Volumes/Data/data/matlab/eeglab/sample_data/eeglab_data_epochs_ica.set';
addpath(fullfile(pwd, '..', 'eeglab'));
if ~exist('pop_loadset')
    eeglab;
end

fileName = '../data/eeglab_data_with_ica_tmp.set';

% call Python function
system([pythonFunc ' ICL_feature_extractor_compare_helper.py ' fileName]);
res = load('python_temp.mat');
% delete('python_temp.mat');

% call EEGLAB function
EEG = pop_loadset(fullfile(pwd, fileName));
features = ICL_feature_extractor(EEG, true);

compare_variables(features, res.grid, 0.03);

