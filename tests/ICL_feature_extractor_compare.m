% extract features for the ICLabel Classifier
% this script compares the MATLAB and Python version of the function
pyenv('Version', '/Users/arno/miniconda3/envs/p39env/bin/python');

% call Python function
system('/Users/arno/miniconda3/envs/p311env/bin/python ICL_feature_extractor_compare_helper.py');
res = load('python_temp.mat');
% delete('python_temp.mat');

% call EEGLAB function
if ~exist('pop_loadset')
    addpath('~/eeglab');
end
eeglabpath = which('eeglab.m');
eeglabpath = eeglabpath(1:end-length('eeglab.m'));
EEG = pop_loadset(fullfile(pwd, 'eeglab_data_with_ica_tmp.set'));
features = ICL_feature_extractor(EEG, true);

compare_variables(features, res.grid, 0.03);

