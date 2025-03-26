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
% fileName = '/System/Volumes/Data/data/data/STUDIES/STERN/S01/Memorize.set';

% call Python function
system([pythonFunc ' eeg_autocorr_compare_helper.py ' fileName]);
res = load('eeg_autocorr_data.mat');
delete('eeg_autocorr_data.mat');

% call EEGLAB function
EEG = pop_loadset(fileName);
temp2 = eeg_autocorr(EEG, 100);

% compare the two
figure('position', [924  835 1276 482])
subplot(1,2,1);
imagesc(temp2); title('MATLAB'); cbar;
subplot(1,2,2);
imagesc(res.grid); title('Python'); cbar;

compare_variables(temp2, res.grid);
