% this script compares the MATLAB and Python version of the function
pyenv('Version', '/Users/arno/miniconda3/envs/p39env/bin/python');

eeglabpath = which('eeglab.m');
eeglabpath = eeglabpath(1:end-length('eeglab.m'));
fileName = fullfile(eeglabpath, 'sample_data', 'eeglab_data_epochs_ica.set');
fileName = '/System/Volumes/Data/data/data/STUDIES/STERN/S01/Memorize.set';

% call Python function
system('/Users/arno/miniconda3/envs/p311env/bin/python eeg_autocorr_compare_helper.py ' fileName);
res = load('eeg_autocorr_data.mat');
delete('eeg_autocorr_data.mat');

% call EEGLAB function
if ~exist('pop_loadset')
    addpath('~/eeglab');
end
eeglabpath = which('eeglab.m');
eeglabpath = eeglabpath(1:end-length('eeglab.m'));
EEG = pop_loadset(fullfile(pwd, 'eeglab_data_with_ica_tmp.set'));
temp2 = eeg_autocorr(EEG, 100);

% compare the two
figure('position', [924  835 1276 482])
subplot(1,2,1);
imagesc(temp2); title('MATLAB'); cbar;
subplot(1,2,2);
imagesc(res.grid); title('Python'); cbar;

compare_variables(temp2, res.grid);
