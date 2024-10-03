% this script compares the MATLAB and Python version of the function

% call Python function
system('/Users/arno/miniconda3/envs/p311env/bin/python topoplot_compare_helper.py')
% system('/Users/arno/miniconda3/envs/p311env/bin/python my_script.py');
res = load('topoplot_data.mat')
delete('topoplot_data.mat')

% call EEGLAB function
addpath('~/eeglab');
eeglab epoch
[~,temp2] = topoplotFast2(EEG.icawinv(:,1), EEG.chanlocs, 'noplot', 'on');

% compare the two
figure('position', [924  835 1276 482])
subplot(1,2,1);
imagesc(temp2); title('MATLAB'); cbar;
subplot(1,2,2);
imagesc(res.grid); title('Python'); cbar;

compare_variables(temp2, res.grid);
