
system('/Users/arno/miniconda3/envs/p311env/bin/python topoplot_script.py')
% system('/Users/arno/miniconda3/envs/p311env/bin/python my_script.py');

addpath('~/eeglab');
eeglab epoch
[~,temp2] = topoplotFast2(EEG.icawinv(:,1), EEG.chanlocs, 'noplot', 'on');

res = load('topoplot.mat')

figure;
subplot(1,2,1);
imagesc(temp2); cbar
subplot(1,2,2);
imagesc(res.grid); cbar
