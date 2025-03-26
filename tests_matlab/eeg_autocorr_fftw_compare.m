clear

% this script compares the MATLAB and Python version of the function
pythonFunc = '../.venv/bin/python';
pyenv('Version', pythonFunc);
dataset = '/System/Volumes/Data/data/matlab/eeglab/sample_data/eeglab_data_epochs_ica.set';
addpath(fullfile(pwd, '..', 'eeglab'));
if ~exist('pop_loadset')
    eeglab;
end

% call Python function
fileName = '../data/eeglab_data_with_ica_tmp.set';
% fileName = '/System/Volumes/Data/data/data/STUDIES/STERN/S01/Memorize.set';

system([ pythonFunc ' eeg_autocorr_fftw_compare_helper.py ' fileName ]);
res = load('eeg_autocorr_data.mat');
%delete('eeg_autocorr_data.mat');
EEG = pop_loadset(fileName);
EEG = pop_reref(EEG, []);

temp2 = eeg_autocorr_fftw(EEG, 100);

%% compare the two
figure('position', [924   752   912   565])
subplot(2,2,1);
imagesc(temp2); 
cl = clim;
ylabel('Component index')
title('MATLAB');

subplot(2,2,3);
imagesc(res.grid); 
xlabel('Frequency (Hz)')
ylabel('Component index')
clim(cl)
title('Python');
cbar;

subplot(2,2,2);
imagesc(temp2-res.grid); 
% ylabel('Component index')
clim(cl-mean(cl))
title('Difference');
cbar;

subplot(2,2,4);
imagesc(temp2-res.grid); 
xlabel('Frequency (Hz)')
% ylabel('Component index')
cl = clim;
clim(cl-mean(cl))
title('Magnified difference');
cbar;

setfont(gcf, 'fontsize', 20)
set(gcf, 'color', 'w')
set(gcf, 'PaperPositionMode', 'auto');
print('-djpeg', '../figures/autocorr_diff.jpg')
print('-depsc', '../figures/autocorr_diff.eps')


%print('-djpeg', 'topoplot_diff.jpg')
compare_variables(temp2, res.grid);

return

comp = 59;
figure('position', [201   715   560   232]); plot(temp2(comp,:));
hold on; plot(res.grid(comp, :), 'r')
set(gcf, 'color', 'w')
xlabel('Frequency');
ylabel('Autocorrelation');
legend({ 'MATLAB' 'Python' })
setfont(gcf, 'fontsize', 16)


