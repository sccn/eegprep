% this script compares the MATLAB and Python version of the function
clear

% this script compares the MATLAB and Python version of the function
pythonFunc = '../.venv/bin/python';
pyenv('Version', pythonFunc);
dataset = '/System/Volumes/Data/data/matlab/eeglab/sample_data/eeglab_data_epochs_ica.set';
addpath(fullfile(pwd, '..', 'eeglab'));
if ~exist('pop_loadset')
    eeglab;
end

eeglabpath = which('eeglab.m');
eeglabpath = eeglabpath(1:end-length('eeglab.m'));
fileName = fullfile(eeglabpath, 'sample_data', 'eeglab_data_epochs_ica.set');
%fileName = fullfile(pwd, '../data/eeglab_data_with_ica_tmp.set');

% call Python function
system([pythonFunc ' eeg_rpsd_compare_helper.py ' fileName]);
res = load('eeg_rpsd_data.mat');
%delete('eeg_rpsd_data.mat');

% call EEGLAB function
EEG = pop_loadset(fileName);
temp2 = eeg_rpsd(EEG, 100);

%% compare the two
figure('position', [924   752   912   565])

subplot(2,2,1);
imagesc(temp2); title('MATLAB'); 
cl = clim;
ylabel('Component index')
title('MATLAB');

subplot(2,2,3);
imagesc(res.grid); title('Python'); 
xlabel('Frequency (Hz)')
ylabel('Component index')
clim(cl)
title('Python');
cbar;

subplot(2,2,2);
imagesc(temp2-res.grid); 
clim(cl-mean(cl))
title('Difference');
cbar;

subplot(2,2,4);
imagesc(temp2-res.grid); 
xlabel('Frequency (Hz)')
cl = clim;
clim(cl-mean(cl))
title('Magnified difference');
cbar;

setfont(gcf, 'fontsize', 20)
set(gcf, 'color', 'w')
set(gcf, 'PaperPositionMode', 'auto');
print('-djpeg', '../figures/ersp_diff.jpg')
print('-depsc', '../figures/ersp_diff.eps')

compare_variables(temp2, res.grid);
