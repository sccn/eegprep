% this script compares the MATLAB and Python version of the function
pyenv('Version', '/Users/arno/miniconda3/envs/p39env/bin/python');

% call Python function
system('/Users/arno/miniconda3/envs/p311env/bin/python eeg_autocorr_fftw_compare_helper.py');
res = load('eeg_autocorr_data.mat');
%delete('eeg_autocorr_data.mat');

% call EEGLAB function
if ~exist('pop_loadset')
    addpath('~/eeglab');
end
eeglabpath = which('eeglab.m');
eeglabpath = eeglabpath(1:end-length('eeglab.m'));
EEG = pop_loadset(fullfile(pwd, 'eeglab_data_with_ica_tmp.set'));
temp2 = eeg_autocorr(EEG, 100);

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
print('-djpeg', 'figures/autocorr_diff.jpg')
print('-depsc', 'figures/autocorr_diff.eps')


%print('-djpeg', 'topoplot_diff.jpg')
compare_variables(temp2, res.grid);
