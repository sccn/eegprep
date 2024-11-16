clear
pyenv('Version', '/Users/arno/miniconda3/envs/p39env/bin/python');
system('/Users/arno/miniconda3/envs/p311env/bin/python iclabel_helper.py eeglab_data_with_ica_tmp.set eeglab_data_with_ica_out.set');
EEGTMP = pop_loadset('eeglab_data_with_ica_out.set');
labels_py4 = EEGTMP.etc.ic_classification.ICLabel.classifications;

% call EEGLAB function
if ~exist('pop_loadset')
    addpath('~/eeglab');
end
eeglabpath = which('eeglab.m');
eeglabpath = eeglabpath(1:end-length('eeglab.m'));
EEG = pop_loadset(fullfile(pwd, 'eeglab_data_with_ica_tmp.set'));
EEG = pop_iclabel(EEG, 'default');

labels_mat = EEG.etc.ic_classification.ICLabel.classifications;

compare_variables(labels_py4, labels_mat);

%% compare the two
figure('position', [924   752   912   565])
subplot(1,3,1);
imagesc(labels_mat); 
cl = clim;
ylabel('Component index')
xlabel('Label category')
title('MATLAB');

subplot(1,3,2);
imagesc(labels_py4); 
xlabel('Label category')
clim(cl)
title('Python');
cbar;

subplot(1,3,3);
imagesc(labels_mat-labels_py4); 
% ylabel('Component index')
clim([-1 1])
xlabel('Label category')
title('Difference');
cbar;

setfont(gcf, 'fontsize', 20)
set(gcf, 'color', 'w')
set(gcf, 'PaperPositionMode', 'auto');
print('-djpeg', 'figures/iclabel_diff.jpg')
print('-depsc', 'figures/iclabel_diff.eps')

[~,class_mat] = max(labels_mat,[],2);
[~,class_py ] = max(labels_py4,[],2);

misclassified = (class_mat~=class_py);
fprintf('Misclassified: %d\n', sum(misclassified));
fprintf('Misclassified max : %2.1f\n', 100*max(max(labels_mat(misclassified,:), [], 2)));
fprintf('Misclassified mean: %2.1f\n', 100*mean(max(labels_mat(misclassified,:), [], 2)));
fprintf('Misclassified std : %2.1f\n', 100*std(max(labels_mat(misclassified,:), [], 2)));
