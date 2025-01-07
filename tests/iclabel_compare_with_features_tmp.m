% original comparison, not in a single Python script yet
clear

pythonFunc = '../.venv/bin/python';
pyenv('Version', pythonFunc);
dataset = '/System/Volumes/Data/data/matlab/eeglab/sample_data/eeglab_data_epochs_ica.set';
addpath(fullfile(pwd, '..', 'eeglab'));
if ~exist('pop_loadset')
    eeglab;
end

pyenv('Version', '/Users/arno/miniconda3/envs/p39env/bin/python');
system([pythonFunc ' ICL_feature_extractor_compare_helper.py']);

res = load('python_temp.mat');
res.grid{1} = single(cat(4, res.grid{1}, -res.grid{1}, res.grid{1}(:, end:-1:1, :, :), -res.grid{1}(:, end:-1:1, :, :)));
res.grid{2} = single(repmat(res.grid{2}, [1 1 1 4]));
res.grid{3} = single(repmat(res.grid{3}, [1 1 1 4]));
save('python_temp_reformated.mat', '-struct', 'res');

system([pythonFunc ' iclabel_net_load_py_measures.py']);
labels_py4 = load('-mat','output4_py.mat');
labels_py4 = reshape(mean(reshape(labels_py4.output', [], 4), 2), 7, [])';
delete('output4_py.mat');

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
