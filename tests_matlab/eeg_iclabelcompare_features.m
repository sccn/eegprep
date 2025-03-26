function res = eeg_iclabelcompare_features(filename, plotFlag)

if nargin < 1
    error('Need file name')
end
if nargin < 2
    plotFlag = false;
end

pythonFunc = '../.venv/bin/python';
pyenv('Version', pythonFunc);
system([pythonFunc ' ICL_feature_extractor_compare_helper.py ' filename]);
respy = load('python_temp.mat');
delete('python_temp.mat');

featurespy = respy.grid;
respy.grid{1} = single(cat(4, respy.grid{1}, -respy.grid{1}, respy.grid{1}(:, end:-1:1, :, :), -respy.grid{1}(:, end:-1:1, :, :)));
respy.grid{2} = single(repmat(respy.grid{2}, [1 1 1 4]));
respy.grid{3} = single(repmat(respy.grid{3}, [1 1 1 4]));
save('python_temp_reformated.mat', '-struct', 'respy');

system([pythonFunc ' iclabel_net_load_py_measures.py']);
labels_py4 = load('-mat','output4_py.mat');
labels_py4 = reshape(mean(reshape(labels_py4.output', [], 4), 2), 7, [])';
delete('output4_py.mat');

% MATLAB
EEG = pop_loadset(filename);
disp 'ICLabel: extracting features...'
features = ICL_feature_extractor(EEG, true);
disp 'ICLabel: calculating labels...'
labels = run_ICL('default', features{:});
disp 'ICLabel: saving results...'
EEG.etc.ic_classification.ICLabel.classes = ...
    {'Brain', 'Muscle', 'Eye', 'Heart', ...
     'Line Noise', 'Channel Noise', 'Other'};
EEG.etc.ic_classification.ICLabel.classifications = labels;
EEG.etc.ic_classification.ICLabel.version = version;
labels_mat = EEG.etc.ic_classification.ICLabel.classifications;

res = compare_variables(labels_py4, labels_mat);
[~,class_mat] = max(labels_mat,[],2);
[~,class_py ] = max(labels_py4,[],2);

misclassified = (class_mat~=class_py);

res.topoplotdiff = squeeze(nanmean(nanmean(nanmean(nanmean(abs((featurespy{1} - features{1})./features{1}),1),2),3),3))';
res.psddiff      = squeeze(nanmean(nanmean(nanmean(nanmean(abs((featurespy{2} - features{2})./features{2}),1),2),3),3))';
res.autocorrdiff = squeeze(nanmean(nanmean(nanmean(nanmean(abs((featurespy{3} - features{3})./features{3}),1),2),3),3))';

% comp = 1; figure; subplot(1,2,1); imagesc(featurespy{1}(:,:,1,comp)); caxis([-1 1]); subplot(1,2,2); imagesc(features{1}(:,:,1,comp)); caxis([-1 1]); 

res.misclassified     = sum(misclassified);
res.misclassifiedmax  = max(max(labels_mat(misclassified,:), [], 2));
res.misclassifiedmean = mean(max(labels_mat(misclassified,:), [], 2));
res.misclassifiedstd  = std(max(labels_mat(misclassified,:), [], 2));

ind2 = find(EEG.etc.ic_classification.ICLabel.classifications(:,2) > 0.9);
ind3 = find(EEG.etc.ic_classification.ICLabel.classifications(:,3) > 0.9);
res.artifacts = length(ind2) + length(ind3);
res.misclassifiedartifacts = length(intersect([ind2;ind3], find(misclassified)));

fprintf('Misclassified: %d\n', res.misclassified);
fprintf('Misclassified max : %2.1f\n', 100*res.misclassifiedmax);
fprintf('Misclassified mean: %2.1f\n', 100*res.misclassifiedmean);
fprintf('Misclassified std : %2.1f\n', 100*res.misclassifiedstd);

if plotFlag
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
    clim([-0.00001 0.00001])
    xlabel('Label category')
    title('Magnified difference');
    cbar;

    setfont(gcf, 'fontsize', 20)
    set(gcf, 'color', 'w')
    set(gcf, 'PaperPositionMode', 'auto');
    print('-depsc', 'figures/iclabel_diff.eps')
    print('-djpeg', 'figures/iclabel_diff.jpg')
end