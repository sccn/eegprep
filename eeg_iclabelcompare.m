function res = eeg_iclabelcompare(filename, plotFlag)

if nargin < 1
    error('Need file name')
end
if nargin < 2
    plotFlag = false;
end

pyenv('Version', '/Users/arno/miniconda3/envs/p39env/bin/python');
system(['/Users/arno/miniconda3/envs/p311env/bin/python iclabel_helper.py ' filename ' tmp.set' ]);
EEGTMP = pop_loadset('tmp.set');
delete('tmp.set')
labels_py4 = EEGTMP.etc.ic_classification.ICLabel.classifications;

EEG = pop_loadset(filename);
EEG = pop_iclabel(EEG, 'default');
labels_mat = EEG.etc.ic_classification.ICLabel.classifications;

res = compare_variables(labels_py4, labels_mat);
[~,class_mat] = max(labels_mat,[],2);
[~,class_py ] = max(labels_py4,[],2);

misclassified = (class_mat~=class_py);

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
    clim([-1 1])
    xlabel('Label category')
    title('Difference');
    cbar;

    setfont(gcf, 'fontsize', 20)
    set(gcf, 'color', 'w')
    set(gcf, 'PaperPositionMode', 'auto');
    print('-djpeg', 'figures/iclabel_diff.jpg')
    print('-depsc', 'figures/iclabel_diff.eps')
end