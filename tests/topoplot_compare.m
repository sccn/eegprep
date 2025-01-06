% this script compares the MATLAB and Python version of the function
pyenv('Version', '/Users/arno/miniconda3/envs/p39env/bin/python');
dataset = '/System/Volumes/Data/data/matlab/eeglab/sample_data/eeglab_data_epochs_ica.set'
EEG = pop_loadset(dataset);

for compInd = 0:31
    
    % call Python function
    system([ '/Users/arno/miniconda3/envs/p311env/bin/python topoplot_compare_helper.py ' dataset ' ' int2str(compInd) ]);
    
    res = load('topoplot_data.mat')
    %delete('topoplot_data.mat')
    
    % call EEGLAB function
    if ~exist('pop_loadset.m')
        addpath('~/eeglab');
        eeglab;
        close;
    end
    [~,temp2] = topoplotFast2(EEG.icawinv(:,compInd+1), EEG.chanlocs, 'noplot', 'on');

    aa = temp2(~isnan(temp2(:)));
    bb = res.grid(~isnan(res.grid(:)));

    diffVal(compInd+1) = mean(abs((aa(:)-bb(:)))/max(abs(aa(:))));
    maxDiffVal(compInd+1) = max(abs((aa(:)-bb(:)))/max(abs(aa(:))));
end

%% compare the two
figure('position', [924 609 723 708])
subplot(2,2,1);
im = imagesc(temp2); title('MATLAB'); axis square; axis off;
set(im, 'AlphaData', ~isnan(temp2)); % Apply transparency
clim([0.37 1.82]);

subplot(2,2,3);
im = imagesc(res.grid); title('Python'); axis square;  axis off; 
set(im, 'AlphaData', ~isnan(res.grid)); % Apply transparency
clim([0.37 1.82]);
cbar;

subplot(2,2,2);
im = imagesc(temp2-res.grid); title('Difference'); axis square; axis off; 
set(im, 'AlphaData', ~isnan(res.grid)); % Apply transparency
clim(([0.37 1.82]-mean([0.37 1.82])));
cbar;

subplot(2,2,4);
im = imagesc(temp2-res.grid); title('Magnified difference'); axis square; axis off; 
set(im, 'AlphaData', ~isnan(res.grid)); % Apply transparency
clim(([0.37 1.82]-mean([0.37 1.82]))/5);
cbar;

setfont(gcf, 'fontsize', 16)
set(gcf, 'color', 'w')
set(gcf, 'PaperPositionMode', 'auto');
print('-djpeg', 'figures/topoplot_diff.jpg')

compare_variables(temp2, res.grid);
