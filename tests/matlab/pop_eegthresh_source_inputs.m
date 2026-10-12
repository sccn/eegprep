function pop_eegthresh_source_inputs(eeglab_root, output_file)
% Reproduce only the random inputs of test_pop_eegthresh.m, not its outputs.
% eeglab_tests ff605546f3f70868916fb8d49c007472b3257b50
% EEGLAB 8ac485f654d6bbb1a6acb8dc9ef3f2eaf3d409ba
% Source: unittesting_popfunc/pop_eegthresh/test_pop_eegthresh.m
% Regenerate after adding this directory to a headless MATLAB path:
% pop_eegthresh_source_inputs('/path/to/eeglab_tests/eeglab', ...
%     '/path/to/tests/matlab/pop_eegthresh_source_inputs.mat');

addpath(eeglab_root);
eeglab('nogui');
rng('default');
rng(1);
EEG = pop_loadset(fullfile(eeglab_root, 'sample_data', 'eeglab_data_epochs_ica.set'));
LThresh = -1*rand(1,EEG.nbchan/2)*1000-50;
HThresh = rand(1,EEG.nbchan/2)*1000+50;
Thresh = rand(1,EEG.nbchan/2)*1000+50;
Elements = randperm(EEG.nbchan);
Elements = Elements(1:EEG.nbchan/2);
% Cases 35/36, 39/40, ..., 63/64 each generate a fresh component permutation.
ComponentElements = zeros(16, size(EEG.icaweights,1)/2);
for case_index = 1:16
    selection = randperm(size(EEG.icaweights,1));
    ComponentElements(case_index,:) = selection(1:size(EEG.icaweights,1)/2);
end
save(output_file, 'LThresh', 'HThresh', 'Thresh', 'Elements', 'ComponentElements', '-v6');
end
