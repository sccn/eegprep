% Reference rejections for tests/test_pop_autorej.py.
%
% Runs EEGLAB pop_autorej on sample_data/eeglab_data_epochs_ica.set for each option
% set below and prints the rejected epoch numbers (sorted; EEGLAB returns them in
% rejection order).
%
% EEGLAB's final kurtosis pass counts EEG.reject.icarejkurt whatever the mode, so for
% channel-based runs it never rejects anything. The script re-applies that pass with
% the channel field (EEG.reject.rejkurt) on the returned dataset and reports the
% combined result, which is what EEGPrep is expected to produce.
%
% Run from tests/matlab with an EEGLAB checkout on the path (or the vendored one).
clear
addpath(fullfile(pwd, '..', '..', 'src', 'eegprep', 'eeglab'));
if ~exist('pop_loadset', 'file')
    eeglab nogui;
end

EEG = pop_loadset('../../sample_data/eeglab_data_epochs_ica.set');
ncomp = size(EEG.icaweights, 1);
cases = { ...
    'default',             {}; ...
    'maxrej2p5',           {'maxrej', 2.5}; ...
    'start3',              {'startprob', 3}; ...
    'start3_maxrej2p5',    {'startprob', 3, 'maxrej', 2.5}; ...
    'start4_maxrej2p5',    {'startprob', 4, 'maxrej', 2.5}; ...
    'maxrej1p25',          {'maxrej', 1.25}; ...
    'elec1to16_maxrej2p5', {'electrodes', 1:16, 'maxrej', 2.5}; ...
    'ica_default',         {'icacomps', 1:ncomp}; ...
    'ica_maxrej2p5',       {'icacomps', 1:ncomp, 'maxrej', 2.5}; ...
    'ica1to10_maxrej2p5',  {'icacomps', 1:10, 'maxrej', 2.5} };

for i = 1:size(cases, 1)
    options = cases{i, 2};
    [EEG2, rmep] = pop_autorej(EEG, 'nogui', 'on', options{:});
    if ~any(strcmp(options, 'icacomps'))
        electrodes = 1:EEG.nbchan;
        idx = find(strcmp(options, 'electrodes'));
        if ~isempty(idx), electrodes = options{idx+1}; end
        EEG3 = pop_rejkurt(EEG2, 1, electrodes, 6, 6, 0, 0);
        remaining = setdiff(1:EEG.trials, rmep);
        rmep = [rmep remaining(find(EEG3.reject.rejkurt))];
    end
    fprintf('REFERENCE %s: %s\n', cases{i, 1}, mat2str(sort(rmep)));
end
