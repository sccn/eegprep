
% call EEGLAB function
if ~exist('pop_loadset')
    addpath('~/eeglab');
end
eeglabpath = which('eeglab.m');
eeglabpath = eeglabpath(1:end-length('eeglab.m'));
EEG = pop_loadset(fullfile(pwd, 'eeglab_data_with_ica_tmp.set'));

EEG2 = pop_reref(EEG, []);

EEGold = EEG;
if length(EEG.icachansind) == EEG.nbchan
    EEG.data = bsxfun(@minus, EEG.data, mean(EEG.data));
    EEG.icawinv = bsxfun(@minus, EEG.icawinv , mean(EEG.icawinv));
    EEG.icaweights = pinv(EEG.icawinv);
    EEG.icasphere = eye(EEG.nbchan);
    EEG.ref = 'average';

    for iChan = 1:length(EEG.chanlocs)
        EEG.chanlocs(iChan).ref = 'average';
    end
end


