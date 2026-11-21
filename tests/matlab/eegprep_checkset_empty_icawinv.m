function result = eegprep_checkset_empty_icawinv(weights, sphere)
% eeg_checkset rebuilds an empty icawinv from icaweights*icasphere.
if exist('eeg_checkset', 'file') ~= 2 || exist('eeg_emptyset', 'file') ~= 2
    error('eegprep_checkset_empty_icawinv:MissingEEGLAB', 'EEGLAB eeg_checkset must be on the path.');
end
EEG = eeg_emptyset;
EEG.nbchan = size(sphere, 2);
EEG.pnts = 8;
EEG.trials = 1;
EEG.srate = 100;
EEG.xmin = 0;
EEG.xmax = (EEG.pnts - 1) / EEG.srate;
EEG.data = zeros(EEG.nbchan, EEG.pnts);
EEG.icaweights = weights;
EEG.icasphere = sphere;
EEG.icawinv = [];
EEG.icaact = [];
EEG.icachansind = 1:EEG.nbchan;
EEG = eeg_checkset(EEG);
result.icawinv = EEG.icawinv;
end
