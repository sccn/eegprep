clear

pythonFunc = '../.venv/bin/python';
pyenv('Version', pythonFunc);
dataset = '/System/Volumes/Data/data/matlab/eeglab/sample_data/eeglab_data_epochs_ica.set';
addpath(fullfile(pwd, '..', 'eeglab'));
if ~exist('pop_loadset')
    eeglab;
end

fileNameIn  = fullfile(pwd, '..', 'data', 'eeglab_data_with_ica_tmp.set');
fileNameOut = [ fileNameIn(1:end-4) '_averef.set' ];

EEG = pop_loadset(fileNameIn);
EEG2 = pop_reref(EEG, []);

EEG3 = EEG;
if length(EEG3.icachansind) == EEG3.nbchan
    EEG3.data = bsxfun(@minus, EEG3.data, mean(EEG3.data));
    EEG3.icawinv = bsxfun(@minus, EEG3.icawinv , mean(EEG3.icawinv));
    EEG3.icaweights = pinv(EEG3.icawinv);
    EEG3.icasphere = eye(EEG3.nbchan);
    EEG3.ref = 'average';

    for iChan = 1:length(EEG3.chanlocs)
        EEG3.chanlocs(iChan).ref = 'average';
    end
end

system([pythonFunc ' pop_reref_compare_helper.py ' fileNameIn ]);
EEG4 = load('-mat', 'python_temp.set'); % do not use pop_loadset or it rescales to RMS
EEG4.icaact = EEG4.icaweights*EEG4.data(:,:);
delete('python_temp.set')

fprintf('EEG2 vs EEG3 icaact:\n')
EEG3.icaact(1:10,1:10) - EEG2.icaact(1:10,1:10)

fprintf('EEG2 vs EEG3 icaweights:\n')
EEG3.icaweights(1:10,1:10) - EEG2.icaweights(1:10,1:10)

fprintf('EEG2 vs EEG3 icawinv:\n')
EEG3.icawinv(1:10,1:10) - EEG2.icawinv(1:10,1:10)

fprintf('**************\n\nEEG2 vs EEG4 icaact:\n')
EEG3.icaact(1:10,1:10) - EEG4.icaact(1:10,1:10)

fprintf('EEG2 vs EEG4 icaweights:\n')
EEG3.icaweights(1:10,1:10) - EEG4.icaweights(1:10,1:10)

fprintf('EEG2 vs EEG4 icawinv:\n')
EEG3.icawinv(1:10,1:10) - EEG4.icawinv(1:10,1:10)