function [data, mapped] = eegprep_test_mmo_preprocess(filename, function_name, function_arguments, rejection, clear_events)
% Preserve live EEG/mmo values through the source preprocessing sequence.
EEG = pop_loadset(filename);
if clear_events
    EEG.event = [];
end
if ~isempty(rejection)
    EEG = eeg_eegrej(EEG, rejection);
end
EEG2 = feval(function_name, EEG, function_arguments{:});
mapped = isa(EEG2.data, 'mmo');
data = EEG2.data(:,:,:);
end
