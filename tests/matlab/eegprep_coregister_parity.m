function result = eegprep_coregister_parity(dataset, template)
% Traditional warp from coregister(..., 'warp', 'auto', 'manual', 'off').
if exist('coregister', 'file') ~= 2 || exist('pop_loadset', 'file') ~= 2
    error('eegprep_coregister_parity:MissingEEGLAB', 'EEGLAB coregister must be on the path.');
end
EEG = pop_loadset(char(dataset));
[~, transform] = coregister(EEG.chanlocs, char(template), 'warp', 'auto', 'manual', 'off');
result.transform = transform(:)';
end
