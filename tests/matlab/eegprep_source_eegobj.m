function lengths = eegprep_source_eegobj(sample_directory)
% Native object operations from eegobj/eegobj_simpletests.m. Keep the object
% in MATLAB; export only the six lengths asserted by the original source.
pop_editoptions('option_eegobject', 1);
try
    EEG = pop_loadset('filename', 'eeglab_data_epochs_ica.set', 'filepath', sample_directory);
    EEG.pnts = 3;
    lengths = length(EEG);
    EEG(2) = EEG;
    lengths(end+1) = length(EEG);
    EEG(4) = EEG(1);
    lengths(end+1) = length(EEG);
    EEG(1:3) = [];
    lengths(end+1) = length(EEG);
    EEG(3) = EEG;
    EEG(4:5) = EEG(2:3);
    lengths(end+1) = length(EEG);
    EEG(2).filename = 'test';
    EEG(1).chanlocs(1).labels = 'E1';
    EEG(end) = EEG(1);
    EEG(end) = [];
    lengths(end+1) = length(EEG);
    pop_editoptions('option_eegobject', 0);
catch err
    pop_editoptions('option_eegobject', 0);
    rethrow(err);
end
end
