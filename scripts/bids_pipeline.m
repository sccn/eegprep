function result_paths = bids_pipeline(rootpath, subjs, runs)
% --- this pipeline is used to run the MATLAB side of the test_bids_preproc() unit test ---

if nargin < 1
    rootpath = '/home/christian/data/OpenNeuro/ds003061-download'; end
if nargin < 2
    % subjs = {'sub-002'};   % for testing
    subjs = {'sub-001','sub-002'};
end
if nargin < 3
    runs = {'1'}; end

% import BIDS
[STUDY, ALLEEG] = pop_importbids(...    
    rootpath, ...
    'subjects', subjs, ...
    'runs', runs);

result_paths = {};
for idx=1:length(ALLEEG)
    EEG = ALLEEG(idx);

    % keep only channels with EEG modality
    chn_modalities = {EEG.chanlocs.type};
    keep = find(strcmp('EEG',chn_modalities));
    EEG = pop_select(EEG, 'channel', keep);

    orig_chanlocs = EEG.chanlocs;

    % resampling
    EEG = pop_resample(EEG, 128);

    % artifact removal
    EEG = clean_artifacts(EEG);

    % PICARD
    EEG = eeg_picard(EEG);  % EEG = pop_runica(EEG, 'icatype', 'picard');    

    % ICLabel
    EEG = pop_iclabel(EEG, 'Default');

    % reinterpolate channels
    EEG = eeg_interp(EEG, orig_chanlocs);

    % epoching
    EEG = pop_epoch(EEG, {}, [-0.2, 0.5]);

    % baseline removal
    EEG = pop_rmbase(EEG, [-200, 0]);

    % common average reference
    EEG = pop_reref(EEG, []);

    % write back as .set
    tmp_path = [tempname(), '.set'];
    pop_saveset(EEG, 'filename', tmp_path);
    result_paths{end+1} = tmp_path;
    
end

return
