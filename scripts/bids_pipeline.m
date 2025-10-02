function result_paths = bids_pipeline(rootpath, subjs, runs, to_stage)
% Basic MATLAB-based equivalent of the bids_preproc pipeline.
% ResultPaths = bids_pipeline(RootPath, Subjects, Runs, ToStage)
%
% The sole purpose of this function is to serve as the MATLAB side of the 
% test_bids_preproc() unit test. This is not intended to be a fully
% configurable preprocessing pipeline; rather, defaults are hardcoded to 
% match what the unit test runs.
% 
% In:
%   RootPath : The directory of a BIDS study, e.g., from OpenNeuro.
%
%   Subjects : A cell array of subject ids to process, with the sub- prefix, 
%              e.g. {'sub-001', 'sub-002'}.
%
%   Runs : A cell array of run IDs to process; excluding the run- prefix.
%          e.g., {'1','2'}. Note that if there are more than 10 runs, a run
%          like '1' will also retain all 1x runs, so this is imperfect.
%
%   ToStage : Run up to and including the processing stage with the
%             specified number, where 1=import, 2=channel selection,
%             3=resample, and so forth. Please confirm in the code what
%             the current maximum stage is. 
% Out:
%   ResultPaths : a cell array of .set file paths where results have been
%                 stored. This is in lieu of returning the processed data
%                 to conserve memory.
%


if nargin < 1
    rootpath = '/home/christian/data/OpenNeuro/ds003061-download'; end
if nargin < 2
    % subjs = {'sub-002'};   % for testing
    subjs = {'sub-001','sub-002'};
end
if nargin < 3
    runs = {'1'}; end
if nargin < 4
    to_stage = 100; end

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
    if to_stage >= 2
        EEG = pop_select(EEG, 'channel', keep); end

    orig_chanlocs = EEG.chanlocs;

    % resampling
    if to_stage >= 3
        EEG = pop_resample(EEG, 128); end

    % artifact removal
    if to_stage >= 4
        EEG = clean_artifacts( ...
            EEG , ...
            'flatline_crit', quickif(to_stage >= 4, 5, 'off'), ...
            'highpass_band', quickif(to_stage >= 5, [0.25 0.75], 'off'), ...
            'chancorr_crit', quickif(to_stage >= 6, 0.8, 'off'), ...
            'line_crit', quickif(to_stage >= 6, 4, 'off'), ...
            'burst_crit', quickif(to_stage >= 7, 5, 'off'), ...
            'window_crit', quickif(to_stage >= 8, 0.25, 'off') ...
            );
    end

    % PICARD
    if to_stage >= 9
        EEG = eeg_picard(EEG); end  % EEG = pop_runica(EEG, 'icatype', 'picard');    

    % ICLabel
    if to_stage >= 10
        EEG = pop_iclabel(EEG, 'Default'); end

    % reinterpolate channels
    if to_stage >= 11
        EEG = eeg_interp(EEG, orig_chanlocs); end

    % epoching
    if to_stage >= 12    
        EEG = pop_epoch(EEG, {}, [-0.2, 0.5]); end

    % baseline removal
    if to_stage >= 13    
        EEG = pop_rmbase(EEG, [-200, 0]); end

    % common average reference
    if to_stage >= 14    
        EEG = pop_reref(EEG, []); end

    % write back as .set
    tmp_path = [tempname(), '.set'];
    pop_saveset(EEG, 'filename', tmp_path);
    result_paths{end+1} = tmp_path;
    
end

return


function res = quickif(cond,a,b)
if cond
    res = a;
else
    res = b;
end
