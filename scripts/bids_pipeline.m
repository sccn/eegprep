function result_paths = bids_pipeline(rootpath, subjs, runs, to_stage, varargin)
% Basic MATLAB-based equivalent of the bids_preproc pipeline.
% ResultPaths = bids_pipeline(RootPath, Subjects, Runs, ToStage, 'key', val, ...)
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
%
%   Optional key-value pairs:
%     'SaveIntermediateStages' : If true, save after each stage (default: false)
%     'IntermediateDir' : Directory for intermediate files (default: tempdir)
%     'ResumeFromStage' : Stage number to resume from (default: 0)
%
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
    runs = {'2'}; end
if nargin < 4
    to_stage = 100; end

% Parse optional arguments
p = inputParser;
addParameter(p, 'SaveIntermediateStages', false);
addParameter(p, 'IntermediateDir', tempdir);
addParameter(p, 'ResumeFromStage', 0);
parse(p, varargin{:});
save_intermediate = p.Results.SaveIntermediateStages;
intermediate_dir = p.Results.IntermediateDir;
resume_from_stage = p.Results.ResumeFromStage;

stage_names = {'import', 'chansel', 'resample', 'flatline', 'highpass', 'chancorr', 'burst', 'window', 'ica', 'iclabel', 'interp', 'epoch', 'baseline', 'reref'};

% import BIDS
if resume_from_stage == 0
    [STUDY, ALLEEG] = pop_importbids(...
        rootpath, ...
        'subjects', subjs, ...
        'runs', runs);
end

result_paths = {};
n_datasets = resume_from_stage > 0 && save_intermediate;  % If resuming, assume single dataset for simplicity
if n_datasets == 0
    n_datasets = length(ALLEEG);
end

for idx=1:n_datasets
    if resume_from_stage > 0
        % Load from saved stage (assume single dataset when resuming)
        fname = sprintf('stage%02d_%s_mat.set', resume_from_stage, stage_names{resume_from_stage});
        fpath = fullfile(intermediate_dir, fname);
        EEG = pop_loadset('filename', fpath);
        fprintf('Resumed from stage %d: %s\n', resume_from_stage, fpath);
    else
        EEG = ALLEEG(idx);
        EEG = eeg_checkset(EEG, 'loaddata');
        % Extract base name for unique identifiers
        [~, base_name, ~] = fileparts(EEG.filename);
        if save_intermediate
            fname = sprintf('%s_stage%02d_%s_mat.set', base_name, 1, stage_names{1});
            pop_saveset(EEG, 'filename', fname, 'filepath', intermediate_dir);
        end
    end

    % temporarily disabled for quicker runs
    % keep only channels with EEG modality
    chn_modalities = {EEG.chanlocs.type};
    keep = find(strcmp('EEG',chn_modalities));
    if to_stage >= 2 && resume_from_stage < 2
        EEG = pop_select(EEG, 'channel', keep);
        if save_intermediate, pop_saveset(EEG, 'filename', sprintf('%s_stage%02d_%s_mat.set', base_name, 2, stage_names{2}), 'filepath', intermediate_dir); end
    end

    orig_chanlocs = EEG.chanlocs;

    % resampling
    if to_stage >= 3 && resume_from_stage < 3
        EEG = pop_resample(EEG, 128);
        if save_intermediate, pop_saveset(EEG, 'filename', sprintf('%s_stage%02d_%s_mat.set', base_name, 3, stage_names{3}), 'filepath', intermediate_dir); end
    end

    % artifact removal
    if to_stage >= 4 && resume_from_stage < 4
        EEG = clean_artifacts( ...
            EEG , ...
            'flatline_crit', quickif(to_stage >= 4, 5, 'off'), ...
            'highpass_band', quickif(to_stage >= 5, [0.25 0.75], 'off'), ...
            'chancorr_crit', quickif(to_stage >= 6, 0.8, 'off'), ...
            'line_crit', quickif(to_stage >= 6, 4, 'off'), ...
            'burst_crit', quickif(to_stage >= 7, 5, 'off'), ...
            'window_crit', quickif(to_stage >= 8, 0.25, 'off') ...
            );
        if save_intermediate, pop_saveset(EEG, 'filename', sprintf('%s_stage%02d_%s_mat.set', base_name, 8, stage_names{8}), 'filepath', intermediate_dir); end
    end

    % ICA with runica
    if to_stage >= 9 && resume_from_stage < 9
        EEG = pop_runica(EEG, 'icatype', 'runica', 'rndreset', 'no');

        % sort components by mean descending activation variance
        [~, windex] = sort(sum(EEG.icawinv.^2).*sum((EEG.icaact').^2), 'descend');
        EEG.icaact = EEG.icaact(windex, :, :);
        EEG.icaweights = EEG.icaweights(windex, :);
        EEG.icawinv = EEG.icawinv(:, windex);

        % normalize components using the same rule as runica()
        [~, ix] = max(abs(EEG.icaact'));
        had_flips = 0;
        ncomps = size(EEG.icaact,1);
        for r=1:ncomps
            if sign(EEG.icaact(r,ix(r))) < 0
                EEG.icaact(r,:) = -EEG.icaact(r,:);
                EEG.icawinv(:,r) = -EEG.icawinv(:,r);
                had_flips = 1;
            end
        end
        if had_flips == 1
            EEG.icaweights = pinv(EEG.icawinv); end
        if save_intermediate, pop_saveset(EEG, 'filename', sprintf('%s_stage%02d_%s_mat.set', base_name, 9, stage_names{9}), 'filepath', intermediate_dir); end
    end

    % ICLabel
    if to_stage >= 10 && resume_from_stage < 10
        EEG = pop_iclabel(EEG, 'Default');
        if save_intermediate, pop_saveset(EEG, 'filename', sprintf('%s_stage%02d_%s_mat.set', base_name, 10, stage_names{10}), 'filepath', intermediate_dir); end
    end

    % reinterpolate channels
    if to_stage >= 11 && resume_from_stage < 11
        EEG = eeg_interp(EEG, orig_chanlocs);
        if save_intermediate, pop_saveset(EEG, 'filename', sprintf('%s_stage%02d_%s_mat.set', base_name, 11, stage_names{11}), 'filepath', intermediate_dir); end
    end

    % epoching
    if to_stage >= 12 && resume_from_stage < 12
        EEG = pop_epoch(EEG, {}, [-0.2, 0.5]);
        if save_intermediate, pop_saveset(EEG, 'filename', sprintf('%s_stage%02d_%s_mat.set', base_name, 12, stage_names{12}), 'filepath', intermediate_dir); end
    end

    % baseline removal
    if to_stage >= 13 && resume_from_stage < 13
        EEG = pop_rmbase(EEG, [-200, 0]);
        if save_intermediate, pop_saveset(EEG, 'filename', sprintf('%s_stage%02d_%s_mat.set', base_name, 13, stage_names{13}), 'filepath', intermediate_dir); end
    end

    % common average reference
    if to_stage >= 14 && resume_from_stage < 14
        EEG = pop_reref(EEG, []);
        if save_intermediate, pop_saveset(EEG, 'filename', sprintf('%s_stage%02d_%s_mat.set', base_name, 14, stage_names{14}), 'filepath', intermediate_dir); end
    end

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
