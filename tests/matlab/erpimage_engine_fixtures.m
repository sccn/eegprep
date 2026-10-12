function erpimage_engine_fixtures(eeglab_root, sample_set, output_file)
% Generate MATLAB reference outputs of erpimage() for tests/test_erpimage_engine.py.
% EEGLAB 8ac485f654d6bbb1a6acb8dc9ef3f2eaf3d409ba (erpimage.m, movav.m and
% phasecoher.m are identical in the local checkout used to regenerate it).
% Recorded with MATLAB R2025b (Signal Processing Toolbox provides pwelch,
% butter and filtfilt). Only EEGLAB function folders are added; eeglab.m is
% never run. Regenerate from a headless MATLAB session:
% erpimage_engine_fixtures('/path/to/eeglab', ...
%     '/path/to/sample_data/eeglab_data_epochs_ica.set', ...
%     '/path/to/tests/matlab/erpimage_engine_fixtures.mat');

for folder = {'sigprocfunc', 'timefreqfunc', 'adminfunc', 'miscfunc', 'guifunc', 'popfunc'}
    addpath(fullfile(eeglab_root, 'functions', folder{1}));
end
set(0, 'DefaultFigureVisible', 'off');
seed = 5489;

% Integer-valued synthetic data: Park-Miller LCG noise plus a trial-shifted square wave.
frames = 44; trials = 24; srate = 128;
times = -125 + (0:frames-1) * 1000 / srate;
state = 20261012;
noise = zeros(frames, trials);
for k = 1:numel(noise)
    state = mod(16807 * state, 2147483647);
    noise(k) = mod(state, 201) - 100;
end
[tt, kk] = ndgrid(0:frames-1, 1:trials);
synth = noise + 40 * (2 * (mod(tt + 3 * kk, 13) < 6) - 1);
synth_nan = synth;
synth_nan([5 60 300]) = NaN;
rt = 20 + mod((1:trials) * 37, 23) * 9;
ties = mod((1:trials) * 7, 5) * 10;
ties([4 11 19]) = NaN;
events = [20 + mod((1:trials)' * 7, 30), 120 + mod((1:trials)' * 11, 40)];
auxvar = [rt - 10; rt + 15];

% Sample data slice: channel 27 (POz), samples 49:112, first 40 epochs.
EEG = load('-mat', sample_set);
EEG = EEG.EEG;
fid = fopen(fullfile(fileparts(sample_set), EEG.data), 'r', 'ieee-le');
raw = fread(fid, [EEG.nbchan, EEG.pnts * EEG.trials], 'float32');
fclose(fid);
raw = reshape(raw, EEG.nbchan, EEG.pnts, EEG.trials);
real_times = linspace(EEG.xmin * 1000, EEG.xmax * 1000, EEG.pnts);
real_pick = 49:112;
real_data = squeeze(raw(27, real_pick, 1:40));
real_sortvar = mod((1:40) * 37, 23) * 13 + 100;

datasets = struct('synth', synth, 'synth_nan', synth_nan, 'real_channel', 27, ...
    'real_samples', real_pick, 'real_epochs', 1:40);

S = {};
% name, data, sortvar, times ('omitted' for nargin<3), avewidth, decimate, options, modes
S{end+1} = {'times_omitted', synth, [], 'omitted', 1, 1, {}, {'plot'}};
S{end+1} = {'times_empty', synth, [], [], 1, 1, {'erp', 'on'}, {'noplot', 'plot'}};
S{end+1} = {'times_compact', synth, rt, [-125 frames srate], 3, 1, {'erp'}, {'noplot'}};
S{end+1} = {'vector_input', reshape(synth, 1, []), 1:trials, times, 2, 1, {}, {'noplot'}};
S{end+1} = {'pop_default', synth, ones(1, trials) * times(end), times, 5, 0, {'erp', 'on', 'cbar', 'on'}, {'noplot', 'plot'}};
S{end+1} = {'nan_ties', synth_nan, ties, times, 3, 1, {'erp', 'on', 'erpstd', 'on'}, {'noplot', 'plot'}};
S{end+1} = {'replace_ties', synth, ties, times, 3, 1, {'replace_ties', 'on', 'erp'}, {'noplot'}};
S{end+1} = {'nosort', synth, rt, times, 5, 2, {'nosort', 'on', 'erp'}, {'noplot'}};
S{end+1} = {'valsort_up', synth, rt, times, 3, 1, {'valsort', [0 100]}, {'noplot'}};
S{end+1} = {'valsort_down', synth, rt, times, 1, 1, {'valsort', [50 50 -1]}, {'noplot'}};
S{end+1} = {'renorm_yes', synth, rt, times, 3, 1, {'renorm', 'yes', 'erp'}, {'noplot'}};
S{end+1} = {'renorm_align', synth, rt, times, 3, 1, {'renorm', '0.5*x+20', 'align', Inf, 'erp'}, {'noplot'}};
S{end+1} = {'align_value', synth, rt, times, 1, 1, {'align', 150, 'erp', 'on'}, {'noplot'}};
S{end+1} = {'smooth_decimate', synth, rt, times, 7, 3, {'erp'}, {'noplot'}};
S{end+1} = {'decimate_fraction', synth, rt, times, 3, 2.5, {}, {'noplot'}};
S{end+1} = {'decimate_large', synth, rt, times, 4, 10, {}, {'noplot'}};
S{end+1} = {'decimate_negative', synth, rt, times, 4, -10, {}, {'noplot'}};
S{end+1} = {'boxcar_clamp', synth, rt, times, 40, 1, {}, {'noplot'}};
S{end+1} = {'gaussian', synth, rt, times, 2, 1, {'avg_type', 'Gaussian', 'erp'}, {'noplot'}};
S{end+1} = {'gaussian_clamp', synth, rt, times, 10, 2, {'avg_type', 'gaussian'}, {'noplot'}};
S{end+1} = {'phasesort_coher', synth, rt, times, 5, 1, {'phasesort', [100 25 10], 'coher', [10 10 0.01], 'erp'}, {'noplot', 'plot'}};
S{end+1} = {'phasesort_band', synth, rt, times, 2, 1, {'phasesort', [100 -20 10 14 90], 'erp'}, {'noplot'}};
S{end+1} = {'phasesort_cycles', synth, rt, times, 3, 1, {'phasesort', [50 0 12], 'cycles', 2}, {'noplot'}};
S{end+1} = {'ampsort', synth, rt, times, 3, 1, {'ampsort', [100 10 10]}, {'noplot'}};
S{end+1} = {'ampsort_band', synth, rt, times, 3, 1, {'ampsort', [100 -10 10 14]}, {'noplot'}};
S{end+1} = {'ampsort_allfreqs', synth, rt, times, 3, 1, {'ampsort', [100 0 -9 11]}, {'noplot'}};
S{end+1} = {'ampsort_sortwin', synth, rt, times, 3, 1, {'ampsort', [Inf 10 10], 'sortwin', [30 60]}, {'noplot'}};
S{end+1} = {'coher_band', synth, rt, times, 3, 1, {'coher', [10 14 0.05], 'baselinedb', [-100 0]}, {'noplot', 'plot'}};
S{end+1} = {'coher_nobase', synth, rt, times, 3, 1, {'coher', 10, 'baselinedb', NaN}, {'noplot', 'plot'}};
% 'signif' with 'coher' alone errors on R2025b (vector operand to ||), so it is recorded with 'plotamps'.
S{end+1} = {'plotamps_signif_basedb', synth, rt, times, 3, 1, {'plotamps', 'on', 'coher', [10 10 0.05], ...
    'signif', [0.5 2 0.4], 'limits', [NaN NaN NaN NaN NaN NaN NaN NaN 3]}, {'noplot'}};
S{end+1} = {'plotamps', synth, rt, times, 3, 1, {'plotamps', 'on', 'coher', [10 10 0.05]}, {'noplot', 'plot'}};
S{end+1} = {'plotamps_phase', synth, rt, times, 3, 1, {'phasesort', [100 0 10], 'plotamps', 'coher', [10 10 0.05]}, {'noplot'}};
S{end+1} = {'erp_ptiles', synth, rt, times, 1, 1, {'erp', 3}, {'noplot', 'plot'}};
S{end+1} = {'rmerp_baseline', synth, rt, times, 1, 1, {'rmerp', 'on', 'baseline', [-100 0], 'erp'}, {'noplot'}};
S{end+1} = {'erpalpha', synth, rt, times, 1, 1, {'erp', 'erpalpha', 0.05}, {'noplot', 'plot'}};
S{end+1} = {'sortvarpercent', synth, rt, times, 5, 1, {'sortvarpercent', [0.1 0.5 0.9]}, {'noplot'}};
S{end+1} = {'sortvar_limits', synth, rt, times, 3, 1, {'sortvar_limits', [60 160], 'erp'}, {'noplot'}};
S{end+1} = {'limits_all', synth, rt, times, 3, 1, {'limits', [0 200 -50 50 -3 3 0 1], 'erp', 'coher', 10}, {'noplot', 'plot'}};
S{end+1} = {'auxvar', synth, rt, times, 3, 1, {'auxvar', auxvar, 'erp'}, {'noplot'}};
S{end+1} = {'timewarp', synth, [], times, 3, 1, {'timewarp', {events}, 'erp'}, {'noplot'}};
S{end+1} = {'filt', synth, rt, times, 3, 1, {'filt', [0 30], 'erp'}, {'noplot'}};
S{end+1} = {'sample_slice', real_data, real_sortvar, real_times(real_pick), 'POz', 5, 3, {'erp', 'cbar'}, {'noplot'}};

out = struct();
stored = {};  % large arrays already saved, for deduplication: {label, value}
out.datasets = datasets;
out.seed = seed;
out.matlab_version = version;
for i = 1:numel(S)
    sc = S{i};
    name = sc{1};
    title = '';
    if ischar(sc{5})  % sample slice carries a title before avewidth
        title = sc{5};
        sc = [sc(1:4), sc(6:end)];
    end
    data = sc{2}; sortvar = sc{3}; times_arg = sc{4};
    avewidth = sc{5}; decimate = sc{6}; options = sc{7}; modes = sc{8};
    record = struct('name', name, 'sortvar', sortvar, 'times_omitted', ischar(times_arg), ...
        'avewidth', avewidth, 'decimate', decimate, 'title', title);
    record.options = options;
    if ischar(times_arg)
        record.times = [];
    else
        record.times = times_arg;
    end
    if isequal(data, synth), record.data_id = 'synth';
    elseif isequal(data, reshape(synth, 1, [])), record.data_id = 'synth_vector';
    elseif isequaln(data, synth_nan), record.data_id = 'synth_nan';
    else, record.data_id = 'real';
    end
    for m = 1:numel(modes)
        fprintf('%s %s\n', name, modes{m});
        res = cell(1, 15);
        rng(seed, 'twister');
        if strcmp(modes{m}, 'noplot')
            args = [{data, sortvar, times_arg, title, avewidth, decimate}, options, {'noplot', 'on'}];
            evalc('[res{:}] = erpimage(args{:});');
            prefix = '';
        else
            close all; figure;
            if ischar(times_arg)
                evalc('[res{:}] = erpimage(data, sortvar);');
            else
                args = [{data, sortvar, times_arg, title, avewidth, decimate}, options];
                evalc('[res{:}] = erpimage(args{:});');
            end
            close all;
            prefix = 'plot_';
        end
        names = {'outdata', 'outvar', 'outtrials', 'limits', 'axhndls', 'erp', 'amps', 'cohers', ...
            'cohsig', 'ampsig', 'outamps', 'phsangls', 'phsamp', 'sortidx', 'erpsig'};
        for k = [1:4, 6:15]
            if ~isempty(prefix) && ~ismember(names{k}, {'limits', 'erpsig'})
                continue
            end
            record.([prefix names{k}]) = res{k};
        end
    end
    if strcmp(name, 'erpalpha')
        % The permutation signs EEGLAB draws (biased sign(randn - 0.5), see
        % https://github.com/sccn/eeglab/issues/976), so the threshold arithmetic
        % can be compared apart from the draw itself.
        nperm = max(500, round(3 / 0.05));
        rng(seed, 'twister');
        draws = zeros(nperm, size(data, 2));
        for n = 1:nperm
            draws(n, :) = randn(1, size(data, 2));
        end
        record.matlab_signs = int8(sign(draws - 0.5));
    end
    [record, stored] = deduplicate(record, name, stored);
    out.(name) = record;
end
save(output_file, '-struct', 'out', '-v7');
end

function [record, stored] = deduplicate(record, name, stored)
% Replace large arrays already stored by another scenario with 'same:<scenario>.<field>'.
fields = fieldnames(record);
for f = 1:numel(fields)
    value = record.(fields{f});
    if ~isnumeric(value) || numel(value) < 64
        continue
    end
    for s = 1:size(stored, 1)
        if isequaln(stored{s, 2}, value)
            record.(fields{f}) = ['same:' stored{s, 1}];
            break
        end
    end
    if isnumeric(record.(fields{f}))
        stored(end + 1, :) = {[name '.' fields{f}], value}; %#ok<AGROW>
    end
end
end
