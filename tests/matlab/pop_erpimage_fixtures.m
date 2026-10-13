function pop_erpimage_fixtures(eeglab_root, sample_set, output_file)
% Generate MATLAB reference outputs of pop_erpimage() for tests/test_pop_erpimage.py.
% Command-line calls on sample_data/eeglab_data_epochs_ica.set with 'erp' and
% 'NoShow','on' (no figure is drawn). The data are converted to double and the
% ICA activity is recomputed, as the Python test does, so the comparison is not
% limited by single precision. Only EEGLAB function folders are added; eeglab.m
% is never run and no option file is read or written. To keep the committed file
% small, the image is stored as row sums and three frames; sortvar is EEGLAB's
% eeg_getepochevent output, from which sortidx (into the trials kept) is mapped.
% Regenerate from a headless MATLAB session:
% pop_erpimage_fixtures('/path/to/eeglab', ...
%     '/path/to/sample_data/eeglab_data_epochs_ica.set', ...
%     '/path/to/tests/matlab/pop_erpimage_fixtures.mat');

for folder = {'sigprocfunc', 'timefreqfunc', 'adminfunc', 'miscfunc', 'guifunc', 'popfunc'}
    addpath(fullfile(eeglab_root, 'functions', folder{1}));
end
set(0, 'DefaultFigureVisible', 'off');

EEG = load('-mat', sample_set);
EEG = EEG.EEG;
fid = fopen(fullfile(fileparts(sample_set), EEG.data), 'r', 'ieee-le');
raw = fread(fid, [EEG.nbchan, EEG.pnts * EEG.trials], 'float32');
fclose(fid);
EEG.data = reshape(raw, EEG.nbchan, EEG.pnts, EEG.trials);
EEG.icaact = [];
EEG.filepath = fileparts(sample_set);
frames = [50 200 350];

% name, typeplot, channel, Python channel label, projchan, smooth, decimate, type, window, field, options
S = {
    {'channel_rt', 1, 14, '', [], 10, 1, {'rt'}, [], 'latency', {}}
    {'label_position_renorm', 1, 27, 'POz', [], 5, 0, {}, [], 'position', {'renorm', 'yes'}}
    {'channels_window', 1, [3 7], '', [], 3, 2, {'rt'}, [200 800], 'latency', {'nosort', 'on'}}
    {'component', 0, 6, '', [], 10, 1, {'rt'}, [], 'latency', {}}
    {'projected', 0, 6, '', 27, 10, 1, {'rt'}, [], 'latency', {'align', Inf}}
};

out = struct();
out.matlab_version = version;
for i = 1:numel(S)
    sc = S{i};
    name = sc{1};
    fprintf('%s\n', name);
    res = cell(1, 15);
    args = [{EEG, sc{2}, sc{3}, sc{5}, '', sc{6}, sc{7}, sc{8}, sc{9}, sc{10}}, sc{11}, {'erp', 'NoShow', 'on'}];
    evalc('[res{:}] = pop_erpimage(args{:});');
    close all;
    outdata = res{1};
    record = struct('typeplot', sc{2}, 'channel', sc{3}, 'channel_label', sc{4}, 'projchan', sc{5}, ...
        'smooth', sc{6}, 'decimate', sc{7}, 'sortingwin', sc{9}, 'field', sc{10});
    record.sortingtype = sc{8};
    record.options = sc{11};
    record.sortvar = eeg_getepochevent(EEG, sc{8}, sc{9}, sc{10});
    record.outvar = res{2};
    record.outtrials = res{3};
    record.sortidx = uint16(res{14});
    record.frames = frames;
    record.outdata_rows = sum(outdata, 1);
    record.outdata_frames = outdata(frames, :);
    record.erp_frames = res{6}(frames);
    record.erp_sum = sum(res{6});
    out.(name) = record;
end
save(output_file, '-struct', 'out', '-v7');
end
