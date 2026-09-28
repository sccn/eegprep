function tests = test_eegprep_structures_expanded
% Added MATLAB-first contracts for pinned EEGLAB 8ac485f654d6bbb1a6acb8dc9ef3f2eaf3d409ba.
% Deterministic sample values expose channel, sample and trial permutations.
tests = functiontests(localfunctions);
end

function test_epochformat_numeric_fields_and_event_indices(testCase)
values = [11 12 13; 21 22 23];
[epochs, fields] = eeg_epochformat(values, 'struct', {'var1'}, [4 9]);
verifyEqual(testCase, fields, {'var1' 'var2' 'var3'});
verifyEqual(testCase, [epochs.var1], [11 21]);
verifyEqual(testCase, [epochs.var2], [12 22]);
verifyEqual(testCase, [epochs.var3], [13 23]);
verifyEqual(testCase, [epochs.event], [4 9]);
end

function test_epochformat_cells_preserve_labels_and_event_lists(testCase)
values = {'target' 125; 'standard' 250};
[epochs, fields] = eeg_epochformat(values, 'struct', {'condition' 'rt'}, {[2 3] [7 8 9]});
verifyEqual(testCase, fields, {'condition' 'rt'});
verifyEqual(testCase, {epochs.condition}, {'target' 'standard'});
verifyEqual(testCase, [epochs.rt], [125 250]);
verifyEqual(testCase, epochs(1).event, [2 3]);
verifyEqual(testCase, epochs(2).event, [7 8 9]);
end

function test_epochformat_array_selects_time_locking_events(testCase)
epochs = event_epochs();
[values, fields] = eeg_epochformat(epochs, 'array');
verifyEqual(testCase, fields, {'event'; 'eventlatency'; 'eventtype'; 'eventcode'});
verifyEqual(testCase, values, {2 0 'stim' 11; 3 0 'stim' 13});
end

function test_epochformat_selected_type_keeps_first_match_and_missing_trial(testCase)
epochs = event_epochs();
epochs(1).eventtype = {'response' 'response'};
[values, fields] = eeg_epochformat(epochs, 'array', {'response'});
verifyEqual(testCase, fields, {'event'; 'eventlatency'; 'eventtype'; 'eventcode'});
verifyEqual(testCase, values, {1 -250 'response' 7; NaN NaN NaN NaN});
end

function test_select_disjoint_samples_retains_fractional_events_and_urevents(testCase)
EEG = continuous_events();
out = pop_select(EEG, 'point', [1 2; 7 8]);
verifyEqual(testCase, out.data, EEG.data(:, [1 2 7 8]));
verifyEqual(testCase, out.pnts, 4);
verifyEqual(testCase, {out.event.type}, {'first' 'left' 'boundary' 'right' 'last'});
verifyEqual(testCase, [out.event.latency], [1 2.25 2.5 2.75 4]);
verifyEqual(testCase, out.event(3).duration, 4);
verifyEqual(testCase, [out.event([1 2 4 5]).urevent], [1 2 4 5]);
verifyEqual(testCase, out.urevent, EEG.urevent);
end

function test_select_crop_keeps_edge_discontinuities_and_exact_samples(testCase)
EEG = continuous_events();
out = pop_select(EEG, 'point', [2 7]);
verifyEqual(testCase, out.data, EEG.data(:, 2:7));
verifyEqual(testCase, [out.event.latency], [0.5 1.25 2 5.75 6.5]);
verifyEqual(testCase, {out.event.type}, {'boundary' 'left' 'middle' 'right' 'boundary'});
verifyEqual(testCase, [out.event([1 5]).duration], [1 1]);
verifyEqual(testCase, [out.event(2:4).urevent], [2 3 4]);
verifyEqual(testCase, out.urevent, EEG.urevent);
end

function test_select_channel_types_keeps_locations_and_removed_metadata(testCase)
EEG = sample_eeg(1);
out = pop_select(EEG, 'chantype', {'EEG' 'EOG'}, 'rmchantype', 'EOG');
verifyEqual(testCase, out.data, EEG.data([1 3], :));
verifyEqual(testCase, out.nbchan, 2);
verifyEqual(testCase, {out.chanlocs.labels}, {'Fz' 'Pz'});
verifyEqual(testCase, [out.chanlocs.X], [1 3]);
verifyEqual(testCase, {out.chaninfo.removedchans.labels}, {'VEOG' 'EMG'});
verifyEqual(testCase, [out.chaninfo.removedchans.X], [2 4]);
end

function test_select_channel_labels_use_original_order_and_exclusion(testCase)
EEG = sample_eeg(1);
out = pop_select(EEG, 'channel', {'EMG' 'Fz' 'Pz'}, 'rmchannel', {'Pz'});
verifyEqual(testCase, out.data, EEG.data([1 4], :));
verifyEqual(testCase, {out.chanlocs.labels}, {'Fz' 'EMG'});
verifyEqual(testCase, {out.chaninfo.removedchans.labels}, {'VEOG' 'Pz'});
end

function test_select_reordered_trials_reindexes_events_without_reordering_urevents(testCase)
EEG = epoched_events(3);
out = pop_select(EEG, 'trial', [3 1], 'sorttrial', 'off');
verifyEqual(testCase, out.data, EEG.data(:, :, [3 1]));
verifyEqual(testCase, out.trials, 2);
for trial = 1:2
    linked = out.event(out.epoch(trial).event);
    verifyEqual(testCase, [linked.epoch], [trial trial]);
    verifyEqual(testCase, [linked.latency], [3 5.5] + 8*(trial-1));
    original = [5 6; 1 2];
    verifyEqual(testCase, [linked.urevent], original(trial, :));
end
% Retain the native regression: eeg_checkset:580-591 requests ascending
% epoch/latency order, as documented by pop_editeventvals:11-13. Its second
% sort currently reapplies indices from the original rather than sorted events.
verifyEqual(testCase, [out.event.latency], [3 5.5 11 13.5]);
verifyEqual(testCase, [out.event.epoch], [1 1 2 2]);
verifyEqual(testCase, [out.event.urevent], [5 6 1 2]);
verifyEqual(testCase, out.epoch(1).event, [1 2]);
verifyEqual(testCase, out.epoch(2).event, [3 4]);
verifyEqual(testCase, cell2mat(out.epoch(1).eventlatency), [0 625]);
verifyEqual(testCase, out.urevent, EEG.urevent);
end

function test_select_duplicate_trials_keep_last_occurrences_in_input_order(testCase)
EEG = epoched_events(3);
out = pop_select(EEG, 'trial', [3 1 3], 'sorttrial', 'off');
% Pinned behavior: unique_bc uses legacy last occurrences, not unique(...,'stable').
verifyEqual(testCase, out.data, EEG.data(:, :, [1 3]));
verifyEqual(testCase, out.trials, 2);
verifyEqual(testCase, [out.event.latency], [3 5.5 11 13.5]);
verifyEqual(testCase, [out.event.epoch], [1 1 2 2]);
verifyEqual(testCase, [out.event.urevent], [1 2 5 6]);
verifyEqual(testCase, out.epoch(1).event, [1 2]);
verifyEqual(testCase, out.epoch(2).event, [3 4]);
verifyEqual(testCase, out.urevent, EEG.urevent);
end

function test_select_one_trial_keeps_sample_axis_and_urevent_origin(testCase)
EEG = epoched_events(3);
out = pop_select(EEG, 'rmtrial', [1 2]);
verifyEqual(testCase, out.data, EEG.data(:, :, 3));
verifyEqual(testCase, [out.nbchan out.pnts out.trials], [4 8 1]);
verifyEqual(testCase, [out.event.latency], [3 5.5]);
verifyEqual(testCase, [out.event.urevent], [5 6]);
verifyFalse(testCase, isfield(out.event, 'epoch'));
verifyEmpty(testCase, out.epoch);
verifyEqual(testCase, out.urevent, EEG.urevent);
end

function test_select_explicit_empty_trial_selection(testCase)
EEG = epoched_events(3);
out = pop_select(EEG, 'rmtrial', [1 2 3], 'erroronempty', 'off');
verifyEmpty(testCase, out.data);
verifyEmpty(testCase, out.event);
verifyEmpty(testCase, out.epoch);
verifyEqual(testCase, out.trials, 0);
end

function test_mergeset_eventless_second_recording_adds_half_sample_boundary(testCase)
first = continuous_events();
second = sample_eeg(1);
second.data = second.data + 100;
out = pop_mergeset(first, second, 0);
verifyEqual(testCase, out.data, [first.data second.data]);
verifyEqual(testCase, [out.nbchan out.pnts out.trials], [4 16 1]);
verifyEqual(testCase, {out.event.type}, {'first' 'left' 'middle' 'right' 'last' 'boundary'});
verifyEqual(testCase, [out.event.latency], [1 2.25 3 6.75 8 8.5]);
verifyEqual(testCase, [out.event(1:5).urevent], 1:5);
verifyEqual(testCase, out.urevent(end).type, 'boundary');
verifyEqual(testCase, out.urevent(end).latency, 8.5);
verifyEqual(testCase, {out.chanlocs.labels}, {first.chanlocs.labels});
end

function test_mergeset_epoched_then_singleton_reconstructs_epoch_references(testCase)
first = epoched_events(2);
second = sample_eeg(1);
second.data = second.data + 100;
second.event = struct('type', 'response', 'latency', 2.5, 'urevent', 1);
second.urevent = rmfield(second.event, 'urevent');
second = rmfield(second, 'epoch');
out = pop_mergeset(first, second, 0);
verifyEqual(testCase, out.data, cat(3, first.data, second.data));
verifyEqual(testCase, [out.pnts out.trials out.xmin out.xmax], [8 3 -0.5 1.25]);
verifyEqual(testCase, [out.event.latency], [3 5.5 11 13.5 18.5]);
verifyEqual(testCase, [out.event.epoch], [1 1 2 2 3]);
verifyEqual(testCase, [out.event.urevent], [1 2 3 4 6]);
verifyEqual(testCase, out.epoch(3).event, 5);
verifyEqual(testCase, out.epoch(3).eventlatency, {-125});
end

function test_mergeset_singleton_then_epoched_offsets_all_events(testCase)
first = sample_eeg(1);
first.event = struct('type', 'start', 'latency', 1.5, 'urevent', 1);
first.urevent = rmfield(first.event, 'urevent');
second = epoched_events(2);
second.data = second.data + 100;
out = pop_mergeset(first, second, 0);
verifyEqual(testCase, out.data, cat(3, first.data, second.data));
verifyEqual(testCase, [out.pnts out.trials out.xmin out.xmax], [8 3 0 1.75]);
verifyEqual(testCase, [out.event.latency], [1.5 11 13.5 19 21.5]);
verifyEqual(testCase, [out.event.epoch], [1 2 2 3 3]);
verifyEqual(testCase, [out.event.urevent], [1 3 4 5 6]);
verifyEqual(testCase, out.epoch(1).eventlatency, {125});
end

function test_checkur_sort_preserves_event_to_original_event_identity(testCase)
EEG = sample_eeg(1);
EEG.event = struct('type', {'late' 'early'}, 'latency', {7 1}, 'urevent', {1 2});
EEG.urevent = rmfield(EEG.event, 'urevent');
out = eeg_checkset(EEG, 'checkur');
verifyEqual(testCase, {out.urevent.type}, {'early' 'late'});
verifyEqual(testCase, [out.urevent.latency], [1 7]);
verifyEqual(testCase, {out.event.type}, {'late' 'early'});
verifyEqual(testCase, [out.event.urevent], [2 1]);
out = eeg_checkset(out, 'eventconsistency');
verifyEqual(testCase, {out.event.type}, {'early' 'late'});
verifyEqual(testCase, [out.event.latency], [1 7]);
verifyEqual(testCase, [out.event.urevent], [1 2]);
verifyEqual(testCase, out.data, EEG.data);
end

function test_eventconsistency_normalizes_initial_boundary_and_removes_nan_event(testCase)
EEG = sample_eeg(1);
EEG.event = struct('type', {'boundary' 'first' 'invalid' 'fractional' 'last'}, ...
    'latency', {0.75 1 NaN 4.5 8}, 'duration', {2 0 0 0 0});
out = eeg_checkset(EEG, 'eventconsistency');
verifyEqual(testCase, {out.event.type}, {'boundary' 'first' 'fractional' 'last'});
verifyEqual(testCase, [out.event.latency], [0.5 1 4.5 8]);
verifyEqual(testCase, [out.event.duration], [2 0 0 0]);
verifyEqual(testCase, out.data, EEG.data);
end

function EEG = sample_eeg(trials)
EEG = eeg_emptyset;
EEG.data = reshape(single(1:4*8*trials), 4, 8, trials);
EEG.nbchan = 4;
EEG.pnts = 8;
EEG.trials = trials;
EEG.srate = 4;
EEG.xmin = 0;
EEG.xmax = 7/4;
EEG.times = (0:7)*250;
EEG.chanlocs = struct('labels', {'Fz' 'VEOG' 'Pz' 'EMG'}, ...
    'type', {'EEG' 'EOG' 'EEG' 'EMG'}, 'X', {1 2 3 4}, 'Y', {0 0 0 0}, 'Z', {0 0 0 0});
end

function EEG = continuous_events()
EEG = sample_eeg(1);
EEG.event = struct('type', {'first' 'left' 'middle' 'right' 'last'}, ...
    'latency', {1 2.25 3 6.75 8}, 'urevent', {1 2 3 4 5});
EEG.urevent = rmfield(EEG.event, 'urevent');
end

function EEG = epoched_events(trials)
EEG = sample_eeg(trials);
EEG.xmin = -0.5;
EEG.xmax = 1.25;
EEG.times = (-2:5)*250;
for trial = 1:trials
    for within = 1:2
        index = 2*(trial-1) + within;
        types = {'stim' 'response'};
        latencies = [3 5.5];
        EEG.event(index).type = types{within};
        EEG.event(index).latency = 8*(trial-1) + latencies(within);
        EEG.event(index).epoch = trial;
        EEG.event(index).urevent = index;
    end
end
EEG.urevent = rmfield(EEG.event, {'urevent' 'epoch'});
EEG = eeg_checkset(EEG, 'eventconsistency');
end

function epochs = event_epochs()
epochs = struct('event', {[1 2] [3 4]}, 'eventlatency', {{-250 0} {0 250}}, ...
    'eventtype', {{'cue' 'stim'} {'stim' 'other'}}, 'eventcode', {{7 11} {13 17}});
end
