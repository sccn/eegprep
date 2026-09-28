function tests = test_eegprep_signal_expanded
% New native scientific contracts; Python translations must preserve these cases.
tests = functiontests(localfunctions);
end

function setupOnce(testCase)
eeglab_options;
testCase.TestData.option_single = option_single;
pop_editoptions('option_single', 0);
end

function teardownOnce(testCase)
pop_editoptions('option_single', testCase.TestData.option_single);
end

function test_fir_symmetric_dc_edges(testCase)
data = [2 4; 4 8; 8 16; 16 32];
expected = [2.5 5; 4.5 9; 9 18; 14 28];
actual = fir_filterdcpadded([0.25 0.5 0.25], 1, data);
verifyEqual(testCase, actual, expected, 'AbsTol', 1e-12);
verifySize(testCase, actual, size(data));
end

function test_fir_causal_asymmetric_dc_padding(testCase)
data = [2 -2; 4 -4; 8 -8; 16 -16];
expected = [2; 14/6; 22/6; 44/6] * [1 -1];
actual = fir_filterdcpadded([1 2 3]/6, 1, data, true, false);
verifyEqual(testCase, actual, expected, 'AbsTol', 1e-12);
end

function test_fir_antisymmetric_linear_ramp(testCase)
data = (-2:2)';
actual = fir_filterdcpadded([-0.5 0 0.5], 1, data, false, false);
verifyEqual(testCase, actual, [-0.5; -1; -1; -1; -0.5], 'AbsTol', 1e-12);
end

function test_fir_fft_single_precision(testCase)
data = single(reshape(1:18, 6, 3));
expected = data;
expected(1,:) = expected(1,:) + 0.25;
expected(end,:) = expected(end,:) - 0.25;
actual = fir_filterdcpadded([0.25 0.5 0.25], 1, data, false, true);
verifyClass(testCase, actual, 'single');
verifyEqual(testCase, actual, expected, 'AbsTol', single(1e-5));
end

function test_fir_fft_complex_signal(testCase)
data = (1:6)' * [1+2i -3+1i];
expected = [1.25; 2; 3; 4; 5; 5.75] * [1+2i -3+1i];
actual = fir_filterdcpadded([0.25 0.5 0.25], 1, data, false, true);
verifyEqual(testCase, actual, expected, 'AbsTol', 1e-12);
end

function test_fir_boundaries_do_not_mix_segments(testCase)
data = [ones(1,9) 100*ones(1,9); 2*ones(1,9) -50*ones(1,9); 1:18];
EEG = signal_eeg(data, 100);
EEG.event = struct('type', 'boundary', 'latency', 9.5, 'duration', 20);
actual = firfilt(EEG, [0.25 0.5 0.25], 7, [1 2]);
verifyEqual(testCase, actual.data, data, 'AbsTol', 1e-12);
verifyEqual(testCase, actual.event, EEG.event);
end

function test_reref_multiple_references_excludes_auxiliary_channel(testCase)
data = reshape(1:24, 4, 3, 2);
expected = data;
expected([1 2 4],:,:) = data([1 2 4],:,:) - mean(data([2 4],:,:), 1);
actual = reref(data, [2 4], 'exclude', 3, 'keepref', 'on');
verifyEqual(testCase, actual, expected, 'AbsTol', 1e-12);
verifyEqual(testCase, actual(3,:,:), data(3,:,:));
verifyEqual(testCase, mean(actual([2 4],:,:), 1), zeros(1,3,2), 'AbsTol', 1e-12);
end

function test_reref_removes_reference_and_updates_locations(testCase)
data = [1 4 7; 2 5 8; 3 6 9];
locs = struct('labels', {'A', 'B', 'C'});
[actual, locations, removed] = reref(data, 2, 'elocs', locs);
verifyEqual(testCase, actual, [-1 -1 -1; 1 1 1]);
verifyEqual(testCase, {locations.labels}, {'A', 'C'});
verifyEqual(testCase, {locations.ref}, {'B', 'B'});
verifyEqual(testCase, removed.labels, 'B');
end

function test_reref_reconstructs_original_reference(testCase)
data = [2 4; 6 8];
locs = struct('labels', {'A', 'B'}, 'theta', {0, 90}, 'radius', {0.5, 0.5});
[actual, locations] = reref(data, [], 'elocs', locs, 'refloc', {'REF', 0, 0});
expected = [-2/3 0; 10/3 4; -8/3 -4];
verifyEqual(testCase, actual, expected, 'AbsTol', 1e-12);
verifyEqual(testCase, sum(actual, 1), [0 0], 'AbsTol', 1e-12);
verifyEqual(testCase, {locations.labels}, {'A', 'B', 'REF'});
verifyEqual(testCase, {locations.ref}, {'average', 'average', 'average'});
end

function test_huber_reference_resists_offset_outlier(testCase)
data = [0; 0; 0; 1000] + [-2 0 3];
actual = reref(data, [], 'huber', 10);
% Three equal inliers balance the clipped outlier at the analytic mean 10/3.
expected = repmat([-10/3; -10/3; -10/3; 1000-10/3], 1, 3);
% The native fixed-point estimator terminates when its mean changes <1e-6.
verifyEqual(testCase, actual, expected, 'AbsTol', 1e-5);
verifyEqual(testCase, actual(4,:)-actual(1,:), [1000 1000 1000], 'AbsTol', 1e-12);
end

function test_huber_reference_without_outlier_is_average(testCase)
data = [-1; 0; 1] + [-2 0 3];
actual = reref(data, [], 'huber', 10);
verifyEqual(testCase, actual, repmat([-1; 0; 1], 1, 3), 'AbsTol', 1e-12);
end

function test_interpolation_kang_preserves_constant_scalp_field(testCase)
EEG = interpolation_eeg;
actual = eeg_interp(EEG, 3, 'sphericalKang');
verify_interpolation(testCase, actual, EEG);
end

function test_interpolation_explicit_spline_parameters(testCase)
EEG = interpolation_eeg;
actual = eeg_interp(EEG, 3, 'spherical', [], [1e-5 4 50]);
verify_interpolation(testCase, actual, EEG);
end

function test_resampling_discontinuities_preserves_dc_and_event_time(testCase)
EEG = signal_eeg([ones(1,3) 100*ones(1,5); -2*ones(1,3) -200*ones(1,5)], 100);
EEG.event = struct('type', {'start','boundary','stim'}, 'latency', {1,3.5,5}, ...
    'duration', {2,10,4}, 'urevent', {1,2,3});
EEG.urevent = rmfield(EEG.event, 'urevent');
actual = pop_resample(EEG, 50, 0.8, 0.4);
verifySize(testCase, actual.data, [2 5]);
verifyEqual(testCase, actual.data, [1 1 100 100 100; -2 -2 -200 -200 -200], 'AbsTol', 1e-10);
verifyEqual(testCase, [actual.event.latency], [1 2.5 3.5], 'AbsTol', 1e-12);
verifyEqual(testCase, [actual.urevent.latency], [1 2.5 3.5], 'AbsTol', 1e-12);
verifyEqual(testCase, [actual.event.duration], [1 5 2]);
verifyEqual(testCase, actual.times, [0 20 40 60 80], 'AbsTol', 1e-12);
verifyEqual(testCase, actual.srate, 50);
end

function test_resampling_epoched_events_and_ica_cache(testCase)
EEG = signal_eeg(reshape(1:40, 2, 10, 2), 100);
EEG.event = struct('type', {'stim','stim'}, 'latency', {3,15}, ...
    'duration', {2,4}, 'epoch', {1,2});
EEG.epoch = struct('event', {1,2});
EEG.urevent = rmfield(EEG.event, 'epoch');
EEG.icaact = ones(2,10,2);
actual = pop_resample(EEG, 50);
verifySize(testCase, actual.data, [2 5 2]);
verifyEqual(testCase, [actual.event.latency], [2 8]);
verifyEqual(testCase, [actual.event.duration], [1 2]);
verifyEmpty(testCase, actual.urevent);
verifyEmpty(testCase, actual.icaact);
verifyEqual(testCase, actual.trials, 2);
verifyEqual(testCase, actual.xmax, 0.08, 'AbsTol', 1e-12);
end

function test_asr_custom_calibration_reconstructs_burst(testCase)
[EEG, calibration, burst] = asr_eeg;
actual = clean_asr(EEG, 20, [], [], [], calibration);
verify_reconstruction(testCase, actual, EEG, burst);
end

function test_clean_artifacts_reconstruction_without_rejection(testCase)
[EEG, calibration, burst] = asr_eeg;
actual = clean_artifacts(EEG, 'FlatlineCriterion', 'off', 'Highpass', 'off', ...
    'ChannelCriterion', 'off', 'LineNoiseCriterion', 'off', 'WindowCriterion', 'off', ...
    'BurstCriterion', 20, 'BurstCriterionRefMaxBadChns', calibration, 'BurstRejection', 'off');
verify_reconstruction(testCase, actual, EEG, burst);
end

function EEG = signal_eeg(data, srate)
EEG = eeg_emptyset;
EEG.data = double(data);
EEG.nbchan = size(data,1);
EEG.pnts = size(data,2);
EEG.trials = size(data,3);
EEG.srate = srate;
EEG.xmin = 0;
EEG.xmax = (EEG.pnts-1)/srate;
EEG.times = (0:EEG.pnts-1)*1000/srate;
for index = 1:EEG.nbchan
    EEG.chanlocs(index).labels = sprintf('C%d', index);
end
end

function EEG = interpolation_eeg
wave = [-2 1 0 4 -1 3];
EEG = signal_eeg(repmat(wave, 6, 1), 100);
coordinates = [1 0 0; 0 1 0; 0 0 1; -1 0 0; 0 -1 0; 0 0 -1];
for index = 1:6
    EEG.chanlocs(index).X = coordinates(index,1);
    EEG.chanlocs(index).Y = coordinates(index,2);
    EEG.chanlocs(index).Z = coordinates(index,3);
    EEG.chanlocs(index).theta = (index-1)*60;
    EEG.chanlocs(index).radius = 0.5;
end
EEG.data(3,:) = 10000;
end

function verify_interpolation(testCase, actual, EEG)
verifyEqual(testCase, actual.data, repmat(EEG.data(1,:),6,1), 'AbsTol', 1e-10);
verifyEqual(testCase, actual.data([1 2 4 5 6],:), EEG.data([1 2 4 5 6],:));
verifyEqual(testCase, {actual.chanlocs.labels}, {EEG.chanlocs.labels});
verifyEqual(testCase, actual.nbchan, 6);
end

function [EEG, calibration, burst] = asr_eeg
% Stationary, zero-mean independent frequencies give a known clean reference;
% an arbitrary real EEG excerpt cannot establish which segments ASR must retain.
time = (0:4095)/128;
data = zeros(8,4096);
for channel = 1:8
    data(channel,:) = 10*sin(2*pi*(5+channel)*time) ...
        + 5*cos(2*pi*(23+channel)*time);
end
EEG = signal_eeg(data, 128);
EEG.event = struct('type', 'stim', 'latency', 2048, 'duration', 1);
calibration = EEG;
burst = 1537:1792;
EEG.data(1,burst) = EEG.data(1,burst) + 1000*sin(2*pi*10*(0:255)/EEG.srate);
end

function verify_reconstruction(testCase, actual, EEG, burst)
verifySize(testCase, actual.data, size(EEG.data));
verifyTrue(testCase, all(isfinite(actual.data(:))));
verifyGreaterThan(testCase, norm(actual.data(:)-EEG.data(:)), 1);
verifyLessThan(testCase, norm(actual.data(:,burst),'fro'), 0.5*norm(EEG.data(:,burst),'fro'));
% Reconstruction must retain the pre-burst signal, not erase the recording.
prefix = 1:1024;
verifyLessThan(testCase, norm(actual.data(:,prefix)-EEG.data(:,prefix),'fro'), ...
    0.01*norm(EEG.data(:,prefix),'fro'));
verifyEqual(testCase, actual.pnts, EEG.pnts);
verifyEqual(testCase, actual.nbchan, EEG.nbchan);
verifyEqual(testCase, actual.srate, EEG.srate);
verifyEqual(testCase, actual.event, EEG.event);
verifyEqual(testCase, {actual.chanlocs.labels}, {EEG.chanlocs.labels});
end
