function data = eegprep_test_detrend_trials(data)
% Arithmetic-only per-trial primitive from test_stdspecplot3/4.
% Python owns the window, FFT call, log powers, means and assertions.
for trial = 1:size(data, 3)
    data(:, :, trial) = detrend(data(:, :, trial)')';
end
end
