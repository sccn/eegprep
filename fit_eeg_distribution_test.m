clear

max_bad_channels = 0.25;
zthresholds = [-inf 7];
window_len = 1;
window_overlap = 0.66;
max_dropout_fraction = 0.1;
min_clean_fraction = 0.25;
truncate_quant = [0.022 0.6];
step_sizes = [0.01 0.01];
shape_range = 1.7:0.15:3.5;

signal.data  = rand(1, 1000);
signal.srate = 100;

[C,S] = size(signal.data);
N = window_len*signal.srate;
wnd = 0:N-1;
offsets = round(1:N*(1-window_overlap):S-N);

X = signal.data(1,:).^2;
X = sqrt(sum(X(bsxfun(@plus,offsets,wnd')))/N);

[signal,sample_mask] = fit_eeg_distribution_simple(X, ...
    max_dropout_fraction, ...
    min_clean_fraction, ...
    truncate_quant, ...
    step_sizes, ...
    shape_range);
