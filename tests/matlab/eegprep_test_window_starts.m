function starts = eegprep_test_window_starts(window_samples, overlap, sample_counts)
% 1-based round(1:window_samples*(1-overlap):S-window_samples) for each length.
step = window_samples * (1 - overlap);
sample_counts = double(sample_counts(:));
count = numel(sample_counts);
rows = cell(count, 1);
width = 0;
for index = 1:count
    rows{index} = round(1:step:(sample_counts(index) - window_samples));
    width = max(width, numel(rows{index}));
end
starts = nan(count, max(width, 1));
for index = 1:count
    starts(index, 1:numel(rows{index})) = rows{index};
end
end
