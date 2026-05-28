# pop_mergeset

Merge two or more EEG datasets.

Datasets must have matching channel counts and sampling rates. Epoched datasets
must also have the same number of points per epoch. Continuous merges concatenate
samples and insert a boundary event between datasets; event latencies in later
datasets are offset to their new positions.

ICA fields are cleared by default unless `keepall` is enabled and the merge can
preserve the first dataset's decomposition.
