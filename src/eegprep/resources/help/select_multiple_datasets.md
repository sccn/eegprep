# Select Multiple Datasets

Select more than one dataset from `ALLEEG`.

The selection order is preserved. Internally `EEGPrepSession.CURRENTSET` stores
EEGLAB-facing 1-based dataset indices; the console exposes a scalar for a single
dataset and a list for multiple datasets.
