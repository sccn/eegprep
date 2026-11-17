# pop_fileio_brainvision_mat

Import a BrainVision Analyzer MATLAB `.mat` export.

The file must contain one numeric variable per channel plus `Channels`,
`SampleRate`, `t` (milliseconds), `SegmentCount`, and `Markers`. A segmented
channel variable is a segments-by-time matrix. A marker with an empty
`Description` is not imported. A channel variable such as `xFp1` is matched
to channel `Fp1`. MATLAB v7.3 (HDF5) files, and any other `.mat` layout, are
refused. Import the `.vhdr` with `pop_loadbv`, or re-export a MATLAB v7
Analyzer file.

The imported dataset does not keep the `.mat` path as its filename. Save it
with File > Save current dataset as.

This reader does not use the `mne` extra and does not require an EEGLAB checkout.
