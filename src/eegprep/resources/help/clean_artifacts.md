CLEAN_ARTIFACTS - Remove common artifacts from continuous EEG data.

Usage:
  EEG = pop_clean_rawdata(EEG)
  EEG, HP, BUR, removed_channels = clean_artifacts(EEG, ...)

The clean_rawdata workflow can remove flatline channels, high-pass drift,
noisy channels, ASR bursts, and bad windows. The pop_clean_rawdata GUI exposes
the same major options as the EEGLAB clean_rawdata plugin.

`Distance` accepts `Euclidian`, `Euclidean`, or `Riemannian`. `Riemannian` is
supported as a calibration-time ASR option. EEGPrep does not implement full
Riemannian ASR processing; use `clean_asr(..., useriemannian='calib')` for the
supported standalone variant. Other distance names raise `ValueError` so typos
do not silently change the ASR path.

`MaxMem` (MB) is the memory budget for ASR block processing, and `availableRAM_GB` sets it from the
RAM you can spare. The default `None` uses 64 MB, or twice the fixed lookahead memory when that is
larger (128 channels at 1000 Hz need 187.5 MB). The block layout can change the output, so EEGPrep
logs a warning when the default budget is raised, when the 10000-block limit makes memory use exceed
the budget, and when blocks are shorter than the lookahead. A larger `MaxMem` gives larger blocks. A
budget below the fixed lookahead memory raises `RuntimeError` with the smallest workable value.
