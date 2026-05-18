CLEAN_ARTIFACTS - Remove common artifacts from continuous EEG data.

Usage:
  EEG = pop_clean_rawdata(EEG)
  EEG, HP, BUR, removed_channels = clean_artifacts(EEG, ...)

The clean_rawdata workflow can remove flatline channels, high-pass drift,
noisy channels, ASR bursts, and bad windows. The pop_clean_rawdata GUI exposes
the same major options as the EEGLAB clean_rawdata plugin.
