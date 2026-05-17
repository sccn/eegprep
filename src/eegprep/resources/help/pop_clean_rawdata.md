POP_CLEAN_RAWDATA - Launch the clean_rawdata artifact-cleaning workflow.

Usage:

    EEG = pop_clean_rawdata(EEG)
    EEG = pop_clean_rawdata(EEG, 'FlatlineCriterion', 5, 'ChannelCriterion', 0.8, ...)
    EEG, command = pop_clean_rawdata(EEG, return_com=True)

Inputs:

- `EEG`: continuous EEGPrep/EEGLAB-style dataset. Epoched data is rejected.
- `Highpass`: `'off'` or `[low high]` transition band in Hz for drift removal.
- `FlatlineCriterion`: maximum tolerated flatline duration in seconds, or `'off'`.
- `ChannelCriterion`: minimum acceptable channel correlation with nearby channels, or `'off'`.
- `LineNoiseCriterion`: maximum acceptable high-frequency noise standard deviation, or `'off'`.
- `BurstCriterion`: ASR burst threshold in standard deviations, or `'off'`.
- `BurstRejection`: `True`/`False`; reject burst periods instead of correcting them.
- `WindowCriterion`: maximum fraction of bad channels tolerated in a time window, or `'off'`.
- `WindowCriterionTolerances`: accepted channel RMS range for bad-window detection.
- `Channels`: optional channel labels or indices to include.
- `Channels_ignore`: optional channel labels or indices to ignore.
- `Distance`: `'Euclidian'` or `'Riemannian'`.

Graphical interface:

Calling `pop_clean_rawdata(EEG)` opens the EEGPrep clean_rawdata dialog. The
dialog follows the EEGLAB plugin layout: drift filtering, channel processing,
ASR correction/rejection, bad-window removal, and an optional rejected-data
viewer checkbox.

Behavior:

- The function calls EEGPrep's `clean_artifacts` backend and returns a cleaned EEG dataset.
- GUI choices are converted to the same named options used by the command-line API.
- The "Pop up scrolling data window with rejected data highlighted" option is present for parity, but the viewer is not yet ported. When selected, EEGPrep shows a user-facing notification instead of silently doing nothing.

Example:

    EEG = pop_clean_rawdata(
        EEG,
        'Highpass', [0.25, 0.75],
        'FlatlineCriterion', 5,
        'ChannelCriterion', 0.8,
        'LineNoiseCriterion', 4,
        'BurstCriterion', 20,
        'WindowCriterion', 0.25,
    )

Notes:

- For command-line work, prefer explicit options so history clearly records the cleaning policy.
- This wrapper is intended to feel familiar to EEGLAB clean_rawdata users while using EEGPrep's Python implementation at runtime.
