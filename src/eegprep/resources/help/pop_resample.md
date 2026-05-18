POP_RESAMPLE - Change the sampling rate of an EEG dataset.

Usage:

    EEG = pop_resample(EEG, freq)
    EEG, command = pop_resample(EEG, freq, return_com=True)

Inputs:

- `EEG`: EEGPrep/EEGLAB-style dataset or list of datasets.
- `freq`: new sampling rate in Hz.
- `fc`: optional anti-aliasing filter cutoff as a fraction of pi radians/sample. Default is `0.9`.
- `df`: optional anti-aliasing transition width as a fraction of pi radians/sample. Default is `0.2`.
- `engine`: optional backend. Use `None` or `"poly"` for the EEGPrep FIR/polyphase implementation, `"scipy"` for SciPy Fourier resampling, or `"matlab"`/`"octave"` to call an external EEGLAB runtime when configured.

Graphical interface:

Calling `pop_resample(EEG)` opens a dialog with a single "New sampling rate"
field. Enter the target rate in Hz and press OK.

Behavior:

- Continuous datasets are resampled in segments split by boundary events, matching EEGLAB's handling of data breaks.
- Epoched datasets are resampled along the time axis of each epoch. As in EEGLAB, resampling epoched data is not generally recommended because anti-aliasing filters can cross epoch boundaries.
- `EEG.srate`, `EEG.pnts`, `EEG.xmax`, and `EEG.times` are updated.
- `EEG.event` latencies and durations are scaled to the new sample grid.
- Continuous `EEG.urevent` latencies are updated. For epoched data, `EEG.urevent` is cleared to match EEGLAB's unsupported urevent remapping path.
- `EEG.icaact` is cleared because ICA activations are tied to the old sampling grid and can be recomputed from the retained ICA weights when needed.

Example:

    EEG = pop_resample(EEG, 250)

Notes:

- Boundary events are recognized using EEGLAB-style `"boundary"` event types and the `option_boundary99` setting for numeric `-99` boundary markers.
- The command history string uses EEGLAB-style syntax:

    EEG = pop_resample( EEG, 250);
