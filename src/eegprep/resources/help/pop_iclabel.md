POP_ICLABEL - Classify independent components with ICLabel.

Usage:

    EEG = pop_iclabel(EEG)
    EEG = pop_iclabel(EEG, 'default')
    EEG, command = pop_iclabel(EEG, 'lite', return_com=True)

Inputs:

- `EEG`: EEGPrep/EEGLAB-style dataset with an ICA decomposition.
- `icversion`: one of `'default'`, `'lite'`, or `'beta'`.

Graphical interface:

Calling `pop_iclabel(EEG)` opens a compact dialog with an ICLabel version
selector. Choose the desired model and press OK.

Behavior:

- ICA weights must already be present. Run `pop_runica` or another supported ICA wrapper first.
- Results are stored in `EEG.etc.ic_classification.ICLabel`.
- The result includes the ICLabel class names, per-component class probabilities, and the selected version string.
- Lists of datasets are processed one dataset at a time with the same selected version.

Example:

    EEG = pop_runica(EEG)
    EEG = pop_iclabel(EEG, 'default')

Notes:

- ICLabel class probabilities are ordered as Brain, Muscle, Eye, Heart, Line Noise, Channel Noise, and Other.
- This wrapper uses EEGPrep's packaged ICLabel implementation and does not require an EEGLAB checkout at runtime.
