POP_RUNICA - Run ICA decomposition on an EEG dataset.

Usage:

    EEG = pop_runica(EEG)
    EEG = pop_runica(EEG, 'icatype', 'runica', ...)
    EEG, command = pop_runica(EEG, 'icatype', 'picard', return_com=True)

Inputs:

- `EEG`: EEGPrep/EEGLAB-style dataset or list of datasets.
- `'icatype'`: ICA backend. EEGPrep supports:
  - `'runica'`: EEGLAB-style Infomax / Extended Infomax.
  - `'picard'`: Picard ICA through EEGPrep's `eeg_picard` wrapper.
  - `'runamica15'` or `'amica'`: AMICA through EEGPrep's `eeg_amica` wrapper. AMICA requires an available AMICA binary.
- `'options'`: key/value options passed to the selected backend.
- `'chanind'`: one-based channel indices, channel labels, or channel types used for ICA.
- `'reorder'`: `'on'` or `'off'`; reorder components by descending activation variance. Default is `'on'`.
- `'dataset'`: one-based dataset indices to process when `EEG` is a list of datasets. Default is all datasets.
- `'concatenate'`: for a list of datasets, `'on'` runs one ICA on concatenated data and copies the decomposition back to each dataset. Default is `'off'`.
- `'concatcond'`: for a list of datasets, `'on'` concatenates datasets that share the same subject and session. Datasets without subject/session metadata are grouped together.

Graphical interface:

Calling `pop_runica(EEG)` opens an EEGLAB-style dialog with:

- An ICA algorithm list.
- A command-line options field.
- A component-reordering checkbox.
- Channel type/index selection controls.
- For multiple datasets, a dataset selector and concatenate controls.

When `pop_runica` is started from the main EEGPrep GUI, the ICA computation
runs behind an indeterminate progress dialog so the window can continue
repainting while the decomposition is being computed.

Behavior:

- Supplying a non-default `icatype` programmatically, for example
  `pop_runica(EEG, icatype='picard')`, runs the selected backend directly
  instead of opening the GUI. Unsupported standalone backends therefore fail
  clearly from the command path instead of opening a dialog first.
- Existing ICA decompositions are saved in `EEG.etc.oldicaweights`,
  `EEG.etc.oldicasphere`, and `EEG.etc.oldicachansind` before being replaced.
- Existing ICLabel classifications are removed when ICA is recomputed because they no longer describe the active components.
- `EEG.icaweights`, `EEG.icasphere`, `EEG.icawinv`, `EEG.icaact`, and `EEG.icachansind` are updated.
- GUI-launched runica adds `'interrupt', 'on'` to the history command, matching EEGLAB's GUI path.
- GUI-launched ICA stores the updated dataset only after the background
  computation finishes successfully. Failed runs leave the current dataset and
  history unchanged.

Examples:

    EEG = pop_runica(EEG)
    EEG = pop_runica(EEG, 'icatype', 'runica', 'extended', 1, 'maxsteps', 512)
    EEG = pop_runica(EEG, 'icatype', 'picard', 'options', {'maxiter', 500, 'mode', 'standard'})
    ALLEEG = pop_runica(ALLEEG, 'concatenate', 'on')

Notes:

- Programmatic channel indices follow EEGLAB user-facing convention and are one-based. Internally, EEGPrep stores `icachansind` as zero-based Python indices.
- AMICA is available only when the AMICA executable can be found through the `amica_binary` argument, `AMICA_BINARY`, a development checkout, or `PATH`.
- EEGLAB algorithms that do not yet have EEGPrep standalone backends, such as
  JADER, SOBI, and FastICA, raise a clear `NotImplementedError`. EEGPrep does
  not shell out to MATLAB toolboxes or silently substitute a different ICA
  algorithm.
