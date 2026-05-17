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

Behavior:

- Existing ICA decompositions are saved in `EEG.etc.oldicaweights`,
  `EEG.etc.oldicasphere`, and `EEG.etc.oldicachansind` before being replaced.
- Existing ICLabel classifications are removed when ICA is recomputed because they no longer describe the active components.
- `EEG.icaweights`, `EEG.icasphere`, `EEG.icawinv`, `EEG.icaact`, and `EEG.icachansind` are updated.
- GUI-launched runica adds `'interrupt', 'on'` to the history command, matching EEGLAB's GUI path.

Examples:

    EEG = pop_runica(EEG)
    EEG = pop_runica(EEG, 'icatype', 'runica', 'extended', 1, 'maxsteps', 512)
    EEG = pop_runica(EEG, 'icatype', 'picard', 'options', {'maxiter', 500, 'mode', 'standard'})
    ALLEEG = pop_runica(ALLEEG, 'concatenate', 'on')

Notes:

- Programmatic channel indices follow EEGLAB user-facing convention and are one-based. Internally, EEGPrep stores `icachansind` as zero-based Python indices.
- AMICA is available only when the AMICA executable can be found through the `amica_binary` argument, `AMICA_BINARY`, a development checkout, or `PATH`.
- EEGLAB algorithms that do not yet have EEGPrep backends, such as JADER, SOBI, and FastICA, raise a clear `NotImplementedError`.
