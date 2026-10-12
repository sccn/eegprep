POP_ICLABEL - Classify independent components with ICLabel.

Usage:

    EEG = pop_iclabel(EEG)
    EEG = pop_iclabel(EEG, 'default')
    EEG, command = pop_iclabel(EEG, 'default', return_com=True)
    EEG = await pop_iclabel_async(EEG, 'default')
    EEG, command = await pop_iclabel_async(EEG, 'default', return_com=True)

Inputs:

- `EEG`: EEGPrep/EEGLAB-style dataset (or a list of datasets) with an ICA decomposition.
- `icversion`: one of `'default'`, `'lite'`, or `'beta'`. The standalone Python engine ships only `'default'`.
- `gui`: open the version dialog. Defaults to true when `icversion` is omitted.
- `renderer`: dialog renderer used instead of the Qt window, for scripted or headless runs.
- `engine`: `None` (default) classifies with EEGPrep's packaged network on `onnxruntime`.
  `'matlab'` or `'octave'` call an EEGLAB ICLabel checkout, which is also the only way to use `'lite'` or `'beta'`.
- `return_com`: also return the history command, `EEG = pop_iclabel(EEG, 'default');`.
  For `pop_iclabel_async` it is `EEG = await pop_iclabel_async(EEG, 'default');`,
  which replays in `eegprep-console`.

Graphical interface:

Calling `pop_iclabel(EEG)` opens a compact dialog with an ICLabel version
selector. Choose the desired model and press OK.

Behavior:

- ICA weights must already be present. Run `pop_runica` or another supported ICA wrapper first.
- Results are stored in `EEG.etc.ic_classification.ICLabel`.
- The result includes the ICLabel class names, per-component class probabilities, and the selected version string.
- Lists of datasets are processed one dataset at a time with the same selected version.
- Standalone Python needs the `iclabel` extra (`pip install "eegprep[iclabel]"`), which installs `onnxruntime`.
  Without it an `ImportError` names the extra. PyTorch is not needed.
- The packaged network is an 8-bit weight-only quantization of the EEGLAB network. How closely its
  probabilities match the full-precision network is reported in the user guide:
  https://sccn.github.io/eegprep/user_guide/ica_rejection.html#iclabel-measured-agreement
- In Pyodide/Emscripten the synchronous `iclabel` and `pop_iclabel` raise `RuntimeError`.
  Use `await iclabel_async(EEG)` or `await pop_iclabel_async(EEG, 'default')`; see `pop_iclabel_async`.

Errors:

- `ValueError` when `icversion` is not one of the three versions or the dataset has no ICA decomposition.
- `NotImplementedError` when `'lite'` or `'beta'` is requested with `engine=None`.
- `ImportError` naming `eegprep[iclabel]` when `onnxruntime` is not installed.
- `RuntimeError` when a synchronous entry point is called under Pyodide/Emscripten.

Example:

    EEG = pop_runica(EEG)
    EEG = pop_iclabel(EEG, 'default')

Notes:

- ICLabel class probabilities are ordered as Brain, Muscle, Eye, Heart, Line Noise, Channel Noise, and Other.
- This wrapper uses EEGPrep's packaged ICLabel implementation and does not require an EEGLAB checkout at runtime.

See also: POP_ICLABEL_ASYNC, POP_ICFLAG, POP_VIEWPROPS
