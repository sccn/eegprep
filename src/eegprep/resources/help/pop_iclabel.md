POP_ICLABEL - Classify independent components with ICLabel.

Usage:

    EEG = pop_iclabel(EEG)
    EEG = pop_iclabel(EEG, 'default')
    EEG, command = pop_iclabel(EEG, 'default', return_com=True)
    EEG = await pop_iclabel_async(EEG, 'default')
    EEG, command = await pop_iclabel_async(EEG, 'default', return_com=True)

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
- Standalone Python EEGPrep ships the default ICLabel network as an ONNX
  artifact (`iclabel.onnx`) and classifies through `onnxruntime`; install the
  `iclabel` extra (`eegprep[iclabel]`) to run it. The EEGLAB `lite` and `beta`
  network artifacts are explicit MATLAB/Octave passthrough choices and raise
  a clear limitation when requested with the standalone Python engine.
- In Pyodide/Emscripten, use `await iclabel_async(EEG)` or
  `await pop_iclabel_async(EEG, 'default')`. The synchronous `iclabel` and
  `pop_iclabel` entry points fail fast because ONNX Runtime Web is asynchronous.
- The browser host must register EEGPrep's ONNX Runtime Web bridge before
  calling the async entry point. A host may run the Pyodide environment in a
  Web Worker to keep the UI responsive.

Example:

    EEG = pop_runica(EEG)
    EEG = pop_iclabel(EEG, 'default')

Notes:

- ICLabel class probabilities are ordered as Brain, Muscle, Eye, Heart, Line Noise, Channel Noise, and Other.
- This wrapper uses EEGPrep's packaged ICLabel implementation and does not require an EEGLAB checkout at runtime.
- Async history commands are replayable from `eegprep-console`; for example,
  use `await eegh(1)` when the selected history entry contains
  `pop_iclabel_async`.
