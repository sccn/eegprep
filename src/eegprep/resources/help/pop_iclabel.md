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
- In Pyodide/Emscripten, use `await iclabel_async(EEG)` or
  `await pop_iclabel_async(EEG, 'default')`. The synchronous `iclabel` and
  `pop_iclabel` entry points fail fast because ONNX Runtime Web is asynchronous.
- The browser host must register EEGPrep's ONNX Runtime Web bridge before
  calling the async entry point. A host may run the Pyodide environment in a
  Web Worker to keep the UI responsive.
- Standalone Python EEGPrep ships the gate-selected weight-only int8 ICLabel
  network as an ONNX artifact (`iclabel.onnx`) and classifies through
  `onnxruntime`; install the `iclabel` extra (`eegprep[iclabel]`) to run it.
  Feature extraction, float32 input normalization, augmentation, and output
  softmax are unchanged. On the frozen 217-component, subject-disjoint set in
  `tools/iclabel/evaluation_manifest.json`, the shipped artifact agreed with
  the preserved float32 teacher on 100% of top-1 labels and 100% of existing
  `pop_icflag` keep-or-reject decisions, with a maximum probability drift of
  0.01346. The calibrated candidate measured 98.1567% top-1 and 100%
  keep-or-reject but exceeded the frozen 0.015 probability-drift gate; the
  smaller weight-only candidate is the shipped default. These are measured
  frozen-set parity results, not a general accuracy claim. The EEGLAB `lite`
  and `beta` network artifacts are
  explicit MATLAB/Octave passthrough choices and raise a clear limitation when
  requested with the standalone Python engine.

Example:

    EEG = pop_runica(EEG)
    EEG = pop_iclabel(EEG, 'default')

Notes:

- ICLabel class probabilities are ordered as Brain, Muscle, Eye, Heart, Line Noise, Channel Noise, and Other.
- This wrapper uses EEGPrep's packaged ICLabel implementation and does not require an EEGLAB checkout at runtime.
- Async history commands are replayable from `eegprep-console`; for example,
  use `await eegh(1)` when the selected history entry contains
  `pop_iclabel_async`.
