POP_ICLABEL_ASYNC - Classify independent components with ICLabel from an awaitable call.

Usage:

    EEG = await pop_iclabel_async(EEG)
    EEG = await pop_iclabel_async(EEG, 'default')
    EEG, command = await pop_iclabel_async(EEG, 'default', return_com=True)

Inputs and results are the same as POP_ICLABEL: `EEG`, `icversion`, `gui`,
`renderer`, `engine`, and `return_com`. The history command is
`EEG = await pop_iclabel_async(EEG, 'default');`, and it replays in
`eegprep-console` with `await`.

Behavior:

- Use it in Pyodide/Emscripten, where the synchronous `pop_iclabel` and `iclabel`
  raise `RuntimeError` because ONNX Runtime Web is asynchronous.
- On native Python it runs the same `onnxruntime` network as `pop_iclabel`, so it needs
  the `iclabel` extra (`pip install "eegprep[iclabel]"`) there as well.
- `engine='matlab'` and `engine='octave'` are not available under Pyodide/Emscripten.
- In `eegprep-console` and the GUI the result is committed to the shared session only if the
  selected dataset is unchanged while the call is running; otherwise a `RuntimeError` reports
  the discarded result. A Save that assigns a new file name or path, such as the first Save of a
  dataset, counts as a change and discards the result; marking the dataset saved alone does not.

Browser host contract:

Before the first call, the host page must set `globalThis.eegprep_iclabel_web` to an object
with one method, and EEGPrep runs the network through it:

    run(imageData, imageShape, psdmedData, psdmedShape, autocorrData, autocorrShape)

- Each `*Data` argument is a flat `Float32Array` and each `*Shape` argument is an array of integers.
- The shapes are `[N, 1, 32, 32]` for `image` and `[N, 1, 1, 100]` for `psdmed` and `autocorr`.
  `N` is four times the number of components, because ICLabel classifies four augmented copies of each.
- `run` returns a `Promise` that resolves to the flat `Float32Array` of the network output named
  `output`: seven class scores for each of the `N` rows, row after row.
- The network inputs are named `image`, `psdmed`, and `autocorr`.

In the EEGPrep source repository, `tools/pyodide/iclabel_web_bridge.mjs` (`createIcLabelWebBridge`)
is the reference implementation: it creates one `onnxruntime-web` session from the bytes of the
packaged `iclabel.onnx`. A host may run Pyodide in a Web Worker to keep the page responsive.
If no bridge is registered, the call raises an error that names it.

See also: POP_ICLABEL
