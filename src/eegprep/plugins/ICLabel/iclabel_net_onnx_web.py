"""ONNX Runtime Web adapter used by the Pyodide/Emscripten backend.

The JavaScript host registers ``eegprep_iclabel_web`` before importing this
module. Its ``run`` method accepts flat Float32Array inputs and their shapes,
then returns a Promise resolving to the flat output Float32Array. Keeping the
adapter at this boundary leaves the ICLabel processing code independent of the
browser host and avoids importing native ``onnxruntime`` in Pyodide.
"""

import numpy as np


async def run_iclabel_net_async(image, psdmed, autocorr):
    """Run ICLabel through the host's asynchronous ONNX Runtime Web bridge."""
    from js import eegprep_iclabel_web  # ty: ignore[unresolved-import]
    from pyodide.ffi import to_js  # ty: ignore[unresolved-import]

    image = np.ascontiguousarray(image, dtype=np.float32)
    psdmed = np.ascontiguousarray(psdmed, dtype=np.float32)
    autocorr = np.ascontiguousarray(autocorr, dtype=np.float32)
    output = await eegprep_iclabel_web.run(
        to_js(image.reshape(-1)),
        to_js(list(image.shape)),
        to_js(psdmed.reshape(-1)),
        to_js(list(psdmed.shape)),
        to_js(autocorr.reshape(-1)),
        to_js(list(autocorr.shape)),
    )
    return np.asarray(output.to_py(), dtype=np.float32).reshape((-1, 7, 1, 1))
