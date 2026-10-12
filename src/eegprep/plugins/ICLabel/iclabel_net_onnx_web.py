"""ONNX Runtime Web adapter used by the Pyodide/Emscripten backend.

The JavaScript host registers ``eegprep_iclabel_web`` before the first call to
``run_iclabel_net_async``. Its ``run`` method accepts flat Float32Array inputs and their shapes,
then returns a Promise resolving to the flat output Float32Array. Keeping the
adapter at this boundary leaves the ICLabel processing code independent of the
browser host and avoids importing native ``onnxruntime`` in Pyodide.
"""

import numpy as np

MISSING_HOST_MESSAGE = (
    "The browser ICLabel backend needs a Pyodide host that registers a global "
    "'eegprep_iclabel_web' object created by createIcLabelWebBridge() "
    "(tools/pyodide/iclabel_web_bridge.mjs) with a run(...) method. This module "
    "only works inside Pyodide, where 'js' and 'pyodide.ffi' exist. Original error: {error}"
)


async def run_iclabel_net_async(image, psdmed, autocorr):
    """Run ICLabel through the host's asynchronous ONNX Runtime Web bridge.

    Raises:
        ImportError: If not running in Pyodide, or the host never registered
            the ``eegprep_iclabel_web`` global.
    """
    # The host registers eegprep_iclabel_web outside Python, so this is a real
    # trust boundary; without the guard a missing registration surfaces as
    # "cannot import name 'eegprep_iclabel_web' from 'js'".
    try:
        from js import eegprep_iclabel_web  # ty: ignore[unresolved-import]
        from pyodide.ffi import to_js  # ty: ignore[unresolved-import]
    except ImportError as exc:
        raise ImportError(MISSING_HOST_MESSAGE.format(error=exc)) from exc

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
