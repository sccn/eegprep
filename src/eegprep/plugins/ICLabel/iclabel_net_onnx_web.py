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
        RuntimeError: If the bridge returns a different number of values than
            seven class probabilities per input row.
    """
    # The host registers eegprep_iclabel_web outside Python. The guard replaces the bare
    # "cannot import name 'eegprep_iclabel_web' from 'js'" with the host contract.
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
    return _network_output(output.to_py(), image.shape[0])


def _network_output(values, batch_size):
    """Shape the bridge's flat output like the native network output, checking its size first."""
    # The host is outside Python's control, so a wrong-sized result must not be
    # silently reshaped into a different number of components.
    output = np.asarray(values, dtype=np.float32)
    expected_shape = (batch_size, 7, 1, 1)
    if output.size != batch_size * 7:
        raise RuntimeError(
            f"ICLabel web bridge returned {output.size} values for a batch of {batch_size}; "
            f"expected {batch_size * 7} (7 class probabilities per input)."
        )
    return output.reshape(expected_shape)
