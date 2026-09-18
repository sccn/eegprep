"""onnxruntime backend for the packaged ICLabel default network.

This is the runtime counterpart to :mod:`iclabel_net` (the torch definition
used only to build and export the network). It loads ``iclabel.onnx``, the
selected packaged artifact produced by the Phase 6 quantization pipeline, and
runs it through onnxruntime so ICLabel classification does not require torch.
"""

import os
import sys

import numpy as np

_INPUT_NAMES = ('image', 'psdmed', 'autocorr')
_OUTPUT_NAME = 'output'
_IS_EMSCRIPTEN = sys.platform == 'emscripten'

_session = None


def _get_session():
    global _session
    if _session is not None:
        return _session
    try:
        import onnxruntime as ort
    except ImportError as e:
        raise ImportError(
            f"onnxruntime is not installed in your environment ({e}). "
            f"To include onnxruntime, install eegprep as eegprep[iclabel] or "
            f"eegprep[all]."
        ) from e
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'iclabel.onnx')
    _session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    return _session


def run_iclabel_net(image, psdmed, autocorr):
    """Run the packaged ICLabel network through onnxruntime.

    Parameters
    ----------
    image, psdmed, autocorr : numpy.ndarray
        NCHW float32 arrays, matching the torch ``ICLabelNet.forward`` inputs.

    Returns
    -------
    numpy.ndarray
        Network output, shaped like the torch model's output.
    """
    if _IS_EMSCRIPTEN:
        raise RuntimeError(
            "ICLabel synchronous ONNX execution is unavailable under Emscripten; use await run_iclabel_net_async(...)."
        )
    session = _get_session()
    inputs = {
        _INPUT_NAMES[0]: np.asarray(image, dtype=np.float32),
        _INPUT_NAMES[1]: np.asarray(psdmed, dtype=np.float32),
        _INPUT_NAMES[2]: np.asarray(autocorr, dtype=np.float32),
    }
    (output,) = session.run([_OUTPUT_NAME], inputs)
    return output


if _IS_EMSCRIPTEN:
    from eegprep.plugins.ICLabel.iclabel_net_onnx_web import run_iclabel_net_async as _run_iclabel_net_async
else:

    async def _run_iclabel_net_async(image, psdmed, autocorr):
        return run_iclabel_net(image, psdmed, autocorr)


async def run_iclabel_net_async(image, psdmed, autocorr):
    """Run the platform-selected ICLabel backend asynchronously."""
    return await _run_iclabel_net_async(image, psdmed, autocorr)
