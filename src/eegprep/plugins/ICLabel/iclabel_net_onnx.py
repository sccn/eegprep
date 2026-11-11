"""ONNX Runtime backend for the packaged ICLabel default network.

This is the runtime counterpart to :mod:`iclabel_net`, the PyTorch definition
used only to export the network. It runs ``iclabel.onnx``, the float32 network
built by ``tools/iclabel/export_iclabel_onnx.py``, through onnxruntime, so
ICLabel classification does not need PyTorch.
"""

import functools
import os

import numpy as np

from eegprep.extras import missing_onnxruntime_message

INPUT_NAMES = ('image', 'psdmed', 'autocorr')
OUTPUT_NAME = 'output'


@functools.cache
def _get_session():
    try:
        import onnxruntime as ort
    except ImportError as e:
        raise ImportError(missing_onnxruntime_message('iclabel')) from e
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'iclabel.onnx')
    return ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])


def run_iclabel_net(image, psdmed, autocorr):
    """Run the packaged ICLabel network through onnxruntime.

    Args:
        image: NCHW float32 topography array, ``(N, 1, 32, 32)``.
        psdmed: NCHW float32 spectrum array, ``(N, 1, 1, 100)``.
        autocorr: NCHW float32 autocorrelation array, ``(N, 1, 1, 100)``.

    Returns:
        numpy.ndarray: Network output ``(N, 7, 1, 1)``, shaped like the torch
        model's output.

    Raises:
        ImportError: If onnxruntime is not installed; the message names the
            ``eegprep[iclabel]`` extra.
    """
    session = _get_session()
    inputs = {
        INPUT_NAMES[0]: np.asarray(image, dtype=np.float32),
        INPUT_NAMES[1]: np.asarray(psdmed, dtype=np.float32),
        INPUT_NAMES[2]: np.asarray(autocorr, dtype=np.float32),
    }
    (output,) = session.run([OUTPUT_NAME], inputs)
    return output
