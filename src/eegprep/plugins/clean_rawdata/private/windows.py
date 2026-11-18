"""MATLAB colon placement for clean_rawdata windows."""

import numpy as np

from ....functions.miscfunc.matlab_colon import matlab_colon
from ....functions.miscfunc.misc import round_mat


def matlab_window_offsets(n_samples: float, window_samples: float, step: float) -> np.ndarray:
    """Return 0-based starts of ``round(1:step:(n_samples-window_samples))``.

    ``clean_windows.m`` and ``asr_calibrate.m`` place RMS windows with that
    colon expression. ``window_samples`` is the same ``N`` used in the stop
    ``S-N``, including a non-integer ``N``.

    Args:
        n_samples: Number of samples, ``S``.
        window_samples: Window length ``N`` in samples.
        step: Colon increment, usually ``N*(1-window_overlap)``.

    Returns:
        0-based integer sample offsets. Empty when the colon is empty.
    """
    starts = matlab_colon(1.0, step, float(n_samples) - float(window_samples))
    if starts.size == 0:
        return np.array([], dtype=int)
    return np.asarray(round_mat(starts), dtype=int) - 1


def asr_update_at(n_samples: int, step_size: int) -> np.ndarray:
    """Return 1-based ``min(step:step:(n+step-1), n)``.

    ``asr_process.m:156`` and ``asr_process_r.m:167`` clamp that colon onto
    the last sample so both ends of a chunk refresh the reconstruction.
    """
    count = int(n_samples)
    step = int(step_size)
    grid = matlab_colon(float(step), float(step), float(count + step - 1))
    return np.asarray(np.rint(np.minimum(grid, float(count))), dtype=np.int64)
