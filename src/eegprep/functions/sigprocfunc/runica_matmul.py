"""Platform-selected matrix products for the float64 runica hot path.

The runica training loops provide the existing ``np.errstate`` suppression
around these calls. Keeping that context at the loop boundary avoids paying
for a new context on every matrix product.
"""

from __future__ import annotations

import sys

import numpy as np
from scipy.linalg import blas

_dgemm = getattr(blas, "dgemm")


def _numpy_matmul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return left @ right


def _blas_matmul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return _dgemm(alpha=1.0, a=left, b=right)


if sys.platform == "emscripten":
    BACKEND = "scipy.linalg.blas.dgemm"
    runica_matmul = _blas_matmul
else:
    BACKEND = "numpy.matmul"
    runica_matmul = _numpy_matmul
