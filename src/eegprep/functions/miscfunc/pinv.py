"""Matrix pseudoinverse computation utilities."""

import numpy as np
from scipy.linalg import lstsq, svd as scipy_svd


def pinv(A, tol=None, method='scipy'):
    """Compute the Moore-Penrose pseudoinverse of a matrix.

    Parameters
    ----------
    A : array_like
        Input matrix to compute pseudoinverse of. Cast to float64.
    tol : float, optional
        Absolute singular-value cutoff, matching MATLAB ``pinv(A, tol)``.
        Singular values less than or equal to ``tol`` are treated as zero.
        The default is ``max(size(A)) * eps(norm(A, 2))``.
    method : str, optional
        Method to use for computation. Options:
        - 'scipy': Economy SVD via ``scipy.linalg.svd`` (default)
        - 'svd': Economy SVD via ``numpy.linalg.svd``
        - 'gelsd': ``scipy.linalg.lstsq`` with the gelsd driver. ``tol`` is ignored.

    Returns
    -------
    ndarray
        The pseudoinverse of A, with shape ``(A.shape[1], A.shape[0])``.
    """
    A = np.asarray(A, dtype=np.float64)

    if method in ('scipy', 'svd'):
        return _svd_pinv(A, tol, use_scipy=method == 'scipy')

    if method == 'gelsd':
        # Use least squares solver for pseudoinverse
        m, n = A.shape
        if m >= n:
            # Tall or square matrix: A_pinv = (A^T A)^(-1) A^T
            # But use lstsq for numerical stability
            identity = np.eye(m, dtype=A.dtype)
            A_pinv, residuals, rank, s = lstsq(A, identity, lapack_driver='gelsd')
            return A_pinv
        # Wide matrix: A_pinv = A^T (A A^T)^(-1)
        identity = np.eye(n, dtype=A.dtype)
        A_pinv, residuals, rank, s = lstsq(A.T, identity, lapack_driver='gelsd')
        return A_pinv.T

    raise ValueError(f"Unknown method: {method}. Use 'scipy', 'svd', or 'gelsd'.")


def _svd_pinv(matrix, tol, *, use_scipy):
    if matrix.ndim != 2:
        raise ValueError('pinv requires a 2-D array.')
    rows, columns = matrix.shape
    if rows == 0 or columns == 0:
        return np.zeros((columns, rows), dtype=matrix.dtype)

    decompose = scipy_svd if use_scipy else np.linalg.svd
    left, singular_values, right = decompose(matrix, full_matrices=False)
    if tol is None:
        # MATLAB pinv: max(size(A)) * eps(norm(s, inf)). norm(s, inf) is the
        # 2-norm of A. Singular values <= this cutoff are zero (s > tol).
        tol = max(rows, columns) * np.spacing(singular_values[0])
    else:
        tol = float(tol)

    rank = int(np.count_nonzero(singular_values > tol))
    if rank == 0:
        return np.zeros((columns, rows), dtype=matrix.dtype)
    factors = 1.0 / singular_values[:rank]
    return (right[:rank].T * factors) @ left[:, :rank].T
