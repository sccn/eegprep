"""Matrix pseudoinverse computation utilities."""

# create a pinv function that uses the pseudoinverse function from scipy

import numpy as np
from scipy.linalg import pinv as scipy_pinv

def pinv(A, tol=None, method='scipy'):
    """Compute the Moore-Penrose pseudoinverse of a matrix.

    Parameters
    ----------
    A : array_like
        Input matrix to compute pseudoinverse of
    tol : float, optional
        Tolerance for small singular values. If None, uses scipy's default.
        This parameter matches MATLAB's pinv(A, tol) interface.
    method : str, optional
        Method to use for computation. Options:
        - 'scipy': Use scipy.linalg.pinv (default)
        - 'svd': Use explicit SVD decomposition for more control
        - 'gelsd': Use scipy.linalg.lstsq with gelsd driver

    Returns
    -------
    ndarray
        The pseudoinverse of A
    """
    A = np.asarray(A, dtype=np.float64)
    
    if method == 'scipy':
        if tol is None:
            return scipy_pinv(A)
        else:
            # Convert tol to rtol for scipy (relative tolerance)
            return scipy_pinv(A, rtol=tol)
    
    elif method == 'svd':
        # Explicit SVD-based pseudoinverse for maximum control
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        
        if tol is None:
            # Use MATLAB's default tolerance: max(size(A)) * eps(max(s))
            tol = max(A.shape) * np.finfo(A.dtype).eps * np.max(s)
        
        # Threshold singular values
        s_inv = np.where(s > tol, 1.0 / s, 0.0)
        
        # Compute pseudoinverse
        return Vt.T @ np.diag(s_inv) @ U.T
    
    elif method == 'gelsd':
        # Use least squares solver for pseudoinverse
        from scipy.linalg import lstsq
        
        m, n = A.shape
        if m >= n:
            # Tall or square matrix: A_pinv = (A^T A)^(-1) A^T
            # But use lstsq for numerical stability
            I = np.eye(m, dtype=A.dtype)
            A_pinv, residuals, rank, s = lstsq(A, I, lapack_driver='gelsd')
            return A_pinv
        else:
            # Wide matrix: A_pinv = A^T (A A^T)^(-1)
            I = np.eye(n, dtype=A.dtype)
            A_pinv, residuals, rank, s = lstsq(A.T, I, lapack_driver='gelsd')
            return A_pinv.T
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'scipy', 'svd', or 'gelsd'.")