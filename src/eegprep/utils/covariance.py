"""Tools for working with covariance matrices or stacks thereof."""

# Copyright (c) 2015-2025 Syntrogi Inc. dba Intheon.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import logging

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ['cov_mean', 'cov_logm', 'cov_expm', 'cov_powm', 'cov_sqrtm', 'cov_rsqrtm', 'cov_sqrtm2', 'cov_shrinkage']


def diag_nd(M):
    """Like np.diag, but in case of a ...,N, returns a ...,N,N array of diag matrices.

    """
    *dims, N = M.shape
    if dims:
        cat = np.concatenate([np.diag(d) for d in M.reshape((-1, N))])
        return np.reshape(cat, dims + [N, N])
    else:
        return np.diag(M)


def cov_logm(C):
    """Calculate the matrix logarithm of a covariance matrix or ...,N,N array.

    """
    D, V = np.linalg.eigh(C)
    return V @ diag_nd(np.log(D)) @ V.swapaxes(-2, -1)


def cov_expm(C):
    """Calculate the matrix exponent of a covariance matrix or ...,N,N array.

    """
    D, V = np.linalg.eigh(C)
    return V @ diag_nd(np.exp(D)) @ V.swapaxes(-2, -1)


def cov_powm(C, exp):
    """Calculate a matrix power of a covariance matrix or ...,N,N array.

    """
    D, V = np.linalg.eigh(C)
    return V @ diag_nd(D**exp) @ V.swapaxes(-2, -1)


def cov_sqrtm(C):
    """Calculate the matrix square root of a covariance matrix or ...,N,N array.

    """
    D, V = np.linalg.eigh(C)
    return V @ diag_nd(np.sqrt(D)) @ V.swapaxes(-2, -1)


def cov_rsqrtm(C):
    """Calculate the matrix reciprocal square root of a covariance matrix or ...,N,N array.

    """
    D, V = np.linalg.eigh(C)
    return V @ diag_nd(1./np.sqrt(D)) @ V.swapaxes(-2, -1)


def cov_sqrtm2(C):
    """Calculate the matrix square root, and its reciprocal, for a covariance matrix or ...,N,N array.

    """
    D, V = np.linalg.eigh(C)
    sqrtD = np.sqrt(D)
    return V @ diag_nd(sqrtD) @ V.swapaxes(-2, -1), V @ diag_nd(1./sqrtD) @ V.swapaxes(-2, -1)


def cov_mean(X, *, weights=None, robust=False, iters=50, tol=1e-5, huber=0,
             nancheck=False, verbose=False):
    """Calculate the (weighted) average of a set of covariance matrices on the manifold of SPD matrices, optionally robustly using the geometric median or Huber mean.

    Args:
        X: a M,N,N array of covariance matrices
        weights: optionally a vector of sample weights (can be unnormalized)
        robust: whether to use a robust estimator
        iters: maximum number of iterations
        huber: huber threshold (delta parameter); can be set to
          * None: use regular least-squares solution
          * 0: use geometric / l1 median
          * >0: use a Huber mean with the given value as the threshold
        tol: tolerance for convergence check
        nancheck: check for NaNs
        verbose: generate verbose output (will print deviations in huber=None mode)

    Returns
    -------
        the N,N mean covariance matrix
    """
    # This algorithm is based on:
    #  [1] Ostresh et al., 1978, "On the Convergence of a Class of Iterative Methods for Solving the Weber Location Problem"
    #  [2] Fletcher et al., 2004, "Principal Geodesic Analysis on Symmetric Spaces: Statistics of Diffusion Tensors"
    #  [3] Fletcher et al. 2010, "The geometric median on Riemannian manifolds with application to robust atlas estimation"
    #  [4] Barachant et al., 2014, "Multiclass Brain-Computer Interface Classification by Riemannian Geometry"
    weights = np.ones(len(X)) if weights is None else np.asarray(weights)
    scales = weights

    mu = np.sum(X * weights[:, None, None], axis=0)/np.sum(weights)
    # step size and divergence check threshold
    step, thresh = 1.0, 1e20
    for i in range(iters):
        mu_sqrt, mu_rsqrt = cov_sqrtm2(mu)
        # linearize around mu (this would be the tangent space, but we omit
        # the pre/post-multiplied mu_sqrt terms since they cancel in both
        # the scale calculation and the exponential map)
        Xt = cov_logm(mu_rsqrt @ X @ mu_rsqrt)
        # geometric-median correction (downweight each pt by its riemannian
        # distance from mu, which we calc here after linearization)
        if robust:
            # deviations/errors per sample
            d = np.sqrt(np.sum(np.square(Xt), axis=(-2, -1)))
            # apply robust scale factor to provided sample weights
            if huber is None:
                scales = weights
                if verbose:
                    logger.info(f"median deviations: {np.median(d)}")
            elif huber == 0:
                scales = weights / d
            else:
                w = np.where(d <= huber, 1, huber / d)
                scales = weights * w
        # get update Jacobian (np.average takes care of renormalization)
        J = np.sum(Xt * scales[:, None, None], axis=0)/np.sum(scales)
        # apply update on manifold
        mu = mu_sqrt @ cov_expm(step * J) @ mu_sqrt
        # convergence checks
        Jnorm = np.sqrt(np.sum(np.square(J)))
        if Jnorm < tol or step < tol:
            break
        h = step * Jnorm
        if h < thresh:
            # exponentially decaying learning rate
            step *= 0.95
            thresh = h
        else:
            # prevent blow-up
            step /= 2
        if nancheck and np.any(np.isnan(mu)):
            raise RuntimeError("NaNs occurred in cov_mean()")
    return mu


def cov_shrinkage(cov, shrinkage=0, *, target='eye'):
    """Regularize the given covariance matrix or stack of matrices using shrinkage.

    Args:
        cov: the covariance matrix (N,N) or stack of matrices (...,N,N).
        shrinkage: degree of shrinkage, between 0 and 1
        target: target matrix to shrink towards; can be:
          'eye': the identity matrix (classic shrinkage; good for small values
            of shrinkage)
          'scaled-eye': the identity matrix, scaled to the average variance
            of the data (can be practical when shrinkage degree is large, since
            otherwise whitening will not have unit variance)
          'diag': the diagonal of the covariance matrix (diagonal shrinkage)

    Returns
    -------
        the regularized covariance matrix or stack of matrices.
    """    
    if not shrinkage:        
        return cov  # early exit

    N = cov.shape[-1]

    if target == 'eye':
        # create a stack of identity matrices matching cov's shape
        eye_target = np.zeros_like(cov)
        eye_target[..., range(N), range(N)] = 1
    elif target == 'scaled-eye':
        # calculate trace for each matrix in the stack (or single matrix)
        # trace_cov will have shape cov.shape[:-2] or be scalar if cov is 2D
        trace_cov = np.trace(cov, axis1=-2, axis2=-1)
        scale = trace_cov / N

        # create a base stack of identity matrices
        eye_base = np.zeros_like(cov)
        eye_base[..., range(N), range(N)] = 1
        
        # apply scaling
        scale_val = scale
        if cov.ndim > 2:
            scale_val = scale[..., np.newaxis, np.newaxis]        
        eye_target = eye_base * scale_val
    elif target == 'diag':
        # get the main diagonal of each matrix in the stack
        main_diagonals = np.diagonal(cov, axis1=-2, axis2=-1)
        # create a stack of diagonal matrices
        eye_target = diag_nd(main_diagonals)
    else:
        raise ValueError(f'Unsupported shrinkage target: {target}')

    cov_regu = shrinkage * eye_target + (1 - shrinkage) * cov
    return cov_regu
