"""Spatial interpolation utilities."""

from typing import *
import numpy as np
from numpy.linalg import pinv


# Helper function (vectorized version of MATLAB's interpMx)
def _interpMx(cosEE, order, tol):
    """Compute the interpolation matrix for a set of point pairs (vectorized).

    Internal helper function for sphericalSplineInterpolate.

    Args:
        cosEE (np.ndarray): Matrix of cosines of angles between points.
        order (int): Order of the polynomial interpolation.
        tol (float): Tolerance for the Legendre polynomial approximation convergence.

    Returns
    -------
        tuple[np.ndarray, np.ndarray]: G and H matrices.
    """
    x = np.asarray(cosEE) # Ensure input is a numpy array

    # Initialize variables for Legendre polynomial recurrence (vectorized)
    # Using float for n even for integers to ensure float division later
    n = 1.0
    Pns1 = np.ones_like(x)
    Pn = x.copy() # Use a copy to avoid modifying input if it was passed by reference

    # Calculate initial terms for G and H sums
    nn_plus_n = n * n + n # = 2.0 when n=1
    # Ensure float exponentiation/division
    tmp = ((2.0 * n + 1.0) * Pn) / (nn_plus_n**float(order))
    G = tmp.copy() # Start sum for G
    H = nn_plus_n * tmp # Start sum for H

    # Initialize convergence tracking variables
    # Initialize dG/dH with the magnitude of the first term; avoids issues if G/H start near zero
    dG = np.abs(G)
    dH = np.abs(H)

    # Summation loop for Legendre polynomial series (vectorized)
    # Max iterations set to 500 as in the MATLAB code
    for n_int in range(2, 501):
        n = float(n_int) # Use float n for calculations

        # Legendre polynomial recurrence relation (vectorized)
        Pns2 = Pns1
        Pns1 = Pn
        Pn = ((2.0 * n - 1.0) * x * Pns1 - (n - 1.0) * Pns2) / n

        # Store old G, H for convergence check (make copies)
        oG = G.copy()
        oH = H.copy()

        # Calculate update term 'tmp' (vectorized)
        nn_plus_n = n * n + n
        # Ensure float exponentiation/division
        tmp = ((2.0 * n + 1.0) * Pn) / (nn_plus_n**float(order))

        # Update G and H sums (vectorized)
        G += tmp        # update function estimate, spline interp
        H += nn_plus_n * tmp # update function estimate, SLAP

        # Update moving average gradient estimate for convergence (vectorized)
        # Add small epsilon to denominator to prevent potential division by zero if dG/dH were zero?
        # Although, initialization above should prevent this. Let's stick to MATLAB logic.
        dG = (np.abs(oG - G) + dG) / 2.0
        dH = (np.abs(oH - H) + dH) / 2.0

        # Check for convergence (break if *all* elements meet tolerance)
        # Using np.all mimics the intent that the sum converges everywhere
        if np.all(dG < tol) and np.all(dH < tol):
            break

    # Final scaling
    G /= (4.0 * np.pi)
    H /= (4.0 * np.pi)

    return G, H

# Main function mirroring the MATLAB sphericalSplineInterpolate
def sphericalSplineInterpolate(src, dest, lambda_reg=1e-5, order=4, type='spline', tol=np.finfo(float).eps):
    """Interpolation matrix for spherical interpolation. Python port of Jason
    Farquhar's MATLAB code.

    Args:
        src (np.ndarray): Source electrode positions [3 x N]. Assumes coordinates are in columns.
        dest (np.ndarray): Destination electrode positions [3 x M]. Assumes coordinates are in columns.
        lambda_reg (float, optional): Regularisation parameter for smoothing estimates. Defaults to 1e-5.
                                      (Renamed from 'lambda' to avoid clash with Python keyword).
        order (int, optional): Order of the polynomial interpolation to use. Defaults to 4.
        type (str, optional): Interpolation type, one of 'spline' or 'slap'. Defaults to 'spline'.
                               'spline' -> spherical Spline
                               'slap'   -> surface Laplician (aka CSD)
        tol (float, optional): Tolerance for the Legendre polynomial approximation convergence.
                               Defaults to machine epsilon for float.

    Returns
    -------
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            W: [M x N] linear mapping matrix between old and new coords.
            Gss: [N x N] interpolation matrix between source points.
            Gds: [M x N] interpolation matrix from source to destination points.
            Hds: [M x N] SLAP interpolation matrix from source to destination points.

    Notes
    -----
        Based upon the paper: Perrin, F., Pernier, J., Bertrand, O., & Echallier, J. F. (1989).
        Spherical splines for scalp potential and current density mapping.
        Electroencephalography and clinical neurophysiology, 72(2), 184-187.

    Original MATLAB Copyright Notice:
        Copyright 2009-     by Jason D.R. Farquhar (jdrf@zepler.org)
        Permission is granted for anyone to copy, use, or modify this
        software and accompanying documents, provided this copyright
        notice is retained, and note is made of any changes that have been
        made. This software and documents are distributed without any
        warranty, express or implied.
    """
    # Ensure inputs are numpy arrays
    src = np.asarray(src)
    dest = np.asarray(dest)

    # Validate input shapes (optional but good practice)
    if src.ndim != 2 or src.shape[0] != 3:
        raise ValueError(f"src must be a 2D array with shape (3, N), got {src.shape}")
    if dest.ndim != 2 or dest.shape[0] != 3:
        raise ValueError(f"dest must be a 2D array with shape (3, M), got {dest.shape}")

    n_src = src.shape[1]
    n_dest = dest.shape[1]

    # Map the positions onto the unit sphere
    # Normalize each column vector
    norm_src = np.sqrt(np.sum(src**2, axis=0, keepdims=True))
    # Avoid division by zero if a source position is exactly at the origin
    norm_src[norm_src == 0] = 1.0
    src_norm = src / norm_src

    norm_dest = np.sqrt(np.sum(dest**2, axis=0, keepdims=True))
    # Avoid division by zero
    norm_dest[norm_dest == 0] = 1.0
    dest_norm = dest / norm_dest

    # Calculate the cosine of the angle between the new and old electrodes.
    # If the vectors are on top of each other, the result is 1.
    # Transpose src_norm (N, 3) and dest_norm (M, 3) for matrix multiplication
    cosSS = src_norm.T @ src_norm  # angles between source positions [N x N]
    cosDS = dest_norm.T @ src_norm # angles between destination positions [M x N]

    # Ensure cosines are within [-1, 1] due to potential floating point errors
    cosSS = np.clip(cosSS, -1.0, 1.0)
    cosDS = np.clip(cosDS, -1.0, 1.0)

    # Compute the interpolation matrices G and H using the helper function
    # Pass the tolerance 'tol' from the main function call
    Gss, _ = _interpMx(cosSS, order, tol)  # [N x N] (Hss not needed)
    Gds, Hds = _interpMx(cosDS, order, tol)  # [M x N]

    # Include the regularisation
    if lambda_reg > 0:
        Gss = Gss + lambda_reg * np.eye(n_src)

    # Compute the mapping to the polynomial coefficients space
    # N.B. this can be numerically unstable so use the PINV to solve..
    muGss = 1.0 # Fixed value as in the MATLAB code (comment mentioned median(diag(Gss)))

    # Construct matrix C
    # C = [ Gss    muGss*ones(N,1) ]
    #     [ muGss*ones(1,N)    0   ]
    C = np.zeros((n_src + 1, n_src + 1))
    C[:n_src, :n_src] = Gss
    C[:n_src, n_src] = muGss # Column of ones * muGss
    C[n_src, :n_src] = muGss # Row of ones * muGss
    # C[n_src, n_src] remains 0

    # Calculate the pseudoinverse of C
    iC = pinv(C) # [N+1 x N+1]

    # Compute the final mapping matrix W based on the specified type
    type_lower = type.lower()
    if type_lower == 'spline':
        # W = [Gds ones(M,1)*muGss] * iC[:, :-1]
        # Construct the [Gds ones(M,1)*muGss] matrix part
        Gds_augmented = np.hstack((Gds, muGss * np.ones((n_dest, 1)))) # [M x N+1]
        # Multiply by the relevant part of iC
        W = Gds_augmented @ iC[:, :n_src] # [M x N+1] @ [N+1 x N] = [M x N]

    elif type_lower == 'slap':
        # W = Hds * iC[:-1, :-1]
        W = Hds @ iC[:n_src, :n_src] # [M x N] @ [N x N] = [M x N]

    else:
        raise ValueError(f"Unknown interpolation type specified: '{type}'. Must be 'spline' or 'slap'.")

    # Return the mapping matrix W and intermediate G/H matrices as per MATLAB signature
    return W, Gss, Gds, Hds

