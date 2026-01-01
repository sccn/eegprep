#!/usr/bin/env python3
"""
Compute AMARI distances and component correlations between all ICA implementations.
"""

import numpy as np
import scipy.io
from pathlib import Path
from scipy.optimize import linear_sum_assignment

# Load all weight and sphere matrices
WEIGHTS_DIR = Path(__file__).parent / 'weights'

implementations = [
    'mne',
    'eegprep',
    'eeglab_runica',
    'runica_simple_matlab',
    'runica_c',
    'binica'
]

def load_matrices(name):
    """Load weights and sphere matrices."""
    weights_file = WEIGHTS_DIR / f"{name}_weights.mat"
    sphere_file = WEIGHTS_DIR / f"{name}_sphere.mat"

    if not weights_file.exists() or not sphere_file.exists():
        return None, None

    weights = scipy.io.loadmat(weights_file)['weights']
    sphere = scipy.io.loadmat(sphere_file)['sphere']

    return weights, sphere


def compute_amari_distance(W1, S1, W2, S2):
    """
    Compute AMARI distance between two ICA solutions.

    AMARI distance measures the difference between two unmixing matrices,
    accounting for permutation and scaling ambiguities.

    Args:
        W1, S1: weights and sphere for first solution (unmixing = W1 @ S1)
        W2, S2: weights and sphere for second solution (unmixing = W2 @ S2)

    Returns:
        amari_distance: float in [0, 1], where 0 = identical (up to permutation/scaling)
    """
    # Compute unmixing matrices
    U1 = W1 @ S1
    U2 = W2 @ S2

    # Compute product matrix P = U1 @ inv(U2)
    P = U1 @ np.linalg.inv(U2)

    # Normalize rows and columns
    n = P.shape[0]

    # Row-wise normalization term
    row_sum = 0.0
    for i in range(n):
        row_max = np.max(np.abs(P[i, :]))
        row_sum += np.sum(np.abs(P[i, :])) / row_max - 1.0

    # Column-wise normalization term
    col_sum = 0.0
    for j in range(n):
        col_max = np.max(np.abs(P[:, j]))
        col_sum += np.sum(np.abs(P[:, j])) / col_max - 1.0

    # AMARI distance
    amari = (row_sum + col_sum) / (2.0 * n)

    return amari


def compute_component_correlation(W1, S1, W2, S2):
    """
    Compute average component correlation after optimal matching.

    Components are columns of the mixing matrix A = inv(W @ S).
    We use absolute correlation to account for sign ambiguity.

    Args:
        W1, S1: weights and sphere for first solution
        W2, S2: weights and sphere for second solution

    Returns:
        mean_corr: average absolute correlation after optimal matching
        min_corr: minimum absolute correlation
        max_corr: maximum absolute correlation
    """
    # Compute mixing matrices (components are columns)
    U1 = W1 @ S1
    U2 = W2 @ S2

    A1 = np.linalg.pinv(U1)
    A2 = np.linalg.pinv(U2)

    n = A1.shape[1]

    # Compute pairwise absolute correlations
    corr_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            corr = np.corrcoef(A1[:, i], A2[:, j])[0, 1]
            corr_matrix[i, j] = np.abs(corr)

    # Find optimal matching using Hungarian algorithm
    # We want to maximize correlation, so minimize negative correlation
    row_ind, col_ind = linear_sum_assignment(-corr_matrix)

    # Extract matched correlations
    matched_corrs = corr_matrix[row_ind, col_ind]

    return np.mean(matched_corrs), np.min(matched_corrs), np.max(matched_corrs)


def main():
    print("="*80)
    print("AMARI Distance and Component Correlation Analysis")
    print("="*80)

    # Load all matrices
    results = {}
    for name in implementations:
        W, S = load_matrices(name)
        if W is not None:
            results[name] = {'weights': W, 'sphere': S}
            print(f"Loaded {name:25s}: {W.shape}")
        else:
            print(f"Skipped {name:25s}: not found")

    available = list(results.keys())
    n_impl = len(available)

    print(f"\nFound {n_impl} implementations")
    print()

    # Compute all pairwise comparisons
    print("="*80)
    print("AMARI Distance Matrix (lower is better, 0 = identical up to permutation)")
    print("="*80)
    print()

    amari_matrix = np.zeros((n_impl, n_impl))

    # Print header
    print(f"{'':25s}", end='')
    for name in available:
        print(f"{name:12s}", end='')
    print()
    print("-" * (25 + 12 * n_impl))

    for i, name1 in enumerate(available):
        print(f"{name1:25s}", end='')
        for j, name2 in enumerate(available):
            if i == j:
                amari_matrix[i, j] = 0.0
                print(f"{'0.000000':>12s}", end='')
            elif i > j:
                # Already computed (symmetric)
                amari_matrix[i, j] = amari_matrix[j, i]
                print(f"{amari_matrix[i, j]:12.6f}", end='')
            else:
                W1, S1 = results[name1]['weights'], results[name1]['sphere']
                W2, S2 = results[name2]['weights'], results[name2]['sphere']

                amari_dist = compute_amari_distance(W1, S1, W2, S2)
                amari_matrix[i, j] = amari_dist
                print(f"{amari_dist:12.6f}", end='')
        print()

    print()
    print("="*80)
    print("Component Correlation After Optimal Matching (higher is better)")
    print("Format: (min - max)")
    print("="*80)
    print()

    # Print header
    print(f"{'':25s}", end='')
    for name in available:
        print(f"{name:>18s}", end='')
    print()
    print("-" * (25 + 18 * n_impl))

    for i, name1 in enumerate(available):
        print(f"{name1:25s}", end='')
        for j, name2 in enumerate(available):
            if i == j:
                print(f"{'(1.000-1.000)':>18s}", end='')
            elif i > j:
                # Skip lower triangle for readability
                print(f"{'':>18s}", end='')
            else:
                W1, S1 = results[name1]['weights'], results[name1]['sphere']
                W2, S2 = results[name2]['weights'], results[name2]['sphere']

                mean_corr, min_corr, max_corr = compute_component_correlation(W1, S1, W2, S2)
                print(f"({min_corr:.3f}-{max_corr:.3f})", end='')
                print(f"{'':>6s}", end='')
        print()

    print()
    print("="*80)
    print("All Pairs Sorted by AMARI Distance (most similar first)")
    print("="*80)

    # Find all pairs with AMARI distance (excluding diagonal)
    amari_flat = []
    for i in range(n_impl):
        for j in range(i+1, n_impl):
            amari_flat.append((amari_matrix[i, j], available[i], available[j]))

    amari_flat.sort()

    print()
    for k, (dist, name1, name2) in enumerate(amari_flat, 1):
        print(f"{k:2d}. {name1:25s} vs {name2:25s}: {dist:.6f}")

    print("\nDone!")


if __name__ == '__main__':
    main()
