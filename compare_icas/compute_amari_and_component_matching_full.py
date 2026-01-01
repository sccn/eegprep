#!/usr/bin/env python3
"""
Compute AMARI distances and component correlations for both standard and extended ICA.
Save results to 3 text files: standard ICA, extended ICA, and sphere comparisons.
"""

import numpy as np
import scipy.io
import sys
from pathlib import Path
from scipy.optimize import linear_sum_assignment
from io import StringIO

# Load all weight and sphere matrices
WEIGHTS_DIR = Path(__file__).parent / 'weights'
OUTPUT_DIR = Path(__file__).parent

implementations = [
    'mne',
    'eegprep',
    'eeglab_runica',
    'runica_simple_matlab',
    'runica_c',
    'binica'
]


def load_matrices(name, extended=False):
    """Load weights and sphere matrices."""
    suffix = "_ext" if extended else ""
    weights_file = WEIGHTS_DIR / f"{name}{suffix}_weights.mat"
    sphere_file = WEIGHTS_DIR / f"{name}{suffix}_sphere.mat"

    if not weights_file.exists() or not sphere_file.exists():
        return None, None

    weights = scipy.io.loadmat(weights_file)['weights']
    sphere = scipy.io.loadmat(sphere_file)['sphere']

    return weights, sphere


def compute_amari_distance(W1, S1, W2, S2):
    """
    Compute AMARI distance between two ICA solutions.
    """
    U1 = W1 @ S1
    U2 = W2 @ S2

    P = U1 @ np.linalg.inv(U2)
    n = P.shape[0]

    row_sum = 0.0
    for i in range(n):
        row_max = np.max(np.abs(P[i, :]))
        row_sum += np.sum(np.abs(P[i, :])) / row_max - 1.0

    col_sum = 0.0
    for j in range(n):
        col_max = np.max(np.abs(P[:, j]))
        col_sum += np.sum(np.abs(P[:, j])) / col_max - 1.0

    amari = (row_sum + col_sum) / (2.0 * n)
    return amari


def compute_component_correlation(W1, S1, W2, S2):
    """
    Compute average component correlation after optimal matching.
    """
    U1 = W1 @ S1
    U2 = W2 @ S2

    A1 = np.linalg.pinv(U1)
    A2 = np.linalg.pinv(U2)

    n = A1.shape[1]

    corr_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            corr = np.corrcoef(A1[:, i], A2[:, j])[0, 1]
            corr_matrix[i, j] = np.abs(corr)

    row_ind, col_ind = linear_sum_assignment(-corr_matrix)
    matched_corrs = corr_matrix[row_ind, col_ind]

    return np.mean(matched_corrs), np.min(matched_corrs), np.max(matched_corrs)


def compute_sphere_correlation(S1, S2):
    """
    Compute correlation between sphere matrices.
    """
    corr = np.corrcoef(S1.flatten(), S2.flatten())[0, 1]
    max_diff = np.max(np.abs(S1 - S2))
    return corr, max_diff


def generate_comparison_report(results, mode_name):
    """Generate comparison report for given mode."""
    output = StringIO()

    output.write("="*80 + "\n")
    output.write(f"AMARI Distance and Component Correlation Analysis - {mode_name}\n")
    output.write("="*80 + "\n\n")

    available = list(results.keys())
    n_impl = len(available)

    output.write(f"Found {n_impl} implementations\n\n")

    # AMARI Distance Matrix
    output.write("="*80 + "\n")
    output.write("AMARI Distance Matrix (lower is better, 0 = identical up to permutation)\n")
    output.write("="*80 + "\n\n")

    amari_matrix = np.zeros((n_impl, n_impl))

    # Print header
    output.write(f"{'':25s}")
    for name in available:
        output.write(f"{name:12s}")
    output.write("\n")
    output.write("-" * (25 + 12 * n_impl) + "\n")

    for i, name1 in enumerate(available):
        output.write(f"{name1:25s}")
        for j, name2 in enumerate(available):
            if i == j:
                amari_matrix[i, j] = 0.0
                output.write(f"{'0.000000':>12s}")
            elif i > j:
                amari_matrix[i, j] = amari_matrix[j, i]
                output.write(f"{amari_matrix[i, j]:12.6f}")
            else:
                W1, S1 = results[name1]['weights'], results[name1]['sphere']
                W2, S2 = results[name2]['weights'], results[name2]['sphere']

                amari_dist = compute_amari_distance(W1, S1, W2, S2)
                amari_matrix[i, j] = amari_dist
                output.write(f"{amari_dist:12.6f}")
        output.write("\n")

    # Component Correlation Matrix
    output.write("\n")
    output.write("="*80 + "\n")
    output.write("Component Correlation After Optimal Matching (higher is better)\n")
    output.write("Format: (min - max)\n")
    output.write("="*80 + "\n\n")

    # Print header
    col_width = 18
    output.write(f"{'':25s}")
    for name in available:
        output.write(name.rjust(col_width))
    output.write("\n")
    output.write("-" * (25 + col_width * n_impl) + "\n")

    for i, name1 in enumerate(available):
        output.write(f"{name1:25s}")
        for j, name2 in enumerate(available):
            if i == j:
                val_str = "(1.000-1.000)"
                output.write(val_str.rjust(col_width))
            elif i > j:
                output.write(" " * col_width)
            else:
                W1, S1 = results[name1]['weights'], results[name1]['sphere']
                W2, S2 = results[name2]['weights'], results[name2]['sphere']

                mean_corr, min_corr, max_corr = compute_component_correlation(W1, S1, W2, S2)
                val_str = f"({min_corr:.3f}-{max_corr:.3f})"
                output.write(val_str.rjust(col_width))
        output.write("\n")

    # All Pairs Sorted
    output.write("\n")
    output.write("="*80 + "\n")
    output.write("All Pairs Sorted by AMARI Distance (most similar first)\n")
    output.write("="*80 + "\n\n")

    amari_flat = []
    for i in range(n_impl):
        for j in range(i+1, n_impl):
            amari_flat.append((amari_matrix[i, j], available[i], available[j]))

    amari_flat.sort()

    for k, (dist, name1, name2) in enumerate(amari_flat, 1):
        output.write(f"{k:2d}. {name1:25s} vs {name2:25s}: {dist:.6f}\n")

    output.write("\n")

    return output.getvalue()


def generate_sphere_comparison_report(results_std, results_ext):
    """Generate sphere matrix comparison report."""
    output = StringIO()

    output.write("="*80 + "\n")
    output.write("Sphere Matrix Comparison\n")
    output.write("="*80 + "\n\n")

    available_std = list(results_std.keys())
    available_ext = list(results_ext.keys())

    # Standard ICA Sphere Correlations
    output.write("="*80 + "\n")
    output.write("Standard ICA - Sphere Matrix Correlations\n")
    output.write("Format: correlation (max_abs_diff)\n")
    output.write("="*80 + "\n\n")

    n_impl = len(available_std)
    col_width = 22

    # Print header
    output.write(f"{'':25s}")
    for name in available_std:
        output.write(name.rjust(col_width))
    output.write("\n")
    output.write("-" * (25 + col_width * n_impl) + "\n")

    for i, name1 in enumerate(available_std):
        output.write(f"{name1:25s}")
        for j, name2 in enumerate(available_std):
            if i == j:
                # Diagonal: check if MNE
                if name1 == 'mne':
                    val_str = "n/a"
                else:
                    val_str = "1.000 (0.0e+00)"
                output.write(val_str.rjust(col_width))
            elif i > j:
                output.write(" " * col_width)
            else:
                # Off-diagonal: check if either is MNE
                if name1 == 'mne' or name2 == 'mne':
                    val_str = "n/a"
                    output.write(val_str.rjust(col_width))
                else:
                    S1 = results_std[name1]['sphere']
                    S2 = results_std[name2]['sphere']

                    corr, max_diff = compute_sphere_correlation(S1, S2)
                    val_str = f"{corr:.3f} ({max_diff:.1e})"
                    output.write(val_str.rjust(col_width))
        output.write("\n")

    # Extended ICA Sphere Correlations
    output.write("\n")
    output.write("="*80 + "\n")
    output.write("Extended ICA - Sphere Matrix Correlations\n")
    output.write("Format: correlation (max_abs_diff)\n")
    output.write("="*80 + "\n\n")

    n_impl = len(available_ext)
    col_width = 22

    # Print header
    output.write(f"{'':25s}")
    for name in available_ext:
        output.write(name.rjust(col_width))
    output.write("\n")
    output.write("-" * (25 + col_width * n_impl) + "\n")

    for i, name1 in enumerate(available_ext):
        output.write(f"{name1:25s}")
        for j, name2 in enumerate(available_ext):
            if i == j:
                # Diagonal: check if MNE
                if name1 == 'mne':
                    val_str = "n/a"
                else:
                    val_str = "1.000 (0.0e+00)"
                output.write(val_str.rjust(col_width))
            elif i > j:
                output.write(" " * col_width)
            else:
                # Off-diagonal: check if either is MNE
                if name1 == 'mne' or name2 == 'mne':
                    val_str = "n/a"
                    output.write(val_str.rjust(col_width))
                else:
                    S1 = results_ext[name1]['sphere']
                    S2 = results_ext[name2]['sphere']

                    corr, max_diff = compute_sphere_correlation(S1, S2)
                    val_str = f"{corr:.3f} ({max_diff:.1e})"
                    output.write(val_str.rjust(col_width))
        output.write("\n")

    output.write("\n")

    return output.getvalue()


def main():
    print("="*80)
    print("Computing AMARI distances and component correlations")
    print("="*80)

    # Load standard ICA results
    print("\nLoading standard ICA results...")
    results_std = {}
    for name in implementations:
        W, S = load_matrices(name, extended=False)
        if W is not None:
            results_std[name] = {'weights': W, 'sphere': S}
            print(f"  Loaded {name:25s}: {W.shape}")

    # Load extended ICA results
    print("\nLoading extended ICA results...")
    results_ext = {}
    for name in implementations:
        W, S = load_matrices(name, extended=True)
        if W is not None:
            results_ext[name] = {'weights': W, 'sphere': S}
            print(f"  Loaded {name:25s}: {W.shape}")

    # Generate reports
    print("\nGenerating comparison reports...")

    # Standard ICA report
    print("  - Standard ICA comparison...")
    std_report = generate_comparison_report(results_std, "Standard ICA")
    std_file = OUTPUT_DIR / "compare_standard_ica.txt"
    with open(std_file, 'w') as f:
        f.write(std_report)
    print(f"    Saved to: {std_file}")

    # Extended ICA report
    print("  - Extended ICA comparison...")
    ext_report = generate_comparison_report(results_ext, "Extended ICA")
    ext_file = OUTPUT_DIR / "compare_extended_ica.txt"
    with open(ext_file, 'w') as f:
        f.write(ext_report)
    print(f"    Saved to: {ext_file}")

    # Sphere comparison report
    print("  - Sphere matrix comparison...")
    sphere_report = generate_sphere_comparison_report(results_std, results_ext)
    sphere_file = OUTPUT_DIR / "compare_sphere.txt"
    with open(sphere_file, 'w') as f:
        f.write(sphere_report)
    print(f"    Saved to: {sphere_file}")

    # Also print to console
    print("\n" + "="*80)
    print("STANDARD ICA RESULTS")
    print("="*80)
    print(std_report)

    print("\n" + "="*80)
    print("EXTENDED ICA RESULTS")
    print("="*80)
    print(ext_report)

    print("\n" + "="*80)
    print("SPHERE COMPARISON")
    print("="*80)
    print(sphere_report)

    print("\nDone!")
    print(f"\nAll results saved to:")
    print(f"  {std_file}")
    print(f"  {ext_file}")
    print(f"  {sphere_file}")


if __name__ == '__main__':
    main()
