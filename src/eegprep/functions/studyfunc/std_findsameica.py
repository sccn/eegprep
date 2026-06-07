"""Find datasets with matching ICA decompositions."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.studyfunc._study_utils import as_alleeg_list


def std_findsameica(
    ALLEEG: list[dict[str, Any]] | dict[str, Any],
    icathreshold: float = 2e-4,
) -> tuple[list[list[int]], list[int]]:
    """Group 1-based dataset indices with near-identical ICA weights*sphere."""
    datasets = as_alleeg_list(ALLEEG)
    if not datasets:
        return [], []
    subjects = [str(eeg.get("subject") or "") for eeg in datasets]
    unique_subjects = _unique(subjects)
    if len([subject for subject in unique_subjects if subject]) > 1:
        clusters: list[list[int]] = []
        for subject in unique_subjects:
            indices = [index for index, value in enumerate(subjects, start=1) if value == subject]
            subset = [datasets[index - 1] for index in indices]
            subset_clusters, _inds = _group_same_ica(subset, float(icathreshold))
            clusters.extend([[indices[item - 1] for item in cluster] for cluster in subset_clusters])
    else:
        clusters, _inds = _group_same_ica(datasets, float(icathreshold))
    cluster_indices = [0] * len(datasets)
    for cluster_index, cluster in enumerate(clusters, start=1):
        for dataset_index in cluster:
            cluster_indices[dataset_index - 1] = cluster_index
    return clusters, cluster_indices


def _group_same_ica(datasets: list[dict[str, Any]], threshold: float) -> tuple[list[list[int]], list[int]]:
    clusters: list[list[int]] = []
    cluster_indices: list[int] = []
    for index, eeg in enumerate(datasets, start=1):
        matrix = _ica_matrix(eeg)
        match_index = 0
        for cluster_index, cluster in enumerate(clusters, start=1):
            reference = _ica_matrix(datasets[cluster[0] - 1])
            if matrix is not None and reference is not None and matrix.shape == reference.shape:
                if float(np.sum(np.abs(matrix - reference))) < threshold:
                    match_index = cluster_index
                    break
        if match_index:
            clusters[match_index - 1].append(index)
            cluster_indices.append(match_index)
        else:
            clusters.append([index])
            cluster_indices.append(len(clusters))
    return clusters, cluster_indices


def _ica_matrix(eeg: dict[str, Any]) -> np.ndarray | None:
    weights = np.asarray(eeg.get("icaweights", []), dtype=float)
    sphere = np.asarray(eeg.get("icasphere", []), dtype=float)
    if weights.size == 0 or sphere.size == 0:
        return None
    if weights.ndim != 2 or sphere.ndim != 2 or weights.shape[1] != sphere.shape[0]:
        return None
    return weights @ sphere


def _unique(values: list[str]) -> list[str]:
    output: list[str] = []
    for value in values:
        if value not in output:
            output.append(value)
    return output


__all__ = ["std_findsameica"]
