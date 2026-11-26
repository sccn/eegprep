"""Mutual information reduction (MIR) of an ICA decomposition."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.miscfunc.misc import finite_matmul
from eegprep.functions.popfunc._ica_utils import flatten_ica_data
from eegprep.functions.sigprocfunc.ica_helpers import _ica_channel_indices

#: Bins of every marginal entropy histogram. Fixed, so that MIR does not change with the recording's length.
MIR_NBINS = 256


def eeg_mir(EEG: dict[str, Any]) -> tuple[float, float]:
    """Return the mutual information reduction (MIR) of the ICA decomposition, in nats.

    MIR is the mutual information that ``icaweights`` removes from the sphered data
    (Delorme et al. 2012, PLoS ONE 7(2):e30135). With ``W = icaweights``,
    ``x = icasphere @ data`` and ``y = W @ x``::

        MIR = log|det W| + sum_i h(x_i) - sum_i h(y_i)

    where ``h`` is a marginal differential entropy. Measuring from the sphered data leaves out the
    decorrelation that sphering does, so MIR rates what the unmixing adds. Identity weights give 0;
    rescaling, reordering or flipping the sign of components does not change it. Larger is better.

    Each ``h`` is the binned estimator of ``getMIR.m`` (bigdelys/pre_ICA_cleaning), as pAMICA ports it
    in ``pamica.metrics.mir``: ``MIR_NBINS`` bins between a row's minimum and maximum, the
    Miller-Madow correction and the log bin width. EEGLAB has no MIR function. getMIR's own bin count,
    ``round(3 * log2(1 + N / 10))`` for ``N`` samples, grows with the recording: on two uniform sources
    it reads MIR 0.046 nats low at 5,000 samples and 0.026 low at 1.2 million, where 256 bins read
    0.005 low at both.

    Epochs are concatenated, only the channels in ``icachansind`` are used, and ``icaact`` is not
    read: the components are recomputed from ``data``. ``icasphere`` may have fewer rows than
    channels, as after AMICA's ``pcakeep``; MIR is then measured in that subspace. ``icaweights``
    must be square.

    Args:
        EEG: Dataset with ``data`` (2-D continuous or 3-D epoched), ``icaweights``, ``icasphere``
            and ``icachansind``.

    Returns:
        MIR in nats, and the variance of that estimate as ``getMIR.m`` computes it.

    Raises:
        ValueError: If the dataset has no decomposition, ``icaweights`` is not square, the matrix
            shapes do not match ``icachansind``, ``icaweights`` or ``icasphere`` is singular, the
            data or matrices hold NaN or infinite values, or a sphered row or a component is constant.
    """
    weights = np.asarray(EEG.get("icaweights", []), dtype=np.float64)
    sphere = np.asarray(EEG.get("icasphere", []), dtype=np.float64)
    if weights.size == 0 or sphere.size == 0:
        raise ValueError("EEG has no ICA decomposition: icaweights or icasphere is empty. Run pop_runica first.")
    if weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
        raise ValueError(
            f"eeg_mir needs square icaweights, got shape {weights.shape}. A runica decomposition reduced "
            "by PCA stores the reduction in icaweights; MIR is defined only when it is in icasphere."
        )
    if sphere.ndim != 2 or sphere.shape[0] != weights.shape[1]:
        raise ValueError(f"icasphere has shape {sphere.shape}; it needs {weights.shape[1]} rows to match icaweights.")

    data = np.asarray(EEG["data"])
    channels = _ica_channel_indices(EEG, data.shape[0])
    if channels.size != sphere.shape[1]:
        raise ValueError(
            f"icachansind names {channels.size} channels and icasphere has {sphere.shape[1]} columns; "
            "they must be the channels the decomposition was computed on."
        )
    data = flatten_ica_data(data[channels])
    for name, values in (("data", data), ("icaweights", weights), ("icasphere", sphere)):
        if not np.all(np.isfinite(values)):
            raise ValueError(f"EEG {name} holds NaN or infinite values; eeg_mir needs finite values.")
    for name, matrix in (("icaweights", weights), ("icasphere", sphere)):
        if np.linalg.matrix_rank(matrix) < matrix.shape[0]:
            raise ValueError(f"{name} is singular: its rows are linearly dependent, so MIR is undefined.")

    sphered = finite_matmul(sphere, data)
    sphered_entropy, sphered_variance = _marginal_entropies(sphered, "row {} of icasphere @ data")
    component_entropy, component_variance = _marginal_entropies(finite_matmul(weights, sphered), "component {}")
    # Sum of log|eigenvalues|, not slogdet, as getMIR.m and pamica.metrics.mir compute log|det W|.
    log_jacobian = np.sum(np.log(np.abs(np.linalg.eigvals(weights))))
    mir = float(log_jacobian + np.sum(sphered_entropy) - np.sum(component_entropy))
    variance = float((np.sum(sphered_variance) + np.sum(component_variance)) / data.shape[1])
    return mir, variance


def _marginal_entropies(rows: np.ndarray, label: str) -> tuple[np.ndarray, np.ndarray]:
    """Binned differential entropy of each row, and the variance of its estimate (``getent4``)."""
    samples = rows.shape[1]
    entropies = np.empty(rows.shape[0])
    variances = np.empty(rows.shape[0])
    for index, row in enumerate(rows):
        low, high = row.min(), row.max()
        if low == high:
            raise ValueError(f"{label.format(index + 1)} is constant, so its entropy is undefined.")
        # getent4 bins with MATLAB round (half away from zero) and np.round rounds half to even; they
        # differ only on a sample exactly halfway between two bin centers.
        bins = np.round((MIR_NBINS - 1) * (row - low) / (high - low)).astype(np.intp)
        counts = np.bincount(bins, minlength=MIR_NBINS)
        probability = counts[counts > 0] / samples
        plugin = -np.sum(probability * np.log(probability))
        variances[index] = np.sum(probability * np.log(probability) ** 2) - plugin**2
        entropies[index] = plugin + (MIR_NBINS - 1) / (2 * samples) + np.log((high - low) / MIR_NBINS)
    return entropies, variances
