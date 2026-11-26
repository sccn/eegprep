"""Mutual information reduction (MIR) of an ICA decomposition."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np

from eegprep.functions.miscfunc.misc import finite_matmul
from eegprep.functions.sigprocfunc.ica_helpers import _ica_channel_indices

#: Bins of every marginal entropy histogram. Fixed, so that MIR does not change with the recording's length.
MIR_NBINS = 256
#: Samples per block. The data are read in blocks, twice, so the working memory is a few arrays of
#: ``2 * components`` rows by this many samples, whatever the recording's length.
_BLOCK_SAMPLES = 8192


def eeg_mir(EEG: dict[str, Any]) -> tuple[float, float]:
    """Return the mutual information reduction (MIR) of the ICA decomposition, in nats.

    MIR is the mutual information that ``icaweights`` removes from the sphered data. With
    ``W = icaweights``, ``x = icasphere @ data`` and ``y = W @ x``::

        MIR = log|det W| + sum_i h(x_i) - sum_i h(y_i)

    where ``h`` is a marginal differential entropy. Identity weights give 0; rescaling, reordering or
    flipping the sign of components does not change it. Larger is better. This is equation 17 of
    Delorme et al. (2012, PLoS ONE 7(2):e30135) with ``x`` taken after sphering, so MIR rates what the
    unmixing adds to the decorrelation that sphering does. The paper measures MIR from the recorded
    channels with the full unmixing and reports it in kbits/s, and the commented-out ``mir`` of NEMAR's
    ``eeg_nemar_dataqual.m`` passes the channel data with ``icaweights @ icasphere``. Their values include
    the reduction that sphering achieves, so they are not comparable with these.

    Each ``h`` is the binned estimator of ``getMIR.m`` by Nima Bigdely-Shamlo (bigdelys/pre_ICA_cleaning),
    as pAMICA ports it in ``pamica.metrics.mir``: ``MIR_NBINS`` bins between a row's minimum and
    maximum, the Miller-Madow correction and the log bin width. EEGLAB has no MIR function. getMIR's
    own bin count, ``round(3 * log2(1 + N / 10))`` for ``N`` samples, grows with the recording: on two
    uniform sources it reads MIR 0.046 nats low at 5,000 samples and 0.026 low at 1.2 million, where
    256 bins read 0.005 low at both. This function is a modified port of ``getMIR.m``, which is under
    the Apache License 2.0; the attribution and the license text are in ``THIRD_PARTY_NOTICES.md``.

    Epochs are concatenated, only the channels in ``icachansind`` are used, and ``icaact`` is not
    read: the components are recomputed from ``data``, a block of samples at a time. ``icasphere``
    may have fewer rows than channels, as after AMICA's ``pcakeep``; MIR is then measured in that
    subspace. ``icaweights`` must be square.

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
    for name, matrix in (("icaweights", weights), ("icasphere", sphere)):
        if not np.all(np.isfinite(matrix)):
            raise ValueError(f"EEG {name} holds NaN or infinite values; eeg_mir needs finite values.")
        if np.linalg.matrix_rank(matrix) < matrix.shape[0]:
            raise ValueError(f"{name} is singular: its rows are linearly dependent, so MIR is undefined.")

    data = np.asarray(EEG["data"])
    channels = _ica_channel_indices(EEG, data.shape[0])
    if channels.size != sphere.shape[1]:
        raise ValueError(
            f"icachansind names {channels.size} channels and icasphere has {sphere.shape[1]} columns; "
            "they must be the channels the decomposition was computed on."
        )
    samples = data[0].size
    selection = slice(None) if np.array_equal(channels, np.arange(data.shape[0])) else channels
    components = weights.shape[0]

    # First pass: the range of each row of x = icasphere @ data (rows 0..n-1) and of y = W @ x (rows n..2n-1).
    low = np.full(2 * components, np.inf)
    high = np.full(2 * components, -np.inf)
    for block in _projected_blocks(data, selection, sphere, weights):
        np.minimum(low, block.min(axis=1), out=low)
        np.maximum(high, block.max(axis=1), out=high)
    for index in np.flatnonzero(low == high):
        name = f"row {index + 1} of icasphere @ data" if index < components else f"component {index - components + 1}"
        raise ValueError(f"{name} is constant, so its entropy is undefined.")

    # Second pass: count each row's samples in its bins.
    counts = np.zeros((2 * components, MIR_NBINS), dtype=np.int64)
    offsets = np.arange(2 * components)[:, np.newaxis] * MIR_NBINS
    for block in _projected_blocks(data, selection, sphere, weights):
        # getent4 bins with MATLAB round (half away from zero) and np.round rounds half to even; they
        # differ only on a sample exactly halfway between two bin centers.
        bins = np.round((MIR_NBINS - 1) * (block - low[:, np.newaxis]) / (high - low)[:, np.newaxis]).astype(np.intp)
        counts += np.bincount((bins + offsets).ravel(), minlength=counts.size).reshape(counts.shape)

    entropies, variances = _binned_entropies(counts, high - low, samples)
    # Sum of log|eigenvalues|, not slogdet, as getMIR.m and pamica.metrics.mir compute log|det W|.
    log_jacobian = np.sum(np.log(np.abs(np.linalg.eigvals(weights))))
    mir = float(log_jacobian + np.sum(entropies[:components]) - np.sum(entropies[components:]))
    variance = float((np.sum(variances[:components]) + np.sum(variances[components:])) / samples)
    return mir, variance


def _projected_blocks(
    data: np.ndarray, channels: np.ndarray | slice, sphere: np.ndarray, weights: np.ndarray
) -> Iterator[np.ndarray]:
    """Yield ``[x; W @ x]`` for each block of samples, after checking that block's data are finite."""
    epochs = data.reshape(data.shape[0], data.shape[1], -1)
    trials_per_block = max(1, _BLOCK_SAMPLES // epochs.shape[1])
    for first in range(0, epochs.shape[2], trials_per_block):
        for start in range(0, epochs.shape[1], _BLOCK_SAMPLES):
            block = epochs[channels, start : start + _BLOCK_SAMPLES, first : first + trials_per_block]
            block = block.reshape(block.shape[0], -1)
            if not np.all(np.isfinite(block)):
                raise ValueError("EEG data holds NaN or infinite values; eeg_mir needs finite values.")
            sphered = finite_matmul(sphere, block)
            yield np.vstack((sphered, finite_matmul(weights, sphered)))


def _binned_entropies(counts: np.ndarray, widths: np.ndarray, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Differential entropy of each histogram row, and the variance of its estimate (``getent4``)."""
    entropies = np.empty(counts.shape[0])
    variances = np.empty(counts.shape[0])
    for index, row in enumerate(counts):
        probability = row[row > 0] / samples
        plugin = -np.sum(probability * np.log(probability))
        variances[index] = np.sum(probability * np.log(probability) ** 2) - plugin**2
        entropies[index] = plugin + (MIR_NBINS - 1) / (2 * samples) + np.log(widths[index] / MIR_NBINS)
    return entropies, variances
