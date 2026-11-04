"""Riemannian ASR processing, port of asr_process_r.m."""

import logging

import numpy as np
import psutil
import scipy.signal

from ...functions.miscfunc.misc import finite_matmul, round_mat
from .private.riemann import asr_subspace_reconstruction, rasr_nonlinear_eigenspace, spd_geodesic_midpoint
from .private.windows import asr_update_at

logger = logging.getLogger(__name__)


def asr_process_r(
    data,
    srate,
    state,
    window_len=0.1,
    lookahead=None,
    step_size=4,
    max_dims=1.0,
    max_mem=None,
    use_gpu=False,
):
    """Clean a chunk with Riemannian Artifact Subspace Reconstruction.

    CleanedData, State = asr_process_r(Data, SamplingRate, State, ...)

    Unlike Euclidean ``asr_process``, each chunk uses one sample covariance.
    When ``state['cov']`` already holds a covariance from a previous chunk, the
    two matrices are replaced by their affine-invariant geodesic midpoint.
    The reconstruction subspace is the nonlinear eigenspace of that matrix.
    ``use_gpu`` is accepted for call compatibility and is not used.

    Args:
        data: Channels-by-samples chunk. Zero-mean, as for calibration.
        srate: Sampling rate in Hz.
        state: State from ``asr_calibrate_r`` or a previous ``asr_process_r`` call.
        window_len: Statistics window length in seconds. Default 0.1, and at
            least ``1.5 * channels / srate``.
        lookahead: Look-ahead in seconds. Default half of ``window_len``.
        step_size: Blend the reconstruction every this many samples. Default 4.
        max_dims: Maximum number or fraction of dimensions to reconstruct.
            Default 1.
        max_mem: Processing memory budget in MB. Default half of free memory.
        use_gpu: Ignored.

    Returns:
        tuple: ``(outdata, outstate)``. ``outdata`` is delayed by ``lookahead``
        samples and has the same length as ``data``.
    """
    if use_gpu:
        logger.debug('use_gpu is ignored; Riemannian ASR runs on CPU.')
    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError('Input data must be a 2D array (channels x samples).')
    channels, samples = data.shape
    if samples == 0:
        return data, state

    srate = float(srate)
    window_len = max(window_len, 1.5 * channels / srate)
    if lookahead is None:
        lookahead = window_len / 2
    if max_mem is None:
        max_mem = psutil.virtual_memory().free / 1024**2 / 2
    if max_dims < 1:
        max_dims_num = int(round_mat(channels * max_dims))
    else:
        max_dims_num = int(max_dims)

    lookahead_samples = int(round_mat(lookahead * srate))

    mixing = state['M']
    thresholds = state['T']
    numerator = state['B']
    denominator = state['A']
    iir_state = state.get('iir')
    if iir_state is None:
        iir_state = np.zeros((channels, max(len(denominator), len(numerator)) - 1))
    carry = state.get('carry')
    covariance = state.get('cov')
    last_R = state.get('last_R')
    last_trivial = state.get('last_trivial', True)

    if carry is None:
        ind = np.mod(np.arange(lookahead_samples + 1, 1, -1) - 1, samples)
        carry = 2 * data[:, [0]] - data[:, ind]
    extended = np.concatenate((carry, data), axis=1)
    extended[~np.isfinite(extended)] = 0

    if max_mem * 1024 * 1024 - channels * channels * lookahead_samples * 8 * 3 < 0:
        logger.warning(
            'Memory too low, increasing it (rejection block size now '
            'depends on available memory so it might not be fully reproducible)...'
        )
        max_mem = psutil.virtual_memory().free / 1024**2 / 2
        if max_mem * 1024 * 1024 - channels * channels * lookahead_samples * 8 * 3 < 0:
            raise RuntimeError('Not enough memory')

    bytes_needed = (
        channels * channels * samples * 8 * 8
        + channels * channels * 8 * samples / step_size
        + channels * samples * 8 * 2
        + samples * 8 * 5
    )
    mem_available = max(max_mem * 1024**2 - channels * channels * lookahead_samples * 8 * 3, 1)
    splits = min(int(np.ceil(bytes_needed / mem_available)), 10000)
    if splits > 1:
        logger.info(f'Cleaning data in {splits} blocks')

    chunk_covariance = covariance
    for split in range(splits):
        chunk_start = int(np.floor(split * samples / splits))
        chunk_end = int(min(samples, np.floor((split + 1) * samples / splits)))
        sample_range = np.arange(chunk_start, chunk_end)
        if len(sample_range) == 0:
            continue

        filtered, iir_state = scipy.signal.lfilter(
            numerator, denominator, extended[:, sample_range + lookahead_samples], axis=1, zi=iir_state
        )
        # asr_process_r.m divides by the full chunk length, including samples
        # that belong to other memory splits.
        sample_covariance = (filtered @ filtered.T) / samples
        chunk_covariance = (
            sample_covariance if covariance is None else spd_geodesic_midpoint(sample_covariance, covariance)
        )

        update_at = asr_update_at(filtered.shape[1], step_size)
        if last_R is None:
            update_at = np.insert(update_at, 0, 1)
            last_R = np.eye(channels)
        update_at = update_at - 1

        vectors, values = rasr_nonlinear_eigenspace(chunk_covariance, channels)
        eigenvalues = np.real(np.diag(values))
        order = np.argsort(eigenvalues, kind='mergesort')
        component_variance = eigenvalues[order]
        basis = np.real(vectors[:, order])
        directional = np.sum(finite_matmul(thresholds, basis) ** 2, axis=0)
        keep = (component_variance < directional) | (np.arange(1, channels + 1) < (channels - max_dims_num))
        trivial = bool(np.all(keep))
        if not trivial:
            try:
                reconstruction = asr_subspace_reconstruction(mixing, basis, keep)
            except np.linalg.LinAlgError:
                logger.warning(f'Failed to calculate inverse at split {split}. Using identity matrix.')
                reconstruction = np.eye(channels)
                trivial = True
        else:
            reconstruction = np.eye(channels)

        last_n = -1
        for n in update_at:
            if not trivial or not last_trivial:
                subrange = range(last_n + 1, int(n) + 1)
                if len(subrange) > 0:
                    blend = (1 - np.cos(np.pi * np.arange(1, len(subrange) + 1) / len(subrange))) / 2
                    idx_in_x = sample_range[subrange]
                    segment = extended[:, idx_in_x]
                    extended[:, idx_in_x] = blend * finite_matmul(reconstruction, segment) + (
                        1 - blend
                    ) * finite_matmul(last_R, segment)
            last_n = int(n)
            last_R = reconstruction
            last_trivial = trivial

        if splits > 1 and split % 10 == 0:
            logger.debug(f'Processing block {split + 1}/{splits}')

    if splits > 1:
        logger.info('Finished cleaning.')

    if lookahead_samples:
        new_carry = extended[:, -lookahead_samples:] if extended.shape[1] >= lookahead_samples else extended
    else:
        new_carry = extended[:, :0]
    outdata = extended[:, lookahead_samples : lookahead_samples + samples]
    outstate = {
        'M': mixing,
        'T': thresholds,
        'B': numerator,
        'A': denominator,
        'iir': iir_state,
        'cov': chunk_covariance,
        'carry': new_carry,
        'last_R': last_R,
        'last_trivial': last_trivial,
    }
    return outdata, outstate
