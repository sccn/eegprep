"""Artifact Subspace Reconstruction (ASR) utilities."""

import logging
from typing import NamedTuple

import numpy as np
import scipy.signal

from ...functions.miscfunc.misc import finite_matmul, finite_pinv, round_mat
from .private.sigproc import moving_average

logger = logging.getLogger(__name__)

# Base memory budget (MB) when max_mem is not given; matches the maxmem=64 default of
# asr_calibrate(). When this budget cannot hold the fixed lookahead overhead, asr_process.m falls
# back to half of the free RAM; plan_blocks() uses twice the overhead instead, which is
# deterministic and needs no RAM probe (psutil is optional and has no WebAssembly build).
DEFAULT_MAX_MEM_MB = 64

# Upper limit on the number of blocks, as in asr_process.m: the block-count formula can ask for
# millions of blocks on long recordings.
MAX_BLOCKS = 10000

_REMEDY = 'Pass a larger max_mem for larger blocks (clean_artifacts: MaxMem or availableRAM_GB; clean_asr: maxmem).'


class BlockPlan(NamedTuple):
    """How asr_process splits the data into blocks to stay within a memory budget."""

    budget_mb: float  # effective budget, either the caller's max_mem or the resolved default
    overhead_mb: float  # fixed memory of the lookahead buffers, independent of the number of blocks
    budget_raised: bool  # the default budget was raised above DEFAULT_MAX_MEM_MB to fit the overhead
    splits: int  # number of blocks, at most MAX_BLOCKS
    block_samples: float
    peak_mb: float  # estimated peak memory: the overhead plus the working memory of one block
    capped: bool  # MAX_BLOCKS made the blocks larger than the budget allows


def plan_blocks(C, S, P, step_size, max_mem=None):
    """Choose the memory budget and the number of blocks for ``S`` samples of ``C`` channels.

    Args:
        C (int): Number of channels.
        S (int): Number of samples to process.
        P (int): Lookahead in samples.
        step_size (int): Samples between updates of the reconstruction matrix.
        max_mem (float, optional): Memory budget in MB. None uses ``DEFAULT_MAX_MEM_MB``, raised to
            twice the fixed lookahead overhead whenever that is larger.

    Returns:
        BlockPlan: The effective budget, the block count and the memory estimates.

    Raises:
        ValueError: If ``max_mem`` is not None and not a finite, positive number.
        RuntimeError: If an explicit ``max_mem`` cannot hold the fixed overhead.
    """
    if max_mem is not None and not (np.isfinite(max_mem) and max_mem > 0):
        raise ValueError(f'max_mem must be a finite, positive number of megabytes or None, got {max_mem!r}')

    overhead_mb = C * C * P * 8 * 3 / 1024**2
    budget_raised = False
    if max_mem is None:
        max_mem = max(DEFAULT_MAX_MEM_MB, 2 * overhead_mb)
        budget_raised = max_mem > DEFAULT_MAX_MEM_MB
    elif max_mem < overhead_mb:
        minimum_mb = int(np.ceil(overhead_mb))
        raise RuntimeError(
            f'Not enough memory: max_mem={max_mem} MB cannot hold the fixed overhead of {overhead_mb:.2f} MB '
            f'for C={C} channels and P={P} lookahead samples (C * C * P * 8 * 3 bytes). '
            f'Pass max_mem of at least {minimum_mb} MB (about {2 * minimum_mb} MB, twice that, gives usable block '
            f'sizes), or max_mem=None to let asr_process choose a budget. Through clean_artifacts set MaxMem '
            f'(or availableRAM_GB), through clean_asr set maxmem; or reduce the number of channels, the sampling '
            f'rate or the lookahead.'
        )

    # Memory formula of asr_process.m, line 138
    bytes_needed = C * C * S * 8 * 8 + C * C * 8 * S / step_size + C * S * 8 * 2 + S * 8 * 5
    mem_available = max(max_mem * 1024**2 - C * C * P * 8 * 3, 1)
    wanted = int(np.ceil(bytes_needed / mem_available))
    splits = min(wanted, MAX_BLOCKS)
    return BlockPlan(
        budget_mb=float(max_mem),
        overhead_mb=overhead_mb,
        budget_raised=budget_raised,
        splits=splits,
        block_samples=S / splits,
        peak_mb=overhead_mb + bytes_needed / splits / 1024**2,
        capped=wanted > MAX_BLOCKS,
    )


def plan_warnings(plan, C, S, P):
    """Return the warnings for a plan that goes beyond the plain default budget and block layout.

    Nothing is returned for the default budget when the blocks are comfortable.
    """
    short_blocks = plan.block_samples < P
    short_text = (
        f'Blocks are shorter than the lookahead ({P} samples), so results can differ materially '
        f'from processing in larger blocks and runtime grows. '
    )
    messages = []
    if plan.budget_raised:
        messages.append(
            f'ASR default memory budget raised from {DEFAULT_MAX_MEM_MB} to {plan.budget_mb:.0f} MB because the '
            f'fixed lookahead buffers need {plan.overhead_mb:.1f} MB for {C} channels and {P} lookahead samples. '
            f'The {S} samples will be processed in {plan.splits} blocks averaging {plan.block_samples:.1f} samples, '
            f'with peak memory about {plan.peak_mb:.0f} MB. The reconstruction matrix is refreshed per block, so the '
            f'output can differ from processing in larger blocks. ' + (short_text if short_blocks else '') + _REMEDY
        )
    elif short_blocks:
        messages.append(
            f'ASR will use {plan.splits} blocks averaging {plan.block_samples:.1f} samples at max_mem='
            f'{plan.budget_mb:g} MB. ' + short_text + _REMEDY
        )
    if plan.capped:
        messages.append(
            f'ASR is limited to {MAX_BLOCKS} blocks, so memory use will exceed the {plan.budget_mb:.0f} MB budget: '
            f'each block needs about {plan.peak_mb - plan.overhead_mb:.0f} MB plus the {plan.overhead_mb:.0f} MB '
            f'fixed overhead, about {plan.peak_mb:.0f} MB in all, for {S} samples of {C} channels. '
            f'Pass a larger max_mem (clean_artifacts: MaxMem or availableRAM_GB; clean_asr: maxmem) or clean the '
            f'recording in segments.'
        )
    return messages


def asr_process(
    data, srate, state, window_len=0.5, lookahead=None, step_size=32, max_dims=0.66, max_mem=None, use_gpu=False
):
    """Process data using the Artifact Subspace Reconstruction (ASR) method.

    CleanedData, State = asr_process(Data, SamplingRate, State, WindowLength, LookAhead, StepSize, MaxDimensions, MaxMemory, UseGPU)

    This function is used to clean multi-channel signal using the ASR method. The required inputs are
    the data matrix, the sampling rate of the data, and the filter state (as initialized by
    asr_calibrate or from the previous call to asr_process).

    Args:
        data (np.ndarray): Chunk of data to process [#channels x #samples]. Assumed to be
                           a continuation of previous data if 'state' is provided.
                           Data should be zero-mean (e.g., high-pass filtered).
        srate (float): Sampling rate of the data in Hz.
        state (dict): State dictionary from asr_calibrate or previous asr_process call.
                      Contains M, T, sos, iir_state, cov, carry, last_R, last_trivial.
        window_len (float, optional): Length of the statistics window in seconds. Should not be much
                                     longer than artifact time scale. Min samples: 1.5x channels. Default: 0.5.
        lookahead (float, optional): Look-ahead amount in seconds (causes delay). Recommended: window_len/2.
                                     Range [0, window_len/2]. Default: window_len/2.
        step_size (int, optional): Update statistics every this many samples. Larger is faster.
                                  Max: window_len * srate. Default: 32.
        max_dims (float or int, optional): Maximum dimensions/fraction of dimensions to remove.
                                         Default: 0.66 (fraction).
        max_mem (float, optional): Memory budget in MB for processing in blocks. Default: None, which
                                 uses DEFAULT_MAX_MEM_MB (64 MB), or twice the fixed lookahead overhead
                                 (C * C * P * 8 * 3 bytes, for C channels and P lookahead samples)
                                 whenever that exceeds 64 MB, that is, whenever the overhead is above
                                 32 MB, as for 128 channels at 1000 Hz. The raised default leaves room
                                 for blocks of only about 0.375 * P samples. The block layout changes
                                 the output, because the reconstruction matrix is refreshed on a
                                 per-block grid, so output can differ materially between budgets for
                                 some data: pass a larger max_mem for bigger blocks. An explicit value
                                 below the overhead raises RuntimeError; zero, negative, infinite and
                                 NaN values raise ValueError.
        use_gpu (bool, optional): Whether to use GPU (not implemented). Default: False.

    Returns
    -------
        tuple: (outdata, outstate)
            outdata (np.ndarray): Cleaned data chunk (delayed by lookahead).
            outstate (dict): Updated state dictionary for subsequent calls.

    Warnings are logged (logger ``eegprep.plugins.clean_rawdata.asr_process``) when processing goes beyond
    the plain default: when the default budget is raised above 64 MB, when the limit of ``MAX_BLOCKS``
    blocks makes memory use exceed the budget (long, high-density recordings), and when blocks are
    shorter than the lookahead, in which case results can differ materially and runtime grows. Each
    message gives the remedy, a larger ``max_mem``.
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError("Input data must be a 2D array (channels x samples).")
    C, S = data.shape

    if S == 0:
        return data, state  # Return empty data as is

    # Parameter handling
    if lookahead is None:
        lookahead = window_len / 2

    window_len = max(window_len, 1.5 * C / srate)

    if max_dims < 1:
        max_dims_num = int(round_mat(C * max_dims))
    else:
        max_dims_num = int(max_dims)

    # Number of samples in sliding window and lookahead
    N = int(round_mat(window_len * srate))
    P = int(round_mat(lookahead * srate))

    plan = plan_blocks(C, S, P, step_size, max_mem)
    for message in plan_warnings(plan, C, S, P):
        logger.warning(message)

    # Fix NaN and Inf values
    data[~np.isfinite(data)] = 0

    M = state['M']  # Mixing matrix
    T = state['T']  # Threshold matrix
    sos = state.get('sos')  # SOS filter representation (None if compatibility='max')
    b = state.get('B')  # Filter numerator coefficients
    a = state.get('A')  # Filter denominator coefficients
    compatibility = state.get('compatibility', 'standard')  # Compatibility mode
    iir_state = state.get('iir_state')  # Filter state
    carry = state.get('carry')  # Carry buffer (previous lookahead data)
    cov = state.get('cov')  # Covariance state (MovAvgState or None)
    # If cov is from an older run and is not a MovAvgState, reset it
    if cov is not None and not hasattr(cov, 'buf'):
        cov = None
    last_R = state.get('last_R')  # Last reconstruction matrix
    last_trivial = state.get('last_trivial', True)  # Was last step trivial (no artifacts)

    if carry is None:
        ind = np.mod(np.arange(P + 1, 1, -1) - 1, S)
        carry = 2 * data[:, [0]] - data[:, ind]

    # Prepend the carry buffer to the data
    X = np.concatenate((carry, data), axis=1)

    splits = plan.splits
    if splits > 1:
        logger.info(f'Cleaning data in {splits} blocks')

    for k in range(splits):
        chunk_start = int(np.floor(k * S / splits))
        chunk_end = int(min(S, np.floor((k + 1) * S / splits)))
        range_ = np.arange(chunk_start, chunk_end)

        if len(range_) == 0:
            continue

        # Get spectrally shaped data for statistics computation (range shifted by lookahead)
        Xraw = X[:, range_ + P]

        if compatibility == 'max':
            # Maximum MATLAB compatibility: use B/A form with lfilter
            Xfilt, iir_state = scipy.signal.lfilter(b, a, Xraw, axis=1, zi=iir_state)
        else:
            # Standard mode: use SOS form
            Xfilt, iir_state = scipy.signal.sosfilt(sos, Xraw, axis=1, zi=iir_state)

        # Calculate per‑sample covariance vectors and compute the running mean
        # covariance using the stateful `moving_average` implementation that
        # replicates MATLAB's `moving_average` helper. This yields a smoothed
        # covariance estimate for *every* sample and updates the internal
        # circular buffer/state stored in `cov`.
        Xcov_sample = np.reshape(np.reshape(Xfilt, (C, 1, -1)) * np.reshape(Xfilt, (1, C, -1)), (C * C, -1))

        # Running mean over a window of N samples (along the last / time axis)
        Xcov_filtered, cov = moving_average(Xcov_sample, N=N, axis=1, Z=cov, init=0)

        update_at = np.arange(step_size, Xfilt.shape[1] + step_size - 1, step_size, dtype=int)
        update_at = np.minimum(update_at, Xfilt.shape[1])

        # If there is no previous R, initialize at first sample
        if last_R is None:
            update_at = np.insert(update_at, 0, 1)
            last_R = np.eye(C)

        update_at -= 1  # prepare for 0-based indexing

        # averaged by the moving window) and reshape to C × C × #updates.
        Xcov_matrices = np.reshape(Xcov_filtered[:, update_at], (C, C, len(update_at)))

        last_n = -1  # MATLAB uses 1‑based indexing; align so first sample is included
        for j, n in enumerate(update_at):
            # Eigendecomposition to find potential artifact components
            try:
                D, V = np.linalg.eigh(Xcov_matrices[:, :, j])
                # Sort in ascending order (eigh already does this)
                # D and V are already sorted in ascending order by eigh
            except np.linalg.LinAlgError:
                # Fallback if eigendecomposition fails
                logger.warning(f"Eigendecomposition failed at update point {j}. Using identity matrix.")
                D, V = np.ones(C), np.eye(C)

            # Determine which components to keep (variance below threshold or not admissible for rejection).
            # No catch-all here: a shape/contract error in T/V must surface rather than silently
            # disable artifact removal for this window. Genuine numerical-singularity cases are
            # handled by the LinAlgError fallbacks for eigendecomposition and the pseudo-inverse.
            thresholds = np.sum(finite_matmul(T, V) ** 2, axis=0)
            keep = (D < thresholds) | (np.arange(1, C + 1) < (C - max_dims_num))
            trivial = np.all(keep)

            if not trivial:
                try:
                    # Following reference implementation:
                    # Get V[:, keep] equivalent by multiplying V by a diagonal selection matrix
                    keep_mask = keep[np.newaxis, :]  # Make column vector
                    A = finite_matmul(V.T, M)  # V.T × M
                    masked_A_T = keep_mask * A.T  # Zero out rows where keep is False
                    Q = masked_A_T.T  # Back to original orientation

                    Z = finite_pinv(Q)
                    R = np.real(finite_matmul(finite_matmul(M, Z), V.T))
                except np.linalg.LinAlgError:
                    logger.warning(f"Failed to calculate inverse at update point {j}. Using identity matrix.")
                    R = np.eye(C)
                    trivial = True
            else:
                R = np.eye(C)

            if not trivial or not last_trivial:
                subrange = range(last_n + 1, n + 1)
                if len(subrange) > 0:
                    # Calculate blend coefficients (raised cosine)
                    blend = (1 - np.cos(np.pi * np.arange(1, len(subrange) + 1) / len(subrange))) / 2

                    # Extract data segment to process (from extended data X)
                    idx_in_X = range_[subrange]
                    segment = X[:, idx_in_X]

                    X[:, idx_in_X] = blend * finite_matmul(R, segment) + (1 - blend) * finite_matmul(last_R, segment)

            last_n = n
            last_R = R
            last_trivial = trivial

        if splits > 1 and k % 10 == 0:
            logger.debug(f'Processing block {k + 1}/{splits}')

    if splits > 1:
        logger.info('Finished cleaning.')

    # Update the carry buffer for next call (last P samples)
    new_carry = X[:, -P:] if X.shape[1] >= P else X

    # Return cleaned data (without the lookahead portion)
    outdata = X[:, P : P + S]

    outstate = {
        'M': M,
        'T': T,
        'sos': sos,
        'iir_state': iir_state,
        'cov': cov,
        'carry': new_carry,
        'last_R': last_R,
        'last_trivial': last_trivial,
        # Include original filter coefficients and compatibility mode
        'B': b,
        'A': a,
        'compatibility': compatibility,
        'useriemannian': state.get('useriemannian'),
    }

    return outdata, outstate
