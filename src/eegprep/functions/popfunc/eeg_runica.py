import copy
import logging

import numpy as np
from ._ica_utils import (
    finalize_ica_fields,
    flatten_ica_data,
    reshape_ica_activations,
    scale_ica_to_rms_microvolt,
)
from ..miscfunc.misc import finite_matmul, finite_pinv
from ..miscfunc.pinv import pinv
from ..sigprocfunc.runica import runica


logger = logging.getLogger(__name__)

# EEGLAB pop_runica.m:475 limits the estimate to 3000 frames. getrank
# (pop_runica.m:670-684) keeps covariance eigenvalues above 1e-7 and, when
# that count disagrees with matrix rank, the smaller one.
_RUNICA_RANK_FRAME_LIMIT = 3000
_RUNICA_RANK_EIGEN_TOLERANCE = 1e-7


def _matlab_eps(value):
    # MATLAB eps(0) is eps(1); np.spacing(0) is a subnormal gap.
    if value == 0.0:
        return float(np.finfo(np.float64).eps)
    return float(np.spacing(value))


def _matlab_matrix_rank(data):
    # MATLAB rank: number of singular values above max(size(A)) * eps(norm(A)).
    singular = np.linalg.svd(data, compute_uv=False)
    if singular.size == 0:
        return 0
    tolerance = max(data.shape) * _matlab_eps(float(singular[0]))
    return int(np.count_nonzero(singular > tolerance))


def eeglab_data_rank(data, *, pca_requested):
    """Return the ``pop_runica`` data rank for ``data``."""
    sample = flatten_ica_data(np.asarray(data, dtype=np.float64))
    sample = sample[:, : min(_RUNICA_RANK_FRAME_LIMIT, sample.shape[1])]
    matrix_rank = _matlab_matrix_rank(sample)
    covariance = np.atleast_2d(np.cov(sample, rowvar=True, ddof=0))
    eigenvalues = np.linalg.eigvalsh(covariance)
    eigenvalue_rank = int(np.count_nonzero(eigenvalues > _RUNICA_RANK_EIGEN_TOLERANCE))
    if matrix_rank == eigenvalue_rank:
        return matrix_rank
    if pca_requested:
        logger.warning(
            "fixing rank computation inconsistency (%d vs %d) most likely because running under "
            "Linux 64-bit Matlab, taking the minimum",
            matrix_rank,
            eigenvalue_rank,
        )
    return min(matrix_rank, eigenvalue_rank)


def apply_runica_data_rank(data, options, *, pca_requested):
    """Return ``options`` with ``pca`` set from :func:`eeglab_data_rank` when needed."""
    options = dict(options)
    if pca_requested:
        logger.warning(
            "you have used PCA to reduce dimensionality so ICA "
            "is not modeling the entire data, only the PCA-reduced data."
        )
        eeglab_data_rank(data, pca_requested=True)
        return options
    rank = eeglab_data_rank(data, pca_requested=False)
    channel_count = flatten_ica_data(np.asarray(data, dtype=np.float64)).shape[0]
    if rank < channel_count:
        logger.info(
            "Data rank (%d) is smaller than the number of channels (%d).",
            rank,
            channel_count,
        )
        options["pca"] = int(rank)
    return options


def eeg_runica(EEG, posact='off', sortcomps='off', **kwargs):
    """
    Perform ICA decomposition using runica (infomax) algorithm.

    When ``pca`` is omitted and the estimated rank is below the channel count,
    that rank is passed to ``runica``. A direct ``runica`` call does not do this.

    When ``option_scaleicarms`` is enabled, components are scaled so each
    mixing column has RMS 1, matching EEGLAB ``eeg_checkset``.

    Parameters
    ----------
    EEG : dict
        EEGLAB-like data structure.
    posact : str | bool, optional
        If 'on' or True, normalize component signs so max(abs(activations)) is positive. Default is 'off'.
    sortcomps : str | bool, optional
        If 'on' or True, sort components by descending activation variance. Default is 'off'.
    kwargs : dict
        Additional keyword arguments to pass to the runica algorithm.

    Returns
    -------
    dict
        The updated EEG structure with ICA fields.
    """
    EEG = copy.deepcopy(EEG)

    data = flatten_ica_data(EEG['data'].astype('float64'))
    if 'pca' not in kwargs:
        kwargs = apply_runica_data_rank(data, kwargs, pca_requested=False)

    weights, sphere, compvars, bias, signs, lrates = runica(data, **kwargs)
    if not np.isfinite(weights).all() or not np.isfinite(sphere).all():
        raise ValueError("runica(): ICA decomposition produced non-finite weights or sphering matrix.")

    EEG['icasphere'] = sphere
    EEG['icaweights'] = weights
    unmixing = finite_matmul(weights, sphere)
    EEG['icawinv'] = finite_pinv(unmixing, solver=pinv)
    if not np.isfinite(EEG['icawinv']).all():
        raise ValueError("runica(): ICA decomposition produced a non-finite inverse weight matrix.")

    EEG['icaact'] = finite_matmul(unmixing, data)
    if not np.isfinite(EEG['icaact']).all():
        raise ValueError("runica(): ICA decomposition produced non-finite activations.")
    EEG['icaact'] = reshape_ica_activations(EEG['icaact'], EEG['pnts'], EEG['trials'])
    EEG['icachansind'] = np.arange(EEG['nbchan'])

    return scale_ica_to_rms_microvolt(finalize_ica_fields(EEG, sortcomps=sortcomps, posact=posact))
