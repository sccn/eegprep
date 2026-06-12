import copy

import numpy as np
from ._ica_utils import finalize_ica_fields, flatten_ica_data, reshape_ica_activations
from ..miscfunc.misc import finite_matmul, finite_pinv
from ..miscfunc.pinv import pinv
from ..sigprocfunc.runica import runica


def eeg_runica(EEG, posact='off', sortcomps='off', **kwargs):
    """
    Perform ICA decomposition using runica (infomax) algorithm.

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

    # Extract data and reshape from 3D to 2D
    data = flatten_ica_data(EEG['data'].astype('float64'))

    # Run runica
    weights, sphere, compvars, bias, signs, lrates = runica(data, **kwargs)
    if not np.isfinite(weights).all() or not np.isfinite(sphere).all():
        raise ValueError("runica(): ICA decomposition produced non-finite weights or sphering matrix.")

    # Update EEG structure with ICA results
    EEG['icasphere'] = sphere
    EEG['icaweights'] = weights
    unmixing = finite_matmul(weights, sphere)
    EEG['icawinv'] = finite_pinv(unmixing, solver=pinv)
    if not np.isfinite(EEG['icawinv']).all():
        raise ValueError("runica(): ICA decomposition produced a non-finite inverse weight matrix.")

    # Compute ICA activations
    EEG['icaact'] = finite_matmul(unmixing, data)
    if not np.isfinite(EEG['icaact']).all():
        raise ValueError("runica(): ICA decomposition produced non-finite activations.")
    # Reshape icaact back to 3D
    EEG['icaact'] = reshape_ica_activations(EEG['icaact'], EEG['pnts'], EEG['trials'])
    EEG['icachansind'] = np.arange(EEG['nbchan'])

    return finalize_ica_fields(EEG, sortcomps=sortcomps, posact=posact)
