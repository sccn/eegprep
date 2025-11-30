from .runica import runica
import numpy as np
from .pinv import pinv


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
    **kwargs : dict
        Additional keyword arguments to be passed to the runica algorithm.

    Returns
    -------
    dict
        The updated EEG structure with ICA fields.
    """
    # Extract data and reshape from 3D to 2D
    data = EEG['data'].astype('float64')
    data = data.reshape(data.shape[0], -1)

    # Run runica
    weights, sphere, compvars, bias, signs, lrates = runica(data, **kwargs)

    # Update EEG structure with ICA results
    EEG['icasphere'] = sphere
    EEG['icaweights'] = weights
    EEG['icawinv'] = pinv(weights @ sphere)

    # Compute ICA activations
    EEG['icaact'] = (weights @ sphere) @ data
    # Reshape icaact back to 3D
    EEG['icaact'] = EEG['icaact'].reshape(EEG['icaact'].shape[0], EEG['pnts'], EEG['trials'])
    EEG['icachansind'] = np.arange(EEG['nbchan'])

    # Optionally sort components by mean descending activation variance
    if sortcomps in ('on', True):
        # Flatten icaact to 2D for variance computation
        icaact_2d = EEG['icaact'].reshape(EEG['icaact'].shape[0], -1)
        # Compute variance metric: sum(icawinv^2) .* sum(icaact^2)
        variance_metric = np.sum(EEG['icawinv'] ** 2, axis=0) * np.sum(icaact_2d ** 2, axis=1)
        # Sort indices in descending order
        windex = np.argsort(variance_metric)[::-1]
        # Reorder components
        EEG['icaact'] = EEG['icaact'][windex, :, :]
        EEG['icaweights'] = EEG['icaweights'][windex, :]
        EEG['icawinv'] = EEG['icawinv'][:, windex]

    # Optionally normalize components using the same rule as runica()
    if posact in ('on', True):
        # Flatten icaact to 2D for finding max abs values
        icaact_2d = EEG['icaact'].reshape(EEG['icaact'].shape[0], -1)
        # Find indices of max absolute values for each component
        ix = np.argmax(np.abs(icaact_2d), axis=1)
        had_flips = False
        ncomps = EEG['icaact'].shape[0]

        for r in range(ncomps):
            if np.sign(icaact_2d[r, ix[r]]) < 0:
                # Flip the activations
                EEG['icaact'][r, :, :] = -EEG['icaact'][r, :, :]
                # Flip the corresponding column of the mixing matrix
                EEG['icawinv'][:, r] = -EEG['icawinv'][:, r]
                had_flips = True

        if had_flips:
            # Recompute unmixing matrix
            EEG['icaweights'] = pinv(EEG['icawinv'])

    return EEG
