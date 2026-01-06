"""Module for performing ICA decomposition using the Picard algorithm."""

from picard import picard
import numpy as np
import os
import tempfile
from .pop_saveset import pop_saveset
from .pop_loadset import pop_loadset
from .eeglabcompat import temp_dir, MatlabWrapper
from .pinv import pinv

def eeg_picard(EEG, engine=None, posact='off', sortcomps='off', **kwargs):
    """Perform ICA decomposition using Picard algorithm.

    This function can use either a Python implementation or an EEGLAB (via MATLAB or Octave) implementation.

    Parameters
    ----------
    EEG : dict
        EEGLAB-like data structure.
    engine : object, optional
        MATLAB or Octave engine instance. If None (default), the Python implementation is used.
    posact : str | bool, optional
        If 'on' or True, normalize component signs so max(abs(activations)) is positive. Default is 'off'.
    sortcomps : str | bool, optional
        If 'on' or True, sort components by descending activation variance. Default is 'off'.
    **kwargs : dict
        Additional keyword arguments to be passed to the Picard algorithm.
        For example, `{'maxiter': 500}`.

    Returns
    -------
    dict
        The updated EEG structure with ICA fields.
    """
    if engine is None:
        # Assuming EEG['data'] contains the EEG data as a numpy array of shape (channels, timepoints)
        # Assuming EEG['data'] contains the EEG data as a numpy array of shape (channels, timepoints)
        data = EEG['data'].astype('float64')
        
        # reshape from 3D to 2D
        data = data.reshape(data.shape[0], -1)
        
        # Parameters to match MATLAB defaults, can be overriden by user kwargs
        params = {
            'ortho': False,
            'verbose': True,
            'random_state': 5489,
            'm': 10
        }
#            'w_init': np.eye(data.shape[0]),
        params.update(kwargs)
        
        weighting_matrix, unmixing_matrix, sources = picard(data, **params)

        # Update EEG['icaweights'] with the separating (unmixing) matrix
        EEG['icasphere'] = np.eye(EEG['nbchan'])
        EEG['icaweights'] = unmixing_matrix @ weighting_matrix
        # use pinv from the imported pinv
        EEG['icawinv'] = pinv(EEG['icaweights'] @ EEG['icasphere'])

        # Calculate the inverse weights (mixing matrix) and store in EEG['icawinv']
        EEG['icaact'] = sources

        # reshape EEG['icaact'] back to 3D as EEG['data']
        EEG['icaact'] = EEG['icaact'].reshape(EEG['icaact'].shape[0], EEG['pnts'], EEG['trials'])
        EEG['icachansind'] = np.arange(EEG['nbchan'])

    else:
        # Use MATLAB/Octave engine
        # (note: this is a minimalist implementation that doesn't have the
        # sorting/normalization options)
        EEG = engine.eeg_picard(EEG, **kwargs)

    # optionally sort components by mean descending activation variance
    if sortcomps in ('on', True):
        # Flatten icaact to 2D for variance computation
        icaact_2d = EEG['icaact'].reshape(EEG['icaact'].shape[0], -1)
        # Compute variance metric: sum(icawinv^2) .* sum(icaact^2)
        variance_metric = np.sum(EEG['icawinv'] ** 2, axis=0) * np.sum(
            icaact_2d ** 2, axis=1)
        # Sort indices in descending order
        windex = np.argsort(variance_metric)[::-1]
        # Reorder components
        EEG['icaact'] = EEG['icaact'][windex, :, :]
        EEG['icaweights'] = EEG['icaweights'][windex, :]
        EEG['icawinv'] = EEG['icawinv'][:, windex]

    # optionally normalize components using the same rule as runica()
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
