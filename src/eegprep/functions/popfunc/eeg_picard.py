"""Module for performing ICA decomposition using the Picard algorithm."""

import copy

from picard import picard
import numpy as np
from ._ica_utils import finalize_ica_fields, flatten_ica_data, reshape_ica_activations
from ..miscfunc.pinv import pinv


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
    kwargs : dict
        Additional keyword arguments to pass to the Picard algorithm. For
        example, ``{"maxiter": 500}``.

    Returns
    -------
    dict
        The updated EEG structure with ICA fields.
    """
    EEG = copy.deepcopy(EEG)

    if engine is None:
        # Assuming EEG['data'] contains the EEG data as a numpy array of shape (channels, timepoints)
        data = EEG['data'].astype('float64')

        # reshape from 3D to 2D
        data = flatten_ica_data(data)

        # Parameters to match MATLAB picard defaults for reproducible parity
        # Using identity w_init ensures deterministic results matching MATLAB
        params = {
            'ortho': False,  # Use standard Picard (not Picard-O)
            'fun': 'tanh',  # Score function (matches MATLAB 'logcosh')
            'verbose': True,
            'm': 10,  # L-BFGS memory size
            'max_iter': 512,  # Match MATLAB python_defaults
            'tol': 1e-7,  # Match MATLAB python_defaults
            'centering': True,  # Center data before ICA
            'whiten': True,  # Whiten data (PCA)
            'w_init': np.eye(data.shape[0]),  # Identity init for reproducibility
        }
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
        EEG['icaact'] = reshape_ica_activations(EEG['icaact'], EEG['pnts'], EEG['trials'])
        EEG['icachansind'] = np.arange(EEG['nbchan'])

    else:
        # Use MATLAB/Octave engine
        # (note: this is a minimalist implementation that doesn't have the
        # sorting/normalization options)
        EEG = engine.eeg_picard(EEG, **kwargs)

    return finalize_ica_fields(EEG, sortcomps=sortcomps, posact=posact)
