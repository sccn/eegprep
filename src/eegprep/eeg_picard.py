from picard import picard
import numpy as np
import os
import tempfile
from .pop_saveset import pop_saveset
from .pop_loadset import pop_loadset
from .eeglabcompat import temp_dir, MatlabWrapper
from .pinv import pinv

def eeg_picard(EEG, engine=None, **kwargs):
    """
    Perform ICA decomposition using Picard algorithm.

    This function can use either a Python implementation or an EEGLAB (via MATLAB or Octave) implementation.

    Parameters
    ----------
    EEG : dict
        EEGLAB-like data structure.
    engine : object, optional
        MATLAB or Octave engine instance. If None (default), the Python implementation is used.
    **kwargs : dict
        Additional keyword arguments to be passed to EEGLAB's pop_runica function.
        For example, `{'maxiter': 500}` would be passed as `'maxiter', 500`.

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
        EEG['icawinv'] = np.linalg.pinv(EEG['icaweights']*EEG['icasphere'])  

        # Calculate the inverse weights (mixing matrix) and store in EEG['icawinv']
        EEG['icaact'] = sources

        # reshape EEG['icaact'] back to 3D as EEG['data']
        EEG['icaact'] = EEG['icaact'].reshape(EEG['icaact'].shape[0], EEG['pnts'], EEG['trials'])
        EEG['icachansind'] = np.arange(EEG['nbchan'])
        
        return EEG
    else:
        # Use MATLAB/Octave engine
        
        # Prepare arguments for pop_runica
        args_list = []
        kwargs['icatype'] = 'picard'
        for k, v in kwargs.items():
            args_list.append(k)
            args_list.append(v)
                    
        return engine.eeg_picard(EEG)
    
    