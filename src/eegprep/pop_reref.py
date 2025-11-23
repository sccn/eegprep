"""EEG data re-referencing functions."""

from copy import deepcopy
import logging

import numpy as np

logger = logging.getLogger(__name__)

def pop_reref(EEG, ref):
    """Re-reference EEG data to average reference.

    Parameters
    ----------
    EEG : dict
        EEG data structure
    ref : list or None
        Reference channels (must be empty or None for average reference)

    Returns
    -------
    EEG : dict
        Re-referenced EEG data structure
    """
    EEG = deepcopy(EEG)

    # check if ref is not empty and not none
    if ref is not None and ref != []:
        raise ValueError('Feature not implemented: The ref parameter must be empty or None')

    # Subtract mean from EEG data using broadcasting in NumPy
    EEG['data'] = EEG['data'] - np.mean(EEG['data'], axis=0)

    # Check if the number of channels in EEG['icachansind'] is the same as the number of channels in EEG['nbchan']
    if len(EEG['icachansind']) and (len(EEG['icachansind']) != EEG['nbchan']):
        logger.error('Feature not implemented: The number of channels in EEG[''icachansind''] '
                     'must be the same as the number of channels in EEG[''nbchan'']. Clearing ICA fields.')
        EEG['icawinv'] = np.array([])
        EEG['icaweights'] = np.array([])
        EEG['icasphere'] = np.array([])
        EEG['icaact'] = np.array([])
    elif len(EEG['icachansind']):
        #print(np.array2string(EEG['icaweights'][:10, :10], precision=3, suppress_small=True))

        # Subtract mean from EEG icawinv using broadcasting in NumPy
        EEG['icawinv'] = EEG['icawinv'] - np.mean(EEG['icawinv'], axis=0)
        
        # show the first 10 rows and columns of EEG['icawinv']
        # print it pretty line by line with 4 digit after the decimal point
        
        # Compute the pseudoinverse of EEG['icawinv']
        EEG['icaweights'] = np.linalg.pinv(EEG['icawinv'])
        #print(' ')
        #print(np.array2string(EEG['icaweights'][:10, :10], precision=3, suppress_small=True))

        EEG['icasphere'] = np.eye(EEG['nbchan'])

        # Compute the ICA activations        
        # data = EEG['data'].reshape(EEG['data'].shape[0], -1)
        # EEG['icaact'] = np.dot(EEG['icaweights'], data)
        # EEG['icaact'] = EEG['icaact'].reshape(EEG['icaweights'].shape[0], EEG['pnts'], EEG['trials'])
                
        # Set EEG['ref'] to 'average'
        EEG['ref'] = 'average'
        
        # Update the reference for each channel in EEG['chanlocs']
        for iChan in range(len(EEG['chanlocs'])):
            EEG['chanlocs'][iChan]['ref'] = 'average'
        
    return EEG
