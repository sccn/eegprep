import numpy as np

def pop_reref(EEG, ref):
    
    # check if ref is not empty and not none
    if ref is not None and ref != []:
        raise ValueError('Feature not implemented: The ref parameter must be empty or None')
            
    # Check if the number of channels in EEG['icachansind'] is the same as the number of channels in EEG['nbchan']
    if len(EEG['icachansind']) == EEG['nbchan']:
        # Subtract mean from EEG data using broadcasting in NumPy
        EEG['data'] = EEG['data'] - np.mean(EEG['data'], axis=0)
        
        # Subtract mean from EEG icawinv using broadcasting in NumPy
        EEG['icawinv'] = EEG['icawinv'] - np.mean(EEG['icawinv'], axis=0)
        
        # Compute the pseudoinverse of EEG['icawinv']
        EEG['icaweights'] = np.linalg.pinv(EEG['icawinv'])
        EEG['icasphere'] = np.eye(EEG['nbchan'])

        # Compute the ICA activations        
        data = EEG['data'].reshape(EEG['data'].shape[0], -1)
        EEG['icaact'] = np.dot(EEG['icaweights'], data)
        EEG['icaact'] = EEG['icaact'].reshape(EEG['icaweights'].shape[0], EEG['pnts'], EEG['trials'])
                
        # Set EEG['ref'] to 'average'
        EEG['ref'] = 'average'
        
        # Update the reference for each channel in EEG['chanlocs']
        for iChan in range(len(EEG['chanlocs'])):
            EEG['chanlocs'][iChan]['ref'] = 'average'
        
        return EEG
            
    else:
        raise ValueError('Feature not implemented: The number of channels in EEG[''icachansind''] must be the same as the number of channels in EEG[''nbchan'']')