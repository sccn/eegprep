import picard
import numpy as np

def eeg_picard(EEG):

    # Assuming EEG['data'] contains the EEG data as a numpy array of shape (channels, timepoints)
    data = EEG['data']
    
    # reshape from 3D to 2D
    data = data.reshape(data.shape[0], -1)
    
    # Note: Specify the number of components (optional). Default is min(timepoints, channels).
    weighting_matrix, unmixing_matrix, sources = picard(data, ortho=False, random_state=0)

    # Update EEG['icaweights'] with the separating (unmixing) matrix
    if weighting_matrix is None:
        EEG['icasphere'] = np.eye(EEG['nbchan'])
    else:
        EEG['icasphere'] = weighting_matrix
    EEG['icaweights'] = unmixing_matrix
    EEG['icawinv'] = np.linalg.pinv(EEG['icaweights']*EEG['icasphere'])  

    # Calculate the inverse weights (mixing matrix) and store in EEG['icawinv']
    EEG['icaact'] = sources

    # reshape EEG['icaact'] back to 3D as EEG['data']
    EEG['icaact'] = EEG['icaact'].reshape(EEG['icaact'].shape[0], EEG['pnts'], EEG['trials'])
    EEG['icachansind'] = np.arange(EEG['nbchan'])
    
    return EEG