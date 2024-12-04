from .ICL_feature_extractor import ICL_feature_extractor
from .iclabel_net import ICLabelNet
import torch
import numpy as np

def iclabel(EEG, algorithm='default'):
    # issue error if algorithm is not 'default'
    if algorithm != 'default':
        raise ValueError('Only the default algorithm is supported')
    
    #ICLABEL Extract ICLabel features from an EEG dataset.
    features = ICL_feature_extractor(EEG, True)
    
    # Equivalent of MATLAB code reshaping
    features[0] = np.single(np.concatenate([features[0],-features[0],features[0][:, ::-1, :, :],-features[0][:, ::-1, :, :]], axis=3))
    features[1] = np.single(np.tile(features[1], (1, 1, 1, 4)))
    features[2] = np.single(np.tile(features[2], (1, 1, 1, 4)))
    print('Feature 0 shape:', features[0].shape)
    print('Feature 1 shape:', features[1].shape)
    print('Feature 2 shape:', features[2].shape)

    # Load the ICLabelNet model
    model = ICLabelNet('netICL.mat')
    
    # Convert the features to torch tensors
    image = torch.tensor(features[0]).permute(-1, 2, 0, 1)
    psdmed = torch.tensor(features[1]).permute(-1, 2, 0, 1)
    autocorr = torch.tensor(features[2]).permute(-1, 2, 0, 1)
    
    # Get the output from the model
    output = model(image, psdmed, autocorr)
    output_np = output.detach().numpy()
    output_np = output_np.T  # Transpose the array
    output_np = np.reshape(output_np, (-1, 4), order='F')  # Reshape to have 4 columns
    output_np = np.mean(output_np, axis=1)  # Compute the mean along the second axis (columns)
    output_np = np.reshape(output_np, (7, -1), order='F')  # Reshape to have 7 rows
    output_np = output_np.T  # Transpose back
    
    if 'ic_classification' not in EEG['etc']:
        EEG['etc']['ic_classification'] = {}
    if 'ICLabel' not in EEG['etc']['ic_classification']:
        EEG['etc']['ic_classification']['ICLabel'] = {}
        
    EEG['etc']['ic_classification']['ICLabel']['classes'] = np.array(
        ['Brain', 'Muscle', 'Eye', 'Heart', 'Line Noise', 'Channel Noise', 'Other'], 
        dtype=np.object
    )
    EEG['etc']['ic_classification']['ICLabel']['classifications'] = output_np
    EEG['etc']['ic_classification']['ICLabel']['version'] = 'default'
    
    return EEG