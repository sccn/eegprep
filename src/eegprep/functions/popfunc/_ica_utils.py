import numpy as np


def flatten_ica_data(data):
    """Flatten channel-major EEG data using EEGLAB/MATLAB epoch ordering."""
    array = np.asarray(data)
    return array.reshape(array.shape[0], -1, order="F")


def reshape_ica_activations(data, pnts, trials):
    """Reshape 2-D ICA activations back to EEGLAB's channel x point x trial form."""
    array = np.asarray(data)
    return array.reshape(array.shape[0], int(pnts), int(trials), order="F")
