"""Utilities for saving data structures to HDF5."""

import numpy as np
import h5py

def save_dict_to_hdf5(data, filename, dataset_name):
    """
    Save a dictionary to an HDF5 file as a structured dataset.

    Parameters
    ----------
    data : dict
        Dictionary to save.
    filename : str
        Path to the HDF5 file.
    dataset_name : str
        Name of the dataset in the HDF5 file.
    """
    # Create a structured dtype from the dictionary keys and their corresponding types
    dtype = []
    for key, value in data.items():
        if isinstance(value, str):
            dtype.append((key, 'S{}'.format(len(value))))
        elif isinstance(value, int):
            dtype.append((key, 'i4'))
        elif isinstance(value, float):
            dtype.append((key, 'f8'))
        elif isinstance(value, np.ndarray):
            dtype.append((key, 'f8', value.shape))            
        elif value is None:
            dtype.append((key, 'S4'))
        else:
            raise TypeError(f"Unsupported data type for key {key}: {type(value)}")

    structured_dtype = np.dtype(dtype)

    # Convert dictionary values to a structured array
    structured_data = np.array([tuple(
        value.encode('utf-8') if isinstance(value, str) else 
        str(value).encode('utf-8') if value is None else 
        value for value in data.values()
    )], dtype=structured_dtype)

    # Save to HDF5
    with h5py.File(filename, 'w') as hdf:
        hdf.create_dataset(dataset_name, data=structured_data)

# Example usage
data = {
    'labels': 'FPz',
    'theta': np.array([0,1,2,3]),
    'radius': 0.5066888888888889,
    'X': 84.98123361344625,
    'Y': 0,
    'Z': -1.7860385037488253,
    'sph_theta': 0,
    'sph_phi': -1.203999999999994,
    'sph_radius': 85,
    'type': 'EEG',
    'urchan': 1,
    'ref': None
}

save_dict_to_hdf5(data, 'data.h5', 'dataset_name')