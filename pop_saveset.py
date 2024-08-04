import scipy.io
import numpy as np
import os

# Allows access using . notation
# class EEG:
#     def __init__(self, **kwargs):
#         self.__dict__.update(kwargs)
#     def __getitem__(self, key):
#         return self.__dict__[key]
#     def __setitem__(self, key, value):
#         self.__dict__[key] = value

default_empty = np.array([])

def flatten_dict_sub(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict_sub(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def flatten_dict(data):
    # Flatten each dictionary and collect the fields and types
    flat_data = [flatten_dict_sub(item) for item in data]
    fields = list(flat_data[0].keys())
    dtypes = []

    # Determine data types
    for field in fields:
        sample_value = flat_data[0][field]
        if isinstance(sample_value, int):
            dtypes.append((field, np.int32))
        elif isinstance(sample_value, float):
            dtypes.append((field, np.float64))
        else:
            dtypes.append((field, 'U{}'.format(max(len(str(item[field])) for item in flat_data))))
            
    # Convert the flattened data to a list of tuples
    data_tuples = [tuple(item[field] for field in fields) for item in flat_data]

    # Create the rec.array
    dtype = np.dtype([(key, 'O') for key in data[0].keys()])
    rec_array = np.array(data_tuples, dtype=dtype).view(np.recarray)
    return rec_array
        
def pop_saveset(EEG, file_path):
    # convert Events to structured array
    # if 'event' in EEG:
    #     EEG['event'] = flatten_dict(EEG['event'])    
        
    # search for array of dictionaries and convert them to flatten_dicts
    for key in EEG:
        if isinstance(EEG[key], np.ndarray) and len(EEG[key]) > 0 and isinstance(EEG[key][0], dict):
            EEG[key] = flatten_dict(EEG[key])

    EEG['icaact'] = default_empty 
    scipy.io.savemat(file_path, EEG)
            
    return EEG

from pop_loadset import pop_loadset

def test_pop_saveset():
    file_path = './eeglab_data_with_ica_tmp.set'
    EEG = pop_loadset(file_path)
    pop_saveset(EEG, 'tmp.set')
    # print the keys of the EEG dictionary
    print(EEG.keys())
    
test_pop_saveset()

# STILL OPEN QUESTION: Better to have empty MATLAB arrays as None for empty numpy arrays (current default).
# The current default is to make it more MALTAB compatible. A lot of MATLAB function start indexing MATLAB
# empty arrays to add values to them. This is not possible with None and would create more conversion and 
# bugs. However, None is more pythonic. 