import scipy.io
import numpy as np
import os
import h5py
from .pop_loadset_h5 import pop_loadset_h5
from .eeg_checkset import eeg_checkset
# Allows access using . notation
# class EEG:
#     def __init__(self, **kwargs):
#         self.__dict__.update(kwargs)
#     def __getitem__(self, key):
#         return self.__dict__[key]
#     def __setitem__(self, key, value):
#         self.__dict__[key] = value

default_empty = np.array([])
#default_empty = None

def loadset(file_path):
    return pop_loadset(file_path)

def pop_loadset(file_path=None):
    if file_path is None:
        raise ValueError("file_path argument is required")
    
    def new_check(obj):
        # check if obj is a dictionary and apply recursively the function to each object not changing the struture of the dictionary
        if isinstance(obj, dict):
            return {key: new_check(obj[key]) for key in obj}
        # check if obj is a numpy array and apply recursively the function to each object not changing the struture of the array
        elif isinstance(obj, list):
            if len(obj) == 0:
                return default_empty
            else:
                return [new_check(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            # check if empty and return none
            if obj.size == 0:
                return default_empty
            # check if it is a numeric array
            elif obj.dtype.kind in ['i', 'u', 'f', 'c']:
                return obj
            else:
                return np.array([new_check(item) for item in obj], dtype=object)
        # check if it is a scalar or a string and return it
        elif np.isscalar(obj) or isinstance(obj, str):
            return obj
        # check if obj is a mat_struct object and convert it to a dictionary
        elif isinstance(obj, scipy.io.matlab.mat_struct) or isinstance(obj, scipy.io.matlab.mio5_params.mat_struct):
            dict_obj = {}
            for field_name in obj._fieldnames:
                field_value = getattr(obj, field_name)
                dict_obj[field_name] = new_check(field_value)
            return dict_obj

    # Load MATLAB file
    print(file_path)  # This will show us the actual path being used
    try:
        EEG = scipy.io.loadmat(file_path, struct_as_record=False, squeeze_me=True, appendmat=False)
        EEG = new_check(EEG)
        if 'EEG' in EEG:
            EEG = EEG['EEG']
    except Exception as e:
        EEG = pop_loadset_h5(file_path)    

    EEG = eeg_checkset(EEG)
    
    return EEG

def test_pop_loadset():
    file_path = './tmp2.set'
    file_path = '/System/Volumes/Data/data/data/STUDIES/STERN/S04/Memorize.set' #'./eeglab_data_with_ica_tmp.set'
    EEG = pop_loadset(file_path)
    
    # print the keys of the EEG dictionary
    print(EEG.keys())
    
# test_pop_loadset()

# STILL OPEN QUESTION: Better to have empty MATLAB arrays as None for empty numpy arrays (current default).
# The current default is to make it more MALTAB compatible. A lot of MATLAB function start indexing MATLAB
# empty arrays to add values to them. This is not possible with None and would create more conversion and 
# bugs. However, None is more pythonic. 