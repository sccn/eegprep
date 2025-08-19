import scipy.io
import numpy as np
import os
import h5py
from eegprep.pop_loadset_h5 import pop_loadset_h5
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
    from eegprep.eeg_checkset import eeg_checkset

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
                if field_name in ['tracking']:
                    # used for fields that this code can't yet parse
                    field_value = '<unsupported>'
                else:
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

    EEG['filepath'] = os.path.dirname(file_path)
    EEG['filename'] = os.path.basename(file_path)

    # delete keys '__header__', '__version__', '__globals__'
    if '__header__' in EEG:
        del EEG['__header__']
    if '__version__' in EEG:
        del EEG['__version__']
    if '__globals__' in EEG:
        del EEG['__globals__']
    
    EEG = eeg_checkset(EEG)

    # subtract 1 to EEG['icachansind'] to make it 0-based
    if 'icachansind' in EEG and EEG['icachansind'].size > 0:
        EEG['icachansind'] = EEG['icachansind'] - 1

    # check if EEG['urchan'] is 0-based
    if len(EEG['chanlocs']) > 0 and 'urchan' in EEG['chanlocs'][0]:
        for i in range(len(EEG['chanlocs'])):
            EEG['chanlocs'][i]['urchan'] = EEG['chanlocs'][i]['urchan'] - 1        

    # check if EEG['chanlocs'][i]['urevent'] is 0-based
    if len(EEG['event']) > 0 and 'urevent' in EEG['event'][0]:
        for i in range(len(EEG['event'])):
            if 'urevent' in EEG['event'][i] and EEG['event'][i]['urevent'] is not None:
                EEG['event'][i]['urevent'] = EEG['event'][i]['urevent'] - 1
    
    return EEG

def test_pop_loadset():
    file_path = './tmp2.set'
    file_path = '/System/Volumes/Data/data/data/STUDIES/STERN/S04/Memorize.set' #'./eeglab_data_with_ica_tmp.set'
    EEG = pop_loadset(file_path)
    
    # print the keys of the EEG dictionary
    print(EEG.keys())
    
if __name__ == "__main__":
    test_pop_loadset()

# STILL OPEN QUESTION: Better to have empty MATLAB arrays as None for empty numpy arrays (current default).
# The current default is to make it more MALTAB compatible. A lot of MATLAB function start indexing MATLAB
# empty arrays to add values to them. This is not possible with None and would create more conversion and 
# bugs. However, None is more pythonic. 