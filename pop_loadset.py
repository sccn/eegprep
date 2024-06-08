import scipy.io
import numpy as np
import os

def pop_loadset(file_path):
    # Load MATLAB file
    EEG = scipy.io.loadmat(file_path, struct_as_record=False, squeeze_me=True)
        
    def check_keys(dict_data):
        """
        Check if entries in dictionary are mat-objects. If yes,
        _to_dict is called to change them to dictionaries.
        Recursively go through the entire structure.
        """
        for key in dict_data:
            if isinstance(dict_data[key], scipy.io.matlab.mat_struct):
                dict_data[key] = to_dict(dict_data[key])
            elif isinstance(dict_data[key], dict):
                dict_data[key] = check_keys(dict_data[key])
            elif isinstance(dict_data[key], np.ndarray) and dict_data[key].dtype == object:
                dict_data[key] = np.array([check_keys({i: item})[i] if isinstance(item, dict) else item for i, item in enumerate(dict_data[key])], dtype=object)
                if dict_data[key].size == 0:
                    dict_data[key] = None
        return dict_data

    def to_dict(matobj):
        """
        A recursive function which constructs from matobjects nested dictionaries.
        """
        dict_data = {}
        for strg in matobj._fieldnames:
            elem = getattr(matobj, strg)
            if isinstance(elem, scipy.io.matlab.mat_struct):
                dict_data[strg] = to_dict(elem)
            elif isinstance(elem, np.ndarray) and elem.dtype == object:
                dict_data[strg] = np.array([to_dict(sub_elem) if isinstance(sub_elem, scipy.io.matlab.mat_struct) else sub_elem for sub_elem in elem], dtype=object)
                if dict_data[strg].size == 0:
                    dict_data[strg] = None
            else:
                dict_data[strg] = elem
        # check if contains empty arrays
        for key in dict_data:
            if isinstance(dict_data[key], np.ndarray) and dict_data[key].size == 0:
                dict_data[key] = np.array([])
                
        return dict_data

    # check if EEG['data'] is a string, and if it the case, read the binary float32 file
    EEG = check_keys(EEG)
    if 'EEG' in EEG:
        EEG = EEG['EEG']
        
    if isinstance(EEG['data'], str):
        file_name = EEG['filepath'] + os.sep + EEG['data']
        EEG['data'] = np.fromfile(file_name, dtype='float32').reshape( EEG['pnts']*EEG['trials'], EEG['nbchan'])
        EEG['data'] = EEG['data'].T.reshape(EEG['nbchan'], EEG['trials'], EEG['pnts']).transpose(0, 2, 1)

    # compute ICA activations
    if 'icaweights' in EEG and 'icasphere' in EEG:
        EEG['icaact'] = np.dot(np.dot(EEG['icaweights'], EEG['icasphere']), EEG['data'].reshape(EEG['nbchan'], -1))
        EEG['icaact'] = EEG['icaact'].reshape(EEG['icaweights'].shape[0], -1, EEG['trials'])
            
    return EEG