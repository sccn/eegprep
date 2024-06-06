import scipy.io
import numpy as np

def pop_loadset(file_path):
    # Load MATLAB file
    mat_data = scipy.io.loadmat(file_path, struct_as_record=False, squeeze_me=True)
        
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

    mat_data = check_keys(mat_data)
    if 'EEG' in mat_data:
        mat_data = mat_data['EEG']
    return mat_data