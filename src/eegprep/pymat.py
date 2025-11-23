"""Python-MATLAB data conversion utilities."""

from typing import *

import numpy as np
import scipy.io

# These are conversions from Python to MATLAB and back
# recarray -> struct array

default_empty = np.array([])
#default_empty = None

# convert list of arbitrary dicts to struct array
def py2mat(dicts):
    """Convert a list of dictionaries to a NumPy structured array.

    Handles nested dictionaries and lists recursively.
    """
    if dicts is None:
        return np.array([], dtype=object)
    
    # Handle single dictionary input by wrapping in a list
    if isinstance(dicts, dict):
        dicts = [dicts]
        
    if not isinstance(dicts, (list, tuple)):
        return dicts
    
    # Check if this is a list of dictionaries (the expected input)
    if dicts and not all(isinstance(item, dict) for item in dicts):
        # If it's a mixed list, we can't convert it to a struct array
        # Return it as an object array instead
        return np.array(dicts, dtype=object)
    
    def process_value(value):
        """Recursively process values, converting nested structures."""
        if value is None:
            # Return None as-is, will be handled later
            return None
        elif isinstance(value, dict):
            # Convert single dict to struct array with one element
            if value:
                return py2mat([value])[0]
            else:
                # Empty dict - return empty array with object dtype
                return np.array([], dtype=object)
        elif isinstance(value, np.ndarray) and value.size > 0:
            try:
                # Check if it's an array of dicts
                if isinstance(value.flat[0], dict):
                    # Convert numpy array of dicts to struct array
                    return py2mat(value.tolist())
                else:
                    # Keep regular numpy arrays as-is
                    return value
            except (IndexError, AttributeError):
                # If we can't access flat[0], just return the array as-is
                return value
        elif isinstance(value, (list, tuple)) and value and isinstance(value[0], dict):
            # Convert list/tuple of dicts to struct array
            return py2mat(value)
        elif isinstance(value, np.ndarray):
            # Keep numpy arrays as-is
            return value
        elif isinstance(value, (list, tuple)) and not isinstance(value, str):
            # Convert regular list/tuple to numpy array (but not strings)
            if not value:
                return np.array([], dtype=object)
            try:
                # Try to create a numpy array, but handle inhomogeneous sequences
                return np.array(value)
            except ValueError:
                # If the sequence is inhomogeneous, create an object array
                return np.array(value, dtype=object)
        else:
            return value
    
    # Collect all unique keys and determine their types and sizes
    all_keys = set()
    key_types = {}
    key_max_lengths = {}
    
    for d in dicts:
        for k, v in d.items():
            all_keys.add(k)
            
            # Process the value recursively
            processed_v = process_value(v)
            
            # Determine the appropriate NumPy dtype for this value
            if isinstance(processed_v, str):
                # For strings, we need to track the maximum length
                if k not in key_max_lengths:
                    key_max_lengths[k] = len(processed_v)
                else:
                    key_max_lengths[k] = max(key_max_lengths[k], len(processed_v))
                key_types[k] = 'U'  # Unicode string type
            elif isinstance(processed_v, (int, np.integer)):
                if k not in key_types:
                    key_types[k] = int
                elif key_types[k] != int and key_types[k] != object:
                    key_types[k] = object
            elif isinstance(processed_v, (float, np.floating)):
                if k not in key_types:
                    key_types[k] = float
                elif key_types[k] not in [float, object]:
                    key_types[k] = object
            elif isinstance(processed_v, bool):
                if k not in key_types:
                    key_types[k] = bool
                elif key_types[k] != bool and key_types[k] != object:
                    key_types[k] = object
            elif isinstance(processed_v, np.ndarray):
                # For arrays (including nested struct arrays), use object type
                key_types[k] = object
            elif processed_v is None:
                # For None values, we'll determine type from other instances
                if k not in key_types:
                    key_types[k] = object
            else:
                # For other types, use object
                key_types[k] = object
    
    # Create dtype from all keys
    dtype_list = []
    for k in sorted(all_keys):
        if key_types[k] == 'U':
            # For Unicode strings, specify the maximum length
            max_len = key_max_lengths.get(k, 1)
            dtype_list.append((k, f'U{max_len}'))
        else:
            dtype_list.append((k, key_types[k]))
    
    dtype = np.dtype(dtype_list)
    
    # Create structured array
    struct_array = np.empty(len(dicts), dtype=dtype)
    
    # Fill the array
    for i, d in enumerate(dicts):
        for k in all_keys:
            value = d.get(k, None)
            processed_value = process_value(value)
            
            if processed_value is None:
                # Handle None values based on the field type
                if key_types[k] == 'U':
                    # For string fields, use empty string instead of None
                    struct_array[i][k] = ''
                elif key_types[k] == int:
                    # For int fields, use 0
                    struct_array[i][k] = 0
                elif key_types[k] == float:
                    # For float fields, use NaN
                    struct_array[i][k] = np.nan
                elif key_types[k] == bool:
                    # For bool fields, use False
                    struct_array[i][k] = False
                else:
                    # For object fields, use empty array with object dtype instead of None
                    struct_array[i][k] = np.array([], dtype=object)
            else:
                struct_array[i][k] = processed_value
    
    return struct_array

# def mat2py(mat_dict):
#     # convert all struct arrays to lists of dicts recursively
#     for k, v in mat_dict.items():
#         if isinstance(v, np.recarray):
#             mat_dict[k] = v.tolist()
#         elif isinstance(v, dict):
#             mat_dict[k] = mat2py(v)
#     return mat_dict

def mat2py(obj):
    """Convert MATLAB data structures to Python equivalents.

    Recursively converts MATLAB structs, arrays, and other types to Python dicts, lists,
    and arrays.
    """
    # check if obj is a dictionary and apply recursively the function to each object not changing the struture of the dictionary
    if isinstance(obj, dict):
        return {key: mat2py(obj[key]) for key in obj}
    # check if obj is a numpy array and apply recursively the function to each object not changing the struture of the array
    elif isinstance(obj, list):
        if len(obj) == 0:
            return default_empty
        else:
            return [mat2py(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        # check if empty and return none
        if obj.size == 0:
            return default_empty
        # check if it is a numeric array
        elif obj.dtype.kind in ['i', 'u', 'f', 'c']:
            if len(obj) == 1:
                if isinstance(obj[0], np.ndarray):
                    # Don't extract further if the inner array has more than one element
                    # This preserves structures like [[5, 8]]
                    if obj[0].size == 1:
                        return obj[0][0]
                    else:
                        return obj[0]
                else:
                    return obj[0]
            else:
                return obj
            # check if it is a string array
        elif obj.dtype.kind in ['U', 'S']:
            if len(obj) == 1:
                return obj[0]
            else:
                return obj.tolist()
        else:
            if isinstance(obj[0], np.ndarray):
                if len(obj) == 1:
                    return mat2py(obj[0])
                else:
                    return [mat2py(row) for row in obj]
            else:
                # Check if dtype has field names (structured array)
                if obj.dtype.names is not None:
                    if len(obj) == 1:
                        return {name: mat2py(obj[0][name]) for name in obj.dtype.names}
                    else:
                        return [{name: mat2py(row[name]) for name in obj.dtype.names} for row in obj]
                else:
                    # Not a structured array, handle as regular array
                    if len(obj) == 1:
                        return mat2py(obj[0])
                    else:
                        return [mat2py(row) for row in obj]
    # check if it is a scalar or a string and return it
    elif np.isscalar(obj) or isinstance(obj, str):
        return obj
    elif isinstance(obj, np.recarray):
        return mat2py(obj.tolist())
    # check if obj is a mat_struct object and convert it to a dictionary
    elif isinstance(obj, scipy.io.matlab.mat_struct):
        dict_obj = {}
        for field_name in obj._fieldnames:
            if field_name in ['tracking']:
                # used for fields that this code can't yet parse
                field_value = '<unsupported>'
            else:
                field_value = getattr(obj, field_name)
            dict_obj[field_name] = mat2py(field_value)
        return dict_obj
    # Handle other objects that have attributes and should be converted to dictionaries
    elif hasattr(obj, '__dict__') and not isinstance(obj, (str, int, float, bool, np.ndarray)):
        # Convert object with attributes to dictionary
        dict_obj = {}
        for attr_name in dir(obj):
            # Skip private/magic attributes and methods
            try:
                if not attr_name.startswith('_') and not callable(getattr(obj, attr_name)):
                    attr_value = getattr(obj, attr_name)
                    dict_obj[attr_name] = mat2py(attr_value)
            except:
                # Skip attributes that can't be accessed or cause errors
                continue
        return dict_obj if dict_obj else obj
    else:
        # Fallback: return the object as-is if no conversion rule applies
        return obj
    
def test_py2mat():
    """Test the py2mat and mat2py conversion functions with various data structures.

    """
    import scipy.io  
    
    # Test basic functionality
    print("=== Basic Test ===")
    dicts = [
        {'a': 'adsaf1', 'b': 2.0},
        {'a': 'adsaf', 'b': 4.0},
        {'a': 'adsaf33', 'b': 7.0}
    ]
    struct_array = py2mat(dicts)
    print("Original: ", dicts)

    dicts2 = mat2py(struct_array)
    scipy.io.savemat('test1.mat', {'struct_array': struct_array})
    struct_array2 = scipy.io.loadmat('test1.mat')
    struct_array2 = struct_array2['struct_array'][0]
    dicts3 = mat2py(struct_array2)
    print("Converted: ", dicts3)

    # Test nested dictionaries
    print("\n=== Nested Dictionary Test ===")
    nested_dicts = [
        {
            'name': 'item1',
            'value': 10.5,
            'config': {'enabled': True, 'threshold': 0.8},
            'tags': ['tag1', 'tag2']
        },
        {
            'name': 'item2', 
            'value': 20.3,
            'config': {'enabled': False, 'threshold': 0.9},
            'tags': ['tag3']
        }
    ]
    nested_struct = py2mat(nested_dicts)
    print("Original: ", nested_dicts)

    nested_dict2 = mat2py(nested_struct)
    print("Converted back (not fully compatible): ", nested_dict2)

    scipy.io.savemat('test2.mat', {'nested_struct': nested_struct})
    nested_struct2 = scipy.io.loadmat('test2.mat')
    nested_struct2 = nested_struct2['nested_struct'][0]
    nested_dict3 = mat2py(nested_struct2)
    print("Converted: ", nested_dict3)

    # Test list of dictionaries as values
    print("\n=== List of Dictionaries Test ===")
    list_dict_data = [
        {
            'id': 1,
            'measurements': [
                {'sensor': 'A', 'reading': 1.2},
                {'sensor': 'B', 'reading': 2.3}
            ]
        },
        {
            'id': 2,
            'measurements': [
                {'sensor': 'A', 'reading': 3.4},
                {'sensor': 'B', 'reading': 4.5},
                {'sensor': 'C', 'reading': 5.6}
            ]
        }
    ]
    list_dict_struct = py2mat(list_dict_data)
    scipy.io.savemat('test3.mat', {'list_dict_struct': list_dict_struct})
    list_dict_struct2 = scipy.io.loadmat('test3.mat')
    list_dict_struct2 = list_dict_struct2['list_dict_struct'][0]
    list_dict_data3   = mat2py(list_dict_struct2)
    print("Original: ", list_dict_data)
    print("Converted: ", list_dict_data3)

    # Test single dictionary input
    print("\n=== Single Dictionary Test ===")
    single_dict = {'x': 1, 'y': 2, 'nested': {'a': 'hello', 'b': 'world'}}
    single_struct = py2mat(single_dict)
    scipy.io.savemat('test4.mat', {'single_struct': single_struct})
    single_struct2 = scipy.io.loadmat('test4.mat')
    single_struct2 = single_struct2['single_struct'][0]
    single_dict2 = mat2py(single_struct2)
    print("Original: ", single_dict)
    print("Converted: ", single_dict2)
    
    
    # Test numpy array of dictionaries
    print("\n=== NumPy Array of Dictionaries Test ===")
    dict_array = np.array([
        {'name': 'sensor1', 'value': 1.1},
        {'name': 'sensor2', 'value': 2.2},
        {'name': 'sensor3', 'value': 3.3}
    ], dtype=object) 
    
    array_dict_data = [
        {
            'id': 'device1',
            'sensors': dict_array
        },
        {
            'id': 'device2', 
            'sensors': np.array([
                {'name': 'sensorA', 'value': 4.4},
                {'name': 'sensorB', 'value': 5.5}
            ], dtype=object)
        }
    ]
    
    array_dict_struct = py2mat(array_dict_data)
    scipy.io.savemat('test5.mat', {'array_dict_struct': array_dict_struct})
    array_dict_struct2 = scipy.io.loadmat('test5.mat')
    array_dict_struct2 = array_dict_struct2['array_dict_struct'][0]
    array_dict_data2 = mat2py(array_dict_struct2)
    print("Original: ", array_dict_data)
    print("Converted: ", array_dict_data2) # Numpy array gets converted to a list of dicts
    
    params = [np.vstack([np.arange(1, 21), np.arange(101, 121)]), [[5, 8]], 10.0, [{'latency': 5.0}, {'latency': 10.0}]]
    params_struct = py2mat(params)
    scipy.io.savemat('test6.mat', {'params_struct': params_struct})
    params_struct2 = scipy.io.loadmat('test6.mat')
    params_struct2 = params_struct2['params_struct'][0]
    params_data2 = mat2py(params_struct2)
    print("Original: ", params)
    print("Converted: ", params_data2)
    
    # EEGLAB dataset    
    eeglab_file_path = '/System/Volumes/Data/data/matlab/eeglab/sample_data/eeglab_data_epochs_ica.set'
    from eegprep.pop_loadset import pop_loadset
    EEG_LOADSET = pop_loadset(eeglab_file_path)
    
    # pop_loadset wihtout index adjustment
    EEG_LOADMAT = scipy.io.loadmat(eeglab_file_path)
    EEG_LOADMAT = mat2py(EEG_LOADMAT['EEG'][0])
    
    # pop_saveset without index adjustment
    EEG_TMP = EEG_LOADMAT.copy()
    EEG_TMP = py2mat(EEG_TMP)
    scipy.io.savemat('test7.set', {'EEG': EEG_TMP})
    
    # load again
    EEG_LOADMAT2 = scipy.io.loadmat('test7.set')
    EEG_LOADMAT2 = mat2py(EEG_LOADMAT2['EEG'][0])
        
    # Limitations
    print("\n=== Limitations ===")
    print("- Conversion back: py2mat then mat2py does not always work for nested structures (works when the file is saved as a .mat file)")
    print("- Numpy arrays of dicts are converted to lists of dicts (an intented feature)")

if __name__ == "__main__":
    test_py2mat()    
