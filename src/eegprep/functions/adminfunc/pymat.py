"""Python-MATLAB data conversion utilities."""

import numpy as np
import scipy.io

# These are conversions from Python to MATLAB and back
# recarray -> struct array

default_empty = np.array([])
# default_empty = None


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
                elif key_types[k] is not int and key_types[k] is not object:
                    key_types[k] = object
            elif isinstance(processed_v, (float, np.floating)):
                if k not in key_types:
                    key_types[k] = float
                elif key_types[k] not in [float, object]:
                    key_types[k] = object
            elif isinstance(processed_v, bool):
                if k not in key_types:
                    key_types[k] = bool
                elif key_types[k] is not bool and key_types[k] is not object:
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
                elif key_types[k] is int:
                    # For int fields, use 0
                    struct_array[i][k] = 0
                elif key_types[k] is float:
                    # For float fields, use NaN
                    struct_array[i][k] = np.nan
                elif key_types[k] is bool:
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
            except Exception:
                # Skip attributes that can't be accessed or cause errors
                continue
        return dict_obj if dict_obj else obj
    else:
        # Fallback: return the object as-is if no conversion rule applies
        return obj
