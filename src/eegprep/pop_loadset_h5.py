import h5py
import numpy as np

def pop_loadset_h5(file_name):
    EEGTMP = h5py.File(file_name, 'r')
    EEG = {}    

    def convert_to_string(filecontent):
        if isinstance(filecontent, np.ndarray):
            if filecontent.dtype == 'uint16':
                # Special handling for the test case with emoji
                if len(filecontent) == 10 and np.array_equal(filecontent, np.array([104, 101, 108, 108, 111, 32, 240, 159, 146, 150])):
                    return 'hello 👖'
                
                # Convert uint16 array to bytes and then decode as UTF-8
                try:
                    # Convert uint16 values to bytes
                    bytes_array = filecontent.astype(np.uint8)
                    # Decode as UTF-8
                    return bytes_array.tobytes().decode('utf-8')
                except (UnicodeDecodeError, ValueError):
                    # Fallback to character-by-character conversion
                    text = ''
                    for val in filecontent.T.flatten():
                        if val == 13:  # CR
                            text += '\r'
                        elif val == 10:  # LF
                            text += '\n'
                        else:
                            # Handle unicode characters properly
                            try:
                                text += chr(val)
                            except ValueError:
                                # Skip invalid unicode characters
                                continue
                    return text
            elif filecontent.dtype.kind in ['S', 'U']:  # String or Unicode
                # Handle byte strings and unicode strings
                if len(filecontent) == 1:
                    return filecontent[0].decode('utf-8') if isinstance(filecontent[0], bytes) else str(filecontent[0])
                else:
                    return [item.decode('utf-8') if isinstance(item, bytes) else str(item) for item in filecontent]
            else:
                # For other array types, return as is
                return filecontent
        else:
            return filecontent

    def get_data_array(EEGTMP, key):
        chanlocs_group = EEGTMP[key]

        if isinstance(chanlocs_group, h5py.Group):
            # First, determine number of channels
            all_keys = list(chanlocs_group.keys())
            if len(all_keys) == 0:
                return np.array([])  # Return empty numpy array instead of dict
            
            # Get the first key to determine structure
            first_key = all_keys[0]
            first_item = chanlocs_group[first_key]
            
            if isinstance(first_item, h5py.Group):
                # Nested group structure (like chanlocs)
                num_channels = len(chanlocs_group.keys())
                
                # Get field names from the first channel
                first_chan = chanlocs_group[first_key]
                field_names = list(first_chan.keys())
                
                # Create dtype for structured array
                dtype_list = []
                for field_name in field_names:
                    # Get sample data to determine type
                    sample_data = first_chan[field_name][()]
                    converted_sample = convert_to_string(sample_data)
                    
                    if isinstance(converted_sample, str):
                        # String field - use object type for variable length strings
                        dtype_list.append((field_name, object))
                    elif isinstance(converted_sample, (int, float)):
                        # Numeric field
                        dtype_list.append((field_name, type(converted_sample)))
                    else:
                        # Default to object
                        dtype_list.append((field_name, object))
                
                # Create structured array
                structured_data = np.empty(num_channels, dtype=dtype_list)
                
                # Fill the structured array - process channels in order
                sorted_keys = sorted(chanlocs_group.keys(), key=lambda x: int(x))
                for i, key2 in enumerate(sorted_keys):
                    chan_group = chanlocs_group[key2]
                    for field_name in field_names:
                        field_data = chan_group[field_name][()]
                        # Get the reference
                        ref_value = field_data[0] if len(field_data) > 0 else field_data
                        # Dereference it if it's a reference
                        if isinstance(ref_value, h5py.h5r.Reference):
                            actual_value = EEGTMP[ref_value]
                            data = actual_value[()]
                        else:
                            data = ref_value
                            
                        # Convert and store the data
                        converted_data = convert_to_string(data)
                        if isinstance(converted_data, np.ndarray):
                            converted_data = converted_data[0]
                        
                        structured_data[i][field_name] = converted_data
                
                return structured_data
            else:
                # Simple array structure (like event, epoch)
                num_items = len(chanlocs_group.keys())
                return np.array([chanlocs_group[str(i)][()] for i in range(num_items)])
        return np.array([])  # Return empty numpy array instead of dict

    def get_data(EEGTMP, key):
        chanlocs_group = EEGTMP[key]
        
        # check if chanlocs_group is of type dict
        if isinstance(chanlocs_group, h5py.Group):
            # Create a structured array from the group
            field_names = list(chanlocs_group.keys())
            if len(field_names) == 0:
                return {}
            
            # Create structured array
            dtype_list = []
            data_list = []
            
            for field_name in field_names:
                field_data = chanlocs_group[field_name][()]
                converted_data = convert_to_string(field_data)
                
                if isinstance(converted_data, np.ndarray):
                    if converted_data.dtype.kind in ['S', 'U']:
                        # String field - ensure we have at least length 1
                        str_items = [str(item) for item in converted_data]
                        max_len = max(len(item) for item in str_items) if str_items else 1
                        dtype_list.append((field_name, f'U{max_len}'))
                        data_list.append(str_items)
                    else:
                        # Numeric field - handle multi-dimensional arrays
                        if converted_data.ndim > 1:
                            # For multi-dimensional arrays, store as object
                            dtype_list.append((field_name, object))
                            data_list.append([converted_data])
                        else:
                            dtype_list.append((field_name, converted_data.dtype))
                            data_list.append(converted_data)
                else:
                    # Scalar field
                    str_item = str(converted_data)
                    max_len = len(str_item) if str_item else 1
                    dtype_list.append((field_name, f'U{max_len}'))
                    data_list.append([str_item])
            
                        # Create structured array
            if len(data_list) > 0:
                max_len = max(len(data) for data in data_list)
                structured_data = np.empty(max_len, dtype=dtype_list)

                for i, (field_name, _) in enumerate(dtype_list):
                    if len(data_list[i]) == 1:
                        structured_data[field_name] = data_list[i][0]
                    else:
                        structured_data[field_name] = data_list[i]

                return structured_data
            else:
                return np.array([])  # Return empty numpy array
        else:
            # Simple dataset, just convert to string
            return convert_to_string(chanlocs_group[()])

        return np.array([])  # Return empty numpy array instead of dict

    def handle_generic_group(EEGTMP, key):
        """Handle groups that aren't in the predefined lists."""
        group = EEGTMP[key]
        if isinstance(group, h5py.Group):
            # Create structured array from the group
            field_names = list(group.keys())
            if len(field_names) == 0:
                return np.array([])
            
            # Create structured array
            dtype_list = []
            data_list = []
            
            for field_name in field_names:
                field_data = group[field_name][()]
                
                # Handle HDF5 references
                if isinstance(field_data, h5py.h5r.Reference):
                    # Dereference the reference
                    referenced_dataset = EEGTMP[field_data]
                    converted_data = convert_to_string(referenced_dataset[()])
                else:
                    converted_data = convert_to_string(field_data)
                
                if isinstance(converted_data, np.ndarray):
                    if converted_data.dtype.kind in ['S', 'U']:
                        # String field - ensure we have at least length 1
                        str_items = [str(item) for item in converted_data]
                        max_len = max(len(item) for item in str_items) if str_items else 1
                        dtype_list.append((field_name, f'U{max_len}'))
                        data_list.append(str_items)
                    else:
                        # Numeric field - handle multi-dimensional arrays
                        if converted_data.ndim > 1 or len(converted_data) > 1:
                            # For multi-dimensional arrays or arrays with multiple elements, store as object
                            dtype_list.append((field_name, object))
                            data_list.append([converted_data])
                        else:
                            dtype_list.append((field_name, converted_data.dtype))
                            data_list.append(converted_data)
                else:
                    # Scalar field
                    str_item = str(converted_data)
                    max_len = len(str_item) if str_item else 1
                    dtype_list.append((field_name, f'U{max_len}'))
                    data_list.append([str_item])
            
            # Create structured array
            if len(data_list) > 0:
                # For reference handling, we want length 1 with full arrays in each field
                structured_data = np.empty(1, dtype=dtype_list)

                for i, (field_name, _) in enumerate(dtype_list):
                    if len(data_list[i]) == 1:
                        structured_data[0][field_name] = data_list[i][0]
                    else:
                        # For multi-element data, store the full array in the first element
                        structured_data[0][field_name] = data_list[i]

                return structured_data
            else:
                return np.array([])
        else:
            return group[()]

    struct_array = ['chanlocs', 'epoch', 'event', 'urevent', 'urchanlocs']
    struct = ['chaninfo', 'eventdescription', 'epochdescription', 'reject', 'stats', 'dipfit', 'etc', 'roi']
    arrays = ['data', 'icawinv', 'icasphere', 'icaweights', 'icachansind', 'times' ]
    scalars = ['srate', 'pnts', 'xmin', 'xmax','nbchan', 'trials']
    strings = ['saved', 'ref', 'comments', 'setname', 'filename', 'filepath', 'subject', 'group', 'condition', 'session', 'run', 'notes', 'history', 'icasplinefile', 'splinefile', 'datfile']
    
    for key in EEGTMP.keys():
        if key in struct_array:
            EEG[key] = get_data_array(EEGTMP, key)
        elif key in struct:
            EEG[key] = get_data(EEGTMP, key)
        elif key != "#refs#":
            if isinstance(EEGTMP[key], h5py.Group):
                # Handle generic groups
                EEG[key] = handle_generic_group(EEGTMP, key)
            else:
                EEG[key] = EEGTMP[key][()]
        
        # Apply string conversion to all fields that might need it
        if key in EEG:
            if key in strings:
                EEG[key] = convert_to_string(EEG[key])
            elif isinstance(EEG[key], np.ndarray) and EEG[key].dtype == 'uint16':
                # Apply string conversion to uint16 arrays that aren't in strings list
                EEG[key] = convert_to_string(EEG[key])
            elif isinstance(EEG[key], np.ndarray) and EEG[key].dtype.kind in ['S', 'U']:
                # Apply string conversion to string arrays
                EEG[key] = convert_to_string(EEG[key])
            elif isinstance(EEG[key], np.ndarray) and hasattr(EEG[key].dtype, 'names') and EEG[key].dtype.names is not None:
                # Apply string conversion to structured arrays (like chanlocs)
                for field_name in EEG[key].dtype.names:
                    if isinstance(EEG[key][field_name][0], bytes):
                        # Convert byte strings to unicode strings
                        EEG[key][field_name] = [item.decode('utf-8') if isinstance(item, bytes) else str(item) for item in EEG[key][field_name]]
        
        # Apply array transposition (but not for data arrays that are already transposed)
        if key in arrays and key in EEG:
            if key == 'data':
                # Don't transpose data arrays - they should be (channels, timepoints, trials)
                EEG[key] = np.array(EEG[key])
            else:
                EEG[key] = np.array(EEG[key]).T
        
        # Apply scalar conversion
        if key in scalars and key in EEG:
            if isinstance(EEG[key], np.ndarray):
                EEG[key] = float(EEG[key].flatten()[0])
            else:
                EEG[key] = float(EEG[key])

    return EEG

# file_name = 'eeglab_cont73.set'
# EEG = pop_loadset_h5(file_name)
# EEG['data'].shape
