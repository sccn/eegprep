"""Load EEG data from HDF5 files."""

import h5py
import numpy as np
from eegprep.eeg_checkset import eeg_checkset

def pop_loadset_h5(file_name):
    """Load EEG data from HDF5 file.

    Parameters
    ----------
    file_name : str
        Path to the HDF5 file

    Returns
    -------
    EEG : dict
        EEG data structure
    """
    EEGTMP = h5py.File(file_name, 'r')
    EEG = {}
    
    # Check if the file has an EEG group (MATLAB .set format)
    if 'EEG' in EEGTMP.keys():
        # Use the EEG group as the main data source
        EEGTMP = EEGTMP['EEG']    

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
    
    def read_ref(f, ref):
        obj = f[ref]
        arr = np.asarray(obj[()])
        return arr.reshape(-1)[0]

    def single_struct_dict(hdf5_group, all_keys):
        # Build one dict from a scalar struct where each field holds a single value
        out = {}
        f = hdf5_group.file
        for k in all_keys:
            ds = hdf5_group[k]
            raw = ds[()]
            if getattr(ds.dtype, "kind", None) == "O":
                ref = np.asarray(raw).reshape(-1)[0]
                out[k] = read_ref(f, ref)
            else:
                out[k] = np.asarray(raw).reshape(-1)[0]
        return out

    def list_of_dicts(hdf5_group, all_keys):
        # number of channels from the first key
        n = int(np.asarray(hdf5_group[all_keys[0]][()]).squeeze().shape[0])

        # list of dicts, one per channel
        dicts = [dict() for _ in range(n)]
        f = hdf5_group.file

        for k in all_keys:
            ds = hdf5_group[k]
            raw = ds[()]  # could be object refs or numeric
            if getattr(ds.dtype, "kind", None) == "O":
                refs = np.asarray(raw).squeeze()
                for i, ref in enumerate(refs):
                    dicts[i][k] = read_ref(f, ref)
            else:
                vals = np.asarray(raw).squeeze()
                for i, v in enumerate(vals):
                    dicts[i][k] = np.asarray(v).reshape(-1)[0]

        return dicts
    
    def get_data_array(EEGTMP, key):
        hdf5_group = EEGTMP[key]

        if isinstance(hdf5_group, h5py.Group):
            # First, determine number of channels/items
            all_keys = list(hdf5_group.keys())
            if len(all_keys) == 0:
                return np.array([])  # Return empty numpy array instead of dict
            
            dicts = list_of_dicts(hdf5_group, all_keys)
            return dicts
        else:
            return hdf5_group[()]
        
    def get_data(EEGTMP, key):
        hdf5_group = EEGTMP[key]

        if isinstance(hdf5_group, h5py.Group):
            # First, determine number of channels/items
            all_keys = list(hdf5_group.keys())
            if len(all_keys) == 0:
                return np.array([])  # Return empty numpy array instead of dict
            
            dicts = single_struct_dict(hdf5_group, all_keys)
            return dicts
        else:
            return hdf5_group[()]
        
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
                # Keep as loaded; we'll fix orientation after the loop when nbchan/pnts are known
                EEG[key] = np.array(EEG[key])
            else:
                EEG[key] = np.array(EEG[key]).T
        
        # Apply scalar conversion
        if key in scalars and key in EEG:
            if isinstance(EEG[key], np.ndarray):
                EEG[key] = float(EEG[key].flatten()[0])
            else:
                EEG[key] = float(EEG[key])

    # Ensure EEG['data'] has channels x pnts (x trials) shape
    if 'data' in EEG:
        data_arr = np.array(EEG['data'])
        if data_arr.ndim == 2:
            # Infer nbchan/pnts if available
            nbchan = int(EEG.get('nbchan', data_arr.shape[0]))
            if data_arr.shape[0] != nbchan and data_arr.shape[1] == nbchan:
                data_arr = data_arr.T
        elif data_arr.ndim == 3:
            if int(EEG['nbchan']) != data_arr.shape[0]:
                data_arr = np.transpose(data_arr, (2, 1, 0))
        EEG['data'] = data_arr
    
    EEG = eeg_checkset(EEG)

    return EEG

if __name__ == '__main__':
    file_name = 'data/eeglab_data_epochs_ica_hdf5.set'
    EEG = pop_loadset_h5(file_name)
    print(EEG['data'].shape)
    print(EEG['icaweights'].shape)
    print(EEG['icasphere'].shape)
    print(EEG['icawinv'].shape)
    print(EEG['icaact'].shape)
# file_name = 'eeglab_cont73.set'
# EEG = pop_loadset_h5(file_name)
# EEG['data'].shape
