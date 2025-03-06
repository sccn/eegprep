import h5py
import numpy as np

def pop_loadset_h5(file_name):
    EEGTMP = h5py.File(file_name, 'r')
    EEG = {}    

    def convert_to_string(filecontent):
        if filecontent.dtype == 'uint16':
            text = ''
            for val in filecontent.T.flatten():
                if val == 13:  # CR
                    text += '\r'
                elif val == 10:  # LF
                    text += '\n'
                else:
                    text += chr(val)
            return text
        else:
            return filecontent

    def get_data_array(EEGTMP, key):
        EEG[key] = {}
        chanlocs_group = EEGTMP[key]

        if isinstance(chanlocs_group, h5py.Group):
            # First, determine number of channels
            all_keys = list(chanlocs_group.keys())
            num_channels = len(chanlocs_group[all_keys[0]][()])

            # Initialize dictionary for each channel
            for i in range(num_channels):
                EEG[key][i] = {}

            # check if key is a reference
            for key2 in chanlocs_group.keys():
                chanlocs_key = chanlocs_group[key2][()]
                for iVal in range(len(chanlocs_key)):
                    # Get the reference
                    ref_value = chanlocs_group[key2][()][iVal][0]
                    # Dereference it if it's a reference
                    if isinstance(ref_value, h5py.h5r.Reference):
                        actual_value = EEGTMP[ref_value]
                        data = actual_value[()]
                    else:
                        data = ref_value
                        
                    # Store the data
                    EEG[key][iVal][key2] = convert_to_string(data)
                    if isinstance(EEG[key][iVal][key2], np.ndarray):
                        EEG[key][iVal][key2] = EEG[key][iVal][key2][0]
                    if isinstance(EEG[key][iVal][key2], np.ndarray):
                        EEG[key][iVal][key2] = EEG[key][iVal][key2][0]

            # convert first dimension of int keys to array
            EEG[key] = np.array([ EEG[key][i] for i in range(len(EEG[key])) ])
        return EEG[key]


    def get_data(EEGTMP, key):
        EEG[key] = {}
        chanlocs_group = EEGTMP[key]
        
        # check if chanlocs_group is of type dict
        if isinstance(chanlocs_group, h5py.Group):
            for key2 in chanlocs_group.keys():
                chanlocs_key = chanlocs_group[key2][()]
            EEG[key][key2] = convert_to_string(chanlocs_key)
            if isinstance(EEG[key][key2], np.ndarray):
                EEG[key][key2] = EEG[key][key2][0]
            
        return EEG[key]

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
            EEG[key] = EEGTMP[key][()]
        if key in arrays:
            EEG[key] = np.array(EEG[key]).T
        if key in scalars:
            EEG[key] = float(EEG[key][0][0])
        if key in strings:
            EEG[key] = convert_to_string(EEG[key])

    return EEG

# file_name = 'eeglab_cont73.set'
# EEG = pop_loadset_h5(file_name)
# EEG['data'].shape
