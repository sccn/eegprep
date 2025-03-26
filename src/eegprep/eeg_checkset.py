import numpy as np
import os

def eeg_checkset(EEG, load_data=True):
      
    # convert EEG['nbchan] to integer
    if 'nbchan' in EEG:
        EEG['nbchan'] = int(EEG['nbchan'])
    if 'trials' in EEG:
        EEG['trials'] = int(EEG['trials'])
    if 'pnts' in EEG:
        EEG['pnts'] = int(EEG['pnts'])
    
    if 'data' in EEG and isinstance(EEG['data'], str) and load_data:
        # get path from file_path
        file_name = EEG['filepath'] + os.sep + EEG['data']
        EEG['data'] = np.fromfile(file_name, dtype='float32').reshape( EEG['pnts']*EEG['trials'], EEG['nbchan'])
        EEG['data'] = EEG['data'].T.reshape(EEG['nbchan'], EEG['trials'], EEG['pnts']).transpose(0, 2, 1)

    # compute ICA activations
    if 'icaweights' in EEG and 'icasphere' in EEG and EEG['icaweights'].size > 0 and EEG['icasphere'].size > 0:
        EEG['icaact'] = np.dot(np.dot(EEG['icaweights'], EEG['icasphere']), EEG['data'].reshape(int(EEG['nbchan']), -1))
        EEG['icaact'] = EEG['icaact'].astype(np.float32)
        EEG['icaact'] = EEG['icaact'].reshape(EEG['icaweights'].shape[0], -1, int(EEG['trials']))
    
    # subtract 1 to EEG['icachansind'] to make it 0-based
    if 'icachansind' in EEG and EEG['icachansind'].size > 0:
        EEG['icachansind'] = EEG['icachansind'] - 1
    
    # type conversion
    EEG['xmin'] = float(EEG['xmin'])
    EEG['xmax'] = float(EEG['xmax'])
    EEG['srate'] = float(EEG['srate'])
         
    # Define the expected types
    expected_types = {
        'setname': str,
        'filename': str,
        'filepath': str,
        'subject': str,
        'group': str,
        'condition': str,
        'session': (str, int),
        'comments': np.ndarray,
        'nbchan': int,
        'trials': int,
        'pnts': int,
        'srate': (float,int),
        'xmin': float,
        'xmax': float,
        'times': np.ndarray,  # Expecting a float numpy array
        'data': np.ndarray,   # Expecting a float numpy array
        'icaact': np.ndarray, # Expecting a float numpy array
        'icawinv': np.ndarray,# Expecting a float numpy array
        'icasphere': np.ndarray, # Expecting a float numpy array
        'icaweights': np.ndarray, # Expecting a float numpy array
        'icachansind': np.ndarray, # Expecting an integer numpy array
        'chanlocs': np.ndarray,    # Expecting numpy array of dictionaries
        'urchanlocs': np.ndarray,  # Expecting numpy array of dictionaries
        'chaninfo': dict,
        'ref': str,
        'event': np.ndarray,       # Expecting numpy array of dictionaries
        'urevent': np.ndarray,     # Expecting numpy array of dictionaries
        'eventdescription': np.ndarray, # Expecting numpy array of strings
        'epoch': np.ndarray,       # Expecting numpy array of dictionaries
        'epochdescription': np.ndarray, # Expecting numpy array of strings
        'reject': dict,
        'stats': dict,
        'specdata': dict,
        'specicaact': dict,
        'splinefile': str,
        'icasplinefile': str,
        'dipfit': dict,
        'history': str,
        'saved': str,
        'etc': dict,
        'datfile': str,
        'run': (str, int),
        'roi': dict,
    }
    
    # Iterate through expected types and check input dictionary
    for field, expected_type in expected_types.items():
        if field not in EEG:
            print(f"Field '{field}' is missing from the EEG dictionnary, adding it.")
            
            # add default values
            if expected_type == str:
                EEG[field] = ''
            elif expected_type == int:
                EEG[field] = np.array([], dtype=int)
            elif expected_type == float:
                EEG[field] = np.array([], dtype=float)
            elif expected_type == dict:
                EEG[field] = {}
            elif expected_type == np.ndarray:
                EEG[field] = np.array([])
            else:
                EEG[field] = np.array([])
            continue
        
        value = EEG[field]
        
        # Special cases for numpy arrays with specific content types
        if isinstance(expected_type, type) and expected_type == np.ndarray:
            if not isinstance(value, np.ndarray):
                print(f"Field '{field}' is expected to be a numpy array but is of type {type(value).__name__}.")
                continue
            # Further checks for numpy array content types
            if field in ['times', 'data', 'icaact', 'icawinv', 'icasphere', 'icaweights']:
                if not np.issubdtype(value.dtype, np.floating):
                    print(f"Field '{field}' is expected to be a numpy array of floats but has dtype {value.dtype}.")
            elif field in ['icachansind']:
                if not np.issubdtype(value.dtype, np.integer):
                    print(f"Field '{field}' is expected to be a numpy array of integers but has dtype {value.dtype}.")
            elif field in ['chanlocs', 'urchanlocs', 'event', 'urevent', 'epoch']:
                if not all(isinstance(item, dict) for item in value):
                    print(f"Field '{field}' is expected to be a numpy array of dictionaries but contains other types.")
            # elif field in ['eventdescription', 'epochdescription']:
            #     if not all(isinstance(item, str) for item in value):
            #         print(f"Field '{field}' is expected to be a numpy array of strings but contains other types.")
        else:
            # General type check
            if not isinstance(value, expected_type):
                # check for empty Ndarray
                if isinstance(value, np.ndarray) and value.size == 0:
                    continue
                print(f"Field '{field}' is expected to be of type {expected_type} but is of type {type(value).__name__}.")  
    
    return EEG

def test_eeg_checkset():
    from eegprep.pop_loadset import pop_loadset

    eeglab_file_path = './data/eeglab_data_with_ica_tmp_out2.set'
    EEG = pop_loadset(eeglab_file_path)
    EEG = eeg_checkset(EEG)
    print('Checkset done')

# test_eeg_checkset()