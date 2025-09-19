import scipy.io
import numpy as np
import os

# Allows access using . notation
# class EEG:
#     def __init__(self, **kwargs):
#         self.__dict__.update(kwargs)
#     def __getitem__(self, key):
#         return self.__dict__[key]
#     def __setitem__(self, key, value):
#         self.__dict__[key] = value

default_empty = np.array([])

def flatten_dict_sub(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict_sub(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def flatten_dict(data):
    # Flatten each dictionary and collect the fields and types
    flat_data = [flatten_dict_sub(item) for item in data]
    fields = list(flat_data[0].keys())
    dtypes = []

    # Determine data types
    for field in fields:
        sample_value = flat_data[0][field]
        if isinstance(sample_value, int):
            dtypes.append((field, np.int32))
        elif isinstance(sample_value, float):
            dtypes.append((field, np.float64))
        else:
            dtypes.append((field, 'U{}'.format(max(len(str(item[field])) for item in flat_data))))
            
    # Convert the flattened data to a list of tuples
    data_tuples = [tuple(item[field] for field in fields) for item in flat_data]

    # Create the rec.array
    dtype = np.dtype([(key, 'O') for key in data[0].keys()])
    rec_array = np.array(data_tuples, dtype=dtype).view(np.recarray)
    return rec_array
        
def saveset(EEG, file_name):
    return pop_saveset(EEG, file_name)

# def dictlist_to_recarray(events):
#     # --- Infer dtype automatically ---
#     dtype_fields = []
#     for key in events[0].keys():
#         values = [e[key] for e in events]

#         # If string: pick Unicode type with max length
#         if all(isinstance(v, str) for v in values):
#             maxlen = max(len(v) for v in values)
#             dtype_fields.append((key, f'U{maxlen}'))

#         # If integer: use int32
#         elif all(isinstance(v, int) for v in values):
#             dtype_fields.append((key, 'i4'))

#         # If float or mixed int/float: use float64
#         elif all(isinstance(v, (float, int)) for v in values):
#             dtype_fields.append((key, 'f8'))

#         else:
#             # fallback: generic object
#             dtype_fields.append((key, object))

#     dtype = np.dtype(dtype_fields)

#     # --- Convert events to recarray ---
#     rec_events = np.array(
#         [tuple(e[k] for k in events[0].keys()) for e in events],
#         dtype=dtype
#     ).view(np.recarray)

#     return rec_events

def pop_saveset_old(EEG, file_path):
    # convert Events to structured array
    # if 'event' in EEG:
    #     EEG['event'] = flatten_dict(EEG['event'])    
        
    # add 1 to EEG['icachansind'] to make it 1-based
    if 'icachansind' in EEG and EEG['icachansind'].size > 0:
        EEG['icachansind'] = EEG['icachansind'] + 1 
        
    # search for array of dictionaries and convert them to flatten_dicts
    for key in EEG:
        if isinstance(EEG[key], np.ndarray) and len(EEG[key]) > 0 and isinstance(EEG[key][0], dict):
            EEG[key] = flatten_dict(EEG[key])

    EEG['icaact'] = default_empty 
    scipy.io.savemat(file_path, EEG)
            
    return EEG


# Example to export MNE epochs to EEGLAB dataset
# Events are not handled correctly in this example but it works

import mne
from mne.datasets import sample
import numpy as np
from scipy.io import savemat

def pop_saveset(EEG, file_name):
    
    eeglab_dict = {
        'setname'         : '',
        'filename'        : '',
        'filepath'        : '',
        'subject'         : '',
        'group'           : '',
        'condition'       : '',
        'session'         : np.array([]),
        'comments'        : '',
        'nbchan'          : float(EEG['nbchan']),
        'trials'          : float(EEG['trials']),
        'pnts'            : float(EEG['pnts']),
        'srate'           : float(EEG['srate']),
        'xmin'            : float(EEG['xmin']),
        'xmax'            : float(EEG['xmax']),
        'times'           : EEG['times'],
        'data'            : EEG['data'],
        'icaact'          : EEG['icaact'],
        'icawinv'         : EEG['icawinv'],
        'icasphere'       : EEG['icasphere'],
        'icaweights'      : EEG['icaweights'],
        'icachansind'     : EEG['icachansind'].copy(),
        'chanlocs'        : EEG['chanlocs'],
        'urchanlocs'      : EEG['urchanlocs'],
        'chaninfo'        : EEG['chaninfo'],
        'ref'             : EEG['ref'],
        'event'           : EEG['event'] if 'event' in EEG else np.array([]),
        'urevent'         : EEG['urevent'] if 'urevent' in EEG else np.array([]),
        'eventdescription': EEG['eventdescription'] if 'eventdescription' in EEG else np.array([]),
        'epoch'           : EEG['epoch'] if 'epoch' in EEG else np.array([]),
        'epochdescription': EEG['epochdescription'] if 'epochdescription' in EEG else np.array([]),
        'reject'          : EEG['reject'] if 'reject' in EEG else np.array([]),
        'stats'           : EEG['stats'] if 'stats' in EEG else np.array([]),
        'specdata'        : EEG['specdata'] if 'specdata' in EEG else np.array([]),
        'specicaact'      : EEG['specicaact'] if 'specicaact' in EEG else np.array([]),
        'splinefile'      : EEG['splinefile'] if 'splinefile' in EEG else np.array([]),
        'icasplinefile'   : EEG['icasplinefile'] if 'icasplinefile' in EEG else np.array([]),
        'dipfit'          : EEG['dipfit'] if 'dipfit' in EEG else np.array([]),
        'history'         : EEG['history'],
        'saved'           : EEG['saved'],
        'etc'             : EEG['etc'],
        'run'             : EEG['run'] if 'run' in EEG else np.array([]),
        'roi'             : EEG['roi'] if 'roi' in EEG else np.array([]),
    }

     # add 1 to EEG['icachansind'] to make it 1-based
    if 'icachansind' in eeglab_dict and eeglab_dict['icachansind'].size > 0:
        eeglab_dict['icachansind'] = eeglab_dict['icachansind'] + 1 

    # check if EEG['urchan'] is 0-based
    if len(eeglab_dict['chanlocs']) > 0 and 'urchan' in eeglab_dict['chanlocs'][0]:
        for i in range(len(eeglab_dict['chanlocs'])):
            eeglab_dict['chanlocs'][i]['urchan'] = eeglab_dict['chanlocs'][i]['urchan'] + 1        
            
    # check if EEG['chanlocs'][i]['urvent'] is 0-based
    if len(eeglab_dict['event']) > 0 and 'urvent' in eeglab_dict['event'][0]:
        for i in range(len(eeglab_dict['event'])):
            eeglab_dict['event'][i]['urvent'] = eeglab_dict['event'][i]['urvent'] + 1  
                   
    # Create the list of dictionaries with a string field
    if 'chanlocs' in EEG and len(EEG['chanlocs']) > 0:
        matlab_null = np.array([])
        d_list = [{
            'labels': c['labels'],
            'theta':  c['theta']   if not isinstance(c.get('theta', matlab_null), np.ndarray) else None,
            'radius': c['radius']  if not isinstance(c.get('radius', matlab_null), np.ndarray) else None,
            'X':      c['X']       if not isinstance(c.get('X', matlab_null), np.ndarray) else None,
            'Y':      c['Y']       if not isinstance(c.get('Y', matlab_null), np.ndarray) else None,
            'Z':      c['Z']       if not isinstance(c.get('Z', matlab_null), np.ndarray) else None,
            'sph_theta':  c['sph_theta']  if not isinstance(c.get('sph_theta', matlab_null), np.ndarray) else None,
            'sph_phi':    c['sph_phi']    if not isinstance(c.get('sph_phi', matlab_null), np.ndarray) else None,
            'sph_radius': c['sph_radius'] if not isinstance(c.get('sph_radius', matlab_null), np.ndarray) else None,
            'type':       c['type']       if not isinstance(c.get('type', matlab_null), np.ndarray) else None,
            'urchan':     c['urchan']     if not isinstance(c.get('urchan', matlab_null), np.ndarray) else None,
            'ref':        c['ref']        if not isinstance(c.get('ref', matlab_null), np.ndarray) else None
        } for c in EEG['chanlocs']]

        # build a list of fields to selectively filter out if all entries are None
        retain_fields = [fld for fld in d_list[0].keys() if not all(d[fld] is None for d in d_list)]

        dtype = np.dtype([(f, t) for f, t in [
            ('labels', 'U100'),      # String up to 100 characters
            ('theta', np.float64),
            ('radius', np.float64),
            ('X', np.float64),
            ('Y', np.float64),
            ('Z', np.float64),
            ('sph_theta', np.float64),
            ('sph_phi', np.float64),
            ('sph_radius', np.float64),
            ('type', 'U10'),         # String up to 10 characters
            ('urchan', np.int32),
            ('ref', 'U100')          # String up to 100 characters
        ] if f in retain_fields])

        # Convert the list of dictionaries to a structured NumPy array
        eeglab_dict['chanlocs'] = np.array([
            tuple(item[fld] for fld in retain_fields)
            for item in d_list
        ], dtype=dtype)
    
    if isinstance(eeglab_dict['event'], list):
        eeglab_dict['event'] = np.array(eeglab_dict['event'])
        
    for key in eeglab_dict:
        if isinstance(eeglab_dict[key], np.ndarray) and len(eeglab_dict[key]) > 0 and isinstance(eeglab_dict[key][0], dict):
            eeglab_dict[key] = flatten_dict(eeglab_dict[key])

    if not os.path.exists(os.path.dirname(file_name)):
        os.makedirs(os.path.dirname(file_name))

    # # Step 4: Save the EEGLAB dataset as a .mat file
    try:
        scipy.io.savemat(file_name, eeglab_dict, appendmat=False)
    except ValueError as e:
        if '31 characters' in str(e):
            # try to save with long_field_names option
            scipy.io.savemat(file_name, eeglab_dict, appendmat=False, long_field_names=True)
        else:
            # the file is likely partial and thus invalid -- delete
            if os.path.exists(file_name):
                os.remove(file_name)
            raise

def test_pop_saveset():
    from eegprep.pop_loadset import pop_loadset
    file_path = './data/eeglab_data_with_ica_tmp.set'
    EEG = pop_loadset(file_path)
    pop_saveset( EEG, '/Users/arno/Python/eegprep/data/tmp.set')
    pop_saveset_old(EEG, '/Users/arno/Python/eegprep/data/tmp2.set') # does not do events and function above is better
    # print the keys of the EEG dictionary
    print(EEG.keys())
    
if __name__ == '__main__':
    test_pop_saveset()

# STILL OPEN QUESTION: Better to have empty MATLAB arrays as None for empty numpy arrays (current default).
# The current default is to make it more MALTAB compatible. A lot of MATLAB function start indexing MATLAB
# empty arrays to add values to them. This is not possible with None and would create more conversion and 
# bugs. However, None is more pythonic. 