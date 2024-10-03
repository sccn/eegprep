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
        
def pop_saveset(EEG, file_path):
    # convert Events to structured array
    # if 'event' in EEG:
    #     EEG['event'] = flatten_dict(EEG['event'])    
        
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

def pop_saveset2(EEG, file_name):
    
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
        'icaact'          : np.array([]),
        'icawinv'         : np.array([]),
        'icasphere'       : np.array([]),
        'icaweights'      : np.array([]),
        'icachansind'     : np.array([]),
        'chanlocs'        : np.array([]),
        'urchanlocs'      : np.array([]),
        'chaninfo'        : np.array([]),
        'ref'             : np.array([]),
        'event'           : np.array([]),
        'urevent'         : np.array([]),
        'eventdescription': np.array([]),
        'epoch'           : np.array([]),
        'epochdescription': np.array([]),
        'reject'          : np.array([]),
        'stats'           : np.array([]),
        'specdata'        : np.array([]),
        'specicaact'      : np.array([]),
        'splinefile'      : np.array([]),
        'icasplinefile'   : np.array([]),
        'dipfit'          : np.array([]),
        'history'         : np.array([]),
        'saved'           : np.array([]),
        'etc'             : np.array([]),
        'datfile'         : np.array([]),
        'run'             : np.array([]),
        'roi'             : np.array([]),
    }
    
    # Create the list of dictionaries with a string field
    d_list = [{
        'labels': c['labels'],
        'theta':  c['theta']   if not isinstance(c['theta'], np.ndarray) else None,
        'radius': c['radius']  if not isinstance(c['radius'], np.ndarray) else None,
        'X':      c['X']       if not isinstance(c['X'], np.ndarray) else None,
        'Y':      c['Y']       if not isinstance(c['Y'], np.ndarray) else None,
        'Z':      c['Z']       if not isinstance(c['Z'], np.ndarray) else None,
        'sph_theta':  c['sph_theta']  if not isinstance(c['sph_theta'], np.ndarray) else None,
        'sph_phi':    c['sph_phi']    if not isinstance(c['sph_phi'], np.ndarray) else None,
        'sph_radius': c['sph_radius'] if not isinstance(c['sph_radius'], np.ndarray) else None,
        'type':       c['type']       if not isinstance(c['type'], np.ndarray) else None,
        'urchan':     c['urchan']     if not isinstance(c['urchan'], np.ndarray) else None,
        'ref':        c['ref']        if not isinstance(c['ref'], np.ndarray) else None
    } for c in EEG['chanlocs']]

    dtype = np.dtype([
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
    ])

    # Convert the list of dictionaries to a structured NumPy array
    eeglab_dict['chanlocs'] = np.array([
        (
            item['labels'],
            item['theta'],
            item['radius'],
            item['X'],
            item['Y'],
            item['Z'],
            item['sph_theta'],
            item['sph_phi'],
            item['sph_radius'],
            item['type'],
            item['urchan'],
            item['ref']
        )
        for item in d_list
    ], dtype=dtype)

    # # Step 4: Save the EEGLAB dataset as a .mat file
    savemat(file_name, eeglab_dict)

from pop_loadset import pop_loadset

def test_pop_saveset():
    file_path = './eeglab_data_with_ica_tmp.set'
    EEG = pop_loadset(file_path)
    pop_saveset( EEG, 'tmp.set')
    pop_saveset2(EEG, 'tmp2.set') # does not do events and function above is better
    # print the keys of the EEG dictionary
    print(EEG.keys())
    
test_pop_saveset()

# STILL OPEN QUESTION: Better to have empty MATLAB arrays as None for empty numpy arrays (current default).
# The current default is to make it more MALTAB compatible. A lot of MATLAB function start indexing MATLAB
# empty arrays to add values to them. This is not possible with None and would create more conversion and 
# bugs. However, None is more pythonic. 