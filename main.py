import os
import json
from eegprep import iclabel
from eegprep import pop_loadset
from eegprep import pop_saveset

# Current path
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

# Populate mne_config.py file with brainlife config.json
#with open(__location__+'/config.json') as config_json:
with open(__location__+'/config.json.example') as config_json:
    config = json.load(config_json)

fname = config['set']

EEG = pop_loadset(fname)
EEG = iclabel(EEG, 'default')
pop_saveset(EEG, os.path.join('out_dir','raw.set'))
