import os
import json
import sys
sys.path.insert(0, '/Users/arno/Python/eegprep/src/')
sys.path.insert(0, '/usr/src/project/src/')
from eegprep import iclabel
from eegprep import pop_loadset
from eegprep import pop_saveset
from eegprep import pop_eegfiltnew
from eegprep import clean_artifacts
from eegprep import eeg_picard

# Current path
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

# Populate mne_config.py file with brainlife config.json
#with open(__location__+'/config.json') as config_json:
with open(__location__+'/config.json.example') as config_json:
    config = json.load(config_json)

fname = config['set']

# create temp file with the same name and _tmp.set suffix
fname_tmp = fname.replace('.set', '_tmp.set')
fname_out = fname.replace('.set', '_out.set')

EEG = pop_loadset(fname)
EEG = pop_eegfiltnew(EEG, locutoff=5,hicutoff=25,revfilt=True,plotfreqz=False)
EEG = clean_artifacts(EEG, FlatlineCriterion=5,ChannelCriterion=0.87, LineNoiseCriterion=4,Highpass=[0.25, 0.75],BurstCriterion= 20, WindowCriterion=0.25, BurstRejection=True, WindowCriterionTolerances=[float('-inf'), 7])
EEG = eeg_picard(EEG)
EEG = iclabel(EEG)
pop_saveset(EEG, fname_out)