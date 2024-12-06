
import os
from scipy.io import savemat
import tempfile # need to do something with tempfile
from .pop_loadset import pop_loadset
from .pop_saveset import pop_saveset
from oct2py import octave as eeglab

def pop_resample( EEG, freq): # 2 additional parameters in MATLAB (never used)
    
    pop_saveset(EEG, 'tmp.set') # 0.8 seconds
        
    path2eeglab = '/System/Volumes/Data/data/matlab/eeglab/' # init >10 seconds
    eeglab.addpath(path2eeglab + '/functions/guifunc')
    eeglab.addpath(path2eeglab + '/functions/popfunc')
    eeglab.addpath(path2eeglab + '/functions/adminfunc')
    eeglab.addpath(path2eeglab + '/plugins/firfilt')
    eeglab.addpath(path2eeglab + '/functions/sigprocfunc')
    eeglab.addpath(path2eeglab + '/functions/miscfunc')
    eeglab.addpath(path2eeglab + '/plugins/dipfit')

    EEG2 = eeglab.pop_loadset('tmp.set') # 2 seconds
    EEG2 = eeglab.pop_resample(EEG2, freq) # 2.4 seconds
    eeglab.pop_saveset(EEG2, 'tmp2.set') # 2.4 seconds

    EEG3 = pop_loadset('tmp2.set') # 0.2 seconds
    
    # delete temporary files
    os.remove('tmp.set')
    os.remove('tmp2.set')
    return EEG3

def test_pop_resample():
    eeglab_file_path = './eeglab_data_with_ica_tmp.set'
    EEG = pop_loadset(eeglab_file_path)
    EEG2 = pop_resample(EEG, 100)
    print(EEG2.keys())
    print(EEG2['srate'])
    
# test_pop_resample()
    
    