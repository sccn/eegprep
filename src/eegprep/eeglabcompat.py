from oct2py import Oct2Py, get_log
import sys
sys.path.insert(0, '/Users/arno/Python/eegprep/src/')
sys.path.insert(0, '/usr/src/project/src/')
from eegprep import pop_loadset
from eegprep import pop_saveset
import logging
import os

eeglab = Oct2Py(logger=get_log())
oc = Oct2Py(logger=get_log())
eeglab.logger = get_log("new_log")
eeglab.logger.setLevel(logging.WARNING)
eeglab.warning('off', 'backtrace')

# On the command line, type "octave-8.4.0" OCTAVE_EXECUTABLE or OCTAVE var
path2eeglab = '/System/Volumes/Data/data/matlab/eeglab/' # init >10 seconds
eeglab.addpath(path2eeglab + '/functions/guifunc')
eeglab.addpath(path2eeglab + '/functions/popfunc')
eeglab.addpath(path2eeglab + '/functions/adminfunc')
eeglab.addpath(path2eeglab + '/plugins/firfilt')
eeglab.addpath(path2eeglab + '/functions/sigprocfunc')
eeglab.addpath(path2eeglab + '/functions/miscfunc')
eeglab.addpath(path2eeglab + '/plugins/dipfit')
eeglab.addpath(path2eeglab + '/plugins/clean_rawdata')
#res = eeglab.version()
#print('Running EEGLAB commands in compatibility mode with Octave ' + res)
eeglab.logger.setLevel(logging.INFO)

def pop_resample( EEG, freq): # 2 additional parameters in MATLAB (never used)
    
    pop_saveset(EEG, './tmp.set') # 0.8 seconds
    EEG2 = eeglab.pop_loadset('./tmp.set') # 2 seconds
    EEG2 = eeglab.pop_resample(EEG2, freq) # 2.4 seconds
    eeglab.pop_saveset(EEG2, './tmp2.set') # 2.4 seconds
    EEG3 = pop_loadset('./tmp2.set') # 0.2 seconds
    
    # delete temporary files
    os.remove('./tmp.set')
    os.remove('./tmp2.set')
    return EEG3


def pop_eegfiltnew(EEG, locutoff=None,hicutoff=None,revfilt=False,plotfreqz=False):
    # error if locutoff and hicutoff are none
    if locutoff==None and hicutoff==None:
        raise('Cannot have low cutoff and high cutoff not defined')
    
    pop_saveset(EEG, './tmp.set') # 0.8 seconds
    EEG2 = eeglab.pop_loadset('./tmp.set') # 2 seconds
    EEG3 = eeglab.pop_eegfiltnew(EEG2, 'locutoff',locutoff,'hicutoff',hicutoff,'revfilt',revfilt,'plotfreqz',plotfreqz)
    eeglab.pop_saveset(EEG2, './tmp2.set') # 2.4 seconds
    EEG4 = pop_loadset('./tmp2.set') # 0.2 seconds
    
    # delete temporary files
    # os.remove('./tmp.set')
    # os.remove('./tmp2.set')
    return EEG4

def clean_artifacts( EEG, ChannelCriterion=False, LineNoiseCriterion=False, FlatlineCriterion=False, BurstCriterion=False, BurstRejection=False, WindowCriterion=0, Highpass=[0.25, 0.75], WindowCriterionTolerances=[float('-inf'), 8]):

    if ChannelCriterion == False or ChannelCriterion == 'off':
        ChannelCriterion='off'
        
    if LineNoiseCriterion == False or LineNoiseCriterion == 'off':
        LineNoiseCriterion='off'
    
    if FlatlineCriterion == False or FlatlineCriterion == 'off':
        FlatlineCriterion='off'

    if BurstCriterion == False or BurstCriterion == 'off':
        BurstCriterion='off'

    if Highpass == False or Highpass == 'off':
        Highpass='off'

    if BurstRejection == False or BurstRejection == 'off':
        BurstRejection='off'           
    else:
        BurstRejection='on'


    pop_saveset(EEG, './tmp.set') # 0.8 seconds
    EEG2 = eeglab.pop_loadset('./tmp.set') # 2 seconds
    EEG3 = eeglab.clean_artifacts(EEG2, 'ChannelCriterion', ChannelCriterion, \
        'LineNoiseCriterion', LineNoiseCriterion, \
        'FlatlineCriterion', FlatlineCriterion, \
        'BurstCriterion', BurstCriterion,  \
        'BurstRejection', BurstRejection,  \
        'WindowCriterion', WindowCriterion, \
        'Highpass',Highpass, \
        'WindowCriterionTolerances', WindowCriterionTolerances)
    eeglab.pop_saveset(EEG3, './tmp2.set') # 2.4 seconds
    EEG4 = pop_loadset('./tmp2.set') # 0.2 seconds
    
    # delete temporary files
    os.remove('./tmp.set')
    os.remove('./tmp2.set')
    return EEG4

# sys.exit()
def test_eeglab_compat():

    eeglab_file_path = '/System/Volumes/Data/data/matlab/eeglab/sample_data/eeglab_data_epochs_ica.set'

    EEG = pop_loadset(eeglab_file_path)
    EEG = pop_eegfiltnew(EEG, locutoff=5,hicutoff=25,revfilt=True,plotfreqz=False)
    EEG = clean_artifacts(EEG, FlatlineCriterion=5,ChannelCriterion=0.87, LineNoiseCriterion=4,Highpass=False,BurstCriterion= 20, WindowCriterion=0.25, BurstRejection=False, WindowCriterionTolerances=[float('-inf'), 7])
        
    # EEG = eeglab.pop_loadset(eeglab_file_path)
    # TMPEEG = eeglab.pop_eegfiltnew(EEG, 'locutoff',5,'hicutoff',25,'revfilt',1,'plotfreqz',0)
    # CLEANEDEEG = eeglab.clean_artifacts(TMPEEG, 'ChannelCriterion', 'off', 
    #     'LineNoiseCriterion', 'off',
    #     'FlatlineCriterion', 'off',
    #     'BurstCriterion', 'off',
    #     'WindowCriterion', 0,
    #     'Highpass',[0.25, 0.75],
    #     'WindowCriterionTolerances', [-10000000, 8])

    # clean_artifacts( EEG, ChannelCriterion='on' )


# test_eeglab_compat()