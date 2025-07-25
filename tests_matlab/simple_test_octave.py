
if __name__ == '__main__':
    from eegprep import pop_resample, pop_loadset
    
    eeglab_file_path = '/Users/arno/Python/eegprep/tmp.set'
    EEG = pop_loadset(eeglab_file_path)
    
    # Test with different engines
    # EEG_python = pop_resample(EEG, 100, engine=None)
    # EEG_matlab = pop_resample(EEG, 100, engine='matlab')
    EEG_octave = pop_resample(EEG, 100, engine='octave')


if 0:
    from oct2py import Oct2Py, get_log
    from oct2py import octave as octave_engine
    import logging
    import os
    #engine = Oct2Py(logger=get_log())
    octave_engine.logger = get_log("new_log")
    octave_engine.logger.setLevel(logging.WARNING)
    octave_engine.warning('off', 'backtrace')

    path2eeglab = '/System/Volumes/Data/data/matlab/eeglab/'
    octave_engine.addpath(path2eeglab + '/functions/guifunc')
    octave_engine.addpath(path2eeglab + '/functions/popfunc')
    octave_engine.addpath(path2eeglab + '/functions/adminfunc')
    octave_engine.addpath(path2eeglab + '/plugins/firfilt')
    octave_engine.addpath(path2eeglab + '/functions/sigprocfunc')
    octave_engine.addpath(path2eeglab + '/functions/miscfunc')
    octave_engine.addpath(path2eeglab + '/plugins/dipfit')


    EEG = octave_engine.pop_loadset('tmp.set')
    EEG = octave_engine.pop_resample(EEG, 100)
    octave_engine.pop_saveset(EEG, 'tmp2.set')