# import sys
# sys.path.insert(0, 'src/')

import tempfile

from .pop_loadset import pop_loadset
from .pop_saveset import pop_saveset
import logging
import os

logger = logging.getLogger(__name__)

# can be either 'OCT' (for Oct2Py) or 'MAT' (MATLAB engine)
default_runtime = 'OCT'


class MatlabWrapper:
    """MATLAB engine wrapper that round-trips calls involving the EEGLAB data structure through files."""

    def __init__(self, engine):
        self.engine = engine

    def __getattr__(self, name):
        def wrapper(*args):
            needs_roundtrip = False
            for a in args:
                if isinstance(a, dict) and a.get('nbchan') is not None:
                    needs_roundtrip = True
                    break
            if needs_roundtrip:
                # passage data through a file
                with tempfile.NamedTemporaryFile(delete=True, suffix='.set') as temp_file:
                    pop_saveset(args[0], temp_file.name)
                    # needs to use eval since returning struct arrays is not supported
                    self.engine.eval(f"EEG = pop_loadset('{temp_file.name}');", nargout=0)
                    # TODO: marshalling of extra arguments should follow octave conventions
                    eval_str = f"EEG = {name}(EEG{',' if args[1:] else ''}{','.join([str(a) for a in args[1:]])});"
                    print(eval_str)
                    self.engine.eval(eval_str, nargout=0)
                    self.engine.eval(f"pop_saveset(EEG, '{temp_file.name}');", nargout=0)
                    return pop_loadset(temp_file.name)
            else:
                # run it directly
                return getattr(self.engine, name)(*args)
        
        return wrapper


class OctaveWrapper:
    """Octave engine wrapper that round-trips calls involving the EEGLAB data structure through files."""

    def __init__(self, engine):
        self.engine = engine
                
    def __getattr__(self, name):
        def wrapper(*args):
            needs_roundtrip = False
            for a in args:
                if isinstance(a, dict) and a.get('nbchan') is not None:
                    needs_roundtrip = True
                    break
            if needs_roundtrip:
                # passage data through a file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.set') as temp_file:
                    pop_saveset(args[0], temp_file.name)
                    # needs to use eval since returning struct arrays is not supported
                    self.engine.eval(f"EEG = pop_loadset('{temp_file.name}');", nargout=0)
                    # TODO: marshalling of extra arguments should follow octave conventions
                    eval_str = f"EEG = {name}(EEG{',' if args[1:] else ''}{','.join([str(a) for a in args[1:]])});"
                    print(eval_str)
                    self.engine.eval(eval_str, nargout=0)
                    self.engine.eval(f"pop_saveset(EEG, '{temp_file.name}');", nargout=0)
                    return pop_loadset(temp_file.name)
            else:
                # run it directly
                return getattr(self.engine, name)(*args)
        
        return wrapper

# noinspection PyDefaultArgument
def get_eeglab(runtime: str = default_runtime, *, auto_file_roundtrip: bool = True, _cache={}):
    """Get a reference to an EEGLAB namespace that is powered
    by the specified runtime (Octave or MATLAB).

    Args:
        runtime: name of the runtime to use ('MAT' or 'OCT')
        auto_file_roundtrip: if set to True (default), EEGLAB data structures
          can be passed as arguments and returned by the engine. This is enabled
          by implicitly performing pop_saveset/pop_loadset with a temporary file
          whenever such a data structure is encountered.
        _cache: reserved for internal use

    """
    rt = runtime.lower()[:3]

    try:
        engine = _cache[rt]
    except KeyError:
        print(f"Loading {runtime} runtime...", end='', flush=True)
        # not yet loaded, do so now
        if rt == 'oct':
            from oct2py import Oct2Py, get_log
            engine = Oct2Py(logger=get_log())
            engine.logger = get_log("new_log")
            engine.logger.setLevel(logging.WARNING)
            engine.warning('off', 'backtrace')
        elif rt == 'mat':
            try:
                import matlab.engine
            except ImportError:
                raise ImportError("""\
                    The MATLAB runtime has not been installed into your Python environment.
                    To do that, make sure you have the pip executable for this python environment
                    on the path, and then run:
                    pip install /your/path/to/matlab/extern/engines/python
                                  
                    This will insert a wrapper package in the python environment that forwards
                    calls to the MATLAB runtime.                                  
                    """)
            engine = matlab.engine.start_matlab()
        else:
            raise ValueError(f"Unsupported runtime: {runtime}. Should be 'OCT' or 'MAT'")

        # On the command line, type "octave-8.4.0" OCTAVE_EXECUTABLE or OCTAVE var
        base_dir = os.path.dirname(os.path.abspath(__file__))
        path2eeglab = os.path.join(base_dir, 'eeglab')
        # path2eeglab = 'eeglab' # init >10 seconds
        engine.addpath(path2eeglab + '/functions/guifunc')
        engine.addpath(path2eeglab + '/functions/popfunc')
        engine.addpath(path2eeglab + '/functions/adminfunc')
        engine.addpath(path2eeglab + '/plugins/firfilt')
        engine.addpath(path2eeglab + '/functions/sigprocfunc')
        engine.addpath(path2eeglab + '/functions/miscfunc')
        engine.addpath(path2eeglab + '/plugins/dipfit')
        engine.addpath(path2eeglab + '/plugins/iclabel')
        engine.addpath(path2eeglab + '/plugins/clean_rawdata')
        engine.addpath(path2eeglab + '/plugins/clean_rawdata2.10')
        #res = eeglab.version()
        #print('Running EEGLAB commands in compatibility mode with Octave ' + res)

        if rt == 'oct':
            engine.logger.setLevel(logging.INFO)

        _cache[rt] = engine
        print('done.')

    # optionally wrap the engine in a file-roundtripping wrapper
    if auto_file_roundtrip:
        if rt == 'oct':
            engine = OctaveWrapper(engine)
        elif rt == 'mat':
            engine = MatlabWrapper(engine)
        else:
            raise ValueError(f"Unsupported runtime: {runtime}. Should be 'OCT' or 'MAT'")

    return engine


def eeg_checkset(EEG, eeglab=None):
    """Reference implementation of eeg_checkset()."""
    if eeglab is None:
        eeglab = get_eeglab()
    return eeglab.eeg_checkset(EEG)


def clean_drifts(EEG, Transition, Attenuation, eeglab=None):
    """Reference implementation of clean_drifts()."""
    if eeglab is None:
        eeglab = get_eeglab()
    return eeglab.clean_drifts(EEG, Transition, Attenuation)


# def pop_resample( EEG, freq): # 2 additional parameters in MATLAB (never used)
#     eeglab = get_eeglab(auto_file_roundtrip=False)
    
#     pop_saveset(EEG, './tmp.set') # 0.8 seconds
#     EEG2 = eeglab.pop_loadset('./tmp.set') # 2 seconds
#     EEG2 = eeglab.pop_resample(EEG2, freq) # 2.4 seconds
#     eeglab.pop_saveset(EEG2, './tmp2.set') # 2.4 seconds
#     EEG3 = pop_loadset('./tmp2.set') # 0.2 seconds
    
#     # delete temporary files
#     os.remove('./tmp.set')
#     os.remove('./tmp2.set')
#     return EEG3


def pop_eegfiltnew(EEG, locutoff=None,hicutoff=None,revfilt=False,plotfreqz=False):
    eeglab = get_eeglab(auto_file_roundtrip=False)
    # error if locutoff and hicutoff are none
    if locutoff==None and hicutoff==None:
        raise('Cannot have low cutoff and high cutoff not defined')
    
    pop_saveset(EEG, './tmp.set') # 0.8 seconds
    EEG2 = eeglab.pop_loadset('./tmp.set') # 2 seconds
    EEG3 = eeglab.pop_eegfiltnew(EEG2, 'locutoff',locutoff,'hicutoff',hicutoff,'revfilt',revfilt,'plotfreqz',plotfreqz)
    eeglab.pop_saveset(EEG3, './tmp2.set') # 2.4 seconds
    EEG4 = pop_loadset('./tmp2.set') # 0.2 seconds
    
    # delete temporary files
    # os.remove('./tmp.set')
    # os.remove('./tmp2.set')
    return EEG4

def clean_artifacts( EEG, ChannelCriterion=False, LineNoiseCriterion=False, FlatlineCriterion=False, BurstCriterion=False, BurstRejection=False, WindowCriterion=0, Highpass=[0.25, 0.75], WindowCriterionTolerances=[float('-inf'), 8]):
    eeglab = get_eeglab(auto_file_roundtrip=False)
    
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