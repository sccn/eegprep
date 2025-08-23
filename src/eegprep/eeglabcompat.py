# import sys
# sys.path.insert(0, 'src/')

import tempfile
from typing import *

from .pop_loadset import pop_loadset
from .pop_saveset import pop_saveset
import logging
import os
import numpy as np
from eegprep.pymat import py2mat
import scipy.io

logger = logging.getLogger(__name__)

# can be either 'OCT' (for Oct2Py) or 'MAT' (MATLAB engine)
default_runtime = 'OCT'

# directory where temporary .set files are written
temp_dir = os.path.abspath(os.path.dirname(__file__) + '../../../temp')
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir, exist_ok=True)

class MatlabWrapper:
    """MATLAB engine wrapper that round-trips calls involving the EEGLAB data structure through files."""

    def __init__(self, engine):
        self.engine = engine

    @staticmethod
    def marshal(a: Any) -> str:
        if a is True:
            return 'true'
        elif a is False:
            return 'false'
        else:
            return repr(a)

    def __getattr__(self, name):
        def wrapper(*args, **kwargs):
            # arg list
            new_args = list(args)
            kwargs_list = []
            for key, value in kwargs.items():
                kwargs_list.append(f'{key}')
                kwargs_list.append(value)      
            new_args.extend(kwargs_list)

            # issue error if kwargs are passed unless it is "nargout"
            needs_roundtrip = False
            
            # Special case for functions that return multiple outputs
            if name == 'epoch':
                eval_str = f"if iscell(args.args), [OUT1,OUT2,OUT3,OUT4,OUT5,OUT6] = {name}(args.args{{:}}); else, [OUT1,OUT2,OUT3,OUT4,OUT5,OUT6] = {name}(args.args); end; OUT = {{OUT1,OUT2,OUT3,OUT4,OUT5,OUT6}};"
            elif name == 'spheric_spline':
                eval_str = f"if iscell(args.args), [OUT1,OUT2,OUT3,OUT4] = {name}(args.args{{:}}); else, [OUT1,OUT2,OUT3,OUT4] = {name}(args.args); end; OUT = {{OUT1,OUT2,OUT3,OUT4}};"
            else:
                eval_str = f"if iscell(args.args), OUT = {name}(args.args{{:}}); else, OUT = {name}(args.args); end;"
                
            if len(args) > 0:
                if isinstance(args[0], dict) and args[0].get('trials') is not None:
                    needs_roundtrip = True
                    new_args = new_args[1:]
                    if name == 'epoch':
                        eval_str = f"if iscell(args.args), [OUT1,OUT2,OUT3,OUT4,OUT5,OUT6] = {name}(EEG,args.args{{:}}); else, [OUT1,OUT2,OUT3,OUT4,OUT5,OUT6] = {name}(EEG,args.args); end; OUT = {{OUT1,OUT2,OUT3,OUT4,OUT5,OUT6}};"
                    elif name == 'spheric_spline':
                        eval_str = f"if iscell(args.args), [OUT1,OUT2,OUT3,OUT4] = {name}(EEG,args.args{{:}}); else, [OUT1,OUT2,OUT3,OUT4] = {name}(EEG,args.args); end; OUT = {{OUT1,OUT2,OUT3,OUT4}};"
                    else:
                        eval_str = f"if iscell(args.args), OUT = {name}(EEG,args.args{{:}}); else, OUT = {name}(EEG,args.args); end;"
            
            # convert numerical list arguments to numpy arrays
            for i, arg in enumerate(new_args):
                if isinstance(arg, list) and all(isinstance(x, (int, float, np.integer, np.floating)) for x in arg):
                    new_args[i] = np.array(arg, dtype=np.float64)
                elif isinstance(arg, np.ndarray) and all(isinstance(x, (int, float, np.integer, np.floating)) for x in np.ravel(arg)):
                    new_args[i] = np.array(arg, dtype=np.float64)
                elif isinstance(arg, (int, float, np.integer, np.floating)):
                    new_args[i] = np.array(arg, dtype=np.float64)
                elif isinstance(arg, str):
                    new_args[i] = arg
                else:
                    new_args[i] = py2mat(arg) # it is unclear if the flatten function of pop_saveset is better here

            try:
                # temporary files
                temp_filename1 = tempfile.mktemp(dir=temp_dir, suffix='.set')
                temp_filename2 = tempfile.mktemp(dir=temp_dir, suffix='.mat')
                result_filename = temp_filename1 + '.result.set'
                print(f"temp_filename1: {temp_filename1}")
                print(f"temp_filename2: {temp_filename2}")
                print(f"result_filename: {result_filename}")

                # save all parameters in the temp_filename which is a .mat file
                if len(new_args) > 0:
                    if len(new_args) > 1:
                        scipy.io.savemat(temp_filename2, {'args': np.array(new_args, dtype=object)}) # object required for passing as cell array
                    else:
                        scipy.io.savemat(temp_filename2, {'args': new_args[0]}) # [0] because other increase dim of array by 1
                    self.engine.eval(f"args = load('{temp_filename2}');", nargout=0)
                else:
                    self.engine.eval("args.args = {};", nargout=0)
                        
                if needs_roundtrip:
                    # passage data through a file
                    pop_saveset(args[0], temp_filename1)
                    self.engine.eval(f"EEG = pop_loadset('{temp_filename1}');", nargout=0)
                    
                print(f"Running in MATLAB/Octave: {eval_str}")
                self.engine.eval(eval_str, nargout=0)
                
                # output
                if needs_roundtrip:
                    self.engine.eval(f"pop_saveset(OUT, '{result_filename}');", nargout=0)
                    OUT = pop_loadset(result_filename)
                    return OUT
                else:
                    self.engine.eval(f"save('-mat', '{result_filename}', 'OUT');", nargout=0)
                    OUT = scipy.io.loadmat(result_filename)['OUT']
                    
                    # Special handling for functions that return multiple outputs
                    if name == 'epoch' and isinstance(OUT, np.ndarray) and OUT.dtype == 'object':
                        # Convert MATLAB cell array to Python tuple
                        return tuple(OUT.flatten())
                    elif name == 'spheric_spline' and isinstance(OUT, np.ndarray) and OUT.dtype == 'object':
                        # Convert MATLAB cell array to Python tuple
                        return tuple(OUT.flatten())
                    else:
                        return OUT

            finally:
                # delete temporary file
                try:
                    # noinspection PyUnboundLocalVariable
                    if os.path.exists(temp_filename1):
                        #os.remove(temp_filename1)
                        pass
                    if os.path.exists(temp_filename2):
                        #os.remove(temp_filename2)
                        pass
                    # noinspection PyUnboundLocalVariable
                    if os.path.exists(result_filename):
                        #os.remove(result_filename)
                        pass
                    if os.path.exists(fdt_file := result_filename.replace('result.set', 'result.fdt')):
                        #os.remove(fdt_file)
                        pass
                except OSError as e:
                    logger.warning(f"Error deleting temporary file {temp_filename}: {e}")
            # else:
            #     # run it directly
            #     return getattr(self.engine, name)(*args)
        
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
        # On the command line, type "octave-8.4.0" OCTAVE_EXECUTABLE or OCTAVE var
        base_dir = os.path.dirname(os.path.abspath(__file__))
        path2eeglab = os.path.join(base_dir, 'eeglab')
        path2localmatlab = os.path.join(base_dir, 'matlab_local_tests')
        print("This is the path2eeglab: ", path2eeglab)

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
            # engine.cd(path2eeglab)
            # engine.eval('eeglab nogui;', nargout=0) # starting EEGLAB is too slow
        else:
            raise ValueError(f"Unsupported runtime: {runtime}. Should be 'OCT' or 'MAT'")

        engine.addpath(path2eeglab + '/functions/guifunc')
        engine.addpath(path2eeglab + '/functions/popfunc')
        engine.addpath(path2eeglab + '/functions/adminfunc')
        engine.addpath(path2eeglab + '/plugins/firfilt')
        engine.addpath(path2eeglab + '/functions/sigprocfunc')
        engine.addpath(path2eeglab + '/functions/miscfunc')
        engine.addpath(path2eeglab + '/plugins/dipfit')
        engine.addpath(path2eeglab + '/plugins/iclabel')
        engine.addpath(path2eeglab + '/plugins/picard')
        engine.addpath(path2eeglab + '/plugins/clean_rawdata')
        engine.addpath(path2eeglab + '/plugins/clean_rawdata2.10')
        engine.addpath(path2localmatlab)
        engine.cd(path2eeglab + '/plugins/clean_rawdata/private')  # to grant access to util funcs for unit testing
        
        # path2eeglab = 'eeglab' # init >10 seconds
        #res = eeglab.version()
        #print('Running EEGLAB commands in compatibility mode with Octave ' + res)

        if rt == 'oct':
            engine.logger.setLevel(logging.INFO)

        _cache[rt] = engine
        print('done.')

    # optionally wrap the engine in a file-roundtripping wrapper
    if auto_file_roundtrip:
        if rt == 'oct':
            engine = MatlabWrapper(engine)
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
