# import sys
# sys.path.insert(0, 'src/')

import tempfile
from typing import *

from .pop_loadset import pop_loadset
from .pop_saveset import pop_saveset
import logging
import os
import numpy as np

logger = logging.getLogger(__name__)

# can be either 'OCT' (for Oct2Py) or 'MAT' (MATLAB engine)
default_runtime = 'OCT'

# directory where temporary .set files are written
temp_dir = os.path.abspath(os.path.dirname(__file__) + '../../../temp')
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir, exist_ok=True)





# convert list of arbitrary dicts to struct array
def py2mat(dicts):
    """
    Convert a list of dictionaries to a NumPy structured array.
    Handles nested dictionaries and lists recursively.
    """
    if dicts is None:
        return np.array([], dtype=object)
    
    # Handle single dictionary input by wrapping in a list
    if isinstance(dicts, dict):
        dicts = [dicts]
        
    if not isinstance(dicts, (list, tuple)):
        return dicts
    
    # Check if this is a list of dictionaries (the expected input)
    if dicts and not all(isinstance(item, dict) for item in dicts):
        # If it's a mixed list, we can't convert it to a struct array
        # Return it as an object array instead
        return np.array(dicts, dtype=object)
    
    def process_value(value):
        """Recursively process values, converting nested structures"""
        if value is None:
            # Return None as-is, will be handled later
            return None
        elif isinstance(value, dict):
            # Convert single dict to struct array with one element
            if value:
                return py2mat([value])[0]
            else:
                # Empty dict - return empty array with object dtype
                return np.array([], dtype=object)
        elif isinstance(value, np.ndarray) and value.size > 0:
            try:
                # Check if it's an array of dicts
                if isinstance(value.flat[0], dict):
                    # Convert numpy array of dicts to struct array
                    return py2mat(value.tolist())
                else:
                    # Keep regular numpy arrays as-is
                    return value
            except (IndexError, AttributeError):
                # If we can't access flat[0], just return the array as-is
                return value
        elif isinstance(value, (list, tuple)) and value and isinstance(value[0], dict):
            # Convert list/tuple of dicts to struct array
            return py2mat(value)
        elif isinstance(value, np.ndarray):
            # Keep numpy arrays as-is
            return value
        elif isinstance(value, (list, tuple)) and not isinstance(value, str):
            # Convert regular list/tuple to numpy array (but not strings)
            if not value:
                return np.array([], dtype=object)
            try:
                # Try to create a numpy array, but handle inhomogeneous sequences
                return np.array(value)
            except ValueError:
                # If the sequence is inhomogeneous, create an object array
                return np.array(value, dtype=object)
        else:
            return value
    
    # Collect all unique keys and determine their types and sizes
    all_keys = set()
    key_types = {}
    key_max_lengths = {}
    
    for d in dicts:
        for k, v in d.items():
            all_keys.add(k)
            
            # Process the value recursively
            processed_v = process_value(v)
            
            # Determine the appropriate NumPy dtype for this value
            if isinstance(processed_v, str):
                # For strings, we need to track the maximum length
                if k not in key_max_lengths:
                    key_max_lengths[k] = len(processed_v)
                else:
                    key_max_lengths[k] = max(key_max_lengths[k], len(processed_v))
                key_types[k] = 'U'  # Unicode string type
            elif isinstance(processed_v, (int, np.integer)):
                if k not in key_types:
                    key_types[k] = int
                elif key_types[k] != int and key_types[k] != object:
                    key_types[k] = object
            elif isinstance(processed_v, (float, np.floating)):
                if k not in key_types:
                    key_types[k] = float
                elif key_types[k] not in [float, object]:
                    key_types[k] = object
            elif isinstance(processed_v, bool):
                if k not in key_types:
                    key_types[k] = bool
                elif key_types[k] != bool and key_types[k] != object:
                    key_types[k] = object
            elif isinstance(processed_v, np.ndarray):
                # For arrays (including nested struct arrays), use object type
                key_types[k] = object
            elif processed_v is None:
                # For None values, we'll determine type from other instances
                if k not in key_types:
                    key_types[k] = object
            else:
                # For other types, use object
                key_types[k] = object
    
    # Create dtype from all keys
    dtype_list = []
    for k in sorted(all_keys):
        if key_types[k] == 'U':
            # For Unicode strings, specify the maximum length
            max_len = key_max_lengths.get(k, 1)
            dtype_list.append((k, f'U{max_len}'))
        else:
            dtype_list.append((k, key_types[k]))
    
    dtype = np.dtype(dtype_list)
    
    # Create structured array
    struct_array = np.empty(len(dicts), dtype=dtype)
    
    # Fill the array
    for i, d in enumerate(dicts):
        for k in all_keys:
            value = d.get(k, None)
            processed_value = process_value(value)
            
            if processed_value is None:
                # Handle None values based on the field type
                if key_types[k] == 'U':
                    # For string fields, use empty string instead of None
                    struct_array[i][k] = ''
                elif key_types[k] == int:
                    # For int fields, use 0
                    struct_array[i][k] = 0
                elif key_types[k] == float:
                    # For float fields, use NaN
                    struct_array[i][k] = np.nan
                elif key_types[k] == bool:
                    # For bool fields, use False
                    struct_array[i][k] = False
                else:
                    # For object fields, use empty array with object dtype instead of None
                    struct_array[i][k] = np.array([], dtype=object)
            else:
                struct_array[i][k] = processed_value
    
    return struct_array















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
            # issue error if kwargs are passed unless it is "nargout"
            needs_roundtrip = False
            for a in args:
                if isinstance(a, dict) and a.get('nbchan') is not None:
                    needs_roundtrip = True
                    break
            # arg list
            new_args = list(args)
            kwargs_list = []
            for key, value in kwargs.items():
                kwargs_list.append(f'{key}')
                kwargs_list.append(value)      
            new_args.extend(kwargs_list)
            
            # convert numerical list arguments to numpy arrays
            for i, arg in enumerate(new_args):
                if i > 0: 
                    if isinstance(arg, list) and all(isinstance(x, (int, float)) for x in arg):
                        new_args[i] = np.array(arg, dtype=np.float64)
                    elif isinstance(arg, (int, float, np.integer, np.floating)):
                        new_args[i] = np.array(arg, dtype=np.float64)
                    else:
                        new_args[i] = py2mat(arg)
            
            if needs_roundtrip:
                # passage data through a file
                try:
                    temp_filename1 = tempfile.mktemp(dir=temp_dir, suffix='.set')
                    temp_filename2 = tempfile.mktemp(dir=temp_dir, suffix='.mat')
                    print(f"temp_filename1: {temp_filename1}")
                    print(f"temp_filename2: {temp_filename2}")
                    result_filename = temp_filename1 + '.result.set'
                    pop_saveset(args[0], temp_filename1)
                    self.engine.eval(f"EEG = pop_loadset('{temp_filename1}');", nargout=0)
                    
                    # save all parameters in the temp_filename which is a .mat file
                    if len(new_args) > 1:
                        # cell_array = np.empty(len(new_args)-1, dtype=object)
                        # cell_array[:] = [py2mat(arg) for arg in new_args[1:]]
                        # cell_array = np.ravel(cell_array)
                        import scipy.io
                        # scipy.io.savemat(temp_filename2, {'args': [py2mat(arg) for arg in new_args[1:]]})
                        scipy.io.savemat(temp_filename2, {'args': new_args[1:]})
                        self.engine.eval(f"args = load('{temp_filename2}');", nargout=0)
                        eval_str = f"if iscell(args.args), EEG = {name}(EEG,args.args{{:}}); else, EEG = {name}(EEG,args.args); end;"
                    else:
                        eval_str = f"EEG = {name}(EEG);"
                    
                    # needs to use eval since returning struct arrays is not supported
                    temp_filename1 = 'adsfdsa' # ***************************************** FILE NOT ERASED
                    temp_filename2 = 'adsfdsa' # ***************************************** FILE NOT ERASED

                    # TODO: marshalling of extra arguments should follow octave conventions
                    print(f"Running in MATLAB: {eval_str}")
                    self.engine.eval(eval_str, nargout=0)
                    self.engine.eval(f"pop_saveset(EEG, '{result_filename}');", nargout=0)
                    return pop_loadset(result_filename)

                finally:
                    # delete temporary file
                    try:
                        # noinspection PyUnboundLocalVariable
                        if os.path.exists(temp_filename1):
                            os.remove(temp_filename1)
                        if os.path.exists(temp_filename2):
                            os.remove(temp_filename2)
                        # noinspection PyUnboundLocalVariable
                        if os.path.exists(result_filename):
                            os.remove(result_filename)
                        if os.path.exists(fdt_file := result_filename.replace('result.set', 'result.fdt')):
                            os.remove(fdt_file)
                    except OSError as e:
                        logger.warning(f"Error deleting temporary file {temp_filename}: {e}")
            else:
                # run it directly
                return getattr(self.engine, name)(*args)
        
        return wrapper


# class OctaveWrapper:
#     """Octave engine wrapper that round-trips calls involving the EEGLAB data structure through files."""

#     def __init__(self, engine):
#         self.engine = engine
                
#     def __getattr__(self, name):
#         def wrapper(*args):
#             needs_roundtrip = False
#             for a in args:
#                 if isinstance(a, dict) and a.get('nbchan') is not None:
#                     needs_roundtrip = True
#                     break
#             if needs_roundtrip:
#                 # passage data through a file
#                 with tempfile.NamedTemporaryFile(dir=temp_dir, delete=True, suffix='.set') as temp_file:
#                     pop_saveset(args[0], temp_file.name)
#                     # Ensure the file is fully written before proceeding
#                     temp_file.flush()
#                     os.fsync(temp_file.fileno())  # Force write to disk to avoid timing issues
#                     # Needs to use eval since returning struct arrays is not supported in Octave
#                     self.engine.eval(f"EEG = pop_loadset('{temp_file.name}');", nargout=0)
#                     # TODO: marshalling of extra arguments should follow octave conventions
#                     eval_str = f"EEG = {name}(EEG{',' if args[1:] else ''}{','.join([repr(a) for a in args[1:]])});"
#                     print("This is the eval_str: ", eval_str)
#                     self.engine.eval(eval_str, nargout=0)
#                     self.engine.eval(f"pop_saveset(EEG, '{temp_file.name}');", nargout=0)
#                     return pop_loadset(temp_file.name)
#             else:
#                 # run it directly
#                 return getattr(self.engine, name)(*args)
        
#         return wrapper

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
        print("This is the path2eeglab: ", path2eeglab)

        # not yet loaded, do so now
        if rt == 'oct':
            from oct2py import Oct2Py, get_log
            engine = Oct2Py(logger=get_log())
            engine.logger = get_log("new_log")
            engine.logger.setLevel(logging.WARNING)
            engine.warning('off', 'backtrace')
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
            engine.cd(path2eeglab)
            engine.eval('eeglab nogui;', nargout=0)
        else:
            raise ValueError(f"Unsupported runtime: {runtime}. Should be 'OCT' or 'MAT'")

        # path2eeglab = 'eeglab' # init >10 seconds
        engine.cd(path2eeglab + '/plugins/clean_rawdata/private')  # to grant access to util funcs for unit testing
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
