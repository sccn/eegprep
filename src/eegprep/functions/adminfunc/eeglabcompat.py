"""EEGLAB compatibility utilities."""

import importlib
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io

from eegprep.functions.adminfunc.pymat import py2mat
from ..popfunc.pop_loadset import pop_loadset
from ..popfunc.pop_saveset import pop_saveset

logger = logging.getLogger(__name__)
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PACKAGE_ROOT.parent.parent
EEGLAB_ROOT_ENV = "EEGPREP_EEGLAB_ROOT"

# can be either 'OCT' (for Oct2Py) or 'MAT' (MATLAB engine)
default_runtime = 'MAT'

# directory where temporary .set files are written
# use environment variable if it exists
if 'TEMP_DIR' in os.environ:
    temp_dir = os.environ['TEMP_DIR']
elif 'TMPDIR' in os.environ:
    temp_dir = os.environ['TMPDIR']
else:
    temp_dir = str(REPO_ROOT / 'temp')
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir, exist_ok=True)


def _prepare_matlab_arg(arg: Any) -> Any:
    """Return one Python value in the shape expected by MATLAB ``savemat``."""
    if isinstance(arg, (list, tuple)) and len(arg) == 0:
        return np.array([], dtype=np.float64)
    if isinstance(arg, (list, tuple)) and all(isinstance(x, str) for x in arg):
        return np.array(arg, dtype=object).reshape(1, -1)
    if isinstance(arg, list) and all(isinstance(x, (int, float, np.integer, np.floating)) for x in arg):
        return np.array(arg, dtype=np.float64)
    if isinstance(arg, np.ndarray) and all(isinstance(x, (int, float, np.integer, np.floating)) for x in np.ravel(arg)):
        return np.array(arg, dtype=np.float64)
    if isinstance(arg, (int, float, np.integer, np.floating)):
        return np.array(arg, dtype=np.float64)
    if isinstance(arg, str):
        return arg
    return py2mat(arg)


def _resolve_eeglab_root() -> Path:
    """Return an external EEGLAB checkout for MATLAB/Octave parity calls."""
    candidates = []
    env_root = os.environ.get(EEGLAB_ROOT_ENV)
    if env_root:
        candidates.append(Path(env_root).expanduser())
    candidates.append(REPO_ROOT.parent / 'eeglab')

    for candidate in candidates:
        if (candidate / 'eeglab.m').is_file():
            return candidate

    raise ImportError(
        "EEGLAB reference checkout not found. Set EEGPREP_EEGLAB_ROOT to an "
        "EEGLAB checkout, or place one alongside the repository, when running "
        "MATLAB/Octave parity helpers."
    )


class MatlabWrapper:
    """MATLAB engine wrapper that round-trips calls involving the EEGLAB data structure through files."""

    def __init__(self, engine):
        """Initialize the MatlabWrapper.

        Parameters
        ----------
        engine : object
            The MATLAB or Octave engine.
        """
        self.engine = engine

    @staticmethod
    def marshal(a: Any) -> str:
        """Marshal a value to string representation.

        Parameters
        ----------
        a : Any
            Value to marshal.

        Returns
        -------
        str
            String representation.
        """
        if a is True:
            return 'true'
        elif a is False:
            return 'false'
        else:
            return repr(a)

    def __getattr__(self, name):
        """Get attribute, returning a wrapper for MATLAB functions.

        Parameters
        ----------
        name : str
            Name of the attribute.

        Returns
        -------
        callable
            Wrapper function.
        """

        def wrapper(*args, **kwargs):
            nargout = kwargs.pop("nargout", None)
            # arg list
            new_args = list(args)
            kwargs_list = []
            for key, value in kwargs.items():
                kwargs_list.append(f'{key}')
                kwargs_list.append(value)
            new_args.extend(kwargs_list)

            needs_roundtrip = False

            # Special case for functions that return multiple outputs
            if nargout is not None and int(nargout) > 1:
                output_names = ",".join(f"OUT{i}" for i in range(1, int(nargout) + 1))
                output_cell = ",".join(f"OUT{i}" for i in range(1, int(nargout) + 1))
                eval_str = f"if iscell(args.args), [{output_names}] = {name}(args.args{{:}}); else, [{output_names}] = {name}(args.args); end; OUT = {{{output_cell}}};"
            elif name == 'epoch':
                eval_str = f"if iscell(args.args), [OUT1,OUT2,OUT3,OUT4,OUT5,OUT6] = {name}(args.args{{:}}); else, [OUT1,OUT2,OUT3,OUT4,OUT5,OUT6] = {name}(args.args); end; OUT = {{OUT1,OUT2,OUT3,OUT4,OUT5,OUT6}};"
            elif name == 'spheric_spline':
                eval_str = f"if iscell(args.args), [OUT1,OUT2,OUT3,OUT4] = {name}(args.args{{:}}); else, [OUT1,OUT2,OUT3,OUT4] = {name}(args.args); end; OUT = {{OUT1,OUT2,OUT3,OUT4}};"
            else:
                eval_str = f"if iscell(args.args), OUT = {name}(args.args{{:}}); else, OUT = {name}(args.args); end;"

            if len(args) > 0:
                if isinstance(args[0], dict) and args[0].get('trials') is not None:
                    needs_roundtrip = True
                    new_args = new_args[1:]
                    if nargout is not None and int(nargout) > 1:
                        output_names = ",".join(f"OUT{i}" for i in range(1, int(nargout) + 1))
                        output_cell = ",".join(f"OUT{i}" for i in range(1, int(nargout) + 1))
                        eval_str = f"if iscell(args.args), [{output_names}] = {name}(EEG,args.args{{:}}); else, [{output_names}] = {name}(EEG,args.args); end; OUT = {{{output_cell}}};"
                    elif name == 'epoch':
                        eval_str = f"if iscell(args.args), [OUT1,OUT2,OUT3,OUT4,OUT5,OUT6] = {name}(EEG,args.args{{:}}); else, [OUT1,OUT2,OUT3,OUT4,OUT5,OUT6] = {name}(EEG,args.args); end; OUT = {{OUT1,OUT2,OUT3,OUT4,OUT5,OUT6}};"
                    elif name == 'spheric_spline':
                        eval_str = f"if iscell(args.args), [OUT1,OUT2,OUT3,OUT4] = {name}(EEG,args.args{{:}}); else, [OUT1,OUT2,OUT3,OUT4] = {name}(EEG,args.args); end; OUT = {{OUT1,OUT2,OUT3,OUT4}};"
                    else:
                        eval_str = f"if iscell(args.args), OUT = {name}(EEG,args.args{{:}}); else, OUT = {name}(EEG,args.args); end;"

            # convert numerical list arguments to numpy arrays
            for i, arg in enumerate(new_args):
                new_args[i] = _prepare_matlab_arg(arg)

            try:
                # temporary files
                with tempfile.NamedTemporaryFile(dir=temp_dir, suffix='.set', delete=False) as temp_file1:
                    temp_filename1 = temp_file1.name
                with tempfile.NamedTemporaryFile(dir=temp_dir, suffix='.mat', delete=False) as temp_file2:
                    temp_filename2 = temp_file2.name
                result_filename = temp_filename1 + '.result.set'
                result_extra_filename = temp_filename1 + '.result.mat'
                logger.debug("MATLAB roundtrip input set path: %s", temp_filename1)
                logger.debug("MATLAB roundtrip args path: %s", temp_filename2)
                logger.debug("MATLAB roundtrip result set path: %s", result_filename)

                # save all parameters in the temp_filename which is a .mat file
                if len(new_args) > 0:
                    if len(new_args) > 1:
                        scipy.io.savemat(
                            temp_filename2, {'args': np.array(new_args, dtype=object)}
                        )  # object required for passing as cell array
                    else:
                        scipy.io.savemat(
                            temp_filename2, {'args': new_args[0]}
                        )  # [0] because other increase dim of array by 1
                    self.engine.eval(f"args = load('{temp_filename2}');", nargout=0)
                else:
                    self.engine.eval("args.args = {};", nargout=0)

                if needs_roundtrip:
                    # passage data through a file
                    pop_saveset(args[0], temp_filename1)
                    self.engine.eval(f"EEG = pop_loadset('{temp_filename1}');", nargout=0)

                logger.debug("Running in MATLAB/Octave: %s", eval_str)
                self.engine.eval(eval_str, nargout=0)

                # output
                # Functions that return numeric arrays instead of EEG structures
                numeric_output_functions = ['eeg_autocorr', 'eeg_autocorr_fftw', 'eeg_autocorr_welch']

                if (
                    needs_roundtrip
                    and nargout is not None
                    and int(nargout) > 1
                    and name not in numeric_output_functions
                ):
                    self.engine.eval(f"pop_saveset(OUT1, '{result_filename}');", nargout=0)
                    OUT = pop_loadset(result_filename)
                    extra_names = ",".join(f"'OUT{i}'" for i in range(2, int(nargout) + 1))
                    self.engine.eval(f"save('-mat', '{result_extra_filename}', {extra_names});", nargout=0)
                    extra_data = scipy.io.loadmat(result_extra_filename, squeeze_me=True)
                    extras = tuple(extra_data[f"OUT{i}"] for i in range(2, int(nargout) + 1))
                    return (OUT, *extras)
                elif (needs_roundtrip or name == 'pop_loadset') and name not in numeric_output_functions:
                    # Always round-trip OUT for pop_loadset to get a proper Python EEG dict
                    self.engine.eval(f"pop_saveset(OUT, '{result_filename}');", nargout=0)
                    OUT = pop_loadset(result_filename)
                    return OUT
                else:
                    self.engine.eval(f"save('-mat', '{result_filename}', 'OUT');", nargout=0)
                    OUT = scipy.io.loadmat(result_filename)['OUT']

                    # Special handling for functions that return multiple outputs
                    if (
                        nargout is not None
                        and int(nargout) > 1
                        and isinstance(OUT, np.ndarray)
                        and OUT.dtype == 'object'
                    ):
                        return tuple(OUT.flatten())
                    elif name == 'epoch' and isinstance(OUT, np.ndarray) and OUT.dtype == 'object':
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
                        os.remove(temp_filename1)
                    if os.path.exists(temp_filename2):
                        os.remove(temp_filename2)
                    # noinspection PyUnboundLocalVariable
                    if os.path.exists(result_filename):
                        os.remove(result_filename)
                    if os.path.exists(result_filename.replace('result.set', 'result.fdt')):
                        os.remove(result_filename.replace('result.set', 'result.fdt'))
                    if os.path.exists(result_extra_filename):
                        os.remove(result_extra_filename)
                except OSError as e:
                    logger.warning(f"Error deleting temporary file(s) in temp dir {temp_dir}: {e}")
            # else:
            #     # run it directly
            #     return getattr(self.engine, name)(*args)

        return wrapper


# noinspection PyDefaultArgument
def get_eeglab(runtime: str = default_runtime, *, auto_file_roundtrip: bool = True, _cache={}):
    """Get a reference to an EEGLAB namespace that is powered by the specified runtime (Octave or MATLAB).

    Args
    ----
    runtime : name of the runtime to use ('MAT' or 'OCT')
    auto_file_roundtrip : if set to True (default), EEGLAB data structures
      can be passed as arguments and returned by the engine. This is enabled
      by implicitly performing pop_saveset/pop_loadset with a temporary file
      whenever such a data structure is encountered.
    _cache : reserved for internal use
    """
    rt = runtime.lower()[:3]

    try:
        engine = _cache[rt]
    except KeyError:
        logger.info("Loading %s runtime...", runtime)
        # On the command line, type "octave-8.4.0" OCTAVE_EXECUTABLE or OCTAVE var
        path2eeglab = str(_resolve_eeglab_root())
        matlab_test_dir = REPO_ROOT / 'tests' / 'matlab'
        scripts_dir = str(REPO_ROOT / 'scripts')
        logger.debug("EEGLAB reference path: %s", path2eeglab)

        # not yet loaded, do so now
        if rt == 'oct':
            from oct2py import Oct2Py, get_log

            engine = Oct2Py(logger=get_log())
            engine.logger = get_log("new_log")
            engine.logger.setLevel(logging.WARNING)
            engine.warning('off', 'backtrace')
        elif rt == 'mat':
            try:
                # Use import_module so ty can run before MATLAB Engine is installed.
                matlab_engine = importlib.import_module("matlab.engine")
            except ImportError:
                raise ImportError("""\
                    The MATLAB runtime has not been installed into your Python environment.
                    To do that, make sure you have the pip executable for this python environment
                    on the path, and then run:
                    pip install /your/path/to/matlab/extern/engines/python

                    This will insert a wrapper package in the python environment that forwards
                    calls to the MATLAB runtime.
                    """)
            engine = matlab_engine.start_matlab()
            # engine.cd(path2eeglab)
            # engine.eval('eeglab nogui;', nargout=0) # starting EEGLAB is too slow
        else:
            raise ValueError(f"Unsupported runtime: {runtime}. Should be 'OCT' or 'MAT'")

        engine.addpath(path2eeglab + '/functions/guifunc')
        engine.addpath(path2eeglab + '/functions/popfunc')
        engine.addpath(path2eeglab + '/functions/adminfunc')
        engine.addpath(path2eeglab + '/functions/studyfunc')
        engine.addpath(path2eeglab + '/plugins/firfilt')
        engine.addpath(path2eeglab + '/functions/sigprocfunc')
        engine.addpath(path2eeglab + '/functions/miscfunc')
        engine.addpath(path2eeglab + '/plugins/dipfit')
        engine.addpath(path2eeglab + '/plugins/ICLabel')
        engine.addpath(path2eeglab + '/plugins/EEG-BIDS')
        engine.addpath(path2eeglab + '/plugins/picard')
        engine.addpath(path2eeglab + '/plugins/picard/matlab_octave')
        engine.addpath(path2eeglab + '/plugins/clean_rawdata')
        amica_path = path2eeglab + '/plugins/amica'
        if os.path.isdir(amica_path):
            engine.addpath(amica_path, nargout=0)
        if matlab_test_dir.is_dir():
            engine.addpath(str(matlab_test_dir))
        engine.addpath(scripts_dir)
        engine.cd(path2eeglab + '/plugins/clean_rawdata/private')  # to grant access to util funcs for unit testing

        # path2eeglab = 'eeglab' # init >10 seconds
        # res = eeglab.version()
        # print('Running EEGLAB commands in compatibility mode with Octave ' + res)

        if rt == 'oct':
            engine.logger.setLevel(logging.INFO)

        _cache[rt] = engine
        logger.info("Loaded %s runtime.", runtime)

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
    """Check the EEG dataset."""
    if eeglab is None:
        eeglab = get_eeglab()
    return eeglab.eeg_checkset(EEG)


def clean_drifts(EEG, Transition, Attenuation, eeglab=None):
    """Remove drifts from EEG data."""
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


def pop_eegfiltnew(EEG, locutoff=None, hicutoff=None, revfilt=False, plotfreqz=False):
    """Filter EEG data using EEGLAB's pop_eegfiltnew.

    Parameters
    ----------
    EEG : dict
        EEG data structure.
    locutoff : float, optional
        Low cutoff frequency.
    hicutoff : float, optional
        High cutoff frequency.
    revfilt : bool, optional
        Reverse filter.
    plotfreqz : bool, optional
        Plot frequency response.

    Returns
    -------
    dict
        Filtered EEG data.
    """
    # error if locutoff and hicutoff are none
    if locutoff is None and hicutoff is None:
        raise ValueError('Cannot have low cutoff and high cutoff not defined')

    # Convert None to empty array for MATLAB
    if locutoff is None:
        locutoff = []
    if hicutoff is None:
        hicutoff = []

    # Use wrapper which handles EEG struct conversion via file roundtrip
    eeglab = get_eeglab(auto_file_roundtrip=True)
    return eeglab.pop_eegfiltnew(
        EEG, 'locutoff', locutoff, 'hicutoff', hicutoff, 'revfilt', revfilt, 'plotfreqz', plotfreqz
    )


def _matlab_false_or_off(value: Any) -> bool:
    if isinstance(value, str):
        return value == 'off'
    if value is False:
        return True
    if isinstance(value, (int, float, np.integer, np.floating)):
        return value == 0
    return False


def clean_artifacts(
    EEG,
    ChannelCriterion=False,
    LineNoiseCriterion=False,
    FlatlineCriterion=False,
    BurstCriterion=False,
    BurstRejection=False,
    WindowCriterion=0,
    Highpass=[0.25, 0.75],
    WindowCriterionTolerances=[float('-inf'), 8],
):
    """Clean artifacts from EEG data using EEGLAB's clean_artifacts.

    Parameters
    ----------
    EEG : dict
        EEG data structure.
    ChannelCriterion : bool or str, optional
        Channel criterion.
    LineNoiseCriterion : bool or str, optional
        Line noise criterion.
    FlatlineCriterion : bool or str, optional
        Flatline criterion.
    BurstCriterion : bool or str, optional
        Burst criterion.
    BurstRejection : bool or str, optional
        Burst rejection.
    WindowCriterion : float, optional
        Window criterion.
    Highpass : list or str, optional
        Highpass filter.
    WindowCriterionTolerances : list, optional
        Window criterion tolerances.

    Returns
    -------
    dict
        Cleaned EEG data.
    """
    eeglab = get_eeglab(auto_file_roundtrip=False)

    if _matlab_false_or_off(ChannelCriterion):
        ChannelCriterion = 'off'

    if _matlab_false_or_off(LineNoiseCriterion):
        LineNoiseCriterion = 'off'

    if _matlab_false_or_off(FlatlineCriterion):
        FlatlineCriterion = 'off'

    if _matlab_false_or_off(BurstCriterion):
        BurstCriterion = 'off'

    if _matlab_false_or_off(Highpass):
        Highpass = 'off'

    if _matlab_false_or_off(BurstRejection):
        BurstRejection = 'off'
    else:
        BurstRejection = 'on'

    with tempfile.TemporaryDirectory(prefix="eegprep_clean_artifacts_") as workdir:
        input_path = Path(workdir) / "input.set"
        output_path = Path(workdir) / "output.set"
        pop_saveset(EEG, input_path)
        EEG2 = eeglab.pop_loadset(str(input_path))
        EEG3 = eeglab.clean_artifacts(
            EEG2,
            'ChannelCriterion',
            ChannelCriterion,
            'LineNoiseCriterion',
            LineNoiseCriterion,
            'FlatlineCriterion',
            FlatlineCriterion,
            'BurstCriterion',
            BurstCriterion,
            'BurstRejection',
            BurstRejection,
            'WindowCriterion',
            WindowCriterion,
            'Highpass',
            Highpass,
            'WindowCriterionTolerances',
            WindowCriterionTolerances,
        )
        eeglab.pop_saveset(EEG3, str(output_path))
        return pop_loadset(output_path)
