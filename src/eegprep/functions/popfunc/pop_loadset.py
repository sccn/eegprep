"""EEGLAB dataset loading utilities."""

import os
from pathlib import Path

import h5py
import numpy as np
import scipy.io

from eegprep.functions.adminfunc.storage import memmap_enabled, memmap_fdt, read_fdt
from eegprep.functions.popfunc._file_io import normalize_icachansind
from eegprep.functions.popfunc._pop_utils import parse_key_value_args
from eegprep.functions.popfunc.pop_loadset_h5 import pop_loadset_h5
# Allows access using . notation
# class EEG:
#     def __init__(self, **kwargs):
#         self.__dict__.update(kwargs)
#     def __getitem__(self, key):
#         return self.__dict__[key]
#     def __setitem__(self, key, value):
#         self.__dict__[key] = value

default_empty = np.array([])
# default_empty = None


def loadset(file_path):
    """Load EEGLAB dataset from file (alias for pop_loadset)."""
    return pop_loadset(file_path)


def pop_loadset(file_path=None, *args, loadmode="all", memmap=None, **kwargs):
    """Load EEGLAB dataset from .set or .mat file.

    Parameters
    ----------
    file_path : str
        Path to the EEGLAB .set file.

    Returns
    -------
    dict
        EEGLAB dataset dictionary.
    """
    from eegprep.functions.adminfunc.eeg_checkset import eeg_checkset

    file_path, loadmode, use_memmap = _load_options(file_path, args, kwargs, loadmode, memmap)
    if loadmode != "all":
        raise NotImplementedError("pop_loadset currently supports loadmode='all' only; storedisk uses eeg_retrieve().")

    def new_check(obj):
        # check if obj is a dictionary and apply recursively the function to each object not changing the struture of the dictionary
        if isinstance(obj, dict):
            return {key: new_check(obj[key]) for key in obj}
        # check if obj is a numpy array and apply recursively the function to each object not changing the struture of the array
        elif isinstance(obj, list):
            if len(obj) == 0:
                return default_empty
            else:
                return [new_check(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            # check if empty and return none
            if obj.size == 0:
                return default_empty
            # check if it is a numeric array
            elif obj.dtype.kind in ['i', 'u', 'f', 'c']:
                return obj
            else:
                return np.array([new_check(item) for item in obj], dtype=object)
        # check if it is a scalar or a string and return it
        elif np.isscalar(obj) or isinstance(obj, str):
            return obj
        # check if obj is a mat_struct object and convert it to a dictionary
        elif isinstance(obj, scipy.io.matlab.mat_struct) or isinstance(obj, scipy.io.matlab.mio5_params.mat_struct):
            dict_obj = {}
            for field_name in obj._fieldnames:
                if field_name in ['tracking']:
                    # used for fields that this code can't yet parse
                    field_value = '<unsupported>'
                else:
                    field_value = getattr(obj, field_name)
                dict_obj[field_name] = new_check(field_value)
            return dict_obj

    # Load MATLAB file. MAT v7.3 files are HDF5; older v5/v7 files are not.
    # Dispatch on the real format instead of treating every scipy error as "must be HDF5".
    loaded_with_h5 = _is_hdf5_file(file_path)
    if loaded_with_h5:
        EEG = pop_loadset_h5(file_path)
    else:
        EEG = scipy.io.loadmat(file_path, struct_as_record=False, squeeze_me=True, appendmat=False)
        EEG = new_check(EEG)
        if 'EEG' in EEG:
            EEG = EEG['EEG']

    EEG['filepath'] = os.path.dirname(file_path)
    EEG['filename'] = os.path.basename(file_path)

    # delete keys '__header__', '__version__', '__globals__'
    if '__header__' in EEG:
        del EEG['__header__']
    if '__version__' in EEG:
        del EEG['__version__']
    if '__globals__' in EEG:
        del EEG['__globals__']

    # Convert MATLAB-loaded 1-based doubles to EEGPrep's 0-based integer indices.
    if 'icachansind' in EEG:
        EEG['icachansind'] = normalize_icachansind(EEG['icachansind'], matlab_one_based=not loaded_with_h5)

    if not loaded_with_h5:
        _load_sidecar_data(EEG, Path(file_path), use_memmap=use_memmap)

    EEG = eeg_checkset(EEG)
    EEG.pop("changes_not_saved", None)
    EEG["saved"] = "justloaded"

    # check if EEG['urchan'] is 0-based
    if len(EEG['chanlocs']) > 0 and 'urchan' in EEG['chanlocs'][0]:
        for i in range(len(EEG['chanlocs'])):
            EEG['chanlocs'][i]['urchan'] = EEG['chanlocs'][i]['urchan'] - 1

    # check if EEG['chanlocs'][i]['urevent'] is 0-based
    if len(EEG['event']) > 0 and 'urevent' in EEG['event'][0]:
        for i in range(len(EEG['event'])):
            if 'urevent' in EEG['event'][i] and EEG['event'][i]['urevent'] is not None:
                EEG['event'][i]['urevent'] = EEG['event'][i]['urevent'] - 1

    return EEG


def _is_hdf5_file(file_path):
    """Return True when the file is HDF5 (MAT v7.3).

    MAT v7.3 files carry a text header in an HDF5 userblock, so the signature is not at
    byte 0; ``h5py.is_hdf5`` checks the userblock offsets HDF5 actually uses.
    """
    return h5py.is_hdf5(os.fspath(file_path))


def _load_options(file_path, args, kwargs, loadmode, memmap):
    known_keys = {"filename", "filepath", "loadmode", "memmap", "check", "verbose", "eeg"}
    if isinstance(file_path, str) and file_path.lower() in known_keys:
        options = parse_key_value_args((file_path, *args), kwargs, lowercase_keys=True, lowercase_kwargs=True)
        filename = options.pop("filename", None)
    else:
        options = parse_key_value_args(args, kwargs, lowercase_keys=True, lowercase_kwargs=True)
        filename = file_path
    filepath = options.pop("filepath", None)
    loadmode = str(options.pop("loadmode", loadmode) or "all").lower()
    memmap = options.pop("memmap", memmap)
    options.pop("check", None)
    options.pop("verbose", None)
    if "eeg" in options:
        eeg = options.pop("eeg")
        filename = str(Path(str(eeg.get("filepath") or "")) / str(eeg.get("filename") or ""))
    if options:
        raise ValueError(f"Unsupported pop_loadset option(s): {', '.join(sorted(options))}")
    if filename is None:
        raise ValueError("file_path argument is required")
    path = Path(os.fspath(filename))
    if filepath not in {None, ""} and not path.is_absolute():
        path = Path(os.fspath(filepath)) / path
    use_memmap = memmap_enabled() if memmap is None else _is_on(memmap)
    return str(path), loadmode, use_memmap


def _load_sidecar_data(EEG, file_path: Path, *, use_memmap: bool) -> None:
    data_value = EEG.get("data")
    datfile = _string_value(EEG.get("datfile"))
    if not datfile and isinstance(data_value, str) and data_value not in {"", "in set file"}:
        datfile = data_value
    if not datfile:
        return
    datfile_path = Path(datfile)
    if not datfile_path.is_absolute():
        datfile_path = file_path.parent / datfile_path.name
    EEG["datfile"] = datfile_path.name
    EEG["data"] = memmap_fdt(datfile_path, EEG) if use_memmap else read_fdt(datfile_path, EEG)


def _string_value(value):
    if value is None:
        return ""
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return ""
        if value.size == 1:
            return str(value.reshape(-1)[0])
    return str(value)


def _is_on(value):
    if isinstance(value, str):
        return value.strip().lower() in {"1", "on", "true", "yes"}
    return bool(value)


# STILL OPEN QUESTION: Better to have empty MATLAB arrays as None for empty numpy arrays (current default).
# The current default is to make it more MALTAB compatible. A lot of MATLAB function start indexing MATLAB
# empty arrays to add values to them. This is not possible with None and would create more conversion and
# bugs. However, None is more pythonic.
