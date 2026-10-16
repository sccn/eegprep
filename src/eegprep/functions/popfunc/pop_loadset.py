"""EEGLAB dataset loading utilities."""

import logging
import os
from pathlib import Path

import h5py
import numpy as np
import scipy.io

from eegprep.functions.adminfunc.storage import memmap_enabled, memmap_fdt, read_fdt
from eegprep.functions.popfunc._file_io import normalize_icachansind, ur_indices_to_zero_based
from eegprep.functions.popfunc._pop_utils import is_on, parse_key_value_args, parse_numeric_sequence
from eegprep.functions.popfunc.pop_loadset_h5 import pop_loadset_h5

logger = logging.getLogger(__name__)

default_empty = np.array([])


def loadset(file_path):
    """Load EEGLAB dataset from file (alias for pop_loadset)."""
    return pop_loadset(file_path)


def pop_loadset(file_path=None, *args, loadmode="all", memmap=None, **kwargs):
    """Load an EEGLAB dataset, its metadata, or selected channels.

    A numeric array stored in the ``.set`` file is the dataset. A ``.fdt``
    sidecar is read only when ``data`` names that file, or when ``data`` is
    ``in set file`` and ``datfile`` names it.

    Parameters
    ----------
    file_path : str
        Path to the EEGLAB .set file.
    loadmode : {"all", "info"} or int or sequence of int, optional
        Load all data, metadata without sample data, or selected channels.
        Channel numbers are 1-based, as in EEGLAB.

    Returns
    -------
    dict
        EEGLAB dataset dictionary.
    """
    from eegprep.functions.adminfunc.eeg_checkset import eeg_checkset

    file_path, loadmode, use_memmap = _load_options(file_path, args, kwargs, loadmode, memmap)
    info_only = isinstance(loadmode, str) and loadmode == "info"
    channel_selection = [] if isinstance(loadmode, str) else loadmode

    def new_check(obj):
        if isinstance(obj, dict):
            return {key: new_check(obj[key]) for key in obj}
        elif isinstance(obj, list):
            if len(obj) == 0:
                return default_empty
            else:
                return [new_check(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            if obj.size == 0:
                return default_empty
            elif obj.dtype.kind in ['i', 'u', 'f', 'c']:
                return obj
            else:
                return np.array([new_check(item) for item in obj], dtype=object)
        elif np.isscalar(obj) or isinstance(obj, str):
            return obj
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
        EEG['icachansind'] = normalize_icachansind(EEG['icachansind'], matlab_one_based=True)

    info_data = None
    if info_only:
        info_data = _info_mode_data(EEG, Path(file_path))
    else:
        _load_sidecar_data(EEG, Path(file_path), use_memmap=use_memmap)

    # Convert 1-based MATLAB urchan/urevent to 0-based before eeg_checkset, which
    # copies event.urevent into epoch.eventurevent.  scipy's squeeze_me returns a
    # bare dict for a 1x1 struct array, so wrap it in a list first.
    for key in ('chanlocs', 'event'):
        if isinstance(EEG.get(key), dict):
            EEG[key] = [EEG[key]]

    ur_indices_to_zero_based(EEG)

    EEG = eeg_checkset(EEG)
    if info_data is not None:
        EEG["data"] = info_data
    if channel_selection:
        _select_loaded_channels(EEG, channel_selection)
    EEG.pop("changes_not_saved", None)
    EEG["saved"] = "justloaded"

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
    loadmode = options.pop("loadmode", loadmode)
    if loadmode is None or (isinstance(loadmode, str) and loadmode == ""):
        loadmode = "all"
    if isinstance(loadmode, str):
        loadmode = loadmode.lower()
        if loadmode not in {"all", "info"}:
            raise ValueError("loadmode must be 'all', 'info', or 1-based channel indices")
    else:
        loadmode = parse_numeric_sequence(loadmode, dtype=int)
        if not loadmode or any(index < 1 for index in loadmode):
            raise ValueError("loadmode channel indices must be positive and 1-based")
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
    use_memmap = memmap_enabled() if memmap is None else is_on(memmap)
    return str(path), loadmode, use_memmap


def _select_loaded_channels(EEG, channel_selection) -> None:
    indices = np.asarray(channel_selection, dtype=int) - 1
    nbchan = int(EEG.get("nbchan", np.asarray(EEG.get("data", [])).shape[0]) or 0)
    if np.any(indices < 0) or np.any(indices >= nbchan):
        raise ValueError("loadmode channel indices exceed the dataset channel count")
    data = np.asarray(EEG.get("data"))
    if data.ndim not in {2, 3}:
        raise ValueError("channel loadmode requires numeric EEG data")
    EEG["datachannel"] = channel_selection[0] if len(channel_selection) == 1 else list(channel_selection)
    EEG["data"] = data[indices]
    EEG["nbchan"] = len(indices)
    chanlocs = EEG.get("chanlocs")
    if chanlocs is not None and len(chanlocs):
        EEG["chanlocs"] = [chanlocs[index] for index in indices]
    for field in ("icachansind", "icaact", "icaweights", "icasphere", "icawinv"):
        EEG[field] = np.array([])


def _load_sidecar_data(EEG, file_path: Path, *, use_memmap: bool) -> None:
    # pop_loadset.m lines 155-160: numeric data is already in the .set file.
    data_value = EEG.get("data")
    data_name = data_value if isinstance(data_value, str) else ""
    if data_name == "in set file":
        logger.warning("Something is wrong with the data file, trying to use the associated .fdt file")
        data_name = _string_value(EEG.get("datfile"))
    if not data_name or data_name == "in set file":
        return
    sidecar = _sidecar_path(data_name, file_path)
    EEG["datfile"] = sidecar.name
    EEG["data"] = memmap_fdt(sidecar, EEG) if use_memmap else read_fdt(sidecar, EEG)


def _info_mode_data(EEG, file_path: Path) -> str | None:
    # pop_loadset.m lines 139-154. None keeps a numeric array loaded from the .set file.
    datfile = _string_value(EEG.get("datfile"))
    if not datfile:
        return "in set file"
    if _sidecar_path(datfile, file_path).is_file():
        return datfile
    data_value = EEG.get("data")
    logger.warning(".fdt file not found, checking if .set contains data")
    if not _is_numeric_data(data_value) and not _is_empty_data(data_value):
        logger.warning(".fdt file not found, but data found in .set EEGLAB file")
        return "in set file"
    if _is_empty_data(data_value) or not _is_numeric_data(data_value):
        logger.warning(".fdt file not found, this EEGLAB file is missing raw data")
    return None


def _sidecar_path(datfile: str, file_path: Path) -> Path:
    path = Path(datfile)
    if path.is_absolute():
        return path
    return file_path.parent / path.name


def _is_numeric_data(value) -> bool:
    if isinstance(value, np.ndarray):
        return value.dtype.kind in "iufc"
    return isinstance(value, np.number)


def _is_empty_data(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value == ""
    if isinstance(value, np.ndarray):
        return value.size == 0
    return False


def _string_value(value):
    if value is None:
        return ""
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return ""
        if value.size == 1:
            return str(value.reshape(-1)[0])
    return str(value)


# STILL OPEN QUESTION: Better to have empty MATLAB arrays as None for empty numpy arrays (current default).
# The current default is to make it more MALTAB compatible. A lot of MATLAB function start indexing MATLAB
# empty arrays to add values to them. This is not possible with None and would create more conversion and
# bugs. However, None is more pythonic.
