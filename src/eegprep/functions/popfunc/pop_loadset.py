"""EEGLAB dataset loading utilities."""

import logging
import os
from pathlib import Path

import h5py
import numpy as np
import scipy.io

from eegprep.functions.adminfunc.storage import eeg_data_shape, memmap_enabled, memmap_fdt, read_fdt
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
        EEG['icachansind'] = normalize_icachansind(EEG['icachansind'], matlab_one_based=not loaded_with_h5)

    info_data = None
    if info_only:
        info_data = _info_mode_data(EEG, Path(file_path))
    elif not loaded_with_h5:
        _load_sidecar_data(EEG, Path(file_path), use_memmap=use_memmap)

    # Convert 1-based MATLAB urchan/urevent to 0-based before eeg_checkset, which
    # copies event.urevent into epoch.eventurevent.  scipy's squeeze_me returns a
    # bare dict for a 1x1 struct array, so wrap it in a list first.
    for key in ('chanlocs', 'event'):
        if isinstance(EEG.get(key), dict):
            EEG[key] = [EEG[key]]

    # pop_loadset_h5 already did this.
    if not loaded_with_h5:
        ur_indices_to_zero_based(EEG)

    _restore_squeezed_matlab_shapes(EEG)
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


def _restore_squeezed_matlab_shapes(EEG: dict) -> None:
    """Restore length-1 axes MATLAB ``load`` keeps for ``size(data, 1) == nbchan``."""
    nbchan = _metadata_int(EEG, "nbchan")
    pnts = _metadata_int(EEG, "pnts")
    trials = _metadata_int(EEG, "trials") or 1
    if nbchan is not None and pnts is not None:
        _assign_unsqueezed(EEG, "data", eeg_data_shape({"nbchan": nbchan, "pnts": pnts, "trials": trials}))
    ica_channels = _ica_channel_count(EEG, nbchan)
    components = _component_count(EEG, ica_channels, pnts, trials)
    if ica_channels is None or components is None:
        return
    _assign_unsqueezed(EEG, "icaweights", (components, ica_channels))
    _assign_unsqueezed(EEG, "icasphere", (ica_channels, ica_channels))
    _assign_unsqueezed(EEG, "icawinv", (ica_channels, components))
    if pnts is not None:
        _assign_unsqueezed(EEG, "icaact", eeg_data_shape({"nbchan": components, "pnts": pnts, "trials": trials}))


def _metadata_int(EEG: dict, key: str) -> int | None:
    value = EEG.get(key)
    if value is None or isinstance(value, (str, bytes)):
        return None
    array = np.asarray(value)
    if array.size != 1 or array.dtype.kind not in "iuf":
        return None
    return int(array.reshape(-1)[0])


def _numeric_size(value) -> int:
    if not _is_numeric_data(value):
        return 0
    return int(np.asarray(value).size)


def _ica_channel_count(EEG: dict, nbchan: int | None) -> int | None:
    count = _numeric_size(EEG.get("icachansind"))
    if count:
        return count
    return nbchan or None


def _component_count(EEG: dict, n_channels: int | None, pnts: int | None, trials: int) -> int | None:
    if n_channels:
        for key in ("icaweights", "icawinv"):
            size = _numeric_size(EEG.get(key))
            if size and size % n_channels == 0:
                return size // n_channels
    frames = (pnts or 0) * trials
    size = _numeric_size(EEG.get("icaact"))
    if frames and size and size % frames == 0:
        return size // frames
    return None


def _assign_unsqueezed(EEG: dict, key: str, shape: tuple[int, ...]) -> None:
    restored = _unsqueezed_numeric(EEG.get(key), shape)
    if restored is not None:
        EEG[key] = restored


def _unsqueezed_numeric(value, shape: tuple[int, ...]):
    if not _is_numeric_data(value):
        return None
    array = np.asarray(value)
    if array.size == 0 or array.shape == shape:
        return None
    expected = 1
    for dim in shape:
        expected *= int(dim)
    if array.size != expected:
        return None
    squeezed = tuple(dim for dim in shape if dim != 1)
    loaded = () if array.ndim == 0 else tuple(int(dim) for dim in array.shape)
    if loaded != squeezed:
        return None
    return np.reshape(array, shape, order="F")


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
