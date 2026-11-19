"""Module that defines short wrapper functions that call their corresponding.

eeg_ or pop_ implementations.

Each wrapper forwards its arguments to the original function. ``epoch`` also
accepts MATLAB positional name/value pairs and returns MATLAB-shaped outputs.

For example, checkset(x) calls eeg_checkset(x). ``epoch`` is the data-matrix
epoch function; dataset epoching is ``pop_epoch``.

Wrappers let you use names without the eeg_ or pop_ prefix while returning the same
results as the originals. Available wrappers: checkset, compare, decodechan, eeg2mne,
eegrej, findboundaries, interp, lat2point, mne2eeg, mne2eeg_epochs, options, picard,
point2lat, epoch, loadset, reref, resample, rmbase, saveset, select.
"""

import numpy as np

from eegprep.functions.adminfunc.eeg_checkset import eeg_checkset
from eegprep.functions.popfunc.eeg_compare import eeg_compare
from eegprep.functions.popfunc.eeg_decodechan import eeg_decodechan
from eegprep.functions.miscfunc.eeg_eeg2mne import eeg_eeg2mne
from eegprep.functions.popfunc.eeg_eegrej import eeg_eegrej
from eegprep.functions.popfunc.eeg_findboundaries import eeg_findboundaries
from eegprep.functions.popfunc.eeg_interp import eeg_interp
from eegprep.functions.popfunc.eeg_lat2point import eeg_lat2point
from eegprep.functions.miscfunc.eeg_mne2eeg import eeg_mne2eeg
from eegprep.functions.miscfunc.eeg_mne2eeg_epochs import eeg_mne2eeg_epochs
from eegprep.functions.adminfunc.eeg_options import EEG_OPTIONS
from eegprep.functions.popfunc.eeg_picard import eeg_picard
from eegprep.functions.popfunc.eeg_point2lat import eeg_point2lat
from eegprep.functions.miscfunc.value_parsing import parse_key_value_args
from eegprep.functions.popfunc.pop_loadset import pop_loadset
from eegprep.functions.sigprocfunc.epoch import epoch as epoch_data
from eegprep.functions.popfunc.pop_reref import pop_reref
from eegprep.functions.popfunc.pop_resample import pop_resample
from eegprep.functions.popfunc.pop_saveset import pop_saveset
from eegprep.functions.popfunc.pop_select import pop_select
from eegprep.functions.sigprocfunc.rmbase import rmbase as sigproc_rmbase


def checkset(*args, **kwargs):
    """Wrap eeg_checkset."""
    return eeg_checkset(*args, **kwargs)


def compare(*args, **kwargs):
    """Wrap eeg_compare."""
    return eeg_compare(*args, **kwargs)


def decodechan(*args, **kwargs):
    """Wrap eeg_decodechan."""
    return eeg_decodechan(*args, **kwargs)


def eeg2mne(*args, **kwargs):
    """Wrap eeg_eeg2mne."""
    kwargs.setdefault("feature", "eeg2mne")
    return eeg_eeg2mne(*args, **kwargs)


def eegrej(*args, **kwargs):
    """Wrap eeg_eegrej."""
    return eeg_eegrej(*args, **kwargs)


def findboundaries(*args, **kwargs):
    """Wrap eeg_findboundaries."""
    return eeg_findboundaries(*args, **kwargs)


def interp(*args, **kwargs):
    """Wrap eeg_interp."""
    return eeg_interp(*args, **kwargs)


def lat2point(*args, **kwargs):
    """Wrap eeg_lat2point."""
    return eeg_lat2point(*args, **kwargs)


def mne2eeg(*args, **kwargs):
    """Wrap eeg_mne2eeg."""
    kwargs.setdefault("feature", "mne2eeg")
    return eeg_mne2eeg(*args, **kwargs)


def mne2eeg_epochs(*args, **kwargs):
    """Wrap eeg_mne2eeg_epochs."""
    return eeg_mne2eeg_epochs(*args, **kwargs)


def options(*args, **kwargs):
    """Wrap EEG_OPTIONS."""
    return EEG_OPTIONS


def picard(*args, **kwargs):
    """Wrap eeg_picard."""
    kwargs.setdefault("feature", "picard")
    return eeg_picard(*args, **kwargs)


def point2lat(*args, **kwargs):
    """Wrap eeg_point2lat."""
    return eeg_point2lat(*args, **kwargs)


def epoch(*args, **kwargs):
    """Extract epochs from a data matrix, matching ``epoch.m``.

    Positional name/value pairs follow the data, event latencies, and time
    limits. Returned event indices are 1-based. ``newtime`` is a row.
    ``alleventout`` and ``alllatencyout`` are a 1-by-N cell of rows, or a
    0-by-0 cell when no event list was given. Dataset epoching is ``pop_epoch``.
    """
    if len(args) < 3:
        raise ValueError("epoch: data, events, and time limits are required")
    options = parse_key_value_args(args[3:], kwargs)
    epochdat, newtime, indexes, alleventout, alllatencyout, reallim = epoch_data(*args[:3], **options)
    return (
        epochdat,
        np.asarray(newtime, dtype=float).reshape(1, -1),
        np.asarray(indexes, dtype=float).reshape(-1, 1) + 1.0,
        _epoch_cells(alleventout, one_based=True),
        _epoch_cells(alllatencyout, one_based=False),
        reallim,
    )


def _epoch_cells(values, *, one_based: bool) -> np.ndarray:
    if not values:
        return np.empty((0, 0), dtype=object)
    cells = np.empty((1, len(values)), dtype=object)
    for index, value in enumerate(values):
        array = np.asarray(value, dtype=float).reshape(-1)
        if one_based and array.size:
            array = array + 1.0
        cells[0, index] = array.reshape(1, -1)
    return cells


def loadset(*args, **kwargs):
    """Wrap pop_loadset."""
    return pop_loadset(*args, **kwargs)


def reref(*args, **kwargs):
    """Wrap pop_reref."""
    return pop_reref(*args, **kwargs)


def resample(*args, **kwargs):
    """Wrap pop_resample."""
    kwargs.setdefault("feature", "resample")
    return pop_resample(*args, **kwargs)


def rmbase(*args, **kwargs):
    """Wrap low-level rmbase."""
    return sigproc_rmbase(*args, **kwargs)


def saveset(*args, **kwargs):
    """Wrap pop_saveset."""
    return pop_saveset(*args, **kwargs)


def select(*args, **kwargs):
    """Wrap pop_select."""
    return pop_select(*args, **kwargs)
