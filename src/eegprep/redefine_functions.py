"""
This module defines short wrapper functions that call their corresponding eeg_ or pop_ implementations.

Each wrapper has the same signature and simply forwards all arguments to the original function.

For example, checkset(x) calls eeg_checkset(x) and epoch(ev) calls pop_epoch(ev).

Wrappers let you use names without the eeg_ or pop_ prefix while returning the same results as the 
originals. Available wrappers: checkset, compare, decodechan, eeg2mne, eegrej, findboundaries, 
interp, lat2point, mne2eeg, mne2eeg_epochs, options, picard, point2lat, epoch, loadset,
reref, resample, rmbase, saveset, select.

"""

from eegprep.eeg_checkset import eeg_checkset
from eegprep.eeg_compare import eeg_compare
from eegprep.eeg_decodechan import eeg_decodechan
from eegprep.eeg_eeg2mne import eeg_eeg2mne
from eegprep.eeg_eegrej import eeg_eegrej
from eegprep.eeg_findboundaries import eeg_findboundaries
from eegprep.eeg_interp import eeg_interp
from eegprep.eeg_lat2point import eeg_lat2point
from eegprep.eeg_mne2eeg import eeg_mne2eeg
from eegprep.eeg_mne2eeg_epochs import eeg_mne2eeg_epochs
from eegprep.eeg_options import EEG_OPTIONS
from eegprep.eeg_picard import eeg_picard
from eegprep.eeg_point2lat import eeg_point2lat
from eegprep.pop_epoch import pop_epoch
from eegprep.pop_loadset import pop_loadset
from eegprep.pop_reref import pop_reref
from eegprep.pop_resample import pop_resample
from eegprep.pop_rmbase import pop_rmbase
from eegprep.pop_saveset import pop_saveset
from eegprep.pop_select import pop_select

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
    return eeg_mne2eeg(*args, **kwargs)

def mne2eeg_epochs(*args, **kwargs):
    """Wrap eeg_mne2eeg_epochs."""
    return eeg_mne2eeg_epochs(*args, **kwargs)

def options(*args, **kwargs):
    """Wrap EEG_OPTIONS."""
    return EEG_OPTIONS

def picard(*args, **kwargs):
    """Wrap eeg_picard."""
    return eeg_picard(*args, **kwargs)

def point2lat(*args, **kwargs):
    """Wrap eeg_point2lat."""
    return eeg_point2lat(*args, **kwargs)

def epoch(*args, **kwargs):
    """Wrap pop_epoch."""
    return pop_epoch(*args, **kwargs)

def loadset(*args, **kwargs):
    """Wrap pop_loadset."""
    return pop_loadset(*args, **kwargs)

def reref(*args, **kwargs):
    """Wrap pop_reref."""
    return pop_reref(*args, **kwargs)

def resample(*args, **kwargs):
    """Wrap pop_resample."""
    return pop_resample(*args, **kwargs)

def rmbase(*args, **kwargs):
    """Wrap pop_rmbase."""
    return pop_rmbase(*args, **kwargs)

def saveset(*args, **kwargs):
    """Wrap pop_saveset."""
    return pop_saveset(*args, **kwargs)

def select(*args, **kwargs):
    """Wrap pop_select."""
    return pop_select(*args, **kwargs)