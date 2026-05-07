"""EEG preprocessing package for MATLAB EEGLAB compatibility."""

import logging

__version__ = "0.2.23"

from .functions.adminfunc.logs import setup_logging
setup_logging(logging.WARNING)

from .plugins.ICLabel.iclabel import iclabel
from .plugins.ICLabel.eeg_icflag import eeg_icflag
from .functions.popfunc.pop_subcomp import pop_subcomp
from .functions.popfunc.pop_saveset import pop_saveset
from .functions.popfunc.pop_loadset import loadset, pop_loadset
from .functions.popfunc.pop_loadset_h5 import pop_loadset_h5
from .functions.popfunc.pop_adjustevents import pop_adjustevents
from .functions.popfunc.pop_epoch import pop_epoch
from .functions.popfunc.pop_resample import pop_resample
from .functions.popfunc.pop_rmbase import pop_rmbase
from .functions.popfunc.pop_select import pop_select
from .functions.adminfunc.eeg_checkset import eeg_checkset, strict_mode as eeg_checkset_strict_mode
from .functions.adminfunc.eeglabcompat import pop_eegfiltnew
from .functions.adminfunc.eeglabcompat import clean_artifacts as eeglab_clean_artifacts
from .plugins.ICLabel.ICL_feature_extractor import ICL_feature_extractor
from .functions.sigprocfunc.topoplot import topoplot
from .plugins.ICLabel.eeg_rpsd import eeg_rpsd
from .plugins.ICLabel.eeg_autocorr_welch import eeg_autocorr_welch
from .plugins.ICLabel.eeg_autocorr import eeg_autocorr
from .plugins.ICLabel.eeg_autocorr_fftw import eeg_autocorr_fftw
from .functions.popfunc.pop_reref import pop_reref
from .functions.popfunc.eeg_amica import eeg_amica
from .functions.popfunc.eeg_picard import eeg_picard
from .functions.popfunc.eeg_runica import eeg_runica
from .plugins.clean_rawdata.clean_flatlines import clean_flatlines
from .plugins.clean_rawdata.clean_drifts import clean_drifts
from .plugins.clean_rawdata.clean_channels_nolocs import clean_channels_nolocs
from .plugins.clean_rawdata.clean_channels import clean_channels
from .plugins.clean_rawdata.clean_asr import clean_asr
from .plugins.clean_rawdata.clean_windows import clean_windows
from .functions.popfunc.eeg_compare import eeg_compare
from .functions.popfunc.eeg_interp import eeg_interp
from .functions.popfunc.eeg_findboundaries import eeg_findboundaries
from .plugins.clean_rawdata.clean_artifacts import clean_artifacts
from .functions.popfunc.pop_load_frombids import pop_load_frombids
from .plugins.EEG_BIDS.bids_list_eeg_files import bids_list_eeg_files
from .plugins.EEG_BIDS.bids_preproc import bids_preproc
from .functions.popfunc.eeg_decodechan import eeg_decodechan
from .functions.sigprocfunc.eegrej import eegrej
from .functions.popfunc.eeg_eegrej import eeg_eegrej
from .functions.eegobj.eegobj import EEGobj
from .functions.redefine_functions import (
    checkset,
    compare,
    decodechan,
    eeg2mne,
    epoch,
    findboundaries,
    interp,
    lat2point,
    mne2eeg,
    mne2eeg_epochs,
    options,
    picard,
    point2lat,
    reref,
    resample,
    rmbase,
    saveset,
    select,
)
from .functions.miscfunc.eeg_eeg2mne import eeg_eeg2mne
from .functions.miscfunc.eeg_mne2eeg import eeg_mne2eeg
from .functions.miscfunc.eeg_mne2eeg_epochs import eeg_mne2eeg_epochs
from .functions.popfunc.eeg_lat2point import eeg_lat2point
from .functions.popfunc.eeg_point2lat import eeg_point2lat
from .functions.adminfunc.eeg_options import EEG_OPTIONS
