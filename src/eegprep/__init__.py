import logging

__version__ = "0.2.23"

from .utils.logs import setup_logging
setup_logging(logging.INFO)

from .iclabel import iclabel
from .pop_saveset import pop_saveset
from .pop_loadset import loadset, pop_loadset
from .pop_loadset_h5 import pop_loadset_h5
from .pop_resample import pop_resample
from .eeg_checkset import eeg_checkset
from .eeglabcompat import pop_eegfiltnew
from .eeglabcompat import clean_artifacts as eeglab_clean_artifacts
from .ICL_feature_extractor import ICL_feature_extractor
from .topoplot import topoplot
from .eeg_rpsd import eeg_rpsd
from .eeg_autocorr_welch import eeg_autocorr_welch
from .eeg_autocorr import eeg_autocorr
from .eeg_autocorr_fftw import eeg_autocorr_fftw
from .pop_reref import pop_reref
from .eeg_picard import eeg_picard
from .clean_flatlines import clean_flatlines
from .clean_drifts import clean_drifts
from .clean_channels_nolocs import clean_channels_nolocs
from .clean_channels import clean_channels
from .clean_asr import clean_asr
from .clean_windows import clean_windows
from .eeg_compare import eeg_compare
from .eeg_interp import eeg_interp
from .clean_artifacts import clean_artifacts
from .pop_load_frombids import pop_load_frombids
from .bids_list_eeg_files import bids_list_eeg_files
from .bids_preproc import bids_preproc
from .eeg_decodechan import eeg_decodechan
from .eegrej import eegrej
from .eeg_eegrej import eeg_eegrej
from .eegobj import EEGobj
from .redefine_functions import *
from .eeg_mne2eeg import eeg_mne2eeg
from .eeg_mne2eeg_epochs import eeg_mne2eeg_epochs
from .eeg_lat2point import eeg_lat2point
from .eeg_point2lat import eeg_point2lat
from .eeg_options import EEG_OPTIONS
from .eeg_checkset import eeg_checkset, strict_mode as eeg_checkset_strict_mode
