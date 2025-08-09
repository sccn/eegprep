import logging

from .utils.logs import setup_logging
setup_logging(logging.INFO)

from .iclabel import iclabel
from .pop_saveset import pop_saveset
from .pop_loadset import loadset, pop_loadset
from .pop_loadset_h5 import pop_loadset_h5
from .pop_resample import pop_resample
from .eeg_checkset import eeg_checkset
from .eeglabcompat import pop_eegfiltnew, pop_biosig
from .eeglabcompat import clean_artifacts as eeglab_clean_artifacts
from .ICL_feature_extractor import ICL_feature_extractor
from .iclabel_net import ICLabelNet
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
# from .eeg_compare import eeg_compare
from .clean_artifacts import clean_artifacts
from .pop_load_frombids import pop_load_frombids
from .bids_list_eeg_files import bids_list_eeg_files
from .bids_preproc import bids_preproc