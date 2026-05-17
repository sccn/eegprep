"""EEG preprocessing package for MATLAB EEGLAB compatibility."""

from __future__ import annotations

import importlib
import logging
from typing import Any

from .functions.adminfunc.logs import setup_logging

__version__ = "0.2.23"

setup_logging(logging.WARNING)

_LAZY_EXPORTS = {
    "EEG_OPTIONS": ("eegprep.functions.adminfunc.eeg_options", "EEG_OPTIONS"),
    "EEGobj": ("eegprep.functions.eegobj.eegobj", "EEGobj"),
    "ICL_feature_extractor": ("eegprep.plugins.ICLabel.ICL_feature_extractor", "ICL_feature_extractor"),
    "bids_list_eeg_files": ("eegprep.plugins.EEG_BIDS.bids_list_eeg_files", "bids_list_eeg_files"),
    "bids_preproc": ("eegprep.plugins.EEG_BIDS.bids_preproc", "bids_preproc"),
    "cart2topo": ("eegprep.functions.sigprocfunc.cart2topo", "cart2topo"),
    "checkset": ("eegprep.functions.redefine_functions", "checkset"),
    "clean_artifacts": ("eegprep.plugins.clean_rawdata.clean_artifacts", "clean_artifacts"),
    "clean_asr": ("eegprep.plugins.clean_rawdata.clean_asr", "clean_asr"),
    "clean_channels": ("eegprep.plugins.clean_rawdata.clean_channels", "clean_channels"),
    "clean_channels_nolocs": ("eegprep.plugins.clean_rawdata.clean_channels_nolocs", "clean_channels_nolocs"),
    "clean_drifts": ("eegprep.plugins.clean_rawdata.clean_drifts", "clean_drifts"),
    "clean_flatlines": ("eegprep.plugins.clean_rawdata.clean_flatlines", "clean_flatlines"),
    "clean_windows": ("eegprep.plugins.clean_rawdata.clean_windows", "clean_windows"),
    "compare": ("eegprep.functions.redefine_functions", "compare"),
    "decodechan": ("eegprep.functions.redefine_functions", "decodechan"),
    "eeg2mne": ("eegprep.functions.redefine_functions", "eeg2mne"),
    "eeg_amica": ("eegprep.functions.popfunc.eeg_amica", "eeg_amica"),
    "eeg_autocorr": ("eegprep.plugins.ICLabel.eeg_autocorr", "eeg_autocorr"),
    "eeg_autocorr_fftw": ("eegprep.plugins.ICLabel.eeg_autocorr_fftw", "eeg_autocorr_fftw"),
    "eeg_autocorr_welch": ("eegprep.plugins.ICLabel.eeg_autocorr_welch", "eeg_autocorr_welch"),
    "eeg_checkset": ("eegprep.functions.adminfunc.eeg_checkset", "eeg_checkset"),
    "eeg_checkset_strict_mode": ("eegprep.functions.adminfunc.eeg_checkset", "strict_mode"),
    "eeg_compare": ("eegprep.functions.popfunc.eeg_compare", "eeg_compare"),
    "eeg_decodechan": ("eegprep.functions.popfunc.eeg_decodechan", "eeg_decodechan"),
    "eeg_eeg2mne": ("eegprep.functions.miscfunc.eeg_eeg2mne", "eeg_eeg2mne"),
    "eeg_eegrej": ("eegprep.functions.popfunc.eeg_eegrej", "eeg_eegrej"),
    "eeg_findboundaries": ("eegprep.functions.popfunc.eeg_findboundaries", "eeg_findboundaries"),
    "eeg_interp": ("eegprep.functions.popfunc.eeg_interp", "eeg_interp"),
    "eeg_lat2point": ("eegprep.functions.popfunc.eeg_lat2point", "eeg_lat2point"),
    "eeg_mne2eeg": ("eegprep.functions.miscfunc.eeg_mne2eeg", "eeg_mne2eeg"),
    "eeg_mne2eeg_epochs": ("eegprep.functions.miscfunc.eeg_mne2eeg_epochs", "eeg_mne2eeg_epochs"),
    "eeg_picard": ("eegprep.functions.popfunc.eeg_picard", "eeg_picard"),
    "eeg_point2lat": ("eegprep.functions.popfunc.eeg_point2lat", "eeg_point2lat"),
    "eeg_rpsd": ("eegprep.plugins.ICLabel.eeg_rpsd", "eeg_rpsd"),
    "eeg_runica": ("eegprep.functions.popfunc.eeg_runica", "eeg_runica"),
    "eeglab": ("eegprep.functions.adminfunc.eeglab", "eeglab"),
    "eeglab_clean_artifacts": ("eegprep.functions.adminfunc.eeglabcompat", "clean_artifacts"),
    "eegrej": ("eegprep.functions.sigprocfunc.eegrej", "eegrej"),
    "epoch": ("eegprep.functions.redefine_functions", "epoch"),
    "findboundaries": ("eegprep.functions.redefine_functions", "findboundaries"),
    "iclabel": ("eegprep.plugins.ICLabel.iclabel", "iclabel"),
    "eeg_icflag": ("eegprep.plugins.ICLabel.eeg_icflag", "eeg_icflag"),
    "interp": ("eegprep.functions.redefine_functions", "interp"),
    "lat2point": ("eegprep.functions.redefine_functions", "lat2point"),
    "loadset": ("eegprep.functions.popfunc.pop_loadset", "loadset"),
    "mne2eeg": ("eegprep.functions.redefine_functions", "mne2eeg"),
    "mne2eeg_epochs": ("eegprep.functions.redefine_functions", "mne2eeg_epochs"),
    "options": ("eegprep.functions.redefine_functions", "options"),
    "picard": ("eegprep.functions.redefine_functions", "picard"),
    "point2lat": ("eegprep.functions.redefine_functions", "point2lat"),
    "pop_adjustevents": ("eegprep.functions.popfunc.pop_adjustevents", "pop_adjustevents"),
    "pop_chansel": ("eegprep.functions.popfunc.pop_chansel", "pop_chansel"),
    "pop_epoch": ("eegprep.functions.popfunc.pop_epoch", "pop_epoch"),
    "pop_eegfiltnew": ("eegprep.functions.adminfunc.eeglabcompat", "pop_eegfiltnew"),
    "pop_interp": ("eegprep.functions.popfunc.pop_interp", "pop_interp"),
    "pop_load_frombids": ("eegprep.functions.popfunc.pop_load_frombids", "pop_load_frombids"),
    "pop_loadset": ("eegprep.functions.popfunc.pop_loadset", "pop_loadset"),
    "pop_loadset_h5": ("eegprep.functions.popfunc.pop_loadset_h5", "pop_loadset_h5"),
    "pop_resample": ("eegprep.functions.popfunc.pop_resample", "pop_resample"),
    "pop_reref": ("eegprep.functions.popfunc.pop_reref", "pop_reref"),
    "pop_rmbase": ("eegprep.functions.popfunc.pop_rmbase", "pop_rmbase"),
    "pop_saveset": ("eegprep.functions.popfunc.pop_saveset", "pop_saveset"),
    "pop_select": ("eegprep.functions.popfunc.pop_select", "pop_select"),
    "pop_subcomp": ("eegprep.functions.popfunc.pop_subcomp", "pop_subcomp"),
    "reref": ("eegprep.functions.redefine_functions", "reref"),
    "resample": ("eegprep.functions.redefine_functions", "resample"),
    "rmbase": ("eegprep.functions.redefine_functions", "rmbase"),
    "saveset": ("eegprep.functions.redefine_functions", "saveset"),
    "select": ("eegprep.functions.redefine_functions", "select"),
    "topoplot": ("eegprep.functions.sigprocfunc.topoplot", "topoplot"),
}

__all__ = ["__version__", *_LAZY_EXPORTS]


def __getattr__(name: str) -> Any:
    """Load public EEGPrep exports on first access."""
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(importlib.import_module(module_name), attr_name)
    # Cache the resolved export so normal attribute lookup skips __getattr__.
    globals()[name] = value
    return value
