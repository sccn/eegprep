"""Pytest collection markers for legacy unittest-style tests."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import pytest


def _preload_matlab_libstdcxx() -> None:
    """Load a modern libstdc++ before MATLAB Engine imports native modules."""
    candidate = os.environ.get("EEGPREP_MATLAB_LIBSTDCXX")
    candidates = [Path(candidate)] if candidate else []

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.append(Path(conda_prefix) / "lib" / "libstdc++.so.6")

    for libstdcxx in candidates:
        if not libstdcxx.exists():
            continue
        try:
            ctypes.CDLL(str(libstdcxx), mode=ctypes.RTLD_GLOBAL)
            return
        except OSError:
            continue


_preload_matlab_libstdcxx()


SLOW_NODEID_PARTS = (
    "tests/test_eeg_amica.py::",
    "tests/test_runamica.py::TestRunamicaIntegration::",
)

VISUAL_FILE_SUFFIXES = (
    "tests/test_visual_parity.py",
)

GUI_FILE_SUFFIXES = (
    "tests/test_gui_pop_adjustevents.py",
    "tests/test_gui_pop_clean_rawdata.py",
    "tests/test_gui_pop_iclabel.py",
    "tests/test_gui_pop_resample.py",
    "tests/test_gui_pop_runica.py",
    "tests/test_gui_pop_select.py",
    "tests/test_gui_main_window.py",
)
GUI_NODEID_PARTS = (
    "::test_gui_",
)

MATLAB_FILE_SUFFIXES = (
    "tests/test_ICL_feature_extractor_parity.py",
    "tests/test_bids_preproc.py",
    "tests/test_clean_rawdata.py",
    "tests/test_eeg_compare.py",
    "tests/test_eeg_eeg2mne.py",
    "tests/test_eeg_eegrej.py",
    "tests/test_eeg_lat2point.py",
    "tests/test_eeg_mne2eeg_epochs.py",
    "tests/test_eeg_point2lat.py",
    "tests/test_eeg_rpsd_parity.py",
    "tests/test_eegfindboundaries.py",
    "tests/test_iclabel.py",
    "tests/test_iclabel_features.py",
    "tests/test_parity_rng.py",
    "tests/test_pinv.py",
    "tests/test_pipeline.py",
    "tests/test_pop_epoch.py",
    "tests/test_pop_loadset_h5.py",
    "tests/test_pop_resample.py",
    "tests/test_pop_rmbase.py",
)

MATLAB_NODEID_PARTS = (
    "tests/test_ICL_feature_extractor.py::TestICLFeatureExtractorParity::",
    "tests/test_eeg_autocorr.py::TestEegAutocorr::test_parity_",
    "tests/test_eeg_autocorr_fftw.py::TestEegAutocorrFftw::test_parity_",
    "tests/test_eeg_autocorr_welch.py::TestEegAutocorrWelch::test_parity_",
    "tests/test_eeg_interp.py::TestComputeGParity::",
    "tests/test_eeg_interp.py::TestEegInterpParity::",
    "tests/test_eeg_interp.py::TestSphericalSplineParity::",
    "tests/test_eeg_picard.py::TestEegPicard::",
    "tests/test_eegrej.py::TestEEGRej::test_compare_to_eeglab",
    "tests/test_eeglabcompat.py::TestCleanDrifts::",
    "tests/test_eeglabcompat.py::TestEegChecksetMatlab::",
    "tests/test_eeglabcompat.py::TestEeglabCompatIntegration::",
    "tests/test_eeglabcompat.py::TestGetEeglab::",
    "tests/test_eeglabcompat.py::TestPopEegfiltnew::",
    "tests/test_epoch.py::TestEpochParity::",
    "tests/test_matlab_path.py::TestMatlabPath::test_get_eeglab_mat",
    "tests/test_matlab_path.py::TestMatlabPath::test_python_matlab_engine",
    "tests/test_matlab_path.py::TestMatlabPath::test_start_matlab_engine",
    "tests/test_pop_reref.py::TestPopReref::test_parity_",
    "tests/test_pop_select.py::TestPopSelectParity::",
    "tests/test_runica.py::TestRunicaParity::",
    "tests/test_topoplot.py::TestTopoplotParity::",
)

OCTAVE_NODEID_PARTS = (
    "tests/test_matlab_path.py::TestMatlabPath::test_get_eeglab_oct",
)


def _path_has_suffix(path: str, suffixes: tuple[str, ...]) -> bool:
    return any(path == suffix or path.endswith(f"/{suffix}") for suffix in suffixes)


def _nodeid_has_part(nodeid: str, parts: tuple[str, ...]) -> bool:
    return any(part in nodeid for part in parts)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    del config

    for item in items:
        nodeid = item.nodeid
        lower_nodeid = nodeid.lower()
        path = item.path.as_posix()

        if _nodeid_has_part(nodeid, SLOW_NODEID_PARTS):
            item.add_marker(pytest.mark.slow)

        if _path_has_suffix(path, VISUAL_FILE_SUFFIXES):
            item.add_marker(pytest.mark.visual)

        if _path_has_suffix(path, GUI_FILE_SUFFIXES) or _nodeid_has_part(
            lower_nodeid, GUI_NODEID_PARTS
        ):
            item.add_marker(pytest.mark.gui)

        if "parity" in lower_nodeid:
            item.add_marker(pytest.mark.parity)

        requires_matlab = _path_has_suffix(path, MATLAB_FILE_SUFFIXES) or _nodeid_has_part(
            nodeid, MATLAB_NODEID_PARTS
        )
        if requires_matlab:
            item.add_marker(pytest.mark.matlab)

        if _nodeid_has_part(nodeid, OCTAVE_NODEID_PARTS):
            item.add_marker(pytest.mark.octave)
