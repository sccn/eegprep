"""Pytest collection markers for legacy unittest-style tests."""

from __future__ import annotations

import ctypes
from functools import partial
import importlib
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from tests.eeglab_tests import EEGLAB_TESTS_EEGLAB_COMMIT, upstream_references
from tests.eeglab_tests.backend import call_matlab, call_python
from tools.eeglab_test_port_audit import validate_suite_checkout


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


def pytest_addoption(parser):
    group = parser.getgroup("EEGLAB reference contracts")
    group.addoption(
        "--eeglab-backend",
        choices=("matlab", "python"),
        default=None,
        help="Opt in to backend-neutral EEGLAB contract tests (MATLAB is required when selected).",
    )
    group.addoption(
        "--eeglab-root",
        default=os.environ.get("EEGPREP_EEGLAB_ROOT"),
        help="Explicit EEGLAB reference checkout containing eeglab.m.",
    )
    group.addoption(
        "--eeglab-suite-root",
        help="Pinned eeglab_tests checkout; defaults to the parent of --eeglab-root.",
    )


@pytest.fixture(scope="session")
def eeglab_suite_root(request):
    root = request.config.getoption("--eeglab-suite-root")
    if not root:
        reference = request.config.getoption("--eeglab-root")
        if not reference:
            pytest.fail("Reference datasets require --eeglab-suite-root or --eeglab-root", pytrace=False)
        root = Path(reference).parent
    root = Path(root).resolve()
    validate_suite_checkout(root)
    return root


@pytest.fixture(params=[("teststudy", "n400clustedit.study")])
def eeglab_sample_study(request, eeglab_backend, eeglab_suite_root):
    """Load fresh source STUDY state as readsamplestudy/readsamplestudy2 do."""
    directory, filename = request.param
    study, alleeg = eeglab_backend(
        "pop_loadstudy",
        filename=filename,
        filepath=str(eeglab_suite_root / "unittesting_studyfunc" / directory),
        nargout=2,
    )
    study = eeglab_backend("std_checkset", study, alleeg)
    return study, alleeg


@pytest.fixture(params=["teststudy"])
def eeglab_writable_study(request, eeglab_suite_root, tmp_path):
    """Copy the original STUDY tree before workflows write measure caches."""
    source = eeglab_suite_root / "unittesting_studyfunc" / request.param
    return Path(shutil.copytree(source, tmp_path / request.param))


@pytest.fixture
def eeglab_options_directory(request, eeglab_backend, tmp_path):
    """Isolate the reference's documented EEGOPTION_PATH configuration hook."""
    directory = tmp_path / "eeglab_options"
    directory.mkdir()
    if request.config.getoption("--eeglab-backend") == "python":
        options = importlib.import_module("eegprep.functions.adminfunc.eeg_options").EEG_OPTIONS
        original_options = options.copy()
        try:
            yield directory
        finally:
            options.clear()
            options.update(original_options)
        return

    engine = request.getfixturevalue("eeglab_matlab_engine")
    original_path, original_directory = engine.path(), engine.pwd()
    original_icadefs = engine.which("icadefs")
    home_options = Path.home() / "eeg_options.m"
    original_home = home_options.read_bytes() if home_options.exists() else None
    if original_home is not None:
        shutil.copyfile(home_options, directory / "eeg_options.m")
    # icadefs explicitly permits a project-local copy. Run the pinned original
    # unchanged, overriding only where user preference writes are stored.
    wrapper = directory / "icadefs.m"
    original_script = original_icadefs.replace("'", "''")
    options_path = str(directory).replace("'", "''")
    wrapper.write_text(
        f"run('{original_script}');\nEEGOPTION_PATH = '{options_path}';\n",
        encoding="utf-8",
    )
    try:
        engine.addpath(str(directory), "-begin", nargout=0)
        engine.eval("clear icadefs eeg_options; icadefs;", nargout=0)
        assert engine.which("icadefs") == str(wrapper)
        assert engine.workspace["EEGOPTION_PATH"] == str(directory)
        yield directory
    finally:
        engine.path(original_path, nargout=0)
        engine.cd(original_directory, nargout=0)
        engine.eval("clear icadefs eeg_options; eeglab_options;", nargout=0)
        assert (home_options.read_bytes() if home_options.exists() else None) == original_home


@pytest.fixture(scope="session")
def eeglab_matlab_engine(request):
    root = request.config.getoption("--eeglab-root")
    if not root or not (Path(root) / "eeglab.m").is_file():
        pytest.fail("MATLAB contracts require --eeglab-root pointing to an EEGLAB checkout", pytrace=False)
    if os.environ.get("EEGPREP_SKIP_MATLAB") == "1":
        pytest.fail("Explicit MATLAB contracts conflict with EEGPREP_SKIP_MATLAB=1", pytrace=False)
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    if revision != EEGLAB_TESTS_EEGLAB_COMMIT:
        pytest.fail(f"EEGLAB is at {revision}; expected pinned {EEGLAB_TESTS_EEGLAB_COMMIT}", pytrace=False)
    # The private cache prevents reusing an engine configured for another root.
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("EEGPREP_EEGLAB_ROOT", str(Path(root).resolve()))
        compat = importlib.import_module("eegprep.functions.adminfunc.eeglabcompat")
        engine = compat.get_eeglab("MAT", auto_file_roundtrip=False, _cache={})
    try:
        # Automated reference plots must not open hundreds of desktop windows.
        engine.set(0.0, "DefaultFigureVisible", "off", nargout=0)
        engine.addpath(str(Path(root).resolve()), str(Path(root).resolve() / "functions"), nargout=0)
        # Use the reference's initialization to activate installed workflow plugins.
        # Unlike add_plugins.m, this does not install missing plugins.
        directory = engine.pwd()
        engine.eeglab("nogui", nargout=0)
        engine.cd(directory, nargout=0)
        engine.addpath(str(Path(__file__).parent / "matlab"), nargout=0)
        yield engine
    finally:
        engine.quit()


@pytest.fixture
def eeglab_backend(request):
    """Call a named reference function on the explicitly selected backend."""
    if request.config.getoption("--eeglab-backend") == "matlab":
        engine = request.getfixturevalue("eeglab_matlab_engine")
        try:
            yield partial(call_matlab, engine)
        finally:
            engine.close("all", "force", nargout=0)
    else:
        yield call_python


@pytest.fixture
def eeglab_working_directory(eeglab_backend, request, tmp_path, monkeypatch):
    """Keep relative input/output workflows in an isolated backend directory."""
    monkeypatch.chdir(tmp_path)
    if request.config.getoption("--eeglab-backend") != "matlab":
        yield tmp_path
        return
    engine = request.getfixturevalue("eeglab_matlab_engine")
    previous = engine.pwd()
    engine.cd(str(tmp_path), nargout=0)
    try:
        yield tmp_path
    finally:
        engine.cd(previous, nargout=0)


SLOW_NODEID_PARTS = (
    "tests/test_eeg_amica.py::",
    "tests/test_runamica.py::TestRunamicaIntegration::",
)

VISUAL_FILE_SUFFIXES = ("tests/test_visual_parity.py",)

GUI_FILE_SUFFIXES = (
    "tests/test_gui_pop_adjustevents.py",
    "tests/test_gui_pop_clean_rawdata.py",
    "tests/test_gui_pop_comments.py",
    "tests/test_gui_pop_editset.py",
    "tests/test_gui_pop_firfilt.py",
    "tests/test_gui_pop_iclabel.py",
    "tests/test_gui_pop_prop_extended.py",
    "tests/test_gui_pop_resample.py",
    "tests/test_gui_pop_runica.py",
    "tests/test_gui_pop_select.py",
    "tests/test_gui_pop_study.py",
    "tests/test_gui_long_task.py",
    "tests/test_gui_main_window.py",
    "tests/test_eegplot_gui.py",
    "tests/test_guifunc_primitives.py",
)
GUI_NODEID_PARTS = ("::test_gui_",)

MATLAB_FILE_SUFFIXES = (
    "tests/test_ICL_feature_extractor_parity.py",
    "tests/test_bids_preproc.py",
    "tests/test_clean_rawdata.py",
    "tests/test_eeg_compare.py",
    "tests/test_eeg_eegrej.py",
    "tests/test_eeg_lat2point.py",
    "tests/test_eeg_point2lat.py",
    "tests/test_eeg_rpsd_parity.py",
    "tests/test_eegfindboundaries.py",
    "tests/test_envtopo_parity.py",
    "tests/test_iclabel.py",
    "tests/test_iclabel_features.py",
    "tests/test_parity_rng.py",
    "tests/test_pinv.py",
    "tests/test_pipeline.py",
    "tests/test_pop_epoch.py",
    "tests/test_pop_loadset_h5.py",
    "tests/test_pop_resample.py",
    "tests/test_spectopo_parity.py",
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
    "tests/test_pop_firfilt.py::TestPopFirfiltParity::",
    "tests/test_pop_rmbase.py::TestPopRmbaseParity::",
    "tests/test_pop_select.py::TestPopSelectParity::",
    "tests/test_runica.py::TestRunicaParity::",
    "tests/test_topoplot.py::TestTopoplotParity::",
)

OCTAVE_NODEID_PARTS = ("tests/test_matlab_path.py::TestMatlabPath::test_get_eeglab_oct",)


def _path_has_suffix(path: str, suffixes: tuple[str, ...]) -> bool:
    return any(path == suffix or path.endswith(f"/{suffix}") for suffix in suffixes)


def _nodeid_has_part(nodeid: str, parts: tuple[str, ...]) -> bool:
    return any(part in nodeid for part in parts)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    backend = config.getoption("--eeglab-backend")

    for item in items:
        nodeid = item.nodeid
        lower_nodeid = nodeid.lower()
        path = item.path.as_posix()

        contract = "eeglab_backend" in getattr(item, "fixturenames", ())
        transport = "eeglab_matlab_engine" in getattr(item, "fixturenames", ())
        if backend == "matlab" and (contract or transport):
            item.add_marker(pytest.mark.matlab)

        if _nodeid_has_part(nodeid, SLOW_NODEID_PARTS):
            item.add_marker(pytest.mark.slow)

        if _path_has_suffix(path, VISUAL_FILE_SUFFIXES):
            item.add_marker(pytest.mark.visual)

        if _path_has_suffix(path, GUI_FILE_SUFFIXES) or _nodeid_has_part(lower_nodeid, GUI_NODEID_PARTS):
            item.add_marker(pytest.mark.gui)

        if "parity" in lower_nodeid:
            item.add_marker(pytest.mark.parity)

        requires_matlab = _path_has_suffix(path, MATLAB_FILE_SUFFIXES) or _nodeid_has_part(nodeid, MATLAB_NODEID_PARTS)
        if requires_matlab:
            item.add_marker(pytest.mark.matlab)

        if _nodeid_has_part(nodeid, OCTAVE_NODEID_PARTS):
            item.add_marker(pytest.mark.octave)


@pytest.hookimpl(specname="pytest_collection_modifyitems", trylast=True)
def pytest_filter_eeglab_contracts(config, items):
    # Apply the reference-lane gate after pytest's explicit -k/-m selection.
    backend = config.getoption("--eeglab-backend")
    deselected = []
    unconverted = []
    for item in items:
        contract = "eeglab_backend" in getattr(item, "fixturenames", ())
        transport = "eeglab_matlab_engine" in getattr(item, "fixturenames", ())
        if (contract and backend is None) or (transport and backend != "matlab"):
            deselected.append(item)
        if backend == "matlab" and not contract and not transport and upstream_references(getattr(item, "obj", None)):
            unconverted.append(item.nodeid)
    if unconverted:
        raise pytest.UsageError(
            "MATLAB mode selected tests that still call Python directly; convert them to "
            "the eeglab_backend fixture first:\n" + "\n".join(unconverted)
        )
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = [item for item in items if item not in deselected]
