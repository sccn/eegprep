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

from tests.eeglab_python_backend_xfails import PYTHON_BACKEND_SUBTEST_XFAILS, PYTHON_BACKEND_XFAILS
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


def _eeglab_support_directory(value):
    directory = Path(value).expanduser().resolve()
    if not directory.is_dir():
        raise pytest.UsageError(f"--eeglab-support-path requires an existing directory: {value}")
    return directory


def _exception_matches(
    exc: BaseException | None, expected: type[BaseException] | tuple[type[BaseException], ...]
) -> bool:
    return exc is not None and isinstance(exc, expected)


def _subtest_label(report) -> str | None:
    """Return the public subtest headline suffix, such as ``(source='name')``."""
    if report.when != "call":
        return None
    domain = report.location[2]
    headline = report.head_line or ""
    prefix = f"{domain} "
    if not headline.startswith(prefix):
        return None
    return headline[len(prefix) :]


class _PythonBackendReports:
    """Strict-xfail individual subtests on the Python backend.

    The parent call returns after a subtest failure is stored, so a parent
    xfail marker would be an unexpected pass. The exception object comes from
    ``pytest_runtest_makereport``; matching uses ``isinstance``, including a
    bare ``AssertionError`` and subclasses.
    """

    def __init__(self, config: pytest.Config) -> None:
        self.config = config
        self._exceptions: dict[str, BaseException | None] = {}
        self._unexpected: dict[str, int] = {}
        self._xfailed: dict[str, int] = {}
        self._in_call: set[str] = set()
        self._passed_parents: set[str] = set()

    @pytest.hookimpl(wrapper=True)
    def pytest_runtest_call(self, item):
        self._in_call.add(item.nodeid)
        try:
            return (yield)
        finally:
            self._in_call.discard(item.nodeid)

    @pytest.hookimpl(wrapper=True, tryfirst=True)
    def pytest_runtest_makereport(self, item, call):
        self._exceptions[item.nodeid] = None if call.excinfo is None else call.excinfo.value
        report = yield
        if (
            call.when == "call"
            and item.nodeid not in self._in_call
            and self.config.getoption("--eeglab-backend") == "python"
            and report.passed
            and not self._unexpected.get(item.nodeid)
            and self._xfailed.get(item.nodeid)
        ):
            self._passed_parents.add(item.nodeid)
        return report

    def _expect_subtest_failure(self, report) -> None:
        if self.config.getoption("--eeglab-backend") != "python":
            return
        label = _subtest_label(report)
        if label is None:
            return
        entry = PYTHON_BACKEND_SUBTEST_XFAILS.get(report.nodeid, {}).get(label)
        if entry is None:
            if report.failed:
                self._unexpected[report.nodeid] = self._unexpected.get(report.nodeid, 0) + 1
            return
        if report.failed:
            if not _exception_matches(self._exceptions.get(report.nodeid), entry.raises):
                self._unexpected[report.nodeid] = self._unexpected.get(report.nodeid, 0) + 1
                return
            report.outcome = "skipped"
            report.wasxfail = entry.reason
            self._xfailed[report.nodeid] = self._xfailed.get(report.nodeid, 0) + 1
            return
        if report.passed:
            report.outcome = "failed"
            report.longrepr = f"[XPASS(strict)] {entry.reason}"
            self._unexpected[report.nodeid] = self._unexpected.get(report.nodeid, 0) + 1

    @pytest.hookimpl(wrapper=True, tryfirst=True)
    def pytest_runtest_logreport(self, report: pytest.TestReport):
        # Runs before the session hook that increments testsfailed.
        self._expect_subtest_failure(report)
        # A killed runner never flushes pytest's capture.
        text = report.longreprtext
        if report.failed and text:
            os.write(2, f"\n=== {report.nodeid} [{report.when}] ===\n{text}\n".encode())
        return (yield)

    @pytest.hookimpl(tryfirst=True)
    def pytest_report_teststatus(self, report):
        """Return passed before ``_pytest/subtests.py`` rewrites the parent.

        ``tryfirst`` runs this hook before that module's ``pytest_report_teststatus``.
        The hook spec is ``firstresult``, so returning a status skips the later
        hook. That hook would mark this passed parent failed, because its count
        still includes subtests recorded here as expected failures. A subtest
        report shares the parent node id, so only the parent headline returns here.
        """
        if report.when != "call" or report.nodeid not in self._passed_parents:
            return None
        if _subtest_label(report) is not None:
            return None
        return "passed", ".", "PASSED"


def pytest_configure(config):
    config.pluginmanager.register(_PythonBackendReports(config), "eegprep-python-backend-reports")
    # CI: MATLAB served by matlab-batch (the batch token cannot license the Engine
    # API), against the vendored EEGLAB. Set here, after option parsing, so that
    # --eeglab-root keeps its default and the pinned eeglab_tests lane stays opted out.
    batch_dir = os.environ.get("EEGPREP_MATLAB_BATCH_DIR")
    if batch_dir:
        from tests.matlab_batch_engine import install

        install(batch_dir)
        os.environ.setdefault("EEGPREP_EEGLAB_ROOT", str(Path(__file__).resolve().parents[1] / "src/eegprep/eeglab"))


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
    group.addoption(
        "--eeglab-support-path",
        action="append",
        default=[],
        type=_eeglab_support_directory,
        metavar="PATH",
        help="Session-local MATLAB support directory, added after plugins; repeatable, last path takes precedence.",
    )
    group.addoption(
        "--run-heavy-reference",
        action="store_true",
        default=False,
        help="Run heavy EEGLAB reference tests that are skipped unless this flag is passed.",
    )
    group.addoption(
        "--record-parity-summary",
        action="store_true",
        default=False,
        help="Overwrite tests/parity_expected/ds003061_pipeline.json with this machine's measured pipeline summary.",
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
        for support_path in request.config.getoption("--eeglab-support-path"):
            engine.addpath(str(support_path), "-begin", nargout=0)
        engine.addpath(str(Path(__file__).parent / "matlab"), nargout=0)
        # nogui does not run eegplugin_* hooks, so nested plugin functions stay off the path.
        engine.eegprep_reference_plugins(str(Path(root).resolve()), nargout=0)
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
            # close('all','force') can fail after a headless pop_* and leave the
            # figure (and its data) alive. Deleting the figures directly keeps a
            # later test from running the runner out of memory.
            try:
                engine.eval("delete(findall(0, 'Type', 'figure'));", nargout=0)
            except Exception as exc:
                os.write(2, f"eeglab_backend figure cleanup failed: {exc}\n".encode())
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
    "tests/test_gui_pop_clean_rawdata.py",
    "tests/test_gui_pop_erpimage.py",
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
    "tests/test_parity_rng.py",
    "tests/test_pinv.py",
    "tests/test_pipeline.py",
    "tests/test_pop_epoch.py",
    "tests/test_pop_resample.py",
    "tests/test_pop_studydesign_matlab.py",
    "tests/test_spectopo_parity.py",
)

# These stay collected so port-audit provenance still sees them. They run only
# with --run-heavy-reference.
HEAVY_REFERENCE_SKIP_REASON = "heavy EEGLAB reference test; pass --run-heavy-reference to run"
HEAVY_REFERENCE_NODEIDS = (
    # LIMO integration, IRLS validation, and preprocessing+stats.
    "tests/test_limo_eeglab_tests.py::test_reference_limo_integration",
    "tests/test_limo_eeglab_tests.py::test_reference_limo_irls_validation",
    "tests/test_limo_eeglab_tests.py::test_reference_limo_preprocessing_and_statistics",
    "tests/test_eeglab_hybrid.py::test_matlab_limo_optional_file_choice_preserves_plots_and_restores_io",
    # STUDY grouped measure plots.
    "tests/test_study_grouped_measure_plots_eeglab_tests.py::test_reference_std_erpplot_channel_cache[teststudy]",
    "tests/test_study_grouped_measure_plots_eeglab_tests.py::test_reference_std_erpplot_component_cache[teststudy]",
    "tests/test_study_grouped_measure_plots_eeglab_tests.py::test_reference_std_erpplot_design_sweep[teststudy2]",
    "tests/test_study_grouped_measure_plots_eeglab_tests.py::test_reference_std_erspplot_channel_cache[teststudy]",
    "tests/test_study_grouped_measure_plots_eeglab_tests.py::test_reference_std_erspplot_component_cache[teststudy]",
    "tests/test_study_grouped_measure_plots_eeglab_tests.py::test_reference_std_erspplot_legacy_channel_cache[teststudy2]",
    "tests/test_study_grouped_measure_plots_eeglab_tests.py::test_reference_std_erspplot_legacy_component_cache[teststudy2]",
    "tests/test_study_grouped_measure_plots_eeglab_tests.py::test_reference_std_specplot_channel_cache[teststudy]",
    "tests/test_study_grouped_measure_plots_eeglab_tests.py::test_reference_std_specplot_component_cache[teststudy]",
    "tests/test_study_grouped_measure_plots_eeglab_tests.py::test_reference_std_specplot_design_sweep[teststudy2]",
    "tests/test_study_grouped_measure_plots_eeglab_tests.py::test_reference_std_topoplot[teststudy]",
    # std_erspplot, std_itcplot, and std_specplot. std_precomp and std_preclust stay runnable.
    "tests/test_study_workflows_eeglab_tests.py::test_reference_std_erspplot[teststudy]",
    "tests/test_study_workflows_eeglab_tests.py::test_reference_std_itcplot[teststudy]",
    "tests/test_study_workflows_eeglab_tests.py::test_reference_std_specplot[teststudy]",
    # Transfers larger than the MAT5 container limits.
    "tests/test_eeglab_hybrid.py::test_matlab_aggregate_outputs_over_two_gib_remain_complete",
    "tests/test_eeglab_hybrid.py::test_matlab_inputs_over_four_gib_remain_complete[False]",
    "tests/test_eeglab_hybrid.py::test_matlab_inputs_over_four_gib_remain_complete[True]",
    # ds003061 import versus MATLAB pop_importbids. The later chain stays separate.
    "tests/test_ds003061_stage_parity.py::test_load_frombids_matches_matlab_pop_importbids",
    # Tutorial reference workflows. The plain BIDS face test is not in this list.
    "tests/test_tutorial_eeglab_tests.py::test_reference_tutorial_bids_p300",
    "tests/test_tutorial_eeglab_tests.py::test_reference_tutorial_event_processing_single_dataset",
    "tests/test_tutorial_eeglab_tests.py::test_reference_tutorial_event_processing_study",
    "tests/test_tutorial_eeglab_tests.py::test_reference_tutorial_history",
    "tests/test_tutorial_eeglab_tests.py::test_reference_tutorial_make_eeg_movie",
    "tests/test_tutorial_eeglab_tests.py::test_reference_tutorial_plot_study_erp[teststudy]",
    "tests/test_tutorial_eeglab_tests.py::test_reference_tutorial_source_reconstruction_advanced",
    "tests/test_tutorial_eeglab_tests.py::test_reference_tutorial_source_reconstruction_eeg",
    "tests/test_tutorial_eeglab_tests.py::test_reference_tutorial_study_script[teststudy]",
    "tests/test_tutorial_eeglab_tests.py::test_reference_tutorial_time_freq_all_elec",
)

MATLAB_NODEID_PARTS = (
    "tests/test_eeg_autocorr.py::TestEegAutocorr::test_parity_",
    "tests/test_eeg_interp.py::TestComputeGParity::",
    "tests/test_eeg_interp.py::TestEegInterpParity::",
    "tests/test_eeg_interp.py::TestSphericalSplineParity::",
    "tests/test_eeg_picard.py::TestEegPicard::",
    "tests/test_erpimage_engine.py::TestErpimageMatlabParity::",
    "tests/test_epoch.py::TestEpochParity::",
    "tests/test_pop_loadset_h5.py::TestPopLoadsetH5Parity::",
    "tests/test_pop_reref.py::TestPopReref::test_parity_",
    "tests/test_pop_firfilt.py::TestPopFirfiltParity::",
    "tests/test_pop_rmbase.py::TestPopRmbaseParity::",
    "tests/test_pop_select.py::TestPopSelectParity::",
    "tests/test_runica.py::TestRunicaParity::",
    "tests/test_topoplot.py::TestTopoplotParity::",
)


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
        if contract:
            item.add_marker(pytest.mark.eeglab_contract)
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

        if not config.getoption("--run-heavy-reference") and nodeid in HEAVY_REFERENCE_NODEIDS:
            item.add_marker(pytest.mark.skip(reason=HEAVY_REFERENCE_SKIP_REASON))

        if backend == "python":
            entry = PYTHON_BACKEND_XFAILS.get(nodeid)
            if entry is not None:
                item.add_marker(pytest.mark.xfail(reason=entry.reason, raises=entry.raises, strict=True))


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
        elif backend == "python" and item.get_closest_marker("matlab_oracle") is not None:
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
