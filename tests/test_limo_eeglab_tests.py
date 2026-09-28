"""Original LIMO workflows and separately retained generated Python checks."""

from __future__ import annotations

from contextlib import contextmanager
from hashlib import sha256
import json
import logging
from pathlib import Path
import shutil
import subprocess

import matplotlib
import numpy as np
import pytest
from scipy import stats
from scipy.io import savemat

matplotlib.use("Agg")

from matplotlib import pyplot as plt

import eegprep
from eegprep.functions.studyfunc._limo_io import save_limo_result
from eegprep.functions.studyfunc.pop_limo import pop_limo
from eegprep.functions.studyfunc.pop_limoresults import pop_limoresults
from eegprep.functions.studyfunc.pop_study import pop_study
from eegprep.functions.studyfunc.std_limo import std_limo
from eegprep.functions.studyfunc.std_limoresults import std_limoresults
from eegprep.functions.studyfunc.std_maketrialinfo import std_maketrialinfo
from eegprep.functions.studyfunc.std_makedesign import std_makedesign
from eegprep.functions.studyfunc.std_readfilelimo import std_readfilelimo
from tests.eeglab_tests import eeglab_test
from tests.eeglab_tests.limo_irls import (
    IRLS_CHANLOCS_ASSIGNMENT,
    IRLS_HELPER_SHA256,
    IRLS_SOURCE,
    IRLS_SOURCE_SHA256,
    RESULT_FIELDS,
    _matlab_max,
    load_irls_mat,
    matlab_binofit,
    prepare_irls_source_overlay,
    reference_limo_glmboot,
)
from tests.test_eeg_store import _source_dataset_row
from tests.test_tutorial_eeglab_tests import _tutorial_redraw, _tutorial_start


LIMO_WRAPPER = "unittesting_limo/limo_wrapperTest.m"
LIMO_PREPROCESSING_SOURCE = "unittesting_limo/limo_preproc_stats_hw.m"
LIMO_PREPROCESSING_SHA256 = "36327e3c301bc8a16455331b69f487205994963021aa10ec4997ee96f05e154a"
LIMO_PREPROCESSING_PATCH = Path(__file__).parent / "matlab" / "limo_preproc_stats_hw.source-corrections.patch"
LIMO_INTEGRATION_SOURCE = "unittesting_limo/limo_test_integration.m"
LIMO_INTEGRATION_SHA256 = "8ff180227af8dcdba48f8deeed3eec5ac32331db2f2d666d7dc60bd99103a6bf"
LIMO_INTEGRATION_REPAIRS = Path(__file__).parent / "matlab" / "limo_test_integration.source-corrections.json"
FACE_EVENTS = tuple(
    f"{face}_{repetition}"
    for face in ("famous", "scrambled", "unfamiliar")
    for repetition in ("new", "second_early", "second_late")
)
logger = logging.getLogger(__name__)


def _cell_row(*values):
    result = np.empty((1, len(values)), dtype=object)
    result[0] = values
    return result


@pytest.fixture
def limo_source_directory(request, eeglab_suite_root, eeglab_options_directory, monkeypatch, tmp_path):
    """The original BIDS subjects, copied before any preprocessing writes."""
    directory = Path(shutil.copytree(eeglab_suite_root / "ds002718", tmp_path / "ds002718"))
    monkeypatch.chdir(directory)
    if request.config.getoption("--eeglab-backend") == "matlab":
        # The options fixture restores the original MATLAB directory afterward.
        request.getfixturevalue("eeglab_matlab_engine").cd(str(directory), nargout=0)
    return directory


def _limo_cd(request, monkeypatch, directory):
    monkeypatch.chdir(directory)
    if request.config.getoption("--eeglab-backend") == "matlab":
        request.getfixturevalue("eeglab_matlab_engine").cd(str(directory), nargout=0)


def _limo_graphics(request, operation, *args, **kwargs):
    if request.config.getoption("--eeglab-backend") == "matlab":
        engine = request.getfixturevalue("eeglab_matlab_engine")
        options = [value for pair in kwargs.items() for value in pair]
        engine.feval(operation, *args, *options, nargout=0)
    elif operation == "subplot":
        plt.subplot(*(int(value) for value in args))
    else:
        getattr(plt, operation)(*args, **kwargs)


@contextmanager
def _limo_status(statuses, section):
    # The source catches each section separately, then asserts all nine passed.
    try:
        yield
    except Exception as error:
        statuses.append((False, f"{section} failed\n{type(error).__name__}: {error}"))
    else:
        statuses.append((True, f"{section} successful"))


def _limo_entries(values):
    # MATLAB cells/struct arrays retain their dimensions in the test transport;
    # Python's equivalents are lists. Index both in MATLAB's linear order.
    return values.ravel(order="F") if isinstance(values, np.ndarray) else values


def prepare_limo_preprocessing_source_overlay(suite_root: Path, directory: Path) -> Path:
    """Apply path/workspace setup repairs and explicit optional-file cancellation."""
    source = suite_root / LIMO_PREPROCESSING_SOURCE
    if sha256(source.read_bytes()).hexdigest() != LIMO_PREPROCESSING_SHA256:
        raise ValueError(f"LIMO preprocessing source differs from the pinned original: {source}")
    source_directory = directory / "unittesting_limo"
    source_directory.mkdir()
    shutil.copy2(source, source_directory / source.name)
    subprocess.run(
        ["git", "apply", "--no-index", "--unidiff-zero", str(LIMO_PREPROCESSING_PATCH)], cwd=directory, check=True
    )
    corrected = source_directory / source.name
    text = corrected.read_text()
    assert text.count("limo_add_plots(") == 8
    for target in ("[STUDY,~,LIMOfiles] = pop_limo(", "limo_batch('contrast only',[],contrast);"):
        assert text.count(target) == 1
        text = text.replace(target, "assignin('base','STUDY',STUDY);\n" + target)
    corrected.write_text(text.replace("limo_add_plots(", "eegprep_test_limo_add_plots("))
    return source_directory


def prepare_limo_integration_source_overlay(suite_root: Path, directory: Path) -> Path:
    """Apply the approved path and base-STUDY setup repairs to a pinned copy."""
    source = suite_root / LIMO_INTEGRATION_SOURCE
    original = source.read_bytes()
    if sha256(original).hexdigest() != LIMO_INTEGRATION_SHA256:
        raise ValueError(f"LIMO integration source differs from the pinned original: {source}")
    corrected = original.decode("utf-8")
    # Exact substrings preserve the original trailing spaces on affected lines;
    # a unified patch would lose those bytes in the required whitespace checks.
    for repair in json.loads(LIMO_INTEGRATION_REPAIRS.read_text()):
        if corrected.count(repair["source"]) != repair["count"]:
            raise ValueError(f"LIMO integration source pattern count differs: {repair['source']}")
        corrected = corrected.replace(repair["source"], repair["replacement"])
    source_directory = directory / "unittesting_limo"
    source_directory.mkdir()
    (source_directory / source.name).write_bytes(corrected.encode("utf-8"))
    return source_directory


def _limo_assign_groups(study):
    for index, info in enumerate(_limo_entries(study["datasetinfo"])):
        info["group"] = "1" if index < 6 else "2" if index < 13 else "3"


def _irls_remove_directory(directory, study_path):
    # The source removes previous derivative/H0 results, never source recordings.
    # Its try/catch permits absent or unremovable directories. Restrict this
    # destructive setup to the explicit copied study before preserving that catch.
    directory.resolve().relative_to(study_path.resolve())
    try:
        shutil.rmtree(directory)
    except OSError as error:
        logger.info("IRLS source cleanup: %s", error)


@eeglab_test(IRLS_SOURCE, "limo_zIRLS_validation_4_Arno")
@pytest.mark.slow
@pytest.mark.gui
def test_reference_limo_irls_validation(eeglab_backend, limo_source_directory, eeglab_suite_root, request):
    """Full standalone IRLS workflow with the explicitly approved input repairs.

    The native overlay supplies chanlocs, captures the generated model paths,
    and discovers the generated H0 filenames.
    No complete MATLAB test or validation-helper script is evaluated here.
    """
    call = eeglab_backend
    native = request.config.getoption("--eeglab-backend") == "matlab"
    overlay = prepare_irls_source_overlay(eeglab_suite_root, limo_source_directory.parent)
    if native:
        call("eegprep_test_base_workspace", "snapshot", nargout=0)
        request.addfinalizer(lambda: call("eegprep_test_base_workspace", "restore", nargout=0))
        call("evalin", "base", "clear variables", nargout=0)
        call("addpath", str(overlay), nargout=0)
    for function, message in (
        ("eeglab", "EEGLAB is not in your path"),
        ("pop_importbids", "EEGLAB BIDS import tools are not your path"),
        ("limo_eeg", "LIMO tools are not in your path"),
    ):
        available = call("exist", f"{function}.m", "file") if native else getattr(eegprep, function, None)
        if not available:
            raise RuntimeError(message)
    if native and not call("exist", "limo_test_glmboot.m", "file"):
        raise RuntimeError("get/set limo validation tools to check type 1 error")
    derivative_path = limo_source_directory / "derivative_IRLS"
    _irls_remove_directory(derivative_path, limo_source_directory)
    window, _alleeg, _eeg, _currentset, allcom = _tutorial_start(call, request, outputs=4)
    study, alleeg = call(
        "pop_importbids",
        str(limo_source_directory),
        bidsevent="on",
        bidschanloc="on",
        studyName="Face_detection",
        eventtype="trial_type",
        outputdir=str(derivative_path),
        nargout=2,
    )
    study["studypath"] = str(derivative_path)
    alleeg = call("pop_select", alleeg, "nochannel", _cell_row("EEG061", "EEG062", "EEG063", "EEG064"))
    eeg = alleeg
    currentset = np.arange(1.0, len(_limo_entries(eeg)) + 1).reshape(1, -1)
    eeg = call(
        "pop_clean_rawdata",
        eeg,
        "FlatlineCriterion",
        5.0,
        "ChannelCriterion",
        0.8,
        "LineNoiseCriterion",
        4.0,
        "Highpass",
        np.array([[0.25, 0.75]]),
        "BurstCriterion",
        "off",
        "WindowCriterion",
        "off",
        "BurstRejection",
        "off",
        "Distance",
        "Euclidian",
        "WindowCriterionTolerances",
        "off",
    )
    eeg = call("pop_reref", eeg, np.empty((0, 0)), "interpchan", np.empty((0, 0)))
    thresholds = np.full((7, 2), np.nan)
    thresholds[1:3] = [0.8, 1.0]
    processed = []
    for dataset in _limo_entries(eeg):
        if native:
            dataset = {field: dataset[field] for field in dataset.dtype.names}
        dataset = call(
            "pop_runica",
            dataset,
            "icatype",
            "picard",
            "concatcond",
            "on",
            "options",
            _cell_row("pca", float(np.asarray(dataset["nbchan"]).item()) - 1),
        )
        dataset = call("pop_iclabel", dataset, "default")
        dataset = call("pop_icflag", dataset, thresholds)
        rejected = np.asarray(dataset["reject"]["gcompreject"])
        components = np.flatnonzero(rejected.ravel(order="F")).astype(float) + 1
        components = (
            components.reshape(1, -1) if rejected.ndim == 1 or rejected.shape[0] == 1 else components.reshape(-1, 1)
        )
        processed.append(call("pop_subcomp", dataset, components, 0.0))
    eeg = _source_dataset_row(*processed) if native else processed
    eeg = call(
        "pop_clean_rawdata",
        eeg,
        "FlatlineCriterion",
        "off",
        "ChannelCriterion",
        "off",
        "LineNoiseCriterion",
        "off",
        "Highpass",
        "off",
        "BurstCriterion",
        20.0,
        "WindowCriterion",
        0.25,
        "BurstRejection",
        "on",
        "Distance",
        "Euclidian",
        "WindowCriterionTolerances",
        np.array([[-np.inf, 7.0]]),
    )
    eeg = call("pop_epoch", eeg, _cell_row(*FACE_EVENTS), np.array([[-0.5, 1.0]]), "epochinfo", "yes")
    eeg = call("eeg_checkset", eeg)
    eeg = call("pop_saveset", eeg, "savemode", "resave")
    alleeg = eeg
    study = call("std_checkset", study, alleeg)
    study = call(
        "std_makedesign",
        study,
        eeg,
        1.0,
        "name",
        "STUDY.FaceRepetition",
        "delfiles",
        "off",
        "defaultdesign",
        "off",
        "variable1",
        "type",
        "values1",
        np.empty((0, 0), dtype=object),
    )
    _tutorial_redraw(
        call,
        request,
        window,
        STUDY=study,
        ALLEEG=alleeg,
        EEG=eeg,
        CURRENTSTUDY=1.0,
        CURRENTSET=currentset,
        ALLCOM=allcom,
    )
    study, eeg = call(
        "std_precomp",
        study,
        eeg,
        np.empty((0, 0), dtype=object),
        "savetrials",
        "on",
        "interp",
        "on",
        "recompute",
        "on",
        "erp",
        "on",
        "erpparams",
        _cell_row("rmbase", np.array([[-200.0, 0.0]])),
        "spec",
        "on",
        "ersp",
        "on",
        "itc",
        "on",
        "specparams",
        _cell_row("specmode", "fft", "logtrials", "off"),
        nargout=2,
    )
    _tutorial_redraw(call, request, window, STUDY=study, EEG=eeg)
    study, _, model_files = call(
        "pop_limo",
        study,
        alleeg,
        "method",
        "IRLS",
        "measure",
        "daterp",
        "timelim",
        np.array([[-50.0, 650.0]]),
        "erase",
        "on",
        "splitreg",
        "off",
        "interaction",
        "off",
        nargout=3,
    )
    _tutorial_redraw(call, request, window, STUDY=study)
    h0 = []
    for subject in range(1, len(_limo_entries(study["datasetinfo"])) + 1):
        logger.info("running bootstrap on subject %g", subject)
        # Approved input repair: consume the actual subject-ordered paths from
        # pop_limo, rather than reproduce a version-dependent naming convention.
        model_file = Path(_limo_entries(model_files["mat"])[subject - 1])
        limo = load_irls_mat(call, native, model_file)["LIMO"]
        limo["design"]["bootstrap"] = 2500.0
        limo["design"]["status"] = "to do"
        model_directory = Path(limo["dir"])
        model_directory.resolve().relative_to(limo_source_directory.resolve())
        savemat(model_directory / "LIMO.mat", {"LIMO": limo}, long_field_names=True)
        _irls_remove_directory(model_directory / "H0", limo_source_directory)
        h0.append(str(model_file.parent / "H0"))
        call("limo_eeg", 4.0, limo, nargout=0)
    # User-approved missing assignment, identical to the checked native patch.
    # std_limo writes this file; no montage or adjacency is synthesized here.
    chanlocs = Path(study["filepath"]) / "derivatives" / "limo_gp_level_chanlocs.mat"
    values = reference_limo_glmboot(call, native, str(chanlocs), h0, step_size=300, Nboot=1000, MinSamp=300)
    results = dict(zip(RESULT_FIELDS, values, strict=True))
    directory = Path(call("pwd")) if native else Path.cwd()
    savemat(directory / "results.mat", {"results": results}, long_field_names=True)


def test_irls_native_overlay_applies_only_approved_input_repairs(request, tmp_path):
    if not (request.config.getoption("--eeglab-suite-root") or request.config.getoption("--eeglab-root")):
        pytest.skip("IRLS overlay verification requires the optional pinned EEGLAB tests checkout")
    eeglab_suite_root = request.getfixturevalue("eeglab_suite_root")
    original = eeglab_suite_root / IRLS_SOURCE
    before = original.read_bytes()
    original_helper = original.with_name("limo_test_glmboot.m")
    helper_before = original_helper.read_bytes()
    directory = prepare_irls_source_overlay(eeglab_suite_root, tmp_path)
    corrected = (directory / original.name).read_text()
    assert corrected.count(IRLS_CHANLOCS_ASSIGNMENT) == 1
    restored = corrected.replace(IRLS_CHANLOCS_ASSIGNMENT, "")
    for source, replacement in (
        ("STUDY = pop_limo(", "[STUDY,~,LIMOfiles] = pop_limo("),
        (
            "    LIMO = load(fullfile(STUDY.datasetinfo(s).filepath,['FaceRepetition_GLM_Channels_Time_IRLS' filesep 'LIMO.mat']));",
            "    LIMO = load(LIMOfiles.mat{s});",
        ),
        (
            "    H0iw{s} = fullfile(STUDY.datasetinfo(s).filepath,['FaceRepetition_GLM_Channels_Time_IRLS' filesep 'H0']);",
            "    H0iw{s} = fullfile(fileparts(LIMOfiles.mat{s}), 'H0');",
        ),
    ):
        assert restored.count(replacement) == 1
        restored = restored.replace(replacement, source)
    assert restored.encode() == before
    assert original.read_bytes() == before
    assert sha256(before).hexdigest() == IRLS_SOURCE_SHA256
    corrected_helper = (directory / original_helper.name).read_text()
    assert corrected_helper.count("'*H0.mat'") == 1
    assert corrected_helper.replace("'*H0.mat'", "'H0_*.mat'").encode() == helper_before
    assert original_helper.read_bytes() == helper_before
    assert sha256(helper_before).hexdigest() == IRLS_HELPER_SHA256


@pytest.mark.parametrize("successes, expected", [(0, (0.0, 0.975)), (1, (0.025, 1.0))])
def test_irls_binomial_interval_retains_source_default_confidence(successes, expected):
    estimate, interval = matlab_binofit(successes, 1)
    assert estimate == successes
    np.testing.assert_allclose(interval, expected, rtol=0, atol=1e-15)


def test_irls_mat_loading_retains_singleton_frequency_axis_and_model_fields(tmp_path):
    file = tmp_path / "H0.mat"
    data = np.arange(48, dtype=np.float32).reshape(2, 1, 3, 2, 4)
    savemat(file, {"H0": data, "LIMO": {"dir": str(tmp_path), "design": {"bootstrap": 2500.0, "status": "to do"}}})
    loaded = load_irls_mat(None, False, file)
    np.testing.assert_array_equal(loaded["H0"], data, strict=True)
    assert loaded["LIMO"]["dir"] == str(tmp_path)
    assert loaded["LIMO"]["design"]["status"] == "to do"
    np.testing.assert_array_equal(loaded["LIMO"]["design"]["bootstrap"], np.array([[2500.0]]))


@pytest.mark.parametrize(
    "values, maximum, index",
    [
        ([[np.nan, 5.0], [2.0, np.nan]], 5.0, 2),
        ([[2.0, 5.0], [5.0, np.nan]], 5.0, 1),
        ([[np.nan, -np.inf]], -np.inf, 1),
        ([[np.nan, np.nan]], np.nan, 0),
    ],
)
def test_irls_maximum_preserves_matlab_nan_and_first_index_semantics(values, maximum, index):
    value, position = _matlab_max(np.asarray(values))
    np.testing.assert_equal(value, maximum)
    assert position == index


def test_limo_preprocessing_native_overlay_preserves_paths_and_explicit_file_choice(request, tmp_path):
    if not (request.config.getoption("--eeglab-suite-root") or request.config.getoption("--eeglab-root")):
        pytest.skip("LIMO overlay verification requires the optional pinned EEGLAB tests checkout")
    suite_root = request.getfixturevalue("eeglab_suite_root")
    original = suite_root / LIMO_PREPROCESSING_SOURCE
    before = original.read_bytes()
    directory = prepare_limo_preprocessing_source_overlay(suite_root, tmp_path)
    corrected = (directory / original.name).read_text()
    modelname_assignment = "[~,modelname] = fileparts(fileparts(LIMOfiles.mat{1}));\n"
    assert corrected.count(modelname_assignment) == 1
    assert corrected.count("eegprep_test_limo_add_plots(") == 8
    restored = corrected.replace(modelname_assignment, "").replace("eegprep_test_limo_add_plots(", "limo_add_plots(")
    assert restored.count("assignin('base','STUDY',STUDY);\n") == 2
    restored = restored.replace("assignin('base','STUDY',STUDY);\n", "")
    replacements = [
        ("STUDY  = pop_limo(", "[STUDY,~,LIMOfiles] = pop_limo("),
        (
            "chanlocs = [STUDY.filepath filesep 'limo_gp_level_chanlocs.mat'];",
            "chanlocs = fullfile(fileparts(LIMOfiles.LIMO), 'limo_gp_level_chanlocs.mat');",
        ),
        (
            "[STUDY.filepath filesep 'LIMO_Face_detection' filesep  'Beta_files_FaceRepetition_GLM_Channels_Time_' mode '.txt']",
            "fullfile(LIMOfiles.LIMO, ['Beta_files_' modelname '.txt'])",
        ),
        (
            "[STUDY.filepath filesep 'LIMO_' STUDY.filename(1:end-6) filesep ...\n"
            "    'LIMO_files_FaceRepetition_GLM_Channels_Time_' mode '.txt']",
            "fullfile(LIMOfiles.LIMO, ['LIMO_files_' modelname '.txt'])",
        ),
    ]
    replacements.extend(
        (
            "[STUDY.filepath filesep 'LIMO_' STUDY.filename(1:end-6) filesep "
            f"'con_{index}_files_FaceRepetition_GLM_Channels_Time_' mode '.txt']",
            f"fullfile(LIMOfiles.LIMO, ['con_{index}_files_' modelname '.txt'])",
        )
        for index in range(1, 4)
    )
    for source, replacement in replacements:
        assert restored.count(replacement) == before.decode().count(source)
        restored = restored.replace(replacement, source)
    assert restored.encode() == before
    assert original.read_bytes() == before
    assert sha256(before).hexdigest() == LIMO_PREPROCESSING_SHA256


def test_limo_integration_native_overlay_applies_only_approved_repairs(request, tmp_path):
    if not (request.config.getoption("--eeglab-suite-root") or request.config.getoption("--eeglab-root")):
        pytest.skip("LIMO overlay verification requires the optional pinned EEGLAB tests checkout")
    suite_root = request.getfixturevalue("eeglab_suite_root")
    original = suite_root / LIMO_INTEGRATION_SOURCE
    before = original.read_bytes()
    directory = prepare_limo_integration_source_overlay(suite_root, tmp_path)
    restored = (directory / original.name).read_text()
    assert (
        "clear variables\n"
        "eegprep_test_base_workspace('snapshot');\n"
        "workspace_cleanup = onCleanup(@() eegprep_test_base_workspace('restore'));\n"
    ) in restored
    assert (
        "    assignin('base','STUDY',STUDY);\n    confiles                 = limo_batch('contrast only',[],contrast);"
    ) in restored
    assert (
        "    Model1_files.con         = confiles.con;\n"
        "    [~,Model1_con1] = fileparts(Model1_files.con{1}{1});\n"
        "    [~,Model1_con2] = fileparts(Model1_files.con{1}{2});"
    ) in restored
    # This inverse is independent of the JSON: unapproved edits cannot be hidden
    # by adding another reversible replacement to the overlay description.
    for assignment in (
        "eegprep_test_base_workspace('snapshot');\n",
        "workspace_cleanup = onCleanup(@() eegprep_test_base_workspace('restore'));\n",
        "    assignin('base','STUDY',STUDY);\n",
        "limo_rootfiles = Model2_files.LIMO;\n",
        "    [~,Model1_name] = fileparts(fileparts(Model1_files.mat{1}));\n",
        "    [~,Model2_name] = fileparts(fileparts(Model2_files.mat{1}));\n",
        "    [~,Model1_con1] = fileparts(Model1_files.con{1}{1});\n",
        "    [~,Model1_con2] = fileparts(Model1_files.con{1}{2});\n",
    ):
        assert restored.count(assignment) == 1
        restored = restored.replace(assignment, "")
    assert "confiles                 = limo_batch('contrast only',[],contrast);" in restored
    replacements = [
        ("['LIMO_files_' Model2_name '.txt']", "'LIMO_files_Face_time_GLM_Channels_Time_WLS.txt'", 1),
        ("Model1_name '.txt'", "STUDY.design(1).name '_GLM_Channels_Time_OLS.txt'", 17),
        ("Model2_name '.txt'", "STUDY.design(2).name '_GLM_Channels_Time_WLS.txt'", 8),
        ("'Beta_files_Gp-'", "'Beta_files_Gp'", 1),
        ("'Betas_desc-H0.mat'", "'H0_Betas.mat'", 2),
    ]
    for group in range(1, 4):
        replacements.append((f"'Beta_files_Gp-{group}_'", f"'Beta_files_Gp{group}_'", 1))
        for contrast, count in ((1, 2), (2, 1)):
            replacements.append(
                (
                    f"Model1_con{contrast} '_files_Gp-{group}_'",
                    f"'con_{contrast}_files_Gp{group}_'",
                    count,
                )
            )
    for replacement, source, count in replacements:
        assert restored.count(replacement) == count
        restored = restored.replace(replacement, source)
    assert restored.encode() == before
    assert original.read_bytes() == before
    assert sha256(before).hexdigest() == LIMO_INTEGRATION_SHA256


@eeglab_test(LIMO_WRAPPER, "limo_test1")
@pytest.mark.slow
@pytest.mark.gui
def test_reference_limo_preprocessing_and_statistics(
    eeglab_backend, limo_source_directory, eeglab_suite_root, request, monkeypatch
):
    """Full original pipeline with approved generated-path repairs; completion is its smoke oracle."""
    # Keep a hash-checked native overlay for the same narrow source correction;
    # this Python-owned workflow never evaluates the complete native script.
    prepare_limo_preprocessing_source_overlay(eeglab_suite_root, limo_source_directory.parent)
    call = eeglab_backend
    native = request.config.getoption("--eeglab-backend") == "matlab"
    plot_function = "eegprep_test_limo_add_plots" if native else "limo_add_plots"
    if native:
        call("eegprep_test_base_workspace", "snapshot", nargout=0)
        request.addfinalizer(lambda: call("eegprep_test_base_workspace", "restore", nargout=0))
    empty = np.empty((0, 0))
    call("pop_editoptions", option_storedisk=1.0, nargout=0)
    study, alleeg = call(
        "pop_importbids",
        str(limo_source_directory),
        "bidsevent",
        "on",
        "bidschanloc",
        "on",
        "studyName",
        "Face_detection",
        "outputdir",
        str(limo_source_directory / "derivatives2"),
        "eventtype",
        "trial_type",
        nargout=2,
    )
    alleeg = call("pop_select", alleeg, "nochannel", _cell_row("EEG061", "EEG062", "EEG063", "EEG064"))
    eeg = call(
        "pop_clean_rawdata",
        alleeg,
        "FlatlineCriterion",
        5.0,
        "ChannelCriterion",
        0.8,
        "LineNoiseCriterion",
        2.5,
        "Highpass",
        np.array([[0.25, 0.75]]),
        "BurstCriterion",
        "off",
        "WindowCriterion",
        "off",
        "BurstRejection",
        "off",
        "Distance",
        "Euclidian",
        "WindowCriterionTolerances",
        "off",
    )
    eeg = call("pop_reref", eeg, empty, "interpchan", empty)
    eeg = call("pop_runica", eeg, "icatype", "runica", "concatcond", "on", "options", _cell_row("pca", -1.0))
    eeg = call("pop_iclabel", eeg, "default")
    thresholds = np.full((7, 2), np.nan)
    thresholds[1:3] = [0.8, 1.0]
    eeg = call("pop_icflag", eeg, thresholds)
    eeg = call("pop_subcomp", eeg, empty, 0.0)
    eeg = call(
        "pop_clean_rawdata",
        eeg,
        "FlatlineCriterion",
        "off",
        "ChannelCriterion",
        "off",
        "LineNoiseCriterion",
        "off",
        "Highpass",
        "off",
        "BurstCriterion",
        20.0,
        "WindowCriterion",
        0.25,
        "BurstRejection",
        "on",
        "Distance",
        "Euclidian",
        "WindowCriterionTolerances",
        np.array([[-np.inf, 7.0]]),
    )
    eeg = call("pop_epoch", eeg, _cell_row(*FACE_EVENTS), np.array([[-0.5, 1.0]]), "epochinfo", "yes")
    eeg = call("eeg_checkset", eeg)
    eeg = call("pop_saveset", eeg, "savemode", "resave")
    alleeg = eeg
    study = call("std_checkset", study, alleeg)
    study, eeg = call(
        "std_precomp",
        study,
        eeg,
        np.empty((0, 0), dtype=object),
        "savetrials",
        "on",
        "interp",
        "on",
        "recompute",
        "on",
        "erp",
        "on",
        "erpparams",
        _cell_row("rmbase", np.array([[-200.0, 0.0]])),
        "spec",
        "off",
        "ersp",
        "off",
        "itc",
        "off",
        nargout=2,
    )
    if native:
        # The script's workspace assignments are GUI setup, not processing.
        for name, value in (
            ("STUDY", study),
            ("ALLEEG", alleeg),
            ("EEG", eeg),
            ("CURRENTSTUDY", 1.0),
            ("CURRENTSET", np.arange(1.0, 19.0)[None, :]),
        ):
            call("assignin", "base", name, value, nargout=0)
    call("eeglab", "redraw", nargout=0)
    study = call(
        "std_makedesign",
        study,
        alleeg,
        1.0,
        "name",
        "FaceRepetition",
        "delfiles",
        "off",
        "defaultdesign",
        "off",
        "variable1",
        "type",
        "values1",
        _cell_row(*FACE_EVENTS),
        "vartype1",
        "categorical",
        "subjselect",
        _cell_row(*(f"sub-{index:03d}" for index in range(2, 20))),
    )
    study, eeg = call("pop_savestudy", study, eeg, "savemode", "resave", nargout=2)
    if native:
        # limo_settings_script reads base STUDY after the new design is saved.
        call("assignin", "base", "STUDY", study, nargout=0)
    study, _, limo_files = call(
        "pop_limo",
        study,
        alleeg,
        "method",
        "WLS",
        "measure",
        "daterp",
        "timelim",
        np.array([[-50.0, 650.0]]),
        "erase",
        "on",
        "splitreg",
        "off",
        "interaction",
        "off",
        nargout=3,
    )
    study_path = Path(study["filepath"])
    assert study_path.is_relative_to(limo_source_directory)
    analysis_path = study_path / "2-ways-ANOVA"
    analysis_path.mkdir()
    _limo_cd(request, monkeypatch, analysis_path)
    # The approved native overlay uses these same returned model paths, keeping
    # the original text-list input forms without guessing old LIMO filenames.
    model_path = Path(limo_files["LIMO"])
    chanlocs = str(model_path.parent / "limo_gp_level_chanlocs.mat")
    model_name = f"{Path(_limo_entries(limo_files['mat'])[0]).parent.name}.txt"
    parameters = tuple(np.arange(start, start + 3, dtype=float)[None, :] for start in (1, 4, 7))
    call(
        "limo_random_select",
        "Repeated Measures ANOVA",
        chanlocs,
        "LIMOfiles",
        _cell_row(str(model_path / f"Beta_files_{model_name}")),
        "analysis_type",
        "Full scalp analysis",
        "parameters",
        _cell_row(*parameters),
        "factor names",
        _cell_row("face", "repetition"),
        "type",
        "Channels",
        "nboot",
        1000.0,
        "tfce",
        0.0,
        "skip design check",
        "yes",
        nargout=0,
    )
    erp_path = analysis_path / "ERPs"
    erp_path.mkdir()
    _limo_cd(request, monkeypatch, erp_path)
    files = str(model_path / f"LIMO_files_{model_name}")
    # The original output spelling 'srambled_faces' is intentional here.
    names = tuple(str(erp_path / name) for name in ("famous_faces", "srambled_faces", "unfamiliar_faces"))
    limo_file = str(analysis_path / "LIMO.mat")
    for estimator, suffix, title in (
        ("Mean", "mean", "Mean Face types at channel 50"),
        ("Weighted mean", "Weighted mean", "Weighted mean Face types at channel 50"),
    ):
        for parameter, name in zip(parameters, names, strict=True):
            call("limo_central_tendency_and_ci", files, parameter, chanlocs, estimator, "Mean", empty, name, nargout=0)
        call(
            plot_function,
            _cell_row(*(f"{name}_Mean_of_{suffix}.mat" for name in names)),
            limo_file,
            "channel",
            50.0,
            nargout=0,
        )
        _limo_graphics(request, "title", title)
    _limo_graphics(request, "figure")
    for index, (name, face) in enumerate(zip(names, ("Famous", "srambled", "unfamiliar"), strict=True), 1):
        _limo_graphics(request, "subplot", 1.0, 3.0, float(index))
        call(
            plot_function,
            _cell_row(f"{name}_Mean_of_mean.mat", f"{name}_Mean_of_Weighted mean.mat"),
            limo_file,
            "channel",
            50.0,
            "figure",
            "hold",
            nargout=0,
        )
        _limo_graphics(request, "title", f"mean and weighed mean {face} Faces", fontsize=12.0)
    _limo_graphics(request, "figure")
    for subject in range(1, 19):
        _limo_graphics(request, "subplot", 3.0, 6.0, float(subject))
        call(
            plot_function,
            _cell_row(
                *(f"{name}_single_subjects_{suffix}.mat" for name in names for suffix in ("Mean", "Weighted mean"))
            ),
            limo_file,
            "variable",
            float(subject),
            "channel",
            50.0,
            "figure",
            "hold",
            nargout=0,
        )
        _limo_graphics(request, "title", f"subject: {subject}", fontsize=12.0)
    _limo_cd(request, monkeypatch, analysis_path)
    _, _, files = call("limo_get_files", empty, empty, empty, str(model_path / f"LIMO_files_{model_name}"), nargout=3)
    contrast = {
        "LIMO_files": files,
        "mat": np.array(
            [
                [1.0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                [0.0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                [0.0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
            ]
        ),
    }
    if native:
        call("assignin", "base", "STUDY", study, nargout=0)
    call("limo_batch", "contrast only", empty, contrast, nargout=0)
    names = []
    for index, face in enumerate(("famous_faces", "scrambled_faces", "unfamiliar_faces"), 1):
        directory = analysis_path / face
        directory.mkdir()
        _limo_cd(request, monkeypatch, directory)
        call(
            "limo_random_select",
            "one sample t-test",
            chanlocs,
            "LIMOfiles",
            _cell_row(str(model_path / f"con_{index}_files_{model_name}")),
            "analysis_type",
            "Full scalp analysis",
            "type",
            "Channels",
            "nboot",
            0.0,
            "tfce",
            0.0,
            nargout=0,
        )
        names.append(str(directory / face))
        call("limo_central_tendency_and_ci", str(directory / "Yr.mat"), "Mean", 50.0, names[-1], nargout=0)
        _limo_cd(request, monkeypatch, analysis_path)
    call(plot_function, _cell_row(*(f"{name}_Mean.mat" for name in names)), limo_file, "channel", 50.0, nargout=0)
    _limo_graphics(request, "title", "Means at channel 50")
    for face, output in (("famous", "diff_to_famous"), ("unfamiliar", "diff_to_unfamiliar")):
        call(
            "limo_plot_difference",
            str(analysis_path / f"{face}_faces" / "Yr.mat"),
            str(analysis_path / "scrambled_faces" / "Yr.mat"),
            "LIMO",
            limo_file,
            "type",
            "paired",
            "percent",
            20.0,
            "alpha",
            0.05,
            "fig",
            "on",
            "name",
            str(analysis_path / output),
            nargout=0,
        )
    call(
        plot_function,
        _cell_row(*(str(analysis_path / name) for name in ("diff_to_famous", "diff_to_unfamiliar"))),
        limo_file,
        "channel",
        50.0,
        nargout=0,
    )
    _limo_graphics(request, "title", "Mean differences at channel 50")
    call("limo_eeg", 5.0, limo_file, nargout=0)


@eeglab_test(LIMO_WRAPPER, "limo_test2")
@pytest.mark.slow
@pytest.mark.gui
def test_reference_limo_integration(eeglab_backend, limo_source_directory, eeglab_suite_root, request, monkeypatch):
    """Original 18-subject, nine-section workflow with approved path/workspace setup."""
    prepare_limo_integration_source_overlay(eeglab_suite_root, limo_source_directory.parent)
    call = eeglab_backend
    native = request.config.getoption("--eeglab-backend") == "matlab"
    if native:
        call("eegprep_test_base_workspace", "snapshot", nargout=0)
        request.addfinalizer(lambda: call("eegprep_test_base_workspace", "restore", nargout=0))
    empty = np.empty((0, 0))
    rng = np.random.default_rng()
    statuses = []
    call("eeglab", nargout=4)
    root = limo_source_directory / "derivatives_integration"
    study, alleeg = call(
        "pop_importbids",
        str(limo_source_directory),
        "bidsevent",
        "on",
        "bidschanloc",
        "on",
        "studyName",
        "Face_detection",
        "outputdir",
        str(root),
        "eventtype",
        "trial_type",
        nargout=2,
    )
    alleeg = call("pop_select", alleeg, "nochannel", _cell_row("EEG061", "EEG062", "EEG063", "EEG064"))
    # Cleaning, rereferencing and ICA are commented out in this source workflow.
    eeg = call("pop_epoch", alleeg, _cell_row(*FACE_EVENTS), np.array([[-0.5, 1.0]]), "epochinfo", "yes")
    eeg = call("eeg_checkset", eeg)
    eeg = call("pop_saveset", eeg, "savemode", "resave")
    alleeg = eeg
    study = call("std_checkset", study, alleeg)
    study, eeg = call(
        "std_precomp",
        study,
        eeg,
        np.empty((0, 0), dtype=object),
        "savetrials",
        "on",
        "interp",
        "on",
        "recompute",
        "on",
        "erp",
        "on",
        "erpparams",
        _cell_row("rmbase", np.array([[-200.0, 0.0]])),
        "spec",
        "off",
        "ersp",
        "off",
        "itc",
        "off",
        nargout=2,
    )
    if request.config.getoption("--eeglab-backend") == "matlab":
        for name, value in (
            ("STUDY", study),
            ("ALLEEG", alleeg),
            ("EEG", eeg),
            ("CURRENTSTUDY", 1.0),
            ("CURRENTSET", np.arange(1.0, 19.0)[None, :]),
        ):
            call("assignin", "base", name, value, nargout=0)
    call("eeglab", "redraw", nargout=0)
    _limo_assign_groups(study)
    study_file = root / "Face_detection.study"
    assert study_file.is_file(), "study file nout found"
    _limo_cd(request, monkeypatch, root)
    eeg = call("eeglab")
    study, alleeg = call("pop_loadstudy", "filename", study_file.name, "filepath", str(root), nargout=2)
    _limo_assign_groups(study)

    subjects = _cell_row(*(f"sub-{index:03d}" for index in range(2, 20)))
    with _limo_status(statuses, "categorical design + contrasts with OLS estimates"):
        study = call(
            "std_makedesign",
            study,
            alleeg,
            1.0,
            "name",
            "FaceRepetition",
            "delfiles",
            "off",
            "defaultdesign",
            "off",
            "variable1",
            "type",
            "values1",
            _cell_row(*FACE_EVENTS),
            "vartype1",
            "categorical",
            "subjselect",
            subjects,
        )
        study, eeg = call("pop_savestudy", study, eeg, "savemode", "resave", nargout=2)
        # The copied source tree has no previous models to clean up.
        study, _, model1 = call(
            "pop_limo",
            study,
            alleeg,
            "method",
            "OLS",
            "measure",
            "daterp",
            "timelim",
            np.array([[-50.0, 650.0]]),
            "erase",
            "on",
            "splitreg",
            "off",
            "interaction",
            "off",
            nargout=3,
        )
        contrast = {
            "LIMO_files": model1["mat"],
            "mat": np.array([[1.0, 1, 1, -1, -1, -1, 0, 0, 0, 0], [0.0, 0, 0, 1, 1, 1, -1, -1, -1, 0]]),
        }
        confiles = call("limo_batch", "contrast only", empty, contrast, study)
        model1["con"] = confiles["con"]

    with _limo_status(statuses, "mixed design with WLS estimates + contrast"):
        study = call(
            "std_makedesign",
            study,
            alleeg,
            2.0,
            "name",
            "Face_time",
            "delfiles",
            "off",
            "defaultdesign",
            "off",
            "variable1",
            "face_type",
            "values1",
            _cell_row("famous", "scrambled", "unfamiliar"),
            "vartype1",
            "categorical",
            "variable2",
            "time_dist",
            "values2",
            empty,
            "vartype2",
            "continuous",
            "subjselect",
            subjects,
        )
        study, eeg = call("pop_savestudy", study, eeg, "savemode", "resave", nargout=2)
        study, _, model2 = call(
            "pop_limo",
            study,
            alleeg,
            "method",
            "WLS",
            "measure",
            "daterp",
            "timelim",
            np.array([[-50.0, 650.0]]),
            "erase",
            "on",
            "splitreg",
            "on",
            "interaction",
            "off",
            nargout=3,
        )
        contrast = {"LIMO_files": model2["mat"], "mat": np.array([[0.0, 0, 0, -1, 0, 1]])}
        if native:
            # Approved setup: native discovery reads base, not caller-local STUDY.
            # Keep the original three-argument call and restore state at test exit.
            call("assignin", "base", "STUDY", study, nargout=0)
        # The second call intentionally omits STUDY, as the source tests discovery.
        confiles = call("limo_batch", "contrast only", empty, contrast)
        model2["con"] = confiles["con"]

    def model_list(prefix, model):
        # The approved overlay resolves the generated GLM name and list root
        # from pop_limo's outputs, keeping the original text-file inputs.
        name = Path(_limo_entries(model["mat"])[0]).parent.name
        return str(Path(model["LIMO"]) / f"{prefix}_{name}.txt")

    def grouped_contrast_list(contrast, group):
        first_subject = _limo_entries(_limo_entries(model1["con"])[0])
        basename = Path(first_subject[contrast - 1]).stem
        return model_list(f"{basename}_files_Gp-{group}", model1)

    second_level_root = root / "2nd_level_tests"
    second_level_root.mkdir()
    _limo_cd(request, monkeypatch, second_level_root)
    channel_vector = call("limo_best_electrodes", model_list("LIMO_files", model2))
    channel_file = second_level_root / "virtual_electrode.mat"
    savemat(channel_file, {"channel_vector": channel_vector})

    def second_level(directory, analysis, *options, nargout=1):
        destination = second_level_root / directory
        destination.mkdir()
        _limo_cd(request, monkeypatch, destination)
        call(
            "limo_random_select",
            analysis,
            study["limo"]["chanloc"],
            *options,
            "nboot",
            101.0,
            "tfce",
            1.0,
            nargout=nargout,
        )

    with _limo_status(statuses, "one sample t-tests"):
        second_level(
            "one_sample",
            "one sample t-test",
            "LIMOfiles",
            model2["con"],
            "analysis_type",
            "Full scalp analysis",
            "type",
            "Channels",
        )
        second_level(
            "one_sample50",
            "one sample t-test",
            "LIMOfiles",
            model_list("con_1_files", model2),
            "analysis_type",
            "1 channel/component only",
            "Channel",
            50.0,
            "type",
            "Channels",
        )
        second_level(
            "one_sampleOPT",
            "one sample t-test",
            "LIMOfiles",
            model2["Beta"],
            "analysis_type",
            "1 channel/component only",
            "Channel",
            channel_vector,
            "type",
            "Channels",
            "parameter",
            _cell_row(np.array([[1.0, 3, 7]])),
        )

    with _limo_status(statuses, "regressions"):
        count = len(_limo_entries(model2["con"]))
        second_level(
            "regression",
            "regression",
            "LIMOfiles",
            model2["con"],
            "regressor_file",
            rng.integers(1, count + 1, (count, 2)).astype(float),
            "analysis_type",
            "Full scalp analysis",
            "type",
            "Channels",
            "zscore",
            "yes",
            "skip design check",
            "yes",
        )
        regressor_file = second_level_root / "regression50" / "reg.mat"
        # Save the same named variable as the source before the file-input call.
        regressor_file.parent.mkdir()
        _limo_cd(request, monkeypatch, regressor_file.parent)
        savemat(regressor_file, {"randomreg": rng.standard_normal((count, 1))})
        call(
            "limo_random_select",
            "regression",
            study["limo"]["chanloc"],
            "regressor_file",
            str(regressor_file),
            "LIMOfiles",
            model_list("con_1_files", model2),
            "analysis_type",
            "1 channel/component only",
            "Channel",
            50.0,
            "type",
            "Channels",
            "zscore",
            "yes",
            "skip design check",
            "yes",
            "nboot",
            101.0,
            "tfce",
            1.0,
        )
        second_level(
            "regressionOPT",
            "regression",
            "LIMOfiles",
            model_list("Beta_files", model2),
            "parameter",
            3.0,
            "regressor_file",
            rng.integers(1, count + 1, (count, 2)).astype(float),
            "analysis_type",
            "1 channel/component only",
            "type",
            "Channels",
            "Channel",
            str(channel_file),
            "zscore",
            "yes",
            "skip design check",
            "yes",
        )

    with _limo_status(statuses, "paired t-test"):
        data = np.empty((2, len(_limo_entries(study["subject"]))), dtype=object)
        for subject, contrasts in enumerate(_limo_entries(model1["con"])):
            for index in range(2):
                # MATLAB's con{N}(index) retains a singleton inner cell.
                data[index, subject] = _cell_row(_limo_entries(contrasts)[index])
        second_level(
            "paired_t-test",
            "paired t-test",
            "LIMOfiles",
            data,
            "analysis_type",
            "Full scalp analysis",
            "type",
            "Channels",
        )
        datafiles = _cell_row(model_list("con_1_files", model1), model_list("con_2_files", model1))
        second_level(
            "paired_t-test50",
            "paired t-test",
            "LIMOfiles",
            datafiles,
            "analysis_type",
            "1 channel/component only",
            "Channel",
            50.0,
            "type",
            "Channels",
        )
        second_level(
            "paired_t-testOPT",
            "paired t-test",
            "LIMOfiles",
            model_list("Beta_files", model1),
            "analysis_type",
            "1 channel/component only",
            "Channel",
            channel_vector,
            "type",
            "Channels",
            "parameter",
            np.array([[1.0, 4.0]]),
        )

    with _limo_status(statuses, "two samples t-test"):
        second_level(
            "two-samples_t-test",
            "two-samples t-test",
            "LIMOfiles",
            data,
            "analysis_type",
            "Full scalp analysis",
            "type",
            "Channels",
        )
        second_level(
            "two-samples_t-test50",
            "two-samples t-test",
            "LIMOfiles",
            datafiles,
            "analysis_type",
            "1 channel/component only",
            "Channel",
            50.0,
            "type",
            "Channels",
        )
        beta_files = _cell_row(model_list("Beta_files", model1), model_list("Beta_files", model2))
        second_level(
            "two-samples_t-testOPT",
            "two-samples t-test",
            "LIMOfiles",
            beta_files,
            "analysis_type",
            "1 channel/component only",
            "Channel",
            np.tile(channel_vector, (2, 1)),
            "type",
            "Channels",
            "parameter",
            np.array([[1.0, 4.0]]),
        )

    with _limo_status(statuses, "1-way ANOVA"):
        data = np.empty((3, 7), dtype=object)
        data.fill(empty)
        for group in range(1, 4):
            indices = [
                index for index, info in enumerate(_limo_entries(study["datasetinfo"])) if str(group) in info["group"]
            ]
            for subject, index in enumerate(indices):
                data[group - 1, subject] = _cell_row(_limo_entries(_limo_entries(model1["con"])[index])[0])
        second_level(
            "N-Ways ANOVA",
            "N-Ways ANOVA",
            "LIMOfiles",
            data.T,
            "analysis_type",
            "Full scalp analysis",
            "type",
            "Channels",
            "skip design check",
            "yes",
        )
        datafiles = _cell_row(*(grouped_contrast_list(1, group) for group in range(1, 4)))
        second_level(
            "N-Ways ANOVA50",
            "N-Ways ANOVA",
            "LIMOfiles",
            datafiles,
            "analysis_type",
            "1 channel/component only",
            "Channel",
            50.0,
            "type",
            "Channels",
            "skip design check",
            "yes",
        )
        beta_files = _cell_row(*(model_list(f"Beta_files_Gp-{group}", model1) for group in range(1, 4)))
        second_level(
            "N-Ways ANOVAOPT",
            "N-Ways ANOVA",
            "LIMOfiles",
            beta_files,
            "analysis_type",
            "1 channel/component only",
            "Channel",
            str(channel_file),
            "type",
            "Channels",
            "parameter",
            _cell_row(np.ones((3, 1))),
            "skip design check",
            "yes",
        )

    with _limo_status(statuses, "ANCOVA + contrast"):
        data = np.empty((3, 7), dtype=object)
        data.fill(empty)
        for group in range(1, 4):
            indices = [
                index for index, info in enumerate(_limo_entries(study["datasetinfo"])) if str(group) in info["group"]
            ]
            for subject, index in enumerate(indices):
                data[group - 1, subject] = _cell_row(_limo_entries(_limo_entries(model1["con"])[index])[0])
        # Unlike ANOVA, this deliberately passes the wrong orientation for LIMO to fix.
        second_level(
            "ANCOVA",
            "ANCOVA",
            "LIMOfiles",
            data,
            "analysis_type",
            "Full scalp analysis",
            "type",
            "Channels",
            "regressor_file",
            rng.standard_normal((18, 2)),
            "skip design check",
            "yes",
        )
        for mode, beta in ((1.0, "Betas.mat"), (2.0, "H0/Betas_desc-H0.mat")):
            directory = second_level_root / "ANCOVA"
            call(
                "limo_contrast",
                str(directory / "Yr.mat"),
                str(directory / beta),
                str(directory / "LIMO.mat"),
                "T",
                mode,
                np.array([[0.0, 0, 0, 1, -1, 0]]),
                nargout=0,
            )
        second_level(
            "ANCOVA50",
            "ANCOVA",
            "LIMOfiles",
            datafiles[:, :2],
            "analysis_type",
            "1 channel/component only",
            "Channel",
            50.0,
            "type",
            "Channels",
            "regressor_file",
            rng.standard_normal((13, 2)),
            "skip design check",
            "yes",
        )
        for mode, beta in ((1.0, "Betas.mat"), (2.0, "H0/Betas_desc-H0.mat")):
            directory = second_level_root / "ANCOVA50"
            call(
                "limo_contrast",
                str(directory / "Yr.mat"),
                str(directory / beta),
                str(directory / "LIMO.mat"),
                "T",
                mode,
                np.array([[0.0, 0, 1, -1, 0]]),
                nargout=0,
            )
        second_level(
            "ANCOVAOPT",
            "ANCOVA",
            "LIMOfiles",
            beta_files,
            "analysis_type",
            "1 channel/component only",
            "Channel",
            np.tile(channel_vector, (2, 1)),
            "regressor_file",
            rng.standard_normal((18, 2)),
            "type",
            "Channels",
            "parameter",
            np.array([[1.0, 4.0, 1.0]]),
            "skip design check",
            "yes",
        )

    with _limo_status(statuses, "Repeated measures ANOVA + contrast"):
        second_level(
            "Rep-ANOVA",
            "Repeated Measures ANOVA",
            "LIMOfiles",
            _cell_row(model_list("Beta_files", model1)),
            "analysis_type",
            "Full scalp analysis",
            "parameters",
            _cell_row(*(np.arange(start, start + 3, dtype=float)[None, :] for start in (1, 4, 7))),
            "factor names",
            _cell_row("face", "repetition"),
            "type",
            "Channels",
            "skip design check",
            "yes",
            nargout=0,
        )
        for mode in (3.0, 4.0):
            directory = second_level_root / "Rep-ANOVA"
            call(
                "limo_contrast",
                str(directory / "Yr.mat"),
                str(directory / "LIMO.mat"),
                mode,
                np.array([[1.0, 1, 1, -2, -2, -2, 1, 1, 1]]),
                nargout=0,
            )
        second_level(
            "GpRep-ANOVA",
            "Repeated Measures ANOVA",
            "LIMOfiles",
            _cell_row(*(model_list(f"Beta_files_Gp-{group}", model2) for group in range(1, 4))).T,
            "analysis_type",
            "Full scalp analysis",
            "parameters",
            _cell_row(np.array([[1.0, 2.0, 3.0]])),
            "factor names",
            _cell_row("face"),
            "type",
            "Channels",
            "skip design check",
            "yes",
            nargout=0,
        )
        for mode in (3.0, 4.0):
            directory = second_level_root / "GpRep-ANOVA"
            call(
                "limo_contrast",
                str(directory / "Yr.mat"),
                str(directory / "LIMO.mat"),
                mode,
                np.array([[1.0, -2, 1]]),
                nargout=0,
            )
        datafiles = _cell_row(
            model_list("con_1_files", model1),
            model_list("con_1_files", model2),
            model_list("con_2_files", model1),
        )
        second_level(
            "Rep-ANOVA50",
            "Repeated Measures ANOVA",
            "LIMOfiles",
            datafiles,
            "analysis_type",
            "1 channel/component only",
            "Channel",
            50.0,
            "factor names",
            _cell_row("face"),
            "parameters",
            _cell_row(np.ones((1, 3))),
            "type",
            "Channels",
            "skip design check",
            "yes",
            nargout=0,
        )
        datafiles = np.empty((3, 2), dtype=object)
        for group in range(1, 4):
            for contrast in range(1, 3):
                datafiles[group - 1, contrast - 1] = grouped_contrast_list(contrast, group)
        second_level(
            "GpRep-ANOVAOPT",
            "Repeated Measures ANOVA",
            "LIMOfiles",
            datafiles,
            "analysis_type",
            "1 channel/component only",
            "Channel",
            channel_vector,
            "factor names",
            _cell_row("face"),
            "parameters",
            _cell_row(*(np.ones((1, 2)) for _ in range(3))).T,
            "type",
            "Channels",
            "skip design check",
            "yes",
            nargout=0,
        )
    _limo_cd(request, monkeypatch, root)
    assert all(success for success, _ in statuses), "\n".join(message for _, message in statuses)


def _limo_eeg(subject_index: int) -> dict:
    rng = np.random.default_rng(900 + subject_index)
    trials = 24
    times = np.linspace(-100.0, 300.0, 9)
    waveform = np.exp(-(((times - 150.0) / 90.0) ** 2))
    faces = np.asarray(["famous"] * (trials // 2) + ["scrambled"] * (trials // 2))
    reaction_time = np.linspace(0.25, 0.75, trials)
    data = np.empty((2, times.size, trials), dtype=float)
    for trial, face in enumerate(faces):
        face_effect = 0.75 if face == "famous" else -0.75
        for channel in range(2):
            data[channel, :, trial] = (
                0.2 * (channel + 1)
                + face_effect * waveform
                + 0.35 * reaction_time[trial]
                + 0.02 * rng.standard_normal(times.size)
            )
    data[:, :, -1] += 3.0
    events = [
        {
            "type": str(face),
            "face": str(face),
            "rt": float(reaction_time[trial]),
            "epoch": trial + 1,
            "latency": trial * times.size + 3,
            "urevent": trial + 1,
        }
        for trial, face in enumerate(faces)
    ]
    return {
        "setname": f"subject_{subject_index:02d}",
        "subject": f"S{subject_index:02d}",
        "condition": "face task",
        "group": "control" if subject_index <= 3 else "patient",
        "data": data,
        "nbchan": 2,
        "pnts": times.size,
        "trials": trials,
        "srate": 20.0,
        "xmin": -0.1,
        "xmax": 0.3,
        "times": times,
        "chanlocs": [{"labels": "Fz"}, {"labels": "Cz"}],
        "event": events,
        "urevent": [{key: value for key, value in event.items() if key != "urevent"} for event in events],
        "epoch": [{"event": [trial + 1], "eventtype": [str(faces[trial])]} for trial in range(trials)],
        "etc": {},
    }


def test_limo_preprocessing_statistics_workflow_fits_wls_and_repeated_measures(tmp_path: Path):
    """Additional generated Python regression, not the original LIMO workflow."""
    datasets = [_limo_eeg(index) for index in range(1, 7)]
    study, alleeg = pop_study(None, datasets, name="Generated face study")
    study, _trialinfo = std_maketrialinfo(study, alleeg)
    study = std_makedesign(
        study,
        alleeg,
        1,
        name="FaceRepetition",
        variable1="face",
        values1=["famous", "scrambled"],
        vartype1="categorical",
        subjselect=[f"S{index:02d}" for index in range(1, 7)],
    )

    study, returned, model_files, command = pop_limo(
        study,
        alleeg,
        method="WLS",
        measure="daterp",
        timelim=[-50, 250],
        outputdir=tmp_path / "models",
        return_com=True,
    )

    assert len(returned) == len(alleeg)
    assert all(left is right for left, right in zip(returned, alleeg))
    assert study["limo"]["method"] == "WLS"
    assert len(model_files["files"]) == 6
    assert command.startswith("STUDY, ALLEEG, model_files = pop_limo(")
    assert all(Path(file).is_file() for file in model_files["files"])
    first = std_readfilelimo(model_files["files"][0])
    np.testing.assert_allclose(first["betas"], model_files["models"][0]["betas"])
    assert first["parameter_names"] == ["face=famous", "face=scrambled", "constant"]
    assert first["weights"][-1] < np.median(first["weights"])

    conditions = std_limoresults(
        model_files,
        "contrast",
        contrast=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    )
    repeated = std_limoresults(conditions["estimates"], "Repeated Measures ANOVA")
    assert repeated["condition_means"].shape == (2, 2, 7)
    assert np.nanmedian(repeated["f"]) > 100.0
    assert np.nanmedian(repeated["p"]) < 1e-6

    differences = conditions["estimates"][:, 0] - conditions["estimates"][:, 1]
    variances = np.full_like(differences, 0.04)
    summary = std_limoresults(
        differences,
        "central tendency",
        estimator="weighted mean",
        variances=variances,
    )
    np.testing.assert_allclose(summary["estimate"], np.mean(differences, axis=0), atol=1e-12)
    assert np.all(summary["ci"][0] <= summary["estimate"])
    assert np.all(summary["ci"][1] >= summary["estimate"])

    updated, contrast_result, result_command = pop_limoresults(
        study,
        model_files,
        analysis="contrast",
        contrast=[1.0, -1.0, 0.0],
        return_com=True,
    )
    assert updated["limo"]["results"][-1]["analysis"] == "contrast"
    assert contrast_result["estimates"].shape == (6, 1, 2, 7)
    assert result_command.startswith("STUDY, result = pop_limoresults(")
    assert eegprep.pop_limo is pop_limo
    assert eegprep.std_limoresults is std_limoresults

    mixed_study = std_makedesign(
        study,
        returned,
        2,
        name="Face_time",
        variable1="face",
        values1=["famous", "scrambled"],
        variable2="rt",
        vartype2="continuous",
    )
    _mixed_study, _mixed_eeg, mixed_files = pop_limo(mixed_study, returned, method="OLS", splitreg="on")
    assert mixed_files["models"][0]["parameter_names"] == [
        "face=famous",
        "face=scrambled",
        "rt|face=famous",
        "rt|face=scrambled",
        "constant",
    ]


def test_limo_integration_covers_first_level_contrasts_and_core_group_models(tmp_path: Path):
    """Additional generated Python regression, not the original integration script."""
    rng = np.random.default_rng(44)
    trials = 30
    condition = np.tile([0.0, 1.0], trials // 2)
    covariate = np.linspace(-1.0, 1.0, trials)
    design = np.column_stack((condition, covariate, np.ones(trials)))
    true_beta = np.asarray(
        [
            [[1.2, 0.5, -0.3], [0.8, -0.25, 0.1], [1.5, 0.2, 0.0]],
            [[-0.4, 0.75, 0.5], [0.6, 0.1, -0.2], [0.2, -0.5, 0.9]],
        ]
    )
    response = np.einsum("tp,cfp->cft", design, true_beta)
    response += 0.01 * rng.standard_normal(response.shape)

    ols = std_limo(response, design, method="OLS", parameter_names=["condition", "rt", "constant"])
    expected = np.linalg.pinv(design) @ np.moveaxis(response, -1, 0).reshape(trials, -1)
    expected = np.moveaxis(expected.reshape((3, 2, 3)), 0, -1)
    np.testing.assert_allclose(ols["betas"], expected, rtol=1e-12, atol=1e-12)
    assert np.min(ols["r2"]) > 0.998
    assert np.all((ols["p"] >= 0.0) & (ols["p"] <= 1.0))

    # Golden output from limo_WLS at LIMO Toolbox bff6d166c5338f05d7ed9b37c1b7615d667492af.
    limo_x = np.column_stack((np.ones(12), np.linspace(-1.0, 1.0, 12)))
    limo_beta = np.asarray([[1.0, 2.0, 3.0, 4.0], [0.5, -1.0, 1.5, -0.25]])
    limo_y = limo_x @ limo_beta
    limo_y[-1] += [5.0, 4.0, 6.0, 3.0]
    wls = std_limo(limo_y.T, limo_x, method="WLS")
    np.testing.assert_allclose(
        wls["betas"].T,
        [
            [1.00173322768859, 2.00138658215087, 3.00207987322631, 4.00103993661315],
            [0.504984652334702, -0.996012278132239, 1.50598158280164, -0.247009208599179],
        ],
        rtol=2e-12,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        wls["weights"],
        [
            0.620542368031203,
            0.846061602506803,
            0.972393649026113,
            0.99996019537856,
            1.0,
            1.0,
            0.999977822653864,
            0.975127729877988,
            0.860292565590187,
            0.657526761562131,
            0.404658627171921,
            0.04,
        ],
        rtol=2e-12,
        atol=2e-12,
    )
    assert wls["weight_reduction"] == 3
    limo_irls = std_limo(limo_y.T, limo_x, method="IRLS")
    np.testing.assert_allclose(limo_irls["betas"].T, limo_beta, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(limo_irls["weights"][:, -1], 0.0, atol=1e-15)

    contaminated = response.copy()
    contaminated[:, :, -1] += 30.0
    contaminated_ols = std_limo(contaminated, design, method="OLS")
    irls = std_limo(contaminated, design, method="IRLS")
    ols_error = np.linalg.norm(contaminated_ols["betas"] - true_beta)
    irls_error = np.linalg.norm(irls["betas"] - true_beta)
    assert irls_error < ols_error * 0.2
    assert np.max(irls["weights"][..., -1]) < 0.1

    file = tmp_path / "first_level.npz"
    saved = save_limo_result({**ols, "dataset_index": 1, "subject": "S01"}, file)
    loaded = std_limoresults(saved, "load")
    np.testing.assert_allclose(loaded["residuals"], ols["residuals"])
    with pytest.raises(NotImplementedError, match="MATLAB LIMO .mat"):
        std_readfilelimo(tmp_path / "LIMO.mat")
    with pytest.raises(NotImplementedError, match="bootstrap and TFCE"):
        std_limoresults(np.ones((6, 2)), nboot=101)

    subjects = 12
    x = np.linspace(-1.0, 1.0, subjects)
    noise = rng.normal(scale=0.05, size=(subjects, 2, 3))
    group_values = 0.4 + 1.7 * x[:, None, None] + noise
    one_sample = std_limoresults(group_values, "one sample t-test", outputfile=tmp_path / "one_sample.npz")
    scipy_one = stats.ttest_1samp(group_values, 0.0, axis=0)
    np.testing.assert_allclose(one_sample["t"], scipy_one.statistic)
    np.testing.assert_allclose(one_sample["p"], scipy_one.pvalue)
    reloaded_one_sample = std_readfilelimo(one_sample["file"])
    np.testing.assert_allclose(reloaded_one_sample["t"], one_sample["t"])

    paired_other = group_values - (0.25 + 0.01 * x[:, None, None])
    paired = std_limoresults(group_values, "paired t-test", data2=paired_other)
    np.testing.assert_allclose(paired["difference"], 0.25, atol=1e-12)
    first_group, second_group = group_values[:6], group_values[6:]
    independent = std_limoresults(first_group, "two samples t-test", data2=second_group)
    scipy_two = stats.ttest_ind(first_group, second_group, axis=0, equal_var=False)
    np.testing.assert_allclose(independent["t"], scipy_two.statistic)
    np.testing.assert_allclose(independent["p"], scipy_two.pvalue)

    regression = std_limoresults(group_values, "regression", regressors=x)
    np.testing.assert_allclose(regression["betas"][..., 0], 1.7, atol=0.1)
    assert np.min(regression["r2"]) > 0.98
    anova = std_limoresults([first_group, second_group], "N-Ways ANOVA")
    scipy_anova = stats.f_oneway(first_group, second_group, axis=0)
    np.testing.assert_allclose(anova["f"], scipy_anova.statistic)

    labels = np.asarray(["control"] * 6 + ["patient"] * 6)
    ancova_data = 1.2 * x[:, None] + (labels == "patient")[:, None] * 0.8 + rng.normal(0.0, 0.02, (subjects, 2))
    ancova = std_limoresults(ancova_data, "ANCOVA", groups=labels, covariates=x)
    assert np.min(ancova["f"]) > 100.0
    assert np.max(ancova["p"]) < 1e-6
