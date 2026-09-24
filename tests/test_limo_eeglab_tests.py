"""Original LIMO workflows and separately retained generated Python checks."""

from __future__ import annotations

from pathlib import Path
import shutil

import matplotlib
import numpy as np
import pytest
from scipy import stats

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


LIMO_WRAPPER = "unittesting_limo/limo_wrapperTest.m"
FACE_EVENTS = tuple(
    f"{face}_{repetition}"
    for face in ("famous", "scrambled", "unfamiliar")
    for repetition in ("new", "second_early", "second_late")
)


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


@eeglab_test(LIMO_WRAPPER, "limo_test1")
def test_reference_limo_preprocessing_and_statistics(eeglab_backend, limo_source_directory, request, monkeypatch):
    """Full original 18-subject pipeline; completion is its smoke oracle."""
    call = eeglab_backend
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
    if request.config.getoption("--eeglab-backend") == "matlab":
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
    study = call(
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
    )
    study_path = Path(study["filepath"])
    assert study_path.is_relative_to(limo_source_directory)
    analysis_path = study_path / "2-ways-ANOVA"
    analysis_path.mkdir()
    _limo_cd(request, monkeypatch, analysis_path)
    chanlocs = str(study_path / "limo_gp_level_chanlocs.mat")
    model_path = study_path / f"LIMO_{study['filename'][:-6]}"
    model_name = "FaceRepetition_GLM_Channels_Time_WLS.txt"
    parameters = tuple(np.arange(start, start + 3, dtype=float)[None, :] for start in (1, 4, 7))
    call(
        "limo_random_select",
        "Repeated Measures ANOVA",
        chanlocs,
        "LIMOfiles",
        _cell_row(str(study_path / "LIMO_Face_detection" / f"Beta_files_{model_name}")),
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
            "limo_add_plots",
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
            "limo_add_plots",
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
            "limo_add_plots",
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
    call("limo_add_plots", _cell_row(*(f"{name}_Mean.mat" for name in names)), limo_file, "channel", 50.0, nargout=0)
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
        "limo_add_plots",
        _cell_row(*(str(analysis_path / name) for name in ("diff_to_famous", "diff_to_unfamiliar"))),
        limo_file,
        "channel",
        50.0,
        nargout=0,
    )
    _limo_graphics(request, "title", "Mean differences at channel 50")
    call("limo_eeg", 5.0, limo_file, nargout=0)


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
