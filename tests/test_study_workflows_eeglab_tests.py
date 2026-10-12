"""Original STUDY workflows and additional generated-fixture Python tests."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from scipy.cluster import vq

import eegprep
from eegprep.functions.studyfunc.pop_corrmap import pop_corrmap
from eegprep.functions.studyfunc.pop_study import pop_study
from eegprep.functions.studyfunc.corrmap import corrmap
from eegprep.functions.studyfunc.std_makedesign import std_makedesign
from eegprep.functions.studyfunc.std_preclust import std_preclust
from eegprep.functions.studyfunc.std_selectdesign import std_selectdesign
from tests.eeglab_tests import eeglab_test
from tests.eeglab_tests.gui import close_reference_gui


STUDYFUNC_ROOT = "unittesting_studyfunc"


def _reference(wrapper: str, test: str):
    source = f"{STUDYFUNC_ROOT}/{wrapper}/studyfunc_{wrapper}_wrapperTest.m"
    return eeglab_test(source, test)


def _deterministic_eeg(
    setname: str,
    subject: str,
    condition: str,
    *,
    offset: float = 0.0,
    n_channels: int = 4,
    n_components: int = 3,
) -> dict:
    srate = 64.0
    pnts = 128
    trials = 4
    seconds = np.arange(pnts, dtype=float) / srate
    data = np.empty((n_channels, pnts, trials), dtype=float)
    activations = np.empty((n_components, pnts, trials), dtype=float)
    for trial in range(trials):
        phase = trial * np.pi / 8
        for channel in range(n_channels):
            frequency = 6.0 + 2.0 * channel
            data[channel, :, trial] = np.sin(2 * np.pi * frequency * seconds + phase) + offset
        for component in range(n_components):
            frequency = 8.0 + 2.0 * component
            activations[component, :, trial] = np.sin(2 * np.pi * frequency * seconds + phase) + offset / 2
    mixing = np.zeros((n_channels, n_components), dtype=float)
    mixing[:n_components, :] = np.eye(n_components)
    weights = np.zeros((n_components, n_channels), dtype=float)
    weights[:, :n_components] = np.eye(n_components)
    chanlocs = []
    for channel in range(n_channels):
        angle = 2 * np.pi * channel / n_channels
        chanlocs.append(
            {
                "labels": f"Ch{channel + 1}",
                "theta": float(np.degrees(angle)),
                "radius": 0.35,
                "X": float(np.cos(angle)),
                "Y": float(np.sin(angle)),
                "Z": 0.0,
            }
        )
    dipoles = [
        {
            "posxyz": [float(component + 1), float((-1) ** component), float(component) / 2],
            "momxyz": [1.0, 0.0, 0.5],
            "rv": 0.05,
        }
        for component in range(n_components)
    ]
    return {
        "setname": setname,
        "subject": subject,
        "condition": condition,
        "group": "control",
        "session": 1,
        "run": 1,
        "data": data,
        "nbchan": n_channels,
        "pnts": pnts,
        "trials": trials,
        "srate": srate,
        "xmin": 0.0,
        "xmax": float(seconds[-1]),
        "times": seconds * 1000.0,
        "chanlocs": chanlocs,
        "icaact": activations,
        "icawinv": mixing,
        "icaweights": weights,
        "icasphere": np.eye(n_channels),
        "icachansind": list(range(n_channels)),
        "event": [
            {"type": "stim", "latency": trial * pnts + 1, "epoch": trial + 1, "urevent": trial + 1}
            for trial in range(trials)
        ],
        "urevent": [{"type": "stim", "latency": trial * pnts + 1, "epoch": trial + 1} for trial in range(trials)],
        "epoch": [{"event": [trial], "eventtype": ["stim"]} for trial in range(trials)],
        "dipfit": {"model": dipoles},
        "etc": {},
    }


def _study_pair(*, n_channels: int = 4, n_components: int = 3) -> tuple[dict, list[dict]]:
    datasets = [
        _deterministic_eeg("s01_target", "S01", "target", n_channels=n_channels, n_components=n_components),
        _deterministic_eeg(
            "s02_standard",
            "S02",
            "standard",
            offset=0.2,
            n_channels=n_channels,
            n_components=n_components,
        ),
    ]
    return pop_study(None, datasets, name="Generated N400 study")


def _corrmap_study(*, scales: tuple[float, float] = (4.0, 0.2)) -> tuple[dict, list[dict]]:
    template = np.array([-2.0, -1.0, -0.2, 0.5, 1.2, 2.0])
    alternating = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    biphasic = np.array([1.0, 0.0, -1.0, 1.0, 0.0, -1.0])
    unrelated = np.array([-0.5, 0.5, 1.0, -1.0, 0.5, -0.5])
    near_template = template + np.array([0.1, -0.05, 0.03, -0.04, 0.02, -0.08])
    inverse_maps = (
        np.column_stack([template, alternating, biphasic]),
        np.column_stack([alternating, -scales[0] * template, unrelated]),
        np.column_stack([biphasic, unrelated, scales[1] * near_template]),
        np.column_stack([alternating, biphasic, unrelated]),
    )
    datasets = []
    for dataset_index, maps in enumerate(inverse_maps, start=1):
        eeg = _deterministic_eeg(
            f"corrmap_{dataset_index}",
            f"S{dataset_index:02d}",
            "target",
            n_channels=6,
            n_components=3,
        )
        eeg["icawinv"] = maps
        eeg["icaweights"] = np.linalg.pinv(maps)
        eeg["icasphere"] = np.eye(6)
        datasets.append(eeg)
    return pop_study(None, datasets, name="Generated CORRMAP study")


def test_pop_corrmap_matches_polarity_builds_cluster_and_is_scale_invariant():
    study, alleeg = _corrmap_study()

    result, matched_study, matched_datasets, command = pop_corrmap(
        study,
        alleeg,
        1,
        1,
        "chanlocs",
        "",
        "th",
        "auto",
        "ics",
        1,
        "title",
        "Cluster test2",
        "clname",
        "test2",
        "badcomps",
        "yes",
        "resetclusters",
        "off",
        return_com=True,
    )

    second_pairs = dict(zip(result["output"]["sets"][1], result["output"]["ics"][1], strict=True))
    second_polarities = dict(zip(result["output"]["sets"][1], result["output"]["polarity"][1], strict=True))
    # Direct output from CORRMAP 6d1b06e on these maps is first-pass sets
    # [2, 3], components [2, 3], then second-pass sets/components [1, 2, 3].
    assert second_pairs == {1: 1, 2: 2, 3: 3}
    assert second_polarities == {1: 1, 2: -1, 3: 1}
    np.testing.assert_array_equal(result["output"]["sets"][0], [2, 3])
    np.testing.assert_array_equal(result["output"]["ics"][0], [2, 3])
    np.testing.assert_allclose(result["corr"]["abs_values"][0][:3], [1.0, 0.9993692940674627, 0.5046949386828399])
    np.testing.assert_array_equal(result["clust"]["sets"]["absent"][0], [1, 4])
    np.testing.assert_array_equal(result["clust"]["sets"]["absent"][1], [4])
    assert result["clust"]["best_th"] == 0.95
    assert result["clust"]["similarity"] > 0.999

    child = matched_study["cluster"][1]
    assert child["name"] == "test2 1"
    assert child["algorithm"][0] == "correlation (CORRMAP)"
    assert dict(zip(child["sets"][0], child["comps"], strict=True)) == second_pairs
    assert matched_study["cluster"][0]["child"] == ["test2 1"]
    assert [eeg.get("badcomps", []) for eeg in matched_datasets] == [[1], [2], [3], []]
    assert all("badcomps" not in eeg for eeg in alleeg)
    assert command.startswith("CORRMAP, STUDY, ALLEEG = pop_corrmap(")

    scaled_study, scaled_alleeg = _corrmap_study(scales=(400.0, 0.002))
    scaled, _study, _datasets = pop_corrmap(scaled_study, scaled_alleeg, 1, 1, th="auto", ics=1)
    np.testing.assert_allclose(result["output"]["average_plot"], scaled["output"]["average_plot"], atol=1e-12)


def test_corrmap_aligns_labelled_montages_and_rejects_unsupported_inputs():
    study, alleeg = _corrmap_study()
    expected, _study, _datasets = corrmap(study, alleeg, 1, 1, th=0.95, ics=1)
    permutation = np.array([5, 3, 1, 4, 2, 0])
    alleeg[1]["icawinv"] = np.asarray(alleeg[1]["icawinv"])[permutation]
    alleeg[1]["icaweights"] = np.linalg.pinv(alleeg[1]["icawinv"])
    alleeg[1]["chanlocs"] = [alleeg[1]["chanlocs"][index] for index in permutation]
    reordered_study, alleeg = pop_study(None, alleeg, name="Reordered CORRMAP study")

    actual, _study, _datasets = corrmap(reordered_study, alleeg, 1, 1, th=0.95, ics=1)

    np.testing.assert_allclose(actual["corr"]["abs_values"], expected["corr"]["abs_values"], atol=1e-12)
    np.testing.assert_allclose(actual["output"]["average_plot"], expected["output"]["average_plot"], atol=1e-12)
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        corrmap(study, alleeg, 1, 1, th=1.0)
    with pytest.raises(ValueError, match="1, 2, or 3"):
        corrmap(study, alleeg, 1, 1, ics=4)
    with pytest.raises(ValueError, match="template component"):
        corrmap(study, alleeg, 1, 4, th=0.8, ics=1)
    with pytest.raises(NotImplementedError, match="summary plotting"):
        corrmap(study, alleeg, 1, 1, th=0.8, ics=1, plot="on")


def _cell_row(*values):
    cells = np.empty((1, len(values)), dtype=object)
    for index, value in enumerate(values):
        cells[0, index] = value
    return cells


@_reference("pop_corrmap", "test_test_pop_corrmap")
@pytest.mark.gui
def test_reference_pop_corrmap(eeglab_backend, eeglab_sample_study):
    version = eeglab_backend("eeg_getversion")
    prefix = version[:2]
    if prefix.rstrip(".").isdigit() and float(prefix) > 9:
        study, alleeg = eeglab_sample_study
        eeglab_backend(
            "pop_corrmap",
            study,
            alleeg,
            1.0,
            1.0,
            chanlocs="",
            th="auto",
            ics=1.0,
            title="Cluster test2",
            clname="test2",
            badcomps="no",
            resetclusters="off",
            nargout=0,
        )


@_reference("std_editset", "test_test_std_editset")
@pytest.mark.gui
def test_reference_std_editset(eeglab_backend, eeglab_suite_root, request):
    directory = eeglab_suite_root / "unittesting_studyfunc/teststudy2/S02"
    matlab = request.config.getoption("--eeglab-backend") == "matlab"
    window = eeglab_backend("eeglab", nargout=0 if matlab else 1)
    # The pinned checkout is the source project's RootFolder.
    study, alleeg = eeglab_backend(
        "std_editset",
        np.empty((0, 0)),
        np.empty((0, 0)),
        commands=_cell_row(
            _cell_row("index", 1.0, "load", str(directory / "Ignore.set")),
            _cell_row("index", 2.0, "load", str(directory / "Probe.set")),
            _cell_row("index", 1.0, "subject", "S01"),
            _cell_row("index", 2.0, "subject", "S01"),
            _cell_row("index", 1.0, "condition", "ignore"),
            _cell_row("index", 2.0, "condition", "probe"),
        ),
        updatedat="off",
        nargout=2,
    )
    study = eeglab_backend(
        "std_makedesign",
        study,
        alleeg,
        1.0,
        "variable1",
        "condition",
        "variable2",
        "load",
        "name",
        "STUDY.design 1",
        "values1",
        _cell_row("ignore", "probe"),
        "values2",
        _cell_row(3.0, 5.0, 7.0),
        "subjselect",
        _cell_row("S01"),
    )
    eeglab_backend(
        "std_editset",
        study,
        alleeg,
        commands=_cell_row(_cell_row("remove", 2.0)),
        updatedat="off",
        nargout=2,
    )
    close_reference_gui(eeglab_backend, request, window=window)


@_reference("std_makedesign", "test_test_std_makedesign")
def test_reference_std_makedesign(eeglab_backend, eeglab_sample_study):
    study, alleeg = eeglab_sample_study
    subjects = _cell_row("S02", "S07", "S08", "S10")
    for index, name, values in (
        (1.0, "STUDY.design 1", _cell_row("non-synonyms", "synonyms")),
        (2.0, "Design 2 test", _cell_row("non-synonyms", _cell_row("non-synonyms", "synonyms"))),
    ):
        study = eeglab_backend(
            "std_makedesign",
            study,
            alleeg,
            index,
            "name",
            name,
            variable1="condition",
            variable2="",
            values1=values,
            subjselect=subjects,
        )
    designs = study["design"]
    if "cell" in designs.dtype.names:
        assert designs["cell"][0, 0].size == 8
        assert designs["cell"][0, 1].size == 8


def _read_writable_n400(backend, directory):
    study, alleeg = backend("pop_loadstudy", filename="n400clustedit.study", filepath=str(directory), nargout=2)
    study = backend("std_checkset", study, alleeg)
    assert Path(study["filepath"]).is_relative_to(directory)
    assert all(Path(path).is_relative_to(directory) for path in alleeg["filepath"].ravel())
    return study, alleeg


@pytest.mark.slow
@_reference("std_itcplot", "test_test_std_itcplot")
@pytest.mark.gui
def test_reference_std_itcplot(eeglab_backend, eeglab_writable_study, request):
    study, alleeg = _read_writable_n400(eeglab_backend, eeglab_writable_study)
    study = eeglab_backend("std_selectdesign", study, alleeg, 1.0)
    for selection in ("components", "channels"):
        study, alleeg = eeglab_backend(
            "std_precomp",
            study,
            alleeg,
            selection,
            recompute="on",
            interp="on",
            itc="on",
            erspparams=_cell_row(
                "cycles",
                np.array([[3.0, 0.8]]),
                "nfreqs",
                10.0,
                "ntimesout",
                10.0,
                "baseline",
                np.nan,
                "verbose",
                "off",
            ),
            nargout=2,
        )
    for options in (
        {"clusters": 3.0, "mode": "centroid"},
        {"clusters": 3.0, "mode": "comps"},
        {"clusters": 3.0, "comps": 4.0},
        {"channels": _cell_row("Cz")},
        {"channels": _cell_row("Cz"), "plotsubjects": "on"},
        {"channels": _cell_row("Cz"), "subject": "S02"},
    ):
        eeglab_backend("std_itcplot", study, alleeg, **options, nargout=0)
        close_reference_gui(eeglab_backend, request)


@pytest.mark.slow
@_reference("std_erspplot", "test_test_std_erspplot")
@pytest.mark.gui
def test_reference_std_erspplot(eeglab_backend, eeglab_writable_study, request):
    study, alleeg = _read_writable_n400(eeglab_backend, eeglab_writable_study)
    labels = alleeg["chanlocs"].flat[0]["labels"].ravel()
    channel, channels, all_channels = _cell_row(labels[15]), _cell_row(*labels[:4]), _cell_row(*labels)
    for selection in ("channels", "components"):
        study, alleeg = eeglab_backend(
            "std_precomp",
            study,
            alleeg,
            selection,
            savetrials="on",
            recompute="on",
            interp="on",
            ersp="on",
            itc="on",
            erspparams=_cell_row("ntimesout", 12.0, "nfreqs", 10.0, "verbose", "off"),
            nargout=2,
        )
    for store, options in (
        (True, {"clusters": np.array([[2.0, 3.0, 4.0]])}),
        (True, {"channels": channels}),
        (False, {"clusters": 2.0}),
        (False, {"clusters": 2.0, "comps": 1.0}),
        (False, {"channels": channel}),
        (False, {"channels": channel, "subject": study["subject"].flat[0]}),
        (False, {"channels": channel, "plotsubjects": "on"}),
        (True, {"channels": all_channels, "topofreq": 5.0, "topotime": 100.0}),
        (False, {"channels": all_channels, "condstats": "on", "topofreq": 5.0, "topotime": 100.0}),
        (
            False,
            {
                "channels": all_channels,
                "subject": study["subject"].flat[0],
                "topofreq": 5.0,
                "topotime": 100.0,
                "caxis": np.array([[-3.0, 3.0]]),
            },
        ),
    ):
        result = eeglab_backend("std_erspplot", study, alleeg, **options, nargout=int(store))
        if store:
            study = result
        close_reference_gui(eeglab_backend, request)


@pytest.mark.slow
@_reference("std_specplot", "test_test_std_specplot")
@pytest.mark.gui
def test_reference_std_specplot(eeglab_backend, eeglab_writable_study, request):
    study, alleeg = _read_writable_n400(eeglab_backend, eeglab_writable_study)
    labels = alleeg["chanlocs"].flat[0]["labels"].ravel()
    channel, all_channels = _cell_row(labels[15]), _cell_row(*labels)
    for selection in ("channels", "components"):
        kwargs = {"interp": "on"} if selection == "channels" else {}
        study, alleeg = eeglab_backend(
            "std_precomp",
            study,
            alleeg,
            selection,
            savetrials="on",
            recompute="on",
            spec="on",
            specparams=_cell_row("specmode", "fft"),
            nargout=2,
            **kwargs,
        )
    study = eeglab_backend("pop_statparams", study, "mode", "eeglab", "method", "param")
    for store, options in (
        (False, {"clusters": np.array([[2.0, 3.0, 4.0, 5.0]])}),
        (False, {"clusters": 3.0}),
        (False, {"clusters": 3.0, "comps": 1.0}),
        (False, {"clusters": 3.0, "condstats": "on"}),
        (False, {"clusters": 3.0, "condstats": "on", "plotconditions": "together"}),
        (False, {"clusters": 3.0, "condstats": "on", "plotconditions": "together", "threshold": 0.05}),
        (
            False,
            {"clusters": 3.0, "condstats": "on", "plotconditions": "together", "threshold": 0.05, "mcorrect": "fdr"},
        ),
        (False, {"channels": channel}),
        (False, {"channels": channel, "plotsubjects": "on"}),
        (False, {"channels": channel, "subject": study["subject"].flat[0], "plotconditions": "together"}),
        (False, {"channels": channel, "condstats": "on"}),
        (True, {"channels": all_channels}),
        (True, {"channels": all_channels, "plotconditions": "together", "condstats": "on"}),
        (False, {"channels": all_channels, "topofreq": 5.0}),
        (False, {"channels": all_channels, "condstats": "on", "topofreq": 5.0}),
    ):
        result = eeglab_backend("std_specplot", study, alleeg, **options, nargout=int(store))
        if store:
            study = result
        close_reference_gui(eeglab_backend, request)


@pytest.mark.slow
@pytest.mark.gui  # The original STUDY/precomputation chain can open MATLAB dialogs.
@_reference("std_precomp", "test_test_std_precomp")
def test_reference_std_precomp(eeglab_backend, eeglab_writable_study):
    for design_phase in (0, 1):
        # Source reloads the original STUDY between the two pairs of calls.
        study, alleeg = _read_writable_n400(eeglab_backend, eeglab_writable_study)
        if design_phase:
            study = eeglab_backend(
                "std_makedesign",
                study,
                alleeg,
                1.0,
                "variable1",
                "condition",
                "variable2",
                "",
                "name",
                "STUDY.design 1",
                "values1",
                _cell_row("non-synonyms", "synonyms"),
                "subjselect",
                _cell_row("S02", "S07", "S08", "S10"),
            )
            study = eeglab_backend(
                "std_makedesign",
                study,
                alleeg,
                2.0,
                "variable1",
                "condition",
                "variable2",
                "",
                "name",
                "Design 2 test",
                "values1",
                _cell_row("non-synonyms", _cell_row("non-synonyms", "synonyms")),
                "subjselect",
                _cell_row("S02"),
            )
            study["currentdesign"] = np.array([[2.0]])
            study = eeglab_backend("std_selectdesign", study, alleeg, 2.0)
        for selection in ("components", "channels"):
            ersp_parameters = ["cycles", np.array([[3.0, 0.8]]), "nfreqs", 2.0, "timesout", 30.0]
            if not design_phase:
                ersp_parameters.extend(("verbose", "off"))
            kwargs = {"scalp": "on"} if selection == "components" else {}
            study, alleeg = eeglab_backend(
                "std_precomp",
                study,
                alleeg,
                selection,
                recompute="on",
                erp="on",
                spec="on",
                specparams=_cell_row("specmode", "fft"),
                ersp="on",
                erspparams=_cell_row(*ersp_parameters),
                itc="on",
                nargout=2,
                **kwargs,
            )


@pytest.mark.slow
@pytest.mark.gui  # Preserve the original precomputation options, including dialog paths.
@_reference("std_preclust", "test_test_std_preclust")
def test_reference_std_preclust(eeglab_backend, eeglab_writable_study):
    study, alleeg = _read_writable_n400(eeglab_backend, eeglab_writable_study)
    study, alleeg = eeglab_backend(
        "std_precomp",
        study,
        alleeg,
        "components",
        savetrials="on",
        recompute="on",
        interp="on",
        erp="on",
        spec="on",
        specparams=_cell_row("specmode", "fft"),
        scalp="on",
        ersp="on",
        itc="on",
        erspparams=_cell_row("ntimesout", 12.0, "nfreqs", 10.0, "verbose", "off"),
        nargout=2,
    )
    empty = np.empty((0, 0))
    eeglab_backend(
        "std_preclust",
        study,
        alleeg,
        1.0,
        _cell_row("spec", "npca", 10.0, "norm", 1.0, "weight", 1.0, "freqrange", np.array([[3.0, 25.0]])),
        _cell_row("erp", "npca", 10.0, "norm", 1.0, "weight", 1.0, "timewindow", empty),
        _cell_row("scalp", "npca", 10.0, "norm", 1.0, "weight", 1.0, "abso", 1.0),
        _cell_row("dipoles", "norm", 1.0, "weight", 10.0),
        _cell_row("ersp", "npca", 10.0, "freqrange", empty, "timewindow", empty, "norm", 1.0, "weight", 1.0),
        _cell_row("itc", "npca", 10.0, "freqrange", empty, "timewindow", empty, "norm", 1.0, "weight", 1.0),
        _cell_row("finaldim", "npca", 10.0),
        nargout=2,
    )


@_reference("pop_clust", "test_test_pop_clust")
def test_reference_pop_clust(eeglab_backend, eeglab_sample_study, request):
    study, alleeg = eeglab_sample_study
    study = eeglab_backend("pop_clust", study, alleeg, algorithm="kmeanscluster", clus_num=10.0)
    if request.config.getoption("--eeglab-backend") == "matlab":
        available = (
            eeglab_backend("license", "checkout", "statistics_toolbox").item()
            and eeglab_backend("exist", "kmean").item()
        )
    else:
        # There is no MATLAB license on Python. Preserve the source's misspelled
        # kmean query in the real public/library namespaces, not corrected kmeans.
        available = any(getattr(module, "kmean", None) is not None for module in (eegprep, vq))
    request.node.user_properties.append(("eeglab_optional_kmeans_branch_entered", bool(available)))
    if available:
        eeglab_backend("pop_clust", study, alleeg, algorithm="kmeans", clus_num=10.0, outliers=3.0)


def test_std_selectdesign_scans_generated_designs_without_corrupting_component_membership():
    study, alleeg = _study_pair()
    study = std_makedesign(study, alleeg, 2, variable1="subject", values1=["S01"], name="S01")
    study = std_makedesign(study, alleeg, 3, variable1="subject", values1=["S02"], name="S02")
    study, alleeg = std_preclust(study, alleeg)
    original_pairs = (deepcopy(study["cluster"][0]["sets"]), deepcopy(study["cluster"][0]["comps"]))

    for design_index in range(1, 4):
        selected = std_selectdesign(study, alleeg, design_index)
        assert selected["currentdesign"] == design_index
        assert (selected["cluster"][0]["sets"], selected["cluster"][0]["comps"]) == original_pairs


@pytest.mark.parametrize("eeglab_sample_study", [("teststudy2", "stern2s.study")], indirect=True)
@_reference("std_selectdesign", "test_test_std_selectdesign")
def test_reference_std_selectdesign(eeglab_backend, eeglab_sample_study):
    study, alleeg = eeglab_sample_study
    version = eeglab_backend("eeg_getversion")
    for design_index in range(study["design"].size):
        design = study["design"].ravel()[design_index]
        skip_design = False
        if version.startswith("9"):
            for variable in design["variable"].ravel():
                values = variable["value"]
                if any(np.asarray(value).dtype == object for value in values.ravel()):
                    skip_design = True
                if (
                    values.size
                    and np.asarray(values.flat[0]).dtype.kind in "biufc"
                    and any(np.asarray(value).size > 1 for value in values.ravel())
                ):
                    skip_design = True
            if any(cell["trials"].flat[0].size == 0 for cell in design["cell"].ravel()):
                skip_design = True
        if skip_design:
            continue
        study = eeglab_backend("std_selectdesign", study, alleeg, float(design_index + 1))
        # The original asserts membership only for EEGLAB <=13; current
        # reference versions exercise every design as a workflow smoke test.
        prefix = version[:2]
        if prefix.rstrip(".").isdigit() and float(prefix) <= 13:
            for cluster in study["cluster"].ravel():
                for indices, sets in zip(cluster["allinds"].ravel(), cluster["setinds"].ravel(), strict=True):
                    for component, dataset_index in zip(indices.ravel(), sets.ravel(), strict=True):
                        selected_set = study["design"]["cell"].ravel()[design_index].ravel()[int(dataset_index) - 1]
                        columns = np.flatnonzero(cluster["comps"].ravel() == component)
                        dataset = selected_set["dataset"]
                        assert dataset.size == 0 or np.isin(dataset, cluster["sets"][:, columns]).any(), (
                            "Clusters corrupted"
                        )
