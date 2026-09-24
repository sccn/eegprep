"""Original std_dipplot workflow and additional Python dipole checks."""

from __future__ import annotations

import ast

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
import numpy as np
import pytest

import eegprep
from eegprep.functions.studyfunc.std_dipplot import std_dipplot
from tests.eeglab_tests import eeglab_test


UPSTREAM = "unittesting_studyfunc/std_dipplot/studyfunc_std_dipplot_wrapperTest.m"


def _dipole(posxyz, momxyz, rv):
    return {"posxyz": posxyz, "momxyz": momxyz, "rv": rv}


def _study_with_dipoles() -> tuple[dict, list[dict]]:
    datasets = [
        {
            "setname": "subject-01",
            "subject": "S01",
            "dipfit": {
                "coordformat": "MNI",
                "model": [
                    _dipole([10.0, 1.0, 20.0], [1.0, 0.0, 1.0], 0.1),
                    _dipole(
                        [[-20.0, 5.0, 30.0], [20.0, 5.0, 30.0]],
                        [[-0.5, 1.0, 0.0], [0.5, 1.0, 0.0]],
                        0.2,
                    ),
                    _dipole([], [], 0.5),
                ],
            },
        },
        {
            "setname": "subject-02",
            "subject": "S02",
            "dipfit": {
                "coordformat": "MNI",
                "model": [
                    _dipole([30.0, -10.0, 40.0], [-1.0, 0.0, 0.0], 0.3),
                    _dipole([50.0, 0.0, 60.0], [0.0, 1.0, 0.0], 0.4),
                ],
            },
        },
    ]
    study = {
        "name": "Localized components",
        "datasetinfo": [
            {"index": 1, "subject": "S01", "comps": [1, 2, 3]},
            {"index": 2, "subject": "S02", "comps": [1, 2]},
        ],
        "cluster": [
            {
                "name": "ParentCluster",
                "sets": [[1, 1, 1, 2, 2]],
                "comps": [1, 2, 3, 1, 2],
            },
            {"name": "Cluster A", "sets": [[1, 2]], "comps": [2, 1]},
            {"name": "Cluster B", "sets": [[1, 2]], "comps": [1, 2]},
        ],
    }
    return study, datasets


@eeglab_test(UPSTREAM, "test_test_std_dipplot")
def test_reference_std_dipplot(eeglab_backend, eeglab_sample_study):
    study, alleeg = eeglab_sample_study
    eeglab_backend("std_dipplot", study, alleeg, clusters="all", mode="joined", nargout=0)
    eeglab_backend("close", nargout=0)
    eeglab_backend("std_dipplot", study, alleeg, clusters=3.0, mode="centroid", nargout=0)
    eeglab_backend("close", nargout=0)
    # The original returns here; the component calls below it are inactive.


def test_current_wrapper_joined_and_centroid_modes_select_and_plot_real_dipfit_values():
    study, alleeg = _study_with_dipoles()

    updated, selections, figures, command = std_dipplot(
        study,
        alleeg,
        "clusters",
        "all",
        "mode",
        "joined",
        return_com=True,
    )

    assert [selection["cluster_index"] for selection in selections] == [2, 3]
    assert [selection["coordformat"] for selection in selections] == ["MNI", "MNI"]
    assert [
        (dipole["study_set"], dipole["dataset_index"], dipole["component"])
        for selection in selections
        for dipole in selection["dipoles"]
    ] == [(1, 1, 2), (2, 2, 1), (1, 1, 1), (2, 2, 2)]
    np.testing.assert_allclose(
        selections[0]["dipoles"][0]["posxyz"],
        [[-20.0, 5.0, 30.0], [20.0, 5.0, 30.0]],
    )
    np.testing.assert_allclose(selections[0]["centroid"]["posxyz"], [[15.0, -2.5, 35.0]])
    np.testing.assert_allclose(selections[0]["centroid"]["momxyz"], [[-0.5, 0.5, 0.0]])
    assert selections[0]["centroid"]["rv"] == pytest.approx(0.25)
    np.testing.assert_allclose(updated["cluster"][1]["dipole"]["posxyz"], [[15.0, -2.5, 35.0]])
    assert len(figures) == 1
    labels = figures[0].axes[0].get_legend_handles_labels()[1]
    assert "Cluster A: S01 IC2 (RV 20.0%)" in labels
    assert "Cluster B centroid (RV 25.0%)" in labels
    assert command.startswith("STUDY, DIPOLES, FIGURES = std_dipplot(")
    ast.parse(command)
    plt.close(figures[0])

    updated, selections, figures = std_dipplot(study, alleeg, "clusters", 3, "mode", "centroid")

    assert len(selections) == 1
    np.testing.assert_allclose(selections[0]["centroid"]["posxyz"], [[30.0, 0.5, 40.0]])
    assert selections[0]["centroid"]["rv"] == pytest.approx(0.25)
    assert figures[0].axes[0].get_legend_handles_labels()[1] == ["Cluster B centroid (RV 25.0%)"]
    np.testing.assert_allclose(updated["cluster"][2]["dipole"]["posxyz"], [[30.0, 0.5, 40.0]])
    plt.close(figures[0])


def test_comps_selects_one_based_cluster_member_positions_without_plotting():
    study, alleeg = _study_with_dipoles()

    _study, selections, figures = std_dipplot(study, alleeg, clusters=2, comps=2, mode="comps", plot=False)

    assert figures == []
    assert len(selections[0]["dipoles"]) == 1
    assert selections[0]["dipoles"][0]["member_index"] == 2
    assert selections[0]["dipoles"][0]["study_set"] == 2
    assert selections[0]["dipoles"][0]["component"] == 1
    np.testing.assert_allclose(selections[0]["dipoles"][0]["posxyz"], [[30.0, -10.0, 40.0]])

    _study, selections, _figures = std_dipplot(study, alleeg, clusters=2, comps="all", plot=False)
    assert [dipole["member_index"] for dipole in selections[0]["dipoles"]] == [1, 2]


def test_unlocalized_members_are_excluded_from_centroid_inputs():
    study, alleeg = _study_with_dipoles()
    study["cluster"][2] = {"name": "Cluster B", "sets": [[1, 2]], "comps": [3, 2]}

    _study, selections, _figures = std_dipplot(study, alleeg, clusters=3, plot=False)

    assert [(dipole["study_set"], dipole["component"]) for dipole in selections[0]["dipoles"]] == [(2, 2)]
    np.testing.assert_allclose(selections[0]["centroid"]["posxyz"], [[50.0, 0.0, 60.0]])
    assert selections[0]["centroid"]["rv"] == pytest.approx(0.4)


def test_malformed_dipfit_values_and_unknown_modes_fail_clearly():
    study, alleeg = _study_with_dipoles()

    with pytest.raises(ValueError, match="mode must be one of"):
        std_dipplot(study, alleeg, clusters=2, mode="silent-no-op", plot=False)

    alleeg[0]["dipfit"]["model"][1]["momxyz"] = [[1.0, 0.0, 0.0]]
    with pytest.raises(ValueError, match="posxyz and momxyz shapes differ"):
        std_dipplot(study, alleeg, clusters=2, plot=False)


def test_incompatible_coordinate_formats_are_not_combined():
    study, alleeg = _study_with_dipoles()
    alleeg[1]["dipfit"]["coordformat"] = "spherical"

    with pytest.raises(ValueError, match="incompatible coordinate formats"):
        std_dipplot(study, alleeg, clusters=2, plot=False)


@pytest.mark.parametrize(
    ("mode", "figure_count", "axes_per_figure"),
    [
        ("apart", 2, [1, 1]),
        ("together", 1, [2]),
        ("multicolor", 1, [1]),
        ("comps", 1, [1]),
    ],
)
def test_supported_member_plot_layouts_render(mode, figure_count, axes_per_figure):
    study, alleeg = _study_with_dipoles()

    _study, _selections, figures = std_dipplot(
        study,
        alleeg,
        clusters="all",
        mode=mode,
        dipcolor=["navy", "darkorange"],
        dipsize=[35, 45],
    )

    assert len(figures) == figure_count
    assert [len(figure.axes) for figure in figures] == axes_per_figure
    for figure in figures:
        plt.close(figure)


def test_std_dipplot_remains_available_from_the_package_api():
    assert eegprep.std_dipplot is std_dipplot
