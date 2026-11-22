"""MATLAB comparisons for the sample ICA dipoles and inside-brain grids."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from eegprep.functions.popfunc.pop_loadset import pop_loadset
from eegprep.plugins.dipfit.dipfit_gridsearch import default_search_grids, fitting_config
from eegprep.plugins.dipfit.eeglab2fieldtrip import eeglab2fieldtrip
from eegprep.plugins.dipfit.ft_dipolefitting import inside_brain_grid, inside_headmodel, prepare_headmodel_sens
from eegprep.plugins.dipfit.pop_dipfit_settings import pop_dipfit_settings
from eegprep.plugins.dipfit.pop_multifit import pop_multifit
from tests.eeglab_tests.backend import call_matlab, skip_without_matlab_plugin


SAMPLE_EPOCHS = Path(__file__).resolve().parents[1] / "sample_data" / "eeglab_data_epochs_ica.set"
_MODELS = ("standardBESA", "standardBEM")
_INSIDE_BRAIN_COUNTS = {"standardBESA": 1912, "standardBEM": 2900}
_SINGLE_DIPOLE_MAX_MM = {"standardBESA": 8.0e-5, "standardBEM": 3.33e-3}
_SYMMETRIC_DIPOLE_MAX_MM = {"standardBESA": 6.42e-3, "standardBEM": 7.52e-3}
_ILL_CONDITIONED_BEM_COMPONENT = 21
_ILL_CONDITIONED_BEM_SYMMETRIC_COMPONENT = 1
_RV_ATOL = 1e-5
_RV_RTOL = 1e-3


def _reference(engine, request, model: str, kind: str):
    eeglab_root = request.config.getoption("--eeglab-root")
    call_matlab(engine, "dipfit_standard_besa_add_fieldtrip", str(eeglab_root), nargout=0)
    skip_without_matlab_plugin(engine, "ft_inside_headmodel", "FieldTrip")
    skip_without_matlab_plugin(engine, "pop_multifit", "DIPFIT")
    return call_matlab(
        engine,
        "eegprep_dipfit_component_parity",
        str(SAMPLE_EPOCHS),
        model,
        kind,
        str(eeglab_root),
        nargout=1,
    )


def _inside_count(model: str) -> int:
    dataset = pop_dipfit_settings(pop_loadset(str(SAMPLE_EPOCHS)), model=model)
    comp = eeglab2fieldtrip(dataset)
    cfg = fitting_config(dataset)
    vol, _sens = prepare_headmodel_sens(cfg, comp["elec"], dataset["chanlocs"])
    _pos, inside = inside_brain_grid(*default_search_grids(dataset), vol)
    return int(np.count_nonzero(inside))


def _fit_positions(model: str, components: list[int], dipoles: int) -> tuple[np.ndarray, np.ndarray]:
    dataset = pop_dipfit_settings(pop_loadset(str(SAMPLE_EPOCHS)), model=model)
    fitted = pop_multifit(dataset, components, threshold=100, dipoles=dipoles, dipplot="off")
    positions = []
    residual = []
    for component in components:
        model_entry = fitted["dipfit"]["model"][component - 1]
        xyz = np.asarray(model_entry["posxyz"], dtype=float).reshape(-1, 3)
        assert xyz.shape[0] >= dipoles
        positions.append(xyz[:dipoles].reshape(-1))
        residual.append(float(model_entry["rv"]))
    return np.vstack(positions), np.asarray(residual, dtype=float)


def _matlab_positions(reference, n_components: int, dipoles: int) -> np.ndarray:
    positions = np.asarray(reference["positions"], dtype=float)
    expected = (n_components, 3 * dipoles)
    if positions.shape != expected and positions.T.shape == expected:
        positions = positions.T
    assert positions.shape == expected
    return positions


@pytest.mark.matlab
@pytest.mark.timeout(7200)
def test_inside_brain_grid_counts_match_eeglab(eeglab_matlab_engine, request):
    for model, expected in _INSIDE_BRAIN_COUNTS.items():
        reference = _reference(eeglab_matlab_engine, request, model, "grid")
        matlab_count = int(np.asarray(reference["count"]).reshape(-1)[0])
        assert matlab_count == expected
        assert _inside_count(model) == matlab_count


@pytest.mark.matlab
@pytest.mark.timeout(7200)
def test_single_dipole_fits_match_eeglab(eeglab_matlab_engine, request):
    for model in _MODELS:
        reference = _reference(eeglab_matlab_engine, request, model, "single")
        components = [int(value) for value in np.asarray(reference["components"]).reshape(-1)]
        assert components == list(range(1, 33))
        python_pos, python_rv = _fit_positions(model, components, 1)
        matlab_pos = _matlab_positions(reference, len(components), 1)
        matlab_rv = np.asarray(reference["rv"], dtype=float).ravel()
        distances = np.linalg.norm(python_pos - matlab_pos, axis=1)
        exact = np.ones(len(components), dtype=bool)
        if model == "standardBEM":
            # Component 21 is ill-conditioned: MATLAB itself reaches several minima under 1e-12 start perturbations.
            ill = components.index(_ILL_CONDITIONED_BEM_COMPONENT)
            exact[ill] = False
            allowance = _RV_ATOL + _RV_RTOL * abs(float(matlab_rv[ill]))
            assert python_rv[ill] <= matlab_rv[ill] + allowance, (
                f"{model} component {_ILL_CONDITIONED_BEM_COMPONENT} residual variance "
                f"{float(python_rv[ill])} exceeds MATLAB {float(matlab_rv[ill])} by more than {allowance}"
            )
        farthest = float(np.max(distances[exact]))
        limit = _SINGLE_DIPOLE_MAX_MM[model]
        assert farthest <= limit, f"{model} farthest dipole is {farthest} mm (limit {limit} mm)"
        np.testing.assert_allclose(python_rv[exact], matlab_rv[exact], atol=_RV_ATOL, rtol=_RV_RTOL)
        if model == "standardBEM":
            dataset = pop_dipfit_settings(pop_loadset(str(SAMPLE_EPOCHS)), model=model)
            comp = eeglab2fieldtrip(dataset)
            vol, _sens = prepare_headmodel_sens(fitting_config(dataset), comp["elec"], dataset["chanlocs"])
            assert inside_headmodel(python_pos[2:3], vol).all()


@pytest.mark.matlab
@pytest.mark.timeout(7200)
def test_symmetric_two_dipole_fits_match_eeglab(eeglab_matlab_engine, request):
    for model in _MODELS:
        reference = _reference(eeglab_matlab_engine, request, model, "symmetric")
        components = [int(value) for value in np.asarray(reference["components"]).reshape(-1)]
        assert components == list(range(1, 11))
        python_pos, python_rv = _fit_positions(model, components, 2)
        matlab_pos = _matlab_positions(reference, len(components), 2)
        matlab_rv = np.asarray(reference["rv"], dtype=float).ravel()
        distances = np.linalg.norm(python_pos.reshape(-1, 3) - matlab_pos.reshape(-1, 3), axis=1)
        exact = np.ones(len(components), dtype=bool)
        exact_dipoles = np.ones(len(components) * 2, dtype=bool)
        if model == "standardBEM":
            # Component 1 is ill-conditioned: MATLAB itself reaches several minima under 1e-12 start perturbations.
            ill = components.index(_ILL_CONDITIONED_BEM_SYMMETRIC_COMPONENT)
            exact[ill] = False
            exact_dipoles[2 * ill : 2 * ill + 2] = False
            allowance = _RV_ATOL + _RV_RTOL * abs(float(matlab_rv[ill]))
            assert python_rv[ill] <= matlab_rv[ill] + allowance, (
                f"{model} symmetric component {_ILL_CONDITIONED_BEM_SYMMETRIC_COMPONENT} residual variance "
                f"{float(python_rv[ill])} exceeds MATLAB {float(matlab_rv[ill])} by more than {allowance}"
            )
        farthest = float(np.max(distances[exact_dipoles]))
        limit = _SYMMETRIC_DIPOLE_MAX_MM[model]
        assert farthest <= limit, f"{model} farthest symmetric dipole is {farthest} mm (limit {limit} mm)"
        np.testing.assert_allclose(python_rv[exact], matlab_rv[exact], atol=_RV_ATOL, rtol=_RV_RTOL)
