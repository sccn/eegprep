"""FieldTrip comparison for the LORETA-KEY leadfield of the selected head model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from eegprep.functions.popfunc.pop_loadset import pop_loadset
from eegprep.plugins.dipfit._fitting import loreta_key_source_points
from eegprep.plugins.dipfit.pop_dipfit_settings import pop_dipfit_settings
from eegprep.plugins.dipfit.pop_leadfield import pop_leadfield
from tests.eeglab_tests.backend import call_matlab, skip_without_matlab_plugin


SAMPLE_EPOCHS = Path(__file__).resolve().parents[1] / "sample_data" / "eeglab_data_epochs_ica.set"
# Same relative leadfield bounds as the standardBEM and standardBESA FieldTrip tests.
_LEADFIELD_RELATIVE = {"standardBEM": 1e-4, "standardBESA": 1e-5}


def _relative_error(actual: np.ndarray, expected: np.ndarray) -> float:
    scale = float(np.max(np.abs(expected)))
    return float(np.max(np.abs(actual - expected)) / scale)


def _stacked_leadfield(fields: list) -> np.ndarray:
    return np.concatenate([np.asarray(field, dtype=float) for field in fields], axis=1)


@pytest.mark.timeout(1800)
def test_loreta_key_leadfield_matches_fieldtrip_for_the_selected_head_model(eeglab_matlab_engine, request):
    eeglab_root = request.config.getoption("--eeglab-root")
    call_matlab(eeglab_matlab_engine, "dipfit_standard_besa_add_fieldtrip", str(eeglab_root), nargout=0)
    skip_without_matlab_plugin(eeglab_matlab_engine, "ft_compute_leadfield", "FieldTrip")
    skip_without_matlab_plugin(eeglab_matlab_engine, "ft_inside_headmodel", "FieldTrip")
    skip_without_matlab_plugin(eeglab_matlab_engine, "pop_dipfit_settings", "DIPFIT")

    dataset = pop_loadset(str(SAMPLE_EPOCHS))
    points = loreta_key_source_points()
    for model, tolerance in _LEADFIELD_RELATIVE.items():
        configured = pop_dipfit_settings(dataset, model=model)
        leadfield = pop_leadfield(configured, sourcemodel="loreta-key")
        source = leadfield["dipfit"]["sourcemodel"]
        python_points = np.asarray(source["pos"], dtype=float)
        np.testing.assert_allclose(python_points, points)
        python_leadfield = _stacked_leadfield(source["leadfield"])

        reference = call_matlab(
            eeglab_matlab_engine,
            "eegprep_dipfit_loreta_key_reference",
            str(SAMPLE_EPOCHS),
            model,
            points,
            str(eeglab_root),
            nargout=1,
        )
        matlab_labels = str(np.asarray(reference["labels"]).reshape(-1)[0]).splitlines()
        order = [matlab_labels.index(label) for label in source["label"]]
        matlab_leadfield = np.asarray(reference["leadfield"], dtype=float)[order]
        matlab_points = np.asarray(reference["points"], dtype=float)
        if matlab_points.shape != python_points.shape:
            matlab_points = matlab_points.T
        np.testing.assert_allclose(matlab_points, python_points)
        assert python_leadfield.shape == matlab_leadfield.shape
        assert _relative_error(python_leadfield, matlab_leadfield) < tolerance
        matlab_inside = np.asarray(reference["inside"], dtype=float).reshape(-1) > 0
        np.testing.assert_array_equal(np.asarray(source["inside"], dtype=bool), matlab_inside)
