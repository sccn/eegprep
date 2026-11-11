"""Four-shell BESA leadfield checks and DIPFIT position parity for standardBESA."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from eegprep.functions.popfunc.pop_loadset import pop_loadset
from eegprep.plugins.dipfit._fitting import default_scan_axes, dipfit_gridsearch
from eegprep.plugins.dipfit.forward.eeg_leadfield1 import eeg_leadfield1
from eegprep.plugins.dipfit.forward.eeg_leadfield4 import eeg_leadfield4
from eegprep.plugins.dipfit.forward.ft_compute_leadfield import ft_compute_leadfield
from eegprep.plugins.dipfit.forward.ft_inside_headmodel import ft_inside_headmodel
from eegprep.plugins.dipfit.forward.ft_prepare_vol_sens import ft_prepare_vol_sens
from eegprep.plugins.dipfit.forward.standard_besa import standard_besa_headmodel
from eegprep.plugins.dipfit.pop_dipfit_settings import pop_dipfit_settings
from eegprep.plugins.dipfit.pop_multifit import pop_multifit
from tests.eeglab_tests.backend import call_matlab, skip_without_matlab_plugin


SAMPLE_EPOCHS = Path(__file__).resolve().parents[1] / "sample_data" / "eeglab_data_epochs_ica.set"
# Positions printed for components 1-3 in https://github.com/sccn/eegprep/issues/600.
EEGLAB_STANDARD_BESA_POSITIONS_MM = np.array(
    [
        [42.6358, -0.7557, -6.2491],
        [-13.3446, -44.4421, 2.0858],
        [68.6342, 1.3846, -38.0883],
    ]
)
POSITION_TOLERANCE_MM = 1.0
LEADFIELD_DIPOLES_MM = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 40.0],
        [30.0, -20.0, 10.0],
        [10.0, -50.0, 20.0],
        [68.6342, 1.3846, -38.0883],
    ]
)


def _electrodes() -> np.ndarray:
    polar = np.linspace(0.35, np.pi - 0.35, 4)
    azimuth = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    points = [
        [85.0 * np.sin(theta) * np.cos(phi), 85.0 * np.sin(theta) * np.sin(phi), 85.0 * np.cos(theta)]
        for theta in polar
        for phi in azimuth
    ]
    return np.asarray(points, dtype=float)


def _relative_error(actual: np.ndarray, expected: np.ndarray) -> float:
    scale = float(np.max(np.abs(expected)))
    return float(np.max(np.abs(actual - expected)) / scale)


def test_equal_conductivities_match_the_single_sphere_leadfield():
    electrodes = _electrodes()
    besa = standard_besa_headmodel()
    equal = {"r": np.asarray(besa["r"], dtype=float), "cond": np.ones(4)}
    single = {"r": np.array([85.0]), "cond": np.array([1.0])}
    cases = (
        (np.zeros(3), 1e-10),
        (np.array([0.0, 0.0, 40.0]), 1e-10),
        (np.array([30.0, -20.0, 10.0]), 1e-10),
        # Order 60 is FieldTrip's series length. At 65 mm the tail is about 1e-7.
        (np.array([0.0, 0.0, 65.0]), 1e-6),
    )
    for position, tolerance in cases:
        four = eeg_leadfield4(position, electrodes, equal)
        one = eeg_leadfield1(position, electrodes, single)
        assert _relative_error(four, one) < tolerance


def test_center_dipole_in_the_besa_model_follows_the_electrode_directions():
    electrodes = _electrodes()
    raw = eeg_leadfield4(np.zeros(3), electrodes, standard_besa_headmodel())
    direction = electrodes / np.linalg.norm(electrodes, axis=1, keepdims=True)
    # At the origin only the n=1 term survives, so each column is one moment axis.
    for axis in range(3):
        scale = float(np.dot(raw[:, axis], direction[:, axis]) / np.dot(direction[:, axis], direction[:, axis]))
        assert scale > 0.0
        np.testing.assert_allclose(raw[:, axis], scale * direction[:, axis], rtol=1e-8, atol=1e-12)


def test_leadfield_is_invariant_under_a_joint_rotation_and_reverses_with_moment():
    electrodes = _electrodes()
    headmodel = standard_besa_headmodel()
    position = np.array([30.0, -20.0, 10.0])
    moment = np.array([0.2, -0.5, 0.7])
    rotation, _ = np.linalg.qr(np.array([[1.0, 2.0, 3.0], [0.0, 1.0, 4.0], [5.0, 6.0, 0.0]]))
    if np.linalg.det(rotation) < 0.0:
        rotation[:, 0] *= -1.0
    original = ft_compute_leadfield(position, {"elecpos": electrodes}, headmodel)
    rotated = ft_compute_leadfield(position @ rotation.T, {"elecpos": electrodes @ rotation.T}, headmodel)
    np.testing.assert_allclose(original @ moment, rotated @ (rotation @ moment), rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(original @ (-moment), -(original @ moment), rtol=1e-12, atol=1e-15)


def test_dipole_between_the_brain_and_scalp_still_has_a_finite_leadfield():
    position = np.array([68.6342, 1.3846, -38.0883])
    assert np.linalg.norm(position) > 71.0
    leadfield = ft_compute_leadfield(position, {"elecpos": _electrodes()}, standard_besa_headmodel())
    assert np.isfinite(leadfield).all()
    assert np.max(np.abs(leadfield)) > 0.0


def test_electrodes_project_onto_the_outer_sphere_and_the_brain_shell_is_strict():
    headmodel = standard_besa_headmodel()
    raw = np.array([[100.0, 0.0, 0.0], [0.0, 40.0, 30.0]])
    _model, sensors = ft_prepare_vol_sens(headmodel, {"elecpos": raw})
    projected = np.asarray(sensors["elecpos"])
    np.testing.assert_allclose(np.linalg.norm(projected, axis=1), 85.0)
    np.testing.assert_allclose(projected[0], [85.0, 0.0, 0.0])
    assert ft_inside_headmodel(np.array([0.0, 0.0, 70.999]), headmodel)[0]
    assert not ft_inside_headmodel(np.array([0.0, 0.0, 71.0]), headmodel)[0]


def test_two_shell_padding_matches_a_repeated_outer_shell():
    electrodes = _electrodes()
    two = {"r": np.array([71.0, 85.0]), "cond": np.array([0.33, 0.33])}
    four = {"r": np.array([71.0, 85.0, 85.0, 85.0]), "cond": np.array([0.33, 0.33, 0.33, 0.33])}
    position = np.array([10.0, -20.0, 30.0])
    np.testing.assert_allclose(
        ft_compute_leadfield(position, {"elecpos": electrodes}, two),
        ft_compute_leadfield(position, {"elecpos": electrodes}, four),
        rtol=1e-10,
        atol=1e-15,
    )


def test_bem_leadfields_stay_unimplemented_for_the_boundary_element_port():
    with pytest.raises(NotImplementedError, match="bem"):
        ft_compute_leadfield(np.zeros(3), {"elecpos": _electrodes()}, {"type": "bem", "bnd": [], "mat": []})


def test_standard_besa_grid_matches_the_dipfit_scan():
    x_values, y_values, z_values = default_scan_axes(85.0)
    assert x_values.size == 24
    assert z_values.size == 12
    np.testing.assert_allclose(x_values[[0, -1]], [-85.0, 85.0])
    np.testing.assert_allclose(z_values[[0, -1]], [0.0, 85.0])
    dataset = pop_dipfit_settings(pop_loadset(SAMPLE_EPOCHS), model="standardBESA")
    scanned = dipfit_gridsearch(dataset, component=[1], xgrid=x_values, ygrid=y_values, zgrid=z_values)
    assert np.asarray(scanned["dipfit"]["model"][0]["posxyz"]).shape == (1, 3)


def test_pop_multifit_standard_besa_positions_match_the_published_eeglab_table():
    dataset = pop_dipfit_settings(pop_loadset(SAMPLE_EPOCHS), model="standardBESA")
    fitted = pop_multifit(dataset, [1, 2, 3], "threshold", 100, "dipplot", "off")
    positions = np.vstack([np.asarray(fitted["dipfit"]["model"][index]["posxyz"], dtype=float) for index in range(3)])
    distances = np.linalg.norm(positions - EEGLAB_STANDARD_BESA_POSITIONS_MM, axis=1)
    assert np.all(distances < POSITION_TOLERANCE_MM), (
        f"3-D distances mm={distances.tolist()} fitted={positions.tolist()}"
    )


@pytest.mark.timeout(1800)
def test_fieldtrip_leadfields_and_pop_multifit_positions_match_eeglab(eeglab_matlab_engine, request):
    eeglab_root = request.config.getoption("--eeglab-root")
    call_matlab(eeglab_matlab_engine, "dipfit_standard_besa_add_fieldtrip", str(eeglab_root), nargout=0)
    skip_without_matlab_plugin(eeglab_matlab_engine, "ft_compute_leadfield", "FieldTrip")
    skip_without_matlab_plugin(eeglab_matlab_engine, "pop_multifit", "DIPFIT")
    reference = call_matlab(
        eeglab_matlab_engine,
        "dipfit_standard_besa_reference",
        str(SAMPLE_EPOCHS),
        eeglab_root,
        nargout=1,
    )
    raw = np.asarray(reference["elec_raw"], dtype=float)
    projected = np.asarray(reference["elec_projected"], dtype=float)
    dipoles = np.asarray(reference["dippos"], dtype=float)
    matlab_leadfield = np.asarray(reference["leadfield"], dtype=float)
    matlab_positions = np.asarray(reference["fitpos"], dtype=float)
    _model, sensors = ft_prepare_vol_sens(standard_besa_headmodel(), {"elecpos": raw})
    np.testing.assert_allclose(np.asarray(sensors["elecpos"]), projected, atol=1e-6)
    python_leadfield = ft_compute_leadfield(dipoles, {"elecpos": raw}, standard_besa_headmodel())
    assert _relative_error(python_leadfield, matlab_leadfield) < 1e-5
    dataset = pop_dipfit_settings(pop_loadset(SAMPLE_EPOCHS), model="standardBESA")
    fitted = pop_multifit(dataset, [1, 2, 3], "threshold", 100, "dipplot", "off")
    positions = np.vstack([np.asarray(fitted["dipfit"]["model"][index]["posxyz"], dtype=float) for index in range(3)])
    distances = np.linalg.norm(positions - matlab_positions.reshape(3, 3), axis=1)
    assert np.all(distances < POSITION_TOLERANCE_MM), f"3-D distances mm={distances.tolist()}"
