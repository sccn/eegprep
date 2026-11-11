"""Internal checks and FieldTrip parity for the MNI boundary-element model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from eegprep import pop_loadset
from eegprep.plugins.dipfit._fitting import default_search_grids, prepare_forward_data
from eegprep.plugins.dipfit._utils import channel_template_filename, lookupchantemplate, template_by_name
from eegprep.plugins.dipfit.fieldtrip.bem import (
    eeg_infinite_dipole,
    eeg_leadfieldb,
    ft_prepare_vol_sens,
    load_bem_headmodel,
)
from eegprep.plugins.dipfit.fieldtrip.ft_compute_leadfield import ft_compute_leadfield
from eegprep.plugins.dipfit.fieldtrip.geometry import project_elec, surface_inside, transfer_elec
from eegprep.plugins.dipfit.pop_dipfit_settings import pop_dipfit_settings, pop_dipfit_settings_dialog_spec
from scripts.dipfit_standard_bem_parity import PARITY_DIPOLES_MM, bem_parity_payload
from tests.eeglab_tests.backend import skip_without_matlab_plugin


_CAP385_TO_MNI = np.asarray([0.832146, -15.6287, 2.41142, 0.0812144, 0.000937391, -1.5732, 1.17419, 1.06011, 1.14846])
_SAMPLE = Path(__file__).resolve().parents[1] / "sample_data" / "eeglab_data_epochs_ica.set"


def _octahedron() -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(
        [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]
    )
    triangles = np.asarray([[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4], [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]])
    return vertices, triangles


def test_infinite_dipole_is_linear_and_antisymmetric():
    dipole = np.asarray([[0.0, 20.0, 5.0]])
    electrode = np.asarray([[10.0, 0.0, 0.0]])
    moment = np.asarray([0.3, -0.2, 0.5])
    forward = eeg_infinite_dipole(dipole, electrode, 1.0)
    reverse = eeg_infinite_dipole(electrode, dipole, 1.0)
    closed = (electrode - dipole) / ((4.0 * np.pi) * np.sum((electrode - dipole) ** 2) ** 1.5)
    np.testing.assert_allclose(forward, closed, atol=1e-12)
    np.testing.assert_allclose(forward @ moment, -(reverse @ moment), atol=1e-12)
    np.testing.assert_allclose(forward @ (2.0 * moment), 2.0 * (forward @ moment), atol=1e-12)


def test_identity_bem_on_a_spherical_mesh_matches_the_infinite_dipole():
    vertices, triangles = _octahedron()
    assert surface_inside([[0.0, 0.0, 0.0]], vertices, triangles)[0]
    volume = {
        "type": "dipoli",
        "cond": np.asarray([1.0]),
        "bnd": [{"pos": vertices, "tri": triangles}],
        "mat": np.eye(vertices.shape[0]),
        "skin_surface": 0,
        "source": 0,
        "unit": "mm",
    }
    dipole = np.asarray([[0.1, -0.2, 0.05]])
    np.testing.assert_allclose(
        eeg_leadfieldb(dipole, vertices, volume),
        eeg_infinite_dipole(dipole, vertices, 1.0),
        atol=1e-12,
    )
    headmodel, sens = ft_prepare_vol_sens(volume, {"elecpos": vertices, "label": [f"V{i}" for i in range(6)]})
    prepared = ft_compute_leadfield(dipole, sens, headmodel)
    referenced = eeg_infinite_dipole(dipole, vertices, 1.0)
    referenced = referenced - np.mean(referenced, axis=0, keepdims=True)
    np.testing.assert_allclose(prepared, referenced, atol=1e-10)
    np.testing.assert_allclose(prepared.sum(axis=0), 0.0, atol=1e-10)


def test_bemcp_and_asa_conductivity_weights_follow_fieldtrip():
    vertices, triangles = _octahedron()
    inner = vertices * 0.5
    boundaries = [{"pos": vertices, "tri": triangles}, {"pos": inner, "tri": triangles}]
    dipole = np.asarray([[0.0, 0.0, 0.0]])
    cond = np.asarray([0.33, 0.0042])
    system = np.eye(vertices.shape[0] + inner.shape[0])
    volume = {
        "type": "bemcp",
        "cond": cond,
        "bnd": boundaries,
        "mat": system,
        "skin_surface": 0,
        "source": 1,
    }
    leadfield = eeg_leadfieldb(dipole, vertices, volume)
    outer = eeg_infinite_dipole(dipole, vertices, (cond[0] + cond[1]) / 2.0)
    inner_field = eeg_infinite_dipole(dipole, inner, cond[1] / 2.0)
    np.testing.assert_allclose(leadfield, np.vstack([outer, inner_field]), atol=1e-12)

    volume["type"] = "asa"
    stacked = np.vstack([eeg_infinite_dipole(dipole, vertices, cond[1]), eeg_infinite_dipole(dipole, inner, cond[1])])
    np.testing.assert_allclose(eeg_leadfieldb(dipole, vertices, volume), stacked, atol=1e-12)


def test_triangle_projection_uses_barycentric_weights():
    vertices = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    triangles = np.asarray([[0, 1, 2]])
    weights, projected = project_elec(np.asarray([[0.2, 0.3, 1.0]]), vertices, triangles)
    np.testing.assert_allclose(projected, [[0.2, 0.3, 0.0]], atol=1e-12)
    np.testing.assert_allclose(weights[0, 1:3], [0.2, 0.3], atol=1e-12)
    transfer = transfer_elec(vertices, triangles, weights)
    np.testing.assert_allclose(transfer.sum(axis=1), [1.0], atol=1e-12)


def test_sample_cap385_file_selects_mni_bem_and_its_coregistration():
    eeg = pop_loadset(str(_SAMPLE))
    template = template_by_name("standardBEM")
    found, transform = lookupchantemplate(channel_template_filename(eeg), template)
    assert found
    np.testing.assert_allclose(transform, _CAP385_TO_MNI, atol=1e-12)

    out = pop_dipfit_settings(eeg, model="standardBEM")
    assert out["dipfit"]["hdmfile"].endswith("standard_vol.mat")
    assert out["dipfit"]["coordformat"] == "MNI"
    np.testing.assert_allclose(out["dipfit"]["coord_transform"], _CAP385_TO_MNI, atol=1e-12)
    assert len(out["dipfit"]["chansel"]) == 30

    spec = pop_dipfit_settings_dialog_spec(eeg)
    coreg = next(control for control in spec.controls if control.tag == "coord_transform")
    np.testing.assert_allclose(np.fromstring(str(coreg.value), sep=" "), _CAP385_TO_MNI, rtol=1e-6, atol=1e-6)
    model = next(control for control in spec.controls if control.tag == "model")
    assert model.value == 2

    x_values, y_values, z_values = default_search_grids(out)
    assert x_values.size == 24 and y_values.size == 24 and z_values.size == 12
    np.testing.assert_allclose([x_values[0], x_values[-1], z_values[0], z_values[-1]], [-85.0, 85.0, 0.0, 85.0])


def test_packaged_dipoli_leadfield_matches_the_prepared_system_matrix():
    eeg = pop_dipfit_settings(pop_loadset(str(_SAMPLE)), model="standardBEM")
    forward = prepare_forward_data(eeg, [1])
    headmodel = forward.headmodel
    assert headmodel is not None
    dipole = PARITY_DIPOLES_MM[:1]
    leadfield = ft_compute_leadfield(dipole, forward.sens, headmodel)
    vertices = np.vstack([boundary["pos"] for boundary in headmodel["bnd"]])
    rebuilt = headmodel["mat"] @ eeg_infinite_dipole(dipole, vertices, 1.0)
    np.testing.assert_allclose(leadfield, rebuilt, atol=1e-12)
    np.testing.assert_allclose(leadfield.sum(axis=0), 0.0, atol=1e-10)
    moment = np.asarray([0.2, -0.4, 0.7])
    np.testing.assert_allclose(leadfield @ (3.0 * moment), 3.0 * (leadfield @ moment), atol=1e-12)
    raw = load_bem_headmodel(eeg["dipfit"]["hdmfile"])
    assert raw["type"] == "dipoli"
    assert raw["mat"].shape == (3000, 3000)
    np.testing.assert_allclose(raw["cond"], [0.33, 0.004125, 0.33], rtol=0, atol=1e-7)


def test_pop_multifit_default_bem_returns_finite_positions():
    payload = bem_parity_payload(_SAMPLE)
    positions = np.asarray(payload["positions"], dtype=float)
    assert positions.shape == (3, 3)
    assert np.all(np.isfinite(positions))
    leadfield = np.asarray(payload["leadfield"], dtype=float)
    assert leadfield.shape[1] == 9
    assert leadfield.shape[0] == len(payload["labels"])


@pytest.mark.matlab
@pytest.mark.parity
def test_bem_leadfield_and_multifit_match_fieldtrip(eeglab_backend, eeglab_matlab_engine):
    skip_without_matlab_plugin(eeglab_matlab_engine, "ft_compute_leadfield", "DIPFIT/FieldTrip")
    skip_without_matlab_plugin(eeglab_matlab_engine, "pop_multifit", "DIPFIT")
    reference = eeglab_backend("eegprep_dipfit_bem_reference", str(_SAMPLE))
    payload = bem_parity_payload(_SAMPLE)
    matlab_labels = str(np.asarray(reference["labels"]).reshape(-1)[0]).splitlines()
    order = [matlab_labels.index(label) for label in payload["labels"]]
    matlab_leadfield = np.asarray(reference["leadfield"], dtype=float)[order]
    python_leadfield = np.asarray(payload["leadfield"], dtype=float)
    relative = np.max(np.abs(python_leadfield - matlab_leadfield)) / np.max(np.abs(matlab_leadfield))
    assert relative < 1e-4

    matlab_positions = np.asarray(reference["positions"], dtype=float)
    python_positions = np.asarray(payload["positions"], dtype=float)
    if matlab_positions.shape != python_positions.shape:
        matlab_positions = matlab_positions.T
    distances = np.linalg.norm(matlab_positions - python_positions, axis=1)
    assert np.all(distances < 1.0)
