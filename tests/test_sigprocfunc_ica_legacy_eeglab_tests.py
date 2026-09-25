"""Current eeglab_tests ports for legacy low-level ICA functions."""

from __future__ import annotations

import numpy as np

from eegprep.functions.miscfunc.misc import finite_matmul
from eegprep.functions.sigprocfunc.icadefs import icadefs
from eegprep.functions.sigprocfunc.kmeanscluster import kmeanscluster
from eegprep.functions.sigprocfunc.posact import posact
from eegprep.functions.sigprocfunc.runica import runica
from eegprep.functions.sigprocfunc.runica_ml2 import runica_ml2
from eegprep.functions.sigprocfunc.runica_mlb import runica_mlb
from tests.eeglab_tests import eeglab_test
from tests.eeglab_tests import assert_matlab_near as _assert_near


SIGPROC_ROOT = "unittesting_sigprocfunc"


@eeglab_test(f"{SIGPROC_ROOT}/icadefs/sigprocfunc_icadefs_wrapperTest.m", "test_pass_general")
def test_reference_icadefs(eeglab_backend, request):
    if request.config.getoption("--eeglab-backend") == "matlab":
        defaults = eeglab_backend("eegprep_test_icadefs")
        binary, sampling_rate = defaults["ICABINARY"], defaults["DEFAULT_SRATE"]
    else:
        defaults = eeglab_backend("icadefs")
        binary, sampling_rate = defaults.ICABINARY, defaults.DEFAULT_SRATE
    assert np.asarray(binary).size
    if isinstance(binary, str):
        assert binary
    assert np.asarray(sampling_rate).size


def test_python_regression_icadefs_current_suite_has_platform_binary_and_sampling_defaults():
    defaults = icadefs()
    assert defaults.ICABINARY in {"ica_linux", "ica_osx", "binica.exe"}
    assert defaults.DEFAULT_SRATE == 256.0175
    assert defaults.DEFAULT_TIMLIM == (-1000, 2000)
    assert len(defaults.BACKCOLOR) == 3


def test_python_regression_kmeanscluster_current_suite_exact_example_and_high_dimensional_case():
    example = np.asarray([[1, 1], [2, 1], [4, 3], [5, 4]], dtype=float)
    labels, centers, previous_labels, unchanged = kmeanscluster(example, 2)
    np.testing.assert_array_equal(labels, [0, 0, 1, 1])
    np.testing.assert_allclose(centers, [[1.5, 1.0], [4.5, 3.5]])
    np.testing.assert_array_equal(previous_labels, labels)
    np.testing.assert_array_equal(unchanged, example)

    random_data = np.random.default_rng(12).normal(size=(100, 20))
    random_labels, random_centers, random_previous, _random_unchanged = kmeanscluster(
        random_data,
        5,
        randomized=True,
        random_state=3,
    )
    assert random_labels.shape == (100,)
    assert random_centers.shape == (5, 20)
    np.testing.assert_array_equal(random_previous, random_labels)
    assert len(np.unique(random_labels)) == 5


def test_python_regression_posact_current_suite_rectangular_weights_exact_outputs():
    data = np.asarray([[-1, 0, 1, 2], [5, -2, 3, -4], [0, 1, -1, 0]], dtype=float)
    weights = np.asarray([[1, 2, -1], [-3, -4, 10]], dtype=float)
    activations, inverse, oriented_weights = posact(data, weights)
    expected_activations = np.asarray([[9, -5, 8, -6], [17, -18, 25, -10]], dtype=float)
    expected_inverse = np.linalg.pinv(weights)
    expected_inverse[:, 1] *= -1
    expected_weights = np.asarray([[1, 2, -1], [3, 4, -10]], dtype=float)
    np.testing.assert_allclose(activations, expected_activations, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(inverse, expected_inverse, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(oriented_weights, expected_weights, rtol=1e-14, atol=1e-14)


def test_python_regression_posact_current_suite_explicit_sphere_exact_outputs():
    data = np.asarray([[-1, 0, 1, 2], [5, -2, 3, -4], [0, 1, -1, 0]], dtype=float)
    weights = np.asarray([[1, 2, -1], [-3, -4, 10]], dtype=float)
    sphere = np.asarray([[-1, 0, 1], [2, 3, -5], [10, -6, 4]], dtype=float)
    activations, inverse, oriented_weights = posact(data, weights, sphere)
    np.testing.assert_allclose(activations, [[67, -37, 42, -62], [-455, 201, -178, 478]])
    np.testing.assert_allclose(inverse, np.linalg.pinv(finite_matmul(weights, sphere)))
    np.testing.assert_array_equal(oriented_weights, weights)


def test_python_regression_posact_current_suite_square_weights_and_all_negative_component():
    data = np.asarray([[-1, 0, 1, 2], [5, -2, 3, -4], [0, 1, -1, 0]], dtype=float)
    weights = np.asarray([[1, 2, -1], [-3, -4, 10], [5, 6, 7]], dtype=float)
    activations, inverse, oriented_weights = posact(data, weights)
    np.testing.assert_allclose(
        activations,
        [[9, -5, 8, -6], [17, -18, 25, -10], [25, -5, 16, -14]],
    )
    np.testing.assert_allclose(finite_matmul(inverse, activations), data, rtol=1e-13, atol=1e-13)
    np.testing.assert_array_equal(oriented_weights[1], -weights[1])

    all_negative, _all_negative_inverse, all_negative_weights = posact(
        np.ones((2, 5)),
        -np.eye(2),
    )
    np.testing.assert_array_equal(all_negative, np.ones((2, 5)))
    np.testing.assert_array_equal(all_negative_weights, np.eye(2))


def _small_ica_data() -> np.ndarray:
    rng = np.random.default_rng(18)
    sources = np.vstack((rng.laplace(size=600), rng.uniform(-2, 2, size=600), rng.normal(size=600) ** 3))
    return finite_matmul(
        np.asarray([[1.0, 0.2, -0.3], [0.1, 1.0, 0.4], [-0.2, 0.3, 1.0]]),
        sources,
    )


def _assert_variant_matches_runica(variant, defaults: dict) -> None:
    data = _small_ica_data()
    options = {"seed": 29, "maxsteps": 3, "verbose": "off"}
    expected = runica(data, **defaults, **options)
    actual = variant(data, **options)
    assert len(actual) == len(expected) == 6
    for actual_value, expected_value in zip(actual, expected):
        np.testing.assert_allclose(actual_value, expected_value, rtol=0, atol=0)


def test_python_regression_runica_ml2_current_suite_corrects_miswired_upstream_call():
    # The MATLAB method named test_runica_ml2 accidentally calls runica rather
    # than runica_ml2. This corrected port exercises the named compatibility API.
    _assert_variant_matches_runica(
        runica_ml2,
        {"extended": 1, "posact": "on", "bias": "off", "anneal": 0.95},
    )


def test_python_regression_runica_mlb_current_suite_delegates_to_qualified_engine():
    _assert_variant_matches_runica(
        runica_mlb,
        {"extended": 1, "posact": "on", "bias": "on", "anneal": 0.98},
    )


@eeglab_test(f"{SIGPROC_ROOT}/kmeanscluster/sigprocfunc_kmeanscluster_wrapperTest.m", "test_test_kmeanscluster")
def test_reference_kmeanscluster(eeglab_backend):
    data = np.array([[1, 1], [2, 1], [4, 3], [5, 4]], dtype=float)
    groups, _centers, _previous, _data = eeglab_backend("kmeanscluster", data, 2.0, nargout=4)
    _assert_near(groups, [[1.0], [1.0], [2.0], [2.0]])
    # The source's second call checks execution only, with defaults unchanged.
    eeglab_backend("kmeanscluster", np.random.default_rng(12).standard_normal((100, 20)), 5.0, nargout=4)


@eeglab_test(f"{SIGPROC_ROOT}/posact/sigprocfunc_posact_wrapperTest.m", "test_pass_general")
@eeglab_test(f"{SIGPROC_ROOT}/posact/sigprocfunc_posact_wrapperTest.m", "test_pass_weights_symmetric")
def test_reference_posact_default_sphere(eeglab_backend):
    data = np.array([[-1, 0, 1, 2], [5, -2, 3, -4], [0, 1, -1, 0]], dtype=float)
    all_weights = np.array([[1, 2, -1], [-3, -4, 10], [5, 6, 7]], dtype=float)
    all_activations = np.array([[9, -5, 8, -6], [17, -18, 25, -10], [25, -5, 16, -14]], dtype=float)
    for components in (2, 3):
        weights = all_weights[:components].copy()
        expected_inverse = np.linalg.pinv(weights)
        expected_inverse[:, 1] *= -1
        expected_weights = weights.copy()
        expected_weights[1] *= -1
        activation, inverse, actual_weights = eeglab_backend("posact", data, weights, nargout=3)
        _assert_near(activation, all_activations[:components])
        _assert_near(inverse, expected_inverse)
        _assert_near(actual_weights, expected_weights)


@eeglab_test(f"{SIGPROC_ROOT}/posact/sigprocfunc_posact_wrapperTest.m", "test_pass_sphere")
def test_reference_posact_sphere(eeglab_backend):
    data = np.array([[-1, 0, 1, 2], [5, -2, 3, -4], [0, 1, -1, 0]], dtype=float)
    weights = np.array([[1, 2, -1], [-3, -4, 10]], dtype=float)
    sphere = np.array([[-1, 0, 1], [2, 3, -5], [10, -6, 4]], dtype=float)
    activation, inverse, actual_weights = eeglab_backend("posact", data, weights, sphere, nargout=3)
    _assert_near(activation, [[67, -37, 42, -62], [-455, 201, -178, 478]])
    _assert_near(inverse, np.linalg.pinv(weights @ sphere))
    _assert_near(actual_weights, weights)


def _reference_ica_recording(eeglab_backend, eeglab_suite_root):
    eeg = eeglab_backend("pop_loadset", str(eeglab_suite_root / "eeglab/sample_data/eeglab_data.set"))
    eeg["pnts"] = 10000.0
    eeg["data"] = eeg["data"][:, :10000]
    return eeglab_backend("eeg_checkset", eeg)


@eeglab_test(f"{SIGPROC_ROOT}/runica_ml2/sigprocfunc_runica_ml2_wrapperTest.m", "test_test_runica_ml2")
def test_reference_runica_ml2_source_calls_runica(eeglab_backend, eeglab_suite_root):
    # Preserve the source call despite its misleading name; the Python-specific
    # runica_ml2 regression above is not a translation of this MATLAB scenario.
    eeg = _reference_ica_recording(eeglab_backend, eeglab_suite_root)
    eeglab_backend("runica", eeg["data"], nargout=2)


@eeglab_test(f"{SIGPROC_ROOT}/runica_mlb/sigprocfunc_runica_mlb_wrapperTest.m", "test_test_runica_mlb")
def test_reference_runica_mlb(eeglab_backend, eeglab_suite_root):
    eeg = _reference_ica_recording(eeglab_backend, eeglab_suite_root)
    eeglab_backend("runica_mlb", eeg["data"], nargout=2)
