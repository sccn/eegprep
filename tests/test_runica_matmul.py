import importlib
import subprocess
import sys

import numpy as np
import pytest

from eegprep.functions.sigprocfunc import runica_matmul


runica_module = importlib.import_module("eegprep.functions.sigprocfunc.runica")


def test_runica_matmul_backends_agree():
    rng = np.random.default_rng(376)
    left = rng.standard_normal((7, 11))
    right = rng.standard_normal((11, 5))

    np.testing.assert_allclose(
        runica_matmul._blas_matmul(left, right),
        runica_matmul._numpy_matmul(left, right),
        rtol=1e-12,
        atol=1e-12,
    )


def test_runica_runs_its_training_products_through_dgemm_on_emscripten(monkeypatch):
    original_platform = sys.platform
    monkeypatch.setattr(sys, "platform", "emscripten")
    try:
        selected = importlib.reload(runica_matmul)
        consumer = importlib.reload(runica_module)
        backend, dgemm = selected.runica_matmul, selected._dgemm
        calls = {"backend": 0, "dgemm": 0}

        def count_backend(left, right):
            calls["backend"] += 1
            return backend(left, right)

        def count_dgemm(**kwargs):
            calls["dgemm"] += 1
            return dgemm(**kwargs)

        # Patching after the reload also shows that runica looks the backend up at call time.
        monkeypatch.setattr(selected, "runica_matmul", count_backend)
        monkeypatch.setattr(selected, "_dgemm", count_dgemm)
        data = np.random.default_rng(385).standard_normal((3, 100))
        consumer.runica(data, maxsteps=1, verbose=False, rndreset="off")
        assert calls["backend"] > 0
        assert calls["dgemm"] == calls["backend"]
    finally:
        monkeypatch.setattr(sys, "platform", original_platform)
        importlib.reload(runica_matmul)
        importlib.reload(runica_module)


@pytest.mark.parametrize("extended", [0, 1], ids=["logistic", "extended"])
def test_runica_blas_backend_matches_numpy_backend(monkeypatch, extended):
    """The dgemm backend used under Emscripten must reproduce the NumPy backend's decomposition.

    Both backends run the real runica training loop on the same Laplace-distributed sources;
    maxsteps stays small so the check is quick while still exercising weight updates. The
    extended loop also runs the backend in its kurtosis-based sign estimation.
    """
    rng = np.random.default_rng(386)
    data = rng.laplace(size=(4, 4000))
    kwargs = {"maxsteps": 30, "rndreset": "off", "verbose": False, "extended": extended}

    monkeypatch.setattr(runica_matmul, "runica_matmul", runica_matmul._numpy_matmul)
    weights_numpy, sphere_numpy, *_ = runica_module.runica(data.copy(), **kwargs)
    monkeypatch.setattr(runica_matmul, "runica_matmul", runica_matmul._blas_matmul)
    weights_blas, sphere_blas, *_ = runica_module.runica(data.copy(), **kwargs)

    np.testing.assert_allclose(weights_blas, weights_numpy, rtol=1e-10, atol=1e-12)
    np.testing.assert_array_equal(sphere_blas, sphere_numpy)


def test_runica_import_does_not_eagerly_load_mne():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            ("import sys; from eegprep.functions.sigprocfunc.runica import runica; assert 'mne' not in sys.modules"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
