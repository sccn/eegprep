import importlib
import subprocess
import sys

import numpy as np

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


def test_runica_matmul_selects_numpy_on_native_platform():
    assert runica_matmul.BACKEND == "numpy.matmul"
    assert runica_matmul.runica_matmul is runica_matmul._numpy_matmul


def test_runica_matmul_selects_dgemm_on_emscripten(monkeypatch):
    original_platform = sys.platform
    monkeypatch.setattr(sys, "platform", "emscripten")
    try:
        reloaded = importlib.reload(runica_matmul)
        assert reloaded.BACKEND == "scipy.linalg.blas.dgemm"
        assert reloaded.runica_matmul is reloaded._blas_matmul
    finally:
        monkeypatch.setattr(sys, "platform", original_platform)
        importlib.reload(runica_matmul)


def test_runica_consumer_reads_the_selected_backend_module(monkeypatch):
    original_platform = sys.platform
    monkeypatch.setattr(sys, "platform", "emscripten")
    try:
        selected = importlib.reload(runica_matmul)
        consumer = importlib.reload(runica_module)
        assert consumer._runica_matmul is selected
        assert consumer._runica_matmul.runica_matmul is selected._blas_matmul
    finally:
        monkeypatch.setattr(sys, "platform", original_platform)
        importlib.reload(runica_matmul)
        importlib.reload(runica_module)


def test_runica_executes_the_selected_backend(monkeypatch):
    original_platform = sys.platform
    monkeypatch.setattr(sys, "platform", "emscripten")
    try:
        selected = importlib.reload(runica_matmul)
        consumer = importlib.reload(runica_module)
        calls = []

        def spy(left, right):
            calls.append((left.shape, right.shape))
            return selected._blas_matmul(left, right)

        monkeypatch.setattr(selected, "runica_matmul", spy)
        data = np.random.default_rng(385).standard_normal((3, 100))
        consumer.runica(data, maxsteps=1, verbose=False, rndreset="off")
        assert calls
    finally:
        monkeypatch.setattr(sys, "platform", original_platform)
        importlib.reload(runica_matmul)
        importlib.reload(runica_module)


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
