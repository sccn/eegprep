from __future__ import annotations

import numpy as np

from eegprep.utils import math_backend


def test_math_backend_info_reports_real_build_and_runtime_state(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    monkeypatch.delenv("MKL_NUM_THREADS", raising=False)

    info = math_backend.get_math_backend_info()

    assert info["numpy_version"] == np.__version__
    assert set(info["numpy_build_dependencies"]) == {"blas", "lapack"}
    assert all("name" in dep for dep in info["numpy_build_dependencies"].values())
    assert isinstance(info["loaded_libraries"], list)
    for library in info["loaded_libraries"]:
        assert {"internal_api", "num_threads"} <= set(library)
    assert info["thread_environment"] == {"OMP_NUM_THREADS": "3"}
    assert info["collection_errors"] == []


def test_numpy_build_dependencies_supports_legacy_numpy_config(monkeypatch):
    # numpy < 2 exposes get_info() instead of CONFIG; not reachable with the installed NumPy.
    monkeypatch.setattr(math_backend.np.__config__, "CONFIG", {}, raising=False)
    monkeypatch.setattr(
        math_backend.np.__config__,
        "get_info",
        lambda name: {"libraries": ["openblas"]} if name == "blas_opt_info" else {},
        raising=False,
    )

    assert math_backend._numpy_build_dependencies() == {
        "blas": {"libraries": ["openblas"]},
        "lapack": {},
    }
