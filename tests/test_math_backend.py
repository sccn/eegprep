from __future__ import annotations

from eegprep.utils import math_backend


def test_math_backend_info_separates_numpy_build_and_loaded_libraries(monkeypatch):
    monkeypatch.setattr(
        math_backend.np.__config__,
        "CONFIG",
        {
            "Build Dependencies": {
                "blas": {"name": "accelerate", "found": True},
                "lapack": {"name": "accelerate", "found": True},
            }
        },
    )
    monkeypatch.setattr(
        math_backend,
        "threadpool_info",
        lambda: [{"user_api": "openmp", "internal_api": "openmp", "num_threads": 4}],
    )
    monkeypatch.setenv("OMP_NUM_THREADS", "4")

    info = math_backend.get_math_backend_info()

    assert info["numpy_build_dependencies"] == {
        "blas": {"name": "accelerate", "found": True},
        "lapack": {"name": "accelerate", "found": True},
    }
    assert info["loaded_libraries"] == [{"user_api": "openmp", "internal_api": "openmp", "num_threads": 4}]
    assert info["thread_environment"]["OMP_NUM_THREADS"] == "4"
    assert info["collection_errors"] == []


def test_math_backend_info_is_non_fatal_when_threadpool_inspection_fails(monkeypatch):
    def fail() -> list[dict[str, object]]:
        raise RuntimeError("inspection unavailable")

    monkeypatch.setattr(math_backend, "threadpool_info", fail)

    info = math_backend.get_math_backend_info()

    assert info["loaded_libraries"] == []
    assert info["collection_errors"] == ["threadpoolctl: RuntimeError: inspection unavailable"]


def test_math_backend_info_is_non_fatal_when_numpy_config_inspection_fails(monkeypatch):
    def fail() -> dict[str, object]:
        raise ValueError("bad build metadata")

    monkeypatch.setattr(math_backend, "_numpy_build_dependencies", fail)
    monkeypatch.setattr(math_backend, "threadpool_info", lambda: [])

    info = math_backend.get_math_backend_info()

    assert info["numpy_build_dependencies"] == {}
    assert info["collection_errors"] == ["numpy config: ValueError: bad build metadata"]


def test_numpy_build_dependencies_supports_legacy_numpy_config(monkeypatch):
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
