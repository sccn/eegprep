"""Read-only diagnostics for NumPy math backends and thread pools."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

import numpy as np
from threadpoolctl import threadpool_info


_THREAD_ENVIRONMENT_VARIABLES = (
    "OMP_NUM_THREADS",
    "OMP_DYNAMIC",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "MKL_DYNAMIC",
    "MKL_THREADING_LAYER",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def get_math_backend_info() -> dict[str, Any]:
    """Return factual build and runtime math-library metadata.

    Collection errors are recorded in the result so diagnostics cannot prevent
    a processing command from writing its manifest.
    """
    errors: list[str] = []
    try:
        build_dependencies = _numpy_build_dependencies()
    except Exception as exc:  # Diagnostic collection must not break processing.
        build_dependencies = {}
        errors.append(f"numpy config: {type(exc).__name__}: {exc}")

    try:
        loaded_libraries = [dict(library) for library in threadpool_info()]
    except Exception as exc:  # Third-party runtime inspection is best-effort.
        loaded_libraries = []
        errors.append(f"threadpoolctl: {type(exc).__name__}: {exc}")

    return {
        "numpy_version": np.__version__,
        "numpy_build_dependencies": build_dependencies,
        "loaded_libraries": loaded_libraries,
        "thread_environment": {
            name: value for name in _THREAD_ENVIRONMENT_VARIABLES if (value := os.environ.get(name)) is not None
        },
        "collection_errors": errors,
    }


def _numpy_build_dependencies() -> dict[str, Any]:
    """Read NumPy 2.x config, with a fallback for older supported NumPy."""
    config = getattr(np.__config__, "CONFIG", {})
    if isinstance(config, Mapping):
        dependencies = config.get("Build Dependencies")
        if isinstance(dependencies, Mapping):
            result: dict[str, Any] = {}
            for name in ("blas", "lapack"):
                value = dependencies.get(name, {})
                result[name] = dict(value) if isinstance(value, Mapping) else {}
            return result

    get_info = getattr(np.__config__, "get_info", None)
    if not callable(get_info):
        return {}
    result = {}
    for name in ("blas", "lapack"):
        value = get_info(f"{name}_opt_info")
        result[name] = dict(value) if isinstance(value, Mapping) else {}
    return result
