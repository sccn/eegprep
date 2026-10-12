"""Fail CI when the ICLabel backend tests would be skipped silently.

The ICLabel runtime and model-regeneration tests skip when onnx, onnxruntime, or torch
cannot be imported. CI test jobs install every extra, so a missing backend there means
those tests stopped running without a failure. The base-install job runs only
tests/test_optional_extras.py and never collects this file.
"""

import importlib
import os

import pytest

_CI_BACKENDS = ("onnx", "onnxruntime", "torch")


@pytest.mark.skipif(not os.environ.get("CI"), reason="only CI test jobs install every optional extra")
@pytest.mark.parametrize("module", _CI_BACKENDS)
def test_ci_installs_the_iclabel_backends(module):
    try:
        importlib.import_module(module)
    except ImportError as exc:
        pytest.fail(f"CI must install {module} (uv sync --all-extras); the tests that need it would skip: {exc}")
