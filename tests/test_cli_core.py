from __future__ import annotations

import json
import os
import platform

import numpy as np

from eegprep.cli import core


def test_software_info_reports_cpu_and_math_backend():
    info = core.software_info()

    assert info["logical_cpu_count"] == os.cpu_count()
    assert info["architecture"] == platform.machine()
    assert info["math_backend_info"]["numpy_version"] == np.__version__
    assert info["math_backend_info"]["collection_errors"] == []
    json.dumps(core.json_safe(info))
