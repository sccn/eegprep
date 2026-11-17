"""Keep the CI and release workflows consistent with the rest of the repository.

No other module reads ``test.yml`` or ``release.yml``. These tests read them as data (and
``uv.lock`` beside them) so a change that breaks one of the rules below fails here, in the
fast suite, instead of in a release run or after a secret has already leaked to a job.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
RELEASE_WORKFLOW = WORKFLOWS / "release.yml"
UV_LOCK = REPO_ROOT / "uv.lock"


def _locked_linux_onnxruntime() -> str:
    """Version of the ``uv.lock`` onnxruntime entry that has a manylinux x86_64 wheel.

    The lock holds more than one onnxruntime entry once an Intel Mac marker splits the
    ``dev`` group, so the entry is chosen by its Linux wheel, not by its position.
    """
    lock = tomllib.loads(UV_LOCK.read_text(encoding="utf-8"))
    versions = [
        package["version"]
        for package in lock["package"]
        if package["name"] == "onnxruntime"
        and any("manylinux_2_28_x86_64" in wheel["url"] for wheel in package.get("wheels", []))
    ]
    assert len(versions) == 1, f"expected one Linux onnxruntime entry in uv.lock, got {versions}"
    return versions[0]


def test_release_smoke_tests_pin_the_locked_onnxruntime() -> None:
    """The wheel and sdist smoke tests run the onnxruntime the test suite validated."""
    pins = re.findall(r"onnxruntime==([0-9][0-9A-Za-z.]*)", RELEASE_WORKFLOW.read_text(encoding="utf-8"))
    assert len(pins) == 2, f"expected a pin in each of the wheel and sdist smoke tests, got {pins}"
    locked = _locked_linux_onnxruntime()
    assert pins == [locked, locked], f"release.yml pins {pins} but uv.lock resolves onnxruntime {locked}"


def test_release_checks_the_full_precision_network_against_the_report() -> None:
    """``release.yml`` compares ``iclabel.onnx`` with ``float32_model.sha256`` in the report."""
    text = RELEASE_WORKFLOW.read_text(encoding="utf-8")
    assert '["float32_model"]["sha256"]' in text, "release.yml does not read float32_model.sha256 from the report"
    hashed = re.search(r"sha256sum src/eegprep/plugins/ICLabel/iclabel\.onnx\b", text)
    assert hashed, "release.yml does not hash src/eegprep/plugins/ICLabel/iclabel.onnx"
    # The comparison must fail the build, mirroring the int8 check beside it.
    check = text[hashed.start() :]
    assert re.search(
        r'if \[ "\$float32_model_sha256" != "\$report_float32_sha256" \]; then\s+echo "::error::.*exit 1', check
    ), "release.yml does not fail when iclabel.onnx differs from the report"
