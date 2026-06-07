from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "src/eegprep"

FORBIDDEN_PATTERNS = (
    ("vendored reference literal", re.compile(r"src/eegprep/eeglab")),
    (
        "package-root eeglab path join",
        re.compile(r"PACKAGE_ROOT[^\n]*(?:os\.path\.join|\bjoinpath\b|/)[^\n]*['\"]eeglab['\"]"),
    ),
    ("eegprep.eeglab import", re.compile(r"(?:from|import)\s+eegprep\.eeglab\b")),
    (
        "importlib resources vendored tree",
        re.compile(r"resources\.(?:files|open_[a-z_]+)\([^\n]*['\"]eegprep\.eeglab['\"]"),
    ),
)


def test_runtime_package_code_does_not_depend_on_vendored_eeglab_tree() -> None:
    offenders: list[str] = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        if "src/eegprep/eeglab" in path.as_posix():
            continue
        text = path.read_text(encoding="utf-8")
        for label, pattern in FORBIDDEN_PATTERNS:
            if pattern.search(text):
                offenders.append(f"{path.relative_to(REPO_ROOT)}: {label}")

    assert offenders == []
