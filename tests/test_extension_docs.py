"""Checks for external extension docs and agent skill references."""

from __future__ import annotations

import importlib
from pathlib import Path

import eegprep
from eegprep import extensions

REPO_ROOT = Path(__file__).resolve().parents[1]
DOC_PATHS = (
    REPO_ROOT / "docs/source/api/extensions.rst",
    REPO_ROOT / "docs/source/user_guide/extensions.rst",
    REPO_ROOT / ".agents/skills/eegprep-extension-development/SKILL.md",
)

SDK_SYMBOLS = (
    "ExtensionAction",
    "ExtensionDependency",
    "ExtensionMenu",
    "ExtensionPopFunction",
    "ExtensionRegistry",
    "ExtensionResource",
    "ExtensionSpec",
    "ExtensionStatus",
    "LazyImport",
    "discover_extensions",
    "validate_extension_spec",
)

INSTALL_COMMANDS = (
    "uv add --editable /path/to/eegprep-ext-foo",
    "uv add git+https://github.com/lab/eegprep-ext-foo",
    "uv add eegprep-ext-foo",
    "uv add --index https://packages.lab.example/simple eegprep-ext-foo",
    "uv add --default-index https://packages.lab.example/simple eegprep-ext-foo",
)


def test_extension_docs_and_skill_reference_importable_sdk_symbols() -> None:
    text = _extension_text()
    sdk_module = importlib.import_module("eegprep.extensions")

    assert extensions.EXTENSION_ENTRY_POINT_GROUP == "eegprep.extensions"
    assert "eegprep.extensions" in text
    for symbol in SDK_SYMBOLS:
        assert hasattr(sdk_module, symbol), symbol
        assert hasattr(eegprep, symbol), symbol
        assert symbol in text


def test_extension_docs_cover_supported_install_paths() -> None:
    text = _extension_text()

    for command in INSTALL_COMMANDS:
        assert command in text


def test_extension_docs_cover_registry_status_language() -> None:
    text = _extension_text()

    for status in extensions.ExtensionStatus:
        assert status.value in text
    assert "scientific endorsement" in text
    assert "does not host extension zips" in text
    assert "does not host arbitrary extension zip files" in text


def _extension_text() -> str:
    return "\n".join(path.read_text(encoding="utf-8") for path in DOC_PATHS)
