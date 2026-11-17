"""EEGLAB-style help browser for pop-function dialogs."""

from __future__ import annotations

from importlib import resources
from importlib.resources.abc import Traversable
import re
from typing import Any

from eegprep.extension_runtime import ExtensionRuntime
from eegprep.extensions import ExtensionResource


HELP_PACKAGE = "eegprep.resources.help"


def pophelp(
    function_name: str,
    nonmatlab: bool = False,
    parent: Any | None = None,
    extension_runtime: ExtensionRuntime | None = None,
) -> Any:
    """Open an EEGLAB-like help browser for a function."""
    try:
        from PySide6 import QtWidgets
    except ImportError as exc:  # pragma: no cover - optional GUI dependency
        raise RuntimeError(
            "PySide6 is required for EEGPrep GUI help. Install it with "
            "`pip install -e .[gui]` or `pip install eegprep[gui]`."
        ) from exc

    function_name = _normalise_function_name(function_name)
    text, _source_path = pophelp_text(function_name, nonmatlab=nonmatlab, extension_runtime=extension_runtime)
    title = f"{function_name} - {function_name.upper()}"
    dialog = QtWidgets.QDialog(parent)
    dialog.setObjectName("pophelp")
    dialog.setWindowTitle(title)
    dialog.resize(720, 520)
    layout = QtWidgets.QVBoxLayout(dialog)
    browser = QtWidgets.QTextBrowser()
    browser.setObjectName("help_browser")
    browser.setOpenExternalLinks(True)
    markdown = text if text.lstrip().startswith("#") else f"# {text}"
    browser.setMarkdown(markdown)
    layout.addWidget(browser)
    button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
    button_box.rejected.connect(dialog.reject)
    button_box.accepted.connect(dialog.accept)
    layout.addWidget(button_box)
    dialog.show()
    dialog.raise_()
    return dialog


def pophelp_text(
    function_name: str,
    nonmatlab: bool = False,
    extension_runtime: ExtensionRuntime | None = None,
) -> tuple[str, str]:
    """Return EEGLAB-style help text and source path for ``function_name``."""
    function_name = _normalise_function_name(function_name)
    source_path = _find_source(function_name, extension_runtime=extension_runtime)
    doc = _read_help_source(source_path, nonmatlab=nonmatlab)
    if function_name.startswith("pop_"):
        called_name = function_name[4:]
        called_source = _find_source(called_name, missing_ok=True, extension_runtime=extension_runtime)
        if called_source is not None:
            called_doc = _read_help_source(called_source, nonmatlab=False)
            doc = _append_called_help(doc, called_doc)
    return doc, _resource_source_name(source_path)


def _normalise_function_name(function_name: str) -> str:
    function_name = str(function_name).strip()
    match = re.fullmatch(r"pophelp\(['\"]([^'\"]+)['\"]\)", function_name)
    if match:
        function_name = match.group(1)
    for suffix in (".md", ".m"):
        if function_name.endswith(suffix):
            function_name = function_name[: -len(suffix)]
    return function_name


def _find_source(
    function_name: str,
    *,
    missing_ok: bool = False,
    extension_runtime: ExtensionRuntime | None = None,
) -> Traversable | ExtensionResource | None:
    direct = resources.files(HELP_PACKAGE).joinpath(f"{function_name}.md")
    if direct.is_file():
        return direct
    if extension_runtime is not None:
        extension_resource = extension_runtime.help_resource(function_name)
        if extension_resource is not None:
            return extension_resource
    if missing_ok:
        return None
    raise FileNotFoundError(
        "Missing packaged EEGPrep help resource for "
        f"{function_name!r}. Add a help snapshot under "
        "src/eegprep/resources/help/."
    )


def _read_help_source(path: Traversable | ExtensionResource | None, *, nonmatlab: bool) -> str:
    if path is None:
        return ""
    if isinstance(path, ExtensionResource):
        function_name = path.path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        return path.read_text(encoding="utf-8").strip() or f"No help found for {function_name}."
    function_name = path.name.rsplit(".", 1)[0]
    return path.read_text(encoding="utf-8").strip() or f"No help found for {function_name}."


def _resource_source_name(path: Traversable | ExtensionResource) -> str:
    if isinstance(path, ExtensionResource):
        return f"{path.package}:{path.path}"
    return f"eegprep/resources/help/{path.name}"


def _append_called_help(doc: str, called_doc: str) -> str:
    return "\n".join(
        [
            doc,
            "",
            "___________________________________________________________________",
            "",
            " The 'pop' function above calls the lower-level function below",
            " and could use some of its optional parameters",
            "___________________________________________________________________",
            "",
            called_doc,
        ]
    )
