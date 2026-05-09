"""EEGLAB-style help browser for pop-function dialogs."""

from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
HELP_ROOT = PACKAGE_ROOT / "resources" / "help"


def pophelp(function_name: str, nonmatlab: bool = False, parent: Any | None = None) -> Any:
    """Open an EEGLAB-like help browser for a function."""
    try:
        from PySide6 import QtWidgets
    except ImportError as exc:  # pragma: no cover - optional GUI dependency
        raise RuntimeError(
            "PySide6 is required for EEGPrep GUI help. Install it with "
            "`pip install -e .[gui]` or `pip install eegprep[gui]`."
        ) from exc

    function_name = _normalise_function_name(function_name)
    text, source_path = pophelp_text(function_name, nonmatlab=nonmatlab)
    title = f"{function_name} - {function_name.upper()}"
    dialog = QtWidgets.QDialog(parent)
    dialog.setObjectName("pophelp")
    dialog.setWindowTitle(title)
    dialog.resize(720, 520)
    layout = QtWidgets.QVBoxLayout(dialog)
    browser = QtWidgets.QTextBrowser()
    browser.setObjectName("help_browser")
    browser.setOpenExternalLinks(True)
    browser.setHtml(_help_html(function_name, source_path, text))
    layout.addWidget(browser)
    button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
    button_box.rejected.connect(dialog.reject)
    button_box.accepted.connect(dialog.accept)
    layout.addWidget(button_box)
    dialog.show()
    dialog.raise_()
    return dialog


def pophelp_text(function_name: str, nonmatlab: bool = False) -> tuple[str, str]:
    """Return EEGLAB-style help text and source path for ``function_name``."""
    function_name = _normalise_function_name(function_name)
    source_path = _find_source(function_name)
    doc = _read_help_source(source_path, nonmatlab=nonmatlab)
    if function_name.startswith("pop_"):
        called_name = function_name[4:]
        called_source = _find_source(called_name, missing_ok=True)
        if called_source is not None:
            called_doc = _read_help_source(called_source, nonmatlab=False)
            doc = _append_called_help(doc, called_doc)
    return doc, str(source_path)


def _normalise_function_name(function_name: str) -> str:
    function_name = str(function_name).strip()
    match = re.fullmatch(r"pophelp\(['\"]([^'\"]+)['\"]\)", function_name)
    if match:
        function_name = match.group(1)
    for suffix in (".md", ".m"):
        if function_name.endswith(suffix):
            function_name = function_name[: -len(suffix)]
    return function_name


def _find_source(function_name: str, *, missing_ok: bool = False) -> Path | None:
    direct = HELP_ROOT / f"{function_name}.md"
    if direct.exists():
        return direct
    if missing_ok:
        return None
    raise FileNotFoundError(
        "Missing packaged EEGPrep help resource for "
        f"{function_name!r}. Add a help snapshot under "
        "src/eegprep/resources/help/."
    )


def _read_help_source(path: Path | None, *, nonmatlab: bool) -> str:
    if path is None:
        return ""
    return path.read_text(encoding="utf-8", errors="replace").strip() or f"No help found for {path.stem}."


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


def _help_html(function_name: str, source_path: str, text: str) -> str:
    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
html, body {{ height: 100%; }}
body {{
  margin: 16px;
  font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
  line-height: 1.4;
  font-size: 13px;
}}
h1 {{ margin: 0 0 8px 0; font-size: 16px; }}
.h1file {{ color: #666; margin: 0 0 12px 0; font-size: 12px; }}
pre {{
  white-space: pre-wrap;
  word-wrap: break-word;
  margin: 0;
  font-family: Consolas, Menlo, Monaco, monospace;
  font-size: 12px;
}}
</style>
</head>
<body>
<h1>{html.escape(function_name.upper())}</h1>
<div class="h1file">{html.escape(source_path)}</div>
<pre>{html.escape(text)}</pre>
</body>
</html>"""
