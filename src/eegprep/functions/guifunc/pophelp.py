"""EEGLAB-style help browser for pop-function dialogs."""

from __future__ import annotations

import html
import importlib
import inspect
import re
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
EEGLAB_ROOTS = (
    PACKAGE_ROOT / "resources" / "eeglab",
    PACKAGE_ROOT / "eeglab",
)
PYTHON_HELP_MODULE_HINTS = (
    "popfunc",
    "sigprocfunc",
    "adminfunc",
    "miscfunc",
    "eegobj",
    "guifunc",
)


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
    source_path = _find_source(function_name, missing_ok=True)
    if source_path is not None:
        doc = _read_help_source(source_path, nonmatlab=nonmatlab)
        if function_name.startswith("pop_"):
            called_name = function_name[4:]
            called_source = _find_source(called_name, missing_ok=True)
            if called_source is not None:
                called_doc = _read_help_source(called_source, nonmatlab=False)
                doc = _append_called_help(doc, called_doc)
        return doc, str(source_path)

    doc, source_label = _read_python_help(function_name, nonmatlab=nonmatlab, missing_ok=False)
    if function_name.startswith("pop_"):
        called_name = function_name[4:]
        called_doc, _called_source = _read_python_help(called_name, nonmatlab=False, missing_ok=True)
        if called_doc:
            doc = _append_called_help(doc, called_doc)
    return doc, source_label


def _normalise_function_name(function_name: str) -> str:
    function_name = str(function_name).strip()
    match = re.fullmatch(r"pophelp\(['\"]([^'\"]+)['\"]\)", function_name)
    if match:
        function_name = match.group(1)
    if function_name.endswith(".m"):
        function_name = function_name[:-2]
    return function_name


def _find_source(function_name: str, *, missing_ok: bool = False) -> Path | None:
    for eeglab_root in EEGLAB_ROOTS:
        if not eeglab_root.exists():
            continue
        direct = eeglab_root / f"{function_name}.m"
        if direct.exists():
            return direct
        matches = sorted(eeglab_root.rglob(f"{function_name}.m"))
        if matches:
            return matches[0]
    if missing_ok:
        return None
    raise FileNotFoundError(f"Could not find EEGLAB help source for {function_name!r}")


def _read_help_source(path: Path | None, *, nonmatlab: bool) -> str:
    if path is None:
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    doc_lines = []
    for line in lines:
        if nonmatlab:
            doc_lines.append(line)
            continue
        if not line.startswith("%"):
            break
        text = line[1:]
        if text.startswith(" "):
            text = text[1:]
        doc_lines.append(text.rstrip())
    return "\n".join(doc_lines).strip() or f"No help found for {path.stem}."


def _append_called_help(doc: str, called_doc: str) -> str:
    return "\n".join(
        [
            doc,
            "",
            "___________________________________________________________________",
            "",
            " The 'pop' function above calls the eponymous Matlab function below",
            " and could use some of its optional parameters",
            "___________________________________________________________________",
            "",
            called_doc,
        ]
    )


def _read_python_help(function_name: str, *, nonmatlab: bool, missing_ok: bool) -> tuple[str, str]:
    for module_name in _python_module_names(function_name):
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue

        target = getattr(module, function_name, None)
        source_obj = target if target is not None else module
        source_file = inspect.getsourcefile(source_obj) or inspect.getsourcefile(module)
        if source_file is not None:
            source_label = str(Path(source_file).resolve())
        else:
            source_label = f"python:{module_name}"

        if nonmatlab and source_file is not None:
            text = Path(source_file).read_text(encoding="utf-8", errors="replace").strip()
        else:
            text = inspect.getdoc(source_obj) or inspect.getdoc(module) or ""
        if text:
            return text, source_label
        return f"No help found for {function_name}.", source_label

    if missing_ok:
        return "", ""
    raise FileNotFoundError(f"Could not find EEGLAB or Python help source for {function_name!r}")


def _python_module_names(function_name: str) -> tuple[str, ...]:
    return tuple(f"eegprep.functions.{group}.{function_name}" for group in PYTHON_HELP_MODULE_HINTS)


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
