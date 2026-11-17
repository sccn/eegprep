"""Render ``eegprep.help()`` from source, without importing the named modules."""

from __future__ import annotations

import ast
import functools
import re
from pathlib import Path

from eegprep.extras import EXTRA_SUMMARY, PUBLIC_EXTRAS, install_command

_PACKAGE_ROOT = Path(__file__).resolve().parent

_ROLE = re.compile(r":[A-Za-z]+:`~?\.?")

_GROUPS: tuple[tuple[str, str], ...] = (
    ("eegprep.functions.popfunc", "Dataset functions"),
    ("eegprep.functions.sigprocfunc", "Signal processing"),
    ("eegprep.plugins.firfilt", "FIR filtering"),
    ("eegprep.plugins.clean_rawdata", "Artifact cleaning"),
    ("eegprep.functions.timefreqfunc", "Time-frequency"),
    ("eegprep.functions.statistics", "Statistics"),
    ("eegprep.functions.studyfunc", "Study"),
    ("eegprep.plugins.EEG_BIDS", "BIDS"),
    ("eegprep.plugins.ICLabel", "Component classification"),
    ("eegprep.plugins.dipfit", "Dipole fitting"),
    ("eegprep.functions.guifunc", "GUI"),
    ("eegprep.functions.adminfunc", "Administration and session"),
    ("eegprep.functions.miscfunc", "Utilities"),
    ("eegprep.functions.eegobj", "EEG object"),
    ("eegprep.functions.redefine_functions", "Short names"),
    ("eegprep.extensions", "Extensions"),
    ("eegprep.extension_catalog", "Extensions"),
    ("eegprep.extension_catalog_validation", "Extensions"),
    ("eegprep.extension_testing", "Extensions"),
)


def render_help(lazy_exports: dict[str, tuple[str, str]], version: str) -> str:
    """Return the text ``eegprep.help()`` prints."""
    exports = tuple((name, module, attr) for name, (module, attr) in lazy_exports.items())
    return _cached_help(version, exports)


@functools.cache
def _cached_help(version: str, exports: tuple[tuple[str, str, str], ...]) -> str:
    return _render_help({name: (module, attr) for name, module, attr in exports}, version)


def _render_help(lazy_exports: dict[str, tuple[str, str]], version: str) -> str:
    titles: list[str] = []
    for _, title in _GROUPS:
        if title not in titles:
            titles.append(title)
    grouped: dict[str, list[tuple[str, str, str]]] = {title: [] for title in titles}
    for name, (module_name, attr_name) in lazy_exports.items():
        title = _group(module_name)
        if title not in grouped:
            raise RuntimeError(f"help() has no group for {name} ({module_name})")
        node = _find_definition(module_name, attr_name, frozenset())
        label = f"{name}()" if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else name
        summary = ""
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            summary = _one_line(ast.get_docstring(node))
        grouped[title].append((label, summary, name))

    labels = [label for entries in grouped.values() for label, _, _ in entries]
    labels.append("__version__")
    width = max(len(label) for label in labels)
    lines = [
        f"eegprep {version}",
        "",
        "Base install: pip install eegprep",
        "Names below come from that install. A bracket names an extra one path of that function needs.",
        "Summaries are the first line of each docstring and do not import the module.",
        "",
        f"  {'__version__':<{width}}  package version",
    ]
    for title in titles:
        entries = grouped[title]
        if not entries:
            continue
        lines.append("")
        lines.append(title)
        for label, summary, name in sorted(entries, key=lambda item: item[2].lower()):
            lines.append(f"  {label:<{width}}  {summary}{_extra_suffix(name)}")

    lines.extend(["", "Extras"])
    for extra, summary in EXTRA_SUMMARY.items():
        lines.append(f"  {extra:<8}  {summary}  {install_command(extra)}")
    lines.extend(
        [
            "",
            "help(eegprep.<name>) prints the full docstring. A name behind an extra needs that extra installed.",
            "from eegprep import * does not replace the builtin help.",
        ]
    )
    return "\n".join(lines)


def _group(module_name: str) -> str:
    for prefix, title in _GROUPS:
        if module_name == prefix or module_name.startswith(prefix + "."):
            return title
    return module_name


def _extra_suffix(name: str) -> str:
    extra = PUBLIC_EXTRAS.get(name)
    if extra is None:
        return ""
    extra_name, detail = extra
    return f"  [{extra_name} extra: {detail}; {install_command(extra_name)}]"


def _one_line(doc: str | None) -> str:
    if not doc:
        return ""
    line = _ROLE.sub("", doc.strip().splitlines()[0].strip()).replace("`", "")
    return " ".join(line.split())


@functools.cache
def _parsed(module_name: str) -> ast.Module | None:
    path = _module_path(module_name)
    if path is None:
        return None
    return ast.parse(path.read_text(encoding="utf-8"))


def _module_path(module_name: str) -> Path | None:
    if not module_name.startswith("eegprep."):
        return None
    relative = Path(*module_name.split(".")[1:])
    package_root = _PACKAGE_ROOT
    candidate = package_root / relative.with_suffix(".py")
    if candidate.is_file():
        return candidate
    init = package_root / relative / "__init__.py"
    if init.is_file():
        return init
    return None


def _find_definition(module_name: str, name: str, seen: frozenset[tuple[str, str]]) -> ast.AST | None:
    key = (module_name, name)
    if key in seen:
        return None
    tree = _parsed(module_name)
    if tree is None:
        return None
    seen = seen | {key}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name:
            return node
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == name:
            return node
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return node
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if (alias.asname or alias.name) != name or alias.name == "*":
                    continue
                imported = _imported_module(module_name, node)
                if imported is None:
                    return node
                return _find_definition(imported, alias.name, seen) or node
    return None


def _imported_module(current: str, node: ast.ImportFrom) -> str | None:
    if node.level:
        parts = current.split(".")
        base = parts[: len(parts) - node.level]
        if node.module:
            base.extend(node.module.split("."))
        return ".".join(base) if base else None
    if node.module and node.module.startswith("eegprep"):
        return node.module
    return None
