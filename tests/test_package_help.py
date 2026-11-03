"""``dir(eegprep)`` and ``eegprep.help()`` list the public names without importing them."""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
from pathlib import Path

import eegprep
from eegprep.extras import OPTIONAL_LIBRARIES, PUBLIC_EXTRAS
from eegprep.package_help import render_help

HEAVY_MODULES = ("numpy", "scipy", "mne", "matplotlib", "h5py", "oct2py")
SRC_ROOT = Path(eegprep.__file__).resolve().parent
# pop_fileio reads .mat without mne. This wrapper rejects every other suffix,
# so the static call to pop_fileio is not an mne path.
_FORMAT_BRANCH_FALSE_POSITIVES = frozenset({"pop_fileio_brainvision_mat"})
_PRIVATE_DIR = {"logging", "Any", "setup_logging", "render_help", "importlib"}


def _run_fresh(code: str) -> str:
    """Run ``code`` in a new interpreter and fail if it imported a heavy library.

    A subprocess, because this process may already have imported those libraries.
    """
    script = (
        "import sys, eegprep\n"
        f"{code}\n"
        f"loaded = [name for name in {HEAVY_MODULES!r} if name in sys.modules]\n"
        "assert not loaded, loaded\n"
    )
    env = dict(os.environ)
    env["EEGPREP_SKIP_MATLAB"] = "1"
    result = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    return result.stdout


def _line_mentions(text: str, name: str) -> bool:
    return re.search(rf"(?m)^  (?:await )?{re.escape(name)}\b", text) is not None


def test_help_is_not_in_dunder_all() -> None:
    assert "help" not in eegprep.__all__
    assert callable(eegprep.help)


def test_dir_lists_every_lazy_name_without_importing_the_stack() -> None:
    names = _run_fresh("print('\\n'.join(dir(eegprep)))").split()

    missing = [name for name in eegprep.__all__ if name not in names]
    assert not missing, f"dir() omits {missing[:12]}"
    assert "help" in names
    assert "pop_loadset" in names
    assert "pop_writeeeg" in names
    assert _PRIVATE_DIR.isdisjoint(names)
    private = [name for name in names if name.startswith("_") and name != "__version__"]
    assert not private, private


def test_help_does_not_bind_render_help() -> None:
    text = _run_fresh(
        "import contextlib, io\n"
        "assert not hasattr(eegprep, 'render_help')\n"
        "with contextlib.redirect_stdout(io.StringIO()):\n"
        "    eegprep.help()\n"
        "assert not hasattr(eegprep, 'render_help')\n"
        "assert 'render_help' not in dir(eegprep)\n"
        "print('ok')"
    )
    assert text.strip() == "ok"


def test_help_describes_every_public_name_without_importing_the_stack() -> None:
    text = _run_fresh("eegprep.help()")

    absent = [name for name in eegprep.__all__ if not _line_mentions(text, name)]
    assert not absent, f"help() does not mention {absent[:12]}"
    assert "Load an EEGLAB dataset" in text
    assert "Plot a 2D topographic map" in text
    assert "Write continuous EEG data to EDF, BDF, or GDF" in text
    for extra in ("eeglab", "mne", "pybids", "picard", "full"):
        assert f'pip install "eegprep[{extra}]"' in text
    assert "restores the previous complete install" in text
    described = [line for line in text.splitlines() if line.startswith("  ") and "  " in line[2:]]
    summaries = [line for line in described if line.split("  ", 2)[-1].strip()]
    assert len(summaries) > 400
    for name, (extra, detail) in PUBLIC_EXTRAS.items():
        line = next(line for line in text.splitlines() if _line_mentions(line, name))
        assert f"[{extra} extra: {detail}" in line


def test_every_public_optional_path_carries_its_marker() -> None:
    """Each public function that can reach a moved library names that extra."""
    reached = _public_optional_extras()
    assert reached == {name: extra for name, (extra, _) in PUBLIC_EXTRAS.items()}
    text = render_help(eegprep._LAZY_EXPORTS, eegprep.__version__)
    for name, extra in reached.items():
        line = next(line for line in text.splitlines() if _line_mentions(line, name))
        assert f"[{extra} extra:" in line


def _module_name(path: Path) -> str:
    parts = list(path.relative_to(SRC_ROOT).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return "eegprep." + ".".join(parts) if parts else "eegprep"


def _imported_module(current: str, node: ast.ImportFrom) -> str | None:
    if node.level:
        base = current.split(".")[: len(current.split(".")) - node.level]
        if node.module:
            base.extend(node.module.split("."))
        return ".".join(base) if base else None
    return node.module


def _import_map(module: str, statements: list[ast.stmt]) -> dict[str, tuple[str, str]]:
    found: dict[str, tuple[str, str]] = {}
    for node in statements:
        if not isinstance(node, ast.ImportFrom):
            continue
        target = _imported_module(module, node)
        if target is None:
            continue
        for alias in node.names:
            if alias.name == "*":
                continue
            found[alias.asname or alias.name] = (target, alias.name)
    return found


def _direct_libraries(func: ast.AST) -> set[str]:
    libraries: set[str] = set()
    for node in ast.walk(func):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id != "import_optional_dependency" or not node.args:
            continue
        argument = node.args[0]
        if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
            libraries.add(argument.value.split(".", 1)[0])
    return libraries


def _public_optional_extras() -> dict[str, str]:
    public = {(module, attr): name for name, (module, attr) in eegprep._LAZY_EXPORTS.items()}
    functions: dict[tuple[str, str], ast.FunctionDef | ast.AsyncFunctionDef] = {}
    module_imports: dict[str, dict[str, tuple[str, str]]] = {}
    for path in sorted(SRC_ROOT.rglob("*.py")):
        if "eeglab" in path.relative_to(SRC_ROOT).parts:
            continue
        module = _module_name(path)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        module_imports[module] = _import_map(module, tree.body)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions[(module, node.name)] = node

    def callees(module: str, func: ast.FunctionDef | ast.AsyncFunctionDef) -> set[tuple[str, str]]:
        visible = dict(module_imports.get(module, {}))
        visible.update(_import_map(module, [node for node in ast.walk(func) if isinstance(node, ast.ImportFrom)]))
        found: set[tuple[str, str]] = set()
        for node in ast.walk(func):
            if not isinstance(node, ast.Name) or not isinstance(node.ctx, ast.Load):
                continue
            if node.id in visible:
                found.add(visible[node.id])
            elif (module, node.id) in functions:
                found.add((module, node.id))
        return found

    calls = {key: callees(key[0], func) for key, func in functions.items()}
    direct = {key: _direct_libraries(func) for key, func in functions.items()}
    public_keys = set(public)
    private_libs = {key: set(direct[key]) for key in functions if key not in public_keys}
    changed = True
    while changed:
        changed = False
        for key, own in private_libs.items():
            for callee in calls[key]:
                downstream = private_libs.get(callee)
                if downstream and not downstream <= own:
                    own.update(downstream)
                    changed = True

    def own_libraries(key: tuple[str, str]) -> set[str]:
        """Libraries this function reaches without going through another public function."""
        libraries = set(direct[key])
        for callee in calls[key]:
            libraries.update(private_libs.get(callee, ()))
        return libraries

    def private_callees(key: tuple[str, str]) -> set[tuple[str, str]]:
        seen: set[tuple[str, str]] = set()
        stack = [key]
        while stack:
            current = stack.pop()
            for callee in calls[current]:
                if callee in functions and callee not in public_keys and callee not in seen:
                    seen.add(callee)
                    stack.append(callee)
        return seen

    # One public hop from this function or its private helpers, plus that
    # callee's own import. pop_runica reaches picard through eeg_picard.
    # A second public hop would mark clean_artifacts: it calls clean_channels,
    # whose Octave import sits behind a subroutine argument clean_artifacts never passes.
    def reached_libraries(key: tuple[str, str]) -> set[str]:
        libraries = own_libraries(key)
        for source in {key, *private_callees(key)}:
            for callee in calls[source]:
                if callee not in functions or callee not in public_keys:
                    continue
                libraries.update(direct[callee])
                for nested in calls[callee]:
                    if nested in functions and nested not in public_keys:
                        libraries.update(direct[nested])
        return libraries

    declared = {name: extra for name, (extra, _detail) in PUBLIC_EXTRAS.items()}
    reached: dict[str, str] = {}
    for key, name in public.items():
        if key not in functions:
            continue
        found = {OPTIONAL_LIBRARIES[library] for library in reached_libraries(key)}
        if name in _FORMAT_BRANCH_FALSE_POSITIVES:
            assert found == {"mne"}, (name, found)
            continue
        if not found:
            continue
        if len(found) == 1:
            (extra,) = found
        else:
            assert name in declared, (name, found)
            extra = declared[name]
            assert extra in found, (name, extra, found)
        reached[name] = extra
    return reached
