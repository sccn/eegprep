"""A base install must work without the optional extras, and must say which extra a feature needs.

The optional packages (``edf``, ``eeglab``, ``sys``, ``iclabel``, ``torch`` extras) are blocked
the way an environment without them behaves: ``sys.modules[name] = None`` makes the import
statement raise ``ImportError``. Checks that must prove an import-time claim run in a subprocess
because the parent process has already imported the packages.
"""

from __future__ import annotations

import ast
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import textwrap
import tomllib
from pathlib import Path

import pytest

from eegprep.functions.adminfunc.eeglabcompat import EEGLAB_ROOT_ENV, get_eeglab
from eegprep.functions.miscfunc.misc import num_jobs_from_reservation

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src" / "eegprep"
SAMPLE_SET = REPO_ROOT / "sample_data" / "eeglab_data.set"

# Packages that only an extra provides and that a base install must run without.
OPTIONAL_PACKAGES = ["psutil", "oct2py", "pyedflib", "onnxruntime", "torch", "onnx"]

# Import name -> PyPI distribution name, where the two differ.
DISTRIBUTION_OF_IMPORT = {
    "IPython": "ipython",
    "PySide6": "pyside6",
    "bids": "pybids",
    "mpl_toolkits": "matplotlib",
    "picard": "python-picard",
    "yaml": "pyyaml",
}

# Provided by the Pyodide host, only imported on the Emscripten code path.
HOST_PROVIDED_MODULES = {"js", "pyodide"}

# Hard requirements of a declared dependency (verified against its metadata below), so they are
# always installed even though pyproject.toml does not list them.
TRANSITIVE_OF_DECLARED = {"nibabel": "pybids", "pandas": "pybids"}

# Modules that may import an extra-only package unguarded at module level, with the reason.
GUI_PACKAGE = "functions/guifunc/"
EXTRA_ONLY_MODULES = {
    "plugins/ICLabel/iclabel_net.py": "torch is imported to export iclabel.onnx and is never imported at runtime",
}


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _requirement_name(requirement: str) -> str:
    return _normalize(re.match(r"[A-Za-z0-9_.-]+", requirement).group(0))


def _declared_distributions() -> tuple[set[str], set[str]]:
    """Return (base dependencies, extra-only dependencies) from pyproject.toml, normalized."""
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["project"]
    base = {_requirement_name(requirement) for requirement in project["dependencies"]}
    extras = {
        _requirement_name(requirement)
        for requirements in project["optional-dependencies"].values()
        for requirement in requirements
    }
    return base, extras - base - {"eegprep"}


def _catches_import_error(handler: ast.ExceptHandler) -> bool:
    if handler.type is None:
        return True
    caught = ast.unparse(handler.type)
    return any(name in caught for name in ("ImportError", "ModuleNotFoundError", "Exception"))


def _scan_imports(path: Path) -> list[tuple[str, bool]]:
    """Return (top-level module, unguarded-at-module-level) for each absolute import in a file.

    An import is guarded when it sits inside a function body or in the body of a ``try`` that
    handles ``ImportError``; both defer or tolerate a missing optional package.
    """
    found: list[tuple[str, bool]] = []

    def visit(node: ast.AST, guarded: bool) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.Import):
                found.extend((alias.name.split(".")[0], not guarded) for alias in child.names)
            elif isinstance(child, ast.ImportFrom) and child.level == 0 and child.module:
                found.append((child.module.split(".")[0], not guarded))
            child_guarded = guarded or isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
            if (
                isinstance(node, ast.Try)
                and child in node.body
                and any(_catches_import_error(h) for h in node.handlers)
            ):
                child_guarded = True
            visit(child, child_guarded)

    visit(ast.parse(path.read_text()), False)
    return found


def _package_modules() -> list[Path]:
    return [path for path in sorted(SRC_ROOT.rglob("*.py")) if path.relative_to(SRC_ROOT).parts[0] != "eeglab"]


def _run_without_optional_extras(
    body: str, tmp_path: Path, blocked: list[str] | None = None, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess:
    """Run ``body`` in a fresh interpreter in which the optional packages cannot be imported."""
    preamble = "import sys\nBLOCKED = {!r}\nfor _name in BLOCKED:\n    sys.modules[_name] = None\n".format(
        blocked or OPTIONAL_PACKAGES
    )
    return subprocess.run(
        [sys.executable, "-c", preamble + textwrap.dedent(body)],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen", **(env or {})},
        timeout=300,
        check=False,
    )


def _json_result(completed: subprocess.CompletedProcess) -> dict:
    assert completed.returncode == 0, completed.stderr[-3000:]
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_every_third_party_import_is_declared() -> None:
    """Each non-stdlib import in the package maps to a base dependency or an extra of pyproject.toml."""
    base, extras = _declared_distributions()
    declared = base | extras
    for name, owner in TRANSITIVE_OF_DECLARED.items():
        requirements = importlib.metadata.requires(owner) or []
        assert any(_requirement_name(r) == name and "extra ==" not in r for r in requirements), (
            f"{owner} no longer hard-requires {name}; declare {name} in pyproject.toml"
        )

    undeclared: dict[str, set[str]] = {}
    for path in _package_modules():
        for module, _ in _scan_imports(path):
            if module in sys.stdlib_module_names or module == "eegprep" or module in HOST_PROVIDED_MODULES:
                continue
            if module in TRANSITIVE_OF_DECLARED:
                continue
            distribution = _normalize(DISTRIBUTION_OF_IMPORT.get(module, module))
            if distribution not in declared:
                undeclared.setdefault(module, set()).add(str(path.relative_to(SRC_ROOT)))
    assert not undeclared, f"imported but declared in neither [project].dependencies nor an extra: {undeclared}"


def test_extra_only_packages_are_never_imported_unguarded_by_base_modules() -> None:
    """A module importable on a base install must not import an extra-only package at module level."""
    _, extras = _declared_distributions()
    offenders: dict[str, set[str]] = {}
    for path in _package_modules():
        relative = path.relative_to(SRC_ROOT).as_posix()
        if relative.startswith(GUI_PACKAGE) or relative in EXTRA_ONLY_MODULES:
            continue
        for module, unguarded in _scan_imports(path):
            distribution = _normalize(DISTRIBUTION_OF_IMPORT.get(module, module))
            if unguarded and distribution in extras:
                offenders.setdefault(relative, set()).add(module)
    assert not offenders, f"unguarded module-level import of an optional package: {offenders}"


def test_every_non_gui_module_imports_without_optional_extras(tmp_path: Path) -> None:
    """Importing the whole package with every optional package blocked fails only for export-only modules."""
    completed = _run_without_optional_extras(
        """
        import importlib, json, pkgutil
        import eegprep

        failures = {}
        for info in pkgutil.walk_packages(eegprep.__path__, "eegprep."):
            if info.name.startswith("eegprep.eeglab") or info.name.endswith(".__main__"):
                continue
            try:
                importlib.import_module(info.name)
            except ImportError as exc:
                if getattr(exc, "name", None) in BLOCKED:
                    failures[info.name] = exc.name
        print(json.dumps(failures))
        """,
        tmp_path,
    )

    assert _json_result(completed) == {"eegprep.plugins.ICLabel.iclabel_net": "torch"}


def test_light_pipeline_runs_without_optional_extras(tmp_path: Path) -> None:
    """Load, filter, re-reference and clean a real dataset with every optional package blocked."""
    completed = _run_without_optional_extras(
        f"""
        import json
        from eegprep import clean_artifacts, pop_eegfiltnew, pop_loadset, pop_reref, pop_select

        EEG = pop_loadset({str(SAMPLE_SET)!r})
        EEG = pop_select(EEG, point=[[0, 3000]])
        EEG = pop_eegfiltnew(EEG, locutoff=1)
        EEG, com = pop_reref(EEG, [], return_com=True)
        cleaned = clean_artifacts(
            EEG, ChannelCriterion="off", LineNoiseCriterion="off", FlatlineCriterion="off",
            Highpass="off", WindowCriterion="off", BurstCriterion=20,
        )[0]
        print(json.dumps({{"command": com, "shape": list(cleaned["data"].shape)}}))
        """,
        tmp_path,
    )

    result = _json_result(completed)
    assert result["command"] == "EEG = pop_reref( EEG, []);"
    assert result["shape"][0] == 32


def test_pop_writeeeg_gdf_works_without_pyedflib(tmp_path: Path) -> None:
    """GDF output has its own writer, so it must not need pyedflib (the ``edf`` extra)."""
    completed = _run_without_optional_extras(
        f"""
        import json
        import numpy as np
        from pathlib import Path
        from eegprep.functions.popfunc.pop_writeeeg import pop_writeeeg

        EEG = {{"data": np.arange(40, dtype=float).reshape(2, 20), "srate": 100.0, "nbchan": 2, "pnts": 20,
               "chanlocs": [{{"labels": "Cz"}}, {{"labels": "Pz"}}], "event": [], "setname": "x"}}
        output = Path({str(tmp_path / "out.gdf")!r})
        pop_writeeeg(EEG, output)
        print(json.dumps({{"bytes": output.stat().st_size}}))
        """,
        tmp_path,
    )

    assert _json_result(completed)["bytes"] > 0


def test_pop_writeeeg_edf_names_the_edf_extra_without_pyedflib(tmp_path: Path) -> None:
    """EDF and BDF need pyedflib; the error must name ``eegprep[edf]`` and chain the original cause."""
    completed = _run_without_optional_extras(
        f"""
        import json
        import numpy as np
        from eegprep.functions.popfunc.pop_writeeeg import pop_writeeeg

        EEG = {{"data": np.arange(40, dtype=float).reshape(2, 20), "srate": 100.0, "nbchan": 2, "pnts": 20,
               "chanlocs": [{{"labels": "Cz"}}, {{"labels": "Pz"}}], "event": []}}
        try:
            pop_writeeeg(EEG, {str(tmp_path / "out.edf")!r})
        except ImportError as exc:
            print(json.dumps({{"message": str(exc), "chained": isinstance(exc.__cause__, ImportError)}}))
        """,
        tmp_path,
    )

    result = _json_result(completed)
    assert "eegprep[edf]" in result["message"]
    assert result["chained"] is True


def test_num_jobs_from_reservation_names_the_sys_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    """A RAM reservation needs psutil; the error names ``eegprep[sys]`` and keeps the real reason."""
    monkeypatch.setitem(sys.modules, "psutil", None)

    with pytest.raises(ImportError) as excinfo:
        num_jobs_from_reservation("4GB")

    assert 'pip install "eegprep[sys]"' in str(excinfo.value)
    assert "None in sys.modules" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, ImportError)


def test_get_eeglab_octave_names_the_eeglab_extra(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The Octave runtime needs oct2py; the error names ``eegprep[eeglab]`` once an EEGLAB root is found."""
    (tmp_path / "eeglab.m").write_text("")
    monkeypatch.setenv(EEGLAB_ROOT_ENV, str(tmp_path))
    monkeypatch.setitem(sys.modules, "oct2py", None)

    with pytest.raises(ImportError) as excinfo:
        get_eeglab("oct")

    assert 'pip install "eegprep[eeglab]"' in str(excinfo.value)
    assert "None in sys.modules" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, ImportError)


def test_ransac_defers_the_eeglab_bridge_but_still_reaches_it(tmp_path: Path) -> None:
    """Importing ransac (and so runica) must not load the file-loading stack; the MATLAB/Octave path still works.

    ``get_eeglab`` lives in a module that imports ``pop_loadset`` and so ``mne``. runica imports
    ransac for ``rand_permutation`` and must stay light, so ransac defers the bridge to its first use.
    """
    (tmp_path / "eeglab.m").write_text("")
    completed = _run_without_optional_extras(
        """
        import json
        import eegprep.plugins.clean_rawdata.private.ransac as ransac

        loaded_on_import = sorted(n for n in ("mne", "eegprep.functions.adminfunc.eeglabcompat") if n in sys.modules)
        try:
            ransac.get_eeglab("OCT")
        except ImportError as exc:
            message = str(exc)
        print(json.dumps({"loaded_on_import": loaded_on_import, "message": message}))
        """,
        tmp_path,
        env={"EEGPREP_EEGLAB_ROOT": str(tmp_path)},
    )

    result = _json_result(completed)
    assert result["loaded_on_import"] == []
    assert 'pip install "eegprep[eeglab]"' in result["message"]
