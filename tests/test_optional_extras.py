"""Libraries that moved to extras fail with an install command, not ModuleNotFoundError."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import textwrap
import tomllib
from pathlib import Path

import pytest

from eegprep.extras import import_optional_dependency

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src" / "eegprep"
MOVED = ("oct2py", "mne", "bids", "picard")
STILL_REQUIRED = ("numpy", "scipy", "matplotlib", "psutil", "pyedflib", "h5py")


def _project() -> dict:
    return tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]


def _requirement_names(requirements: list[str]) -> set[str]:
    names = set()
    for requirement in requirements:
        names.add(requirement.split(";", 1)[0].split(">", 1)[0].split("=", 1)[0].split("<", 1)[0].strip().lower())
    return names


def test_base_install_keeps_plotting_and_edf_and_moves_heavy_libraries() -> None:
    project = _project()
    base = _requirement_names(project["dependencies"])
    extras = project["optional-dependencies"]

    assert set(STILL_REQUIRED) <= base
    assert {"oct2py", "mne", "pybids", "python-picard", "neo"}.isdisjoint(base)
    assert "sys" not in extras
    assert "edf" not in extras
    assert _requirement_names(extras["eeglab"]) == {"oct2py"}
    assert _requirement_names(extras["mne"]) == {"mne"}
    assert _requirement_names(extras["pybids"]) == {"pybids"}
    assert _requirement_names(extras["picard"]) == {"python-picard"}
    assert set(extras["full"]) == {"eegprep[eeglab]", "eegprep[mne]", "eegprep[pybids]", "eegprep[picard]"}
    assert {"eegprep[eeglab]", "eegprep[mne]", "eegprep[pybids]", "eegprep[picard]"} <= set(extras["all"])
    assert "eegprep[torch]" in extras["all"]


def _catches_import_error(handler: ast.ExceptHandler) -> bool:
    if handler.type is None:
        return True
    caught = ast.unparse(handler.type)
    return any(name in caught for name in ("ImportError", "ModuleNotFoundError", "Exception"))


def _unguarded_imports(path: Path) -> set[str]:
    """Top-level modules imported outside a function and outside an ImportError handler."""
    found: set[str] = set()

    def visit(node: ast.AST, guarded: bool) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.Import):
                if not guarded:
                    found.update(alias.name.split(".", 1)[0] for alias in child.names)
            elif isinstance(child, ast.ImportFrom) and child.level == 0 and child.module and not guarded:
                found.add(child.module.split(".", 1)[0])
            child_guarded = guarded or isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
            if (
                isinstance(node, ast.Try)
                and child in node.body
                and any(_catches_import_error(h) for h in node.handlers)
            ):
                child_guarded = True
            visit(child, child_guarded)

    visit(ast.parse(path.read_text(encoding="utf-8")), False)
    return found


def test_moved_libraries_have_no_unguarded_module_import() -> None:
    offenders: dict[str, set[str]] = {}
    for path in sorted(SRC_ROOT.rglob("*.py")):
        if "eeglab" in path.relative_to(SRC_ROOT).parts:
            continue
        hit = _unguarded_imports(path) & set(MOVED)
        if hit:
            offenders[path.relative_to(SRC_ROOT).as_posix()] = hit
    assert not offenders, offenders


def _run_blocked(body: str, tmp_path: Path, env: dict[str, str] | None = None) -> dict:
    preamble = "import sys\nfor _name in {moved!r}:\n    sys.modules[_name] = None\n".format(moved=MOVED)
    completed = subprocess.run(
        [sys.executable, "-c", preamble + textwrap.dedent(body)],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen", "EEGPREP_SKIP_MATLAB": "1", **(env or {})},
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_moved_functions_name_their_extra_when_the_library_is_absent(tmp_path: Path) -> None:
    eeglab_root = tmp_path / "eeglab"
    eeglab_root.mkdir()
    (eeglab_root / "eeglab.m").write_text("", encoding="utf-8")
    output = tmp_path / "out"
    result = _run_blocked(
        f"""
        import importlib
        import json
        import pkgutil
        from pathlib import Path

        import numpy as np

        import eegprep
        from eegprep.functions.adminfunc.eeglabcompat import get_eeglab
        from eegprep.functions.miscfunc.eeg_eeg2mne import eeg_eeg2mne
        from eegprep.functions.miscfunc.misc import num_jobs_from_reservation
        from eegprep.functions.popfunc.eeg_picard import eeg_picard
        from eegprep.functions.popfunc.pop_biosig import pop_biosig
        from eegprep.functions.popfunc.pop_fileio import pop_fileio
        from eegprep.functions.popfunc.pop_writeeeg import pop_writeeeg
        from eegprep.plugins.EEG_BIDS.bids_list_eeg_files import bids_list_eeg_files

        failures = {{}}
        for info in pkgutil.walk_packages(eegprep.__path__, "eegprep."):
            if info.name.startswith("eegprep.eeglab") or info.name.endswith(".__main__"):
                continue
            try:
                importlib.import_module(info.name)
            except ImportError as exc:
                root = (exc.name or "").split(".", 1)[0]
                text = str(exc)
                if root in {MOVED!r} or any(name in text for name in {MOVED!r}):
                    failures[info.name] = root or text

        def call(name, fn):
            try:
                fn()
            except ImportError as exc:
                return {{
                    "message": str(exc),
                    "wrapped": not isinstance(exc, ModuleNotFoundError),
                    "chained": isinstance(exc.__cause__, ImportError),
                }}
            return {{"message": "no ImportError", "wrapped": False, "chained": False}}

        eeg = {{
            "data": np.arange(40, dtype=float).reshape(2, 20),
            "srate": 100.0,
            "nbchan": 2,
            "pnts": 20,
            "chanlocs": [{{"labels": "Cz"}}, {{"labels": "Pz"}}],
            "event": [],
            "setname": "x",
            "subject": "",
        }}
        gdf = Path({str(output / "out.gdf")!r})
        pop_writeeeg(eeg, gdf)
        edf = Path({str(output / "edf" / "out.edf")!r})
        pop_writeeeg(eeg, edf)
        calls = {{
            "get_eeglab": call("get_eeglab", lambda: get_eeglab("oct")),
            "eeg_eeg2mne": call("eeg_eeg2mne", lambda: eeg_eeg2mne(eeg)),
            "eeg_picard": call("eeg_picard", lambda: eeg_picard(eeg)),
            "pop_biosig": call("pop_biosig", lambda: pop_biosig("recording.edf")),
            "pop_fileio": call("pop_fileio", lambda: pop_fileio("recording.edf")),
            "bids_list_eeg_files": call("bids_list_eeg_files", lambda: bids_list_eeg_files(".")),
        }}
        print(json.dumps({{
            "import_failures": failures,
            "calls": calls,
            "cpu_jobs": num_jobs_from_reservation("1CPU"),
            "memory_jobs": num_jobs_from_reservation("4GB"),
            "gdf_bytes": gdf.stat().st_size,
            "edf_bytes": edf.stat().st_size,
        }}))
        """,
        tmp_path,
        env={"EEGPREP_EEGLAB_ROOT": str(eeglab_root)},
    )

    assert result["import_failures"] == {}
    assert result["gdf_bytes"] > 0
    assert result["edf_bytes"] > 0
    assert result["cpu_jobs"] >= 1
    assert result["memory_jobs"] >= 1
    expected = {
        "get_eeglab": ("oct2py", "eegprep[eeglab]", "get_eeglab"),
        "eeg_eeg2mne": ("mne", "eegprep[mne]", "eeg_eeg2mne"),
        "eeg_picard": ("picard", "eegprep[picard]", "eeg_picard"),
        "pop_biosig": ("mne", "eegprep[mne]", "pop_biosig"),
        "pop_fileio": ("mne", "eegprep[mne]", "pop_fileio"),
        "bids_list_eeg_files": ("bids", "eegprep[pybids]", "bids_list_eeg_files"),
    }
    for name, (library, extra, feature) in expected.items():
        call = result["calls"][name]
        assert call["wrapped"] is True, call
        assert call["chained"] is True, call
        assert library in call["message"]
        assert feature in call["message"]
        assert f'pip install "{extra}"' in call["message"]
        assert 'pip install "eegprep[full]"' in call["message"]


def test_import_optional_dependency_returns_an_installed_library() -> None:
    pytest.importorskip("oct2py")
    module = import_optional_dependency("oct2py", feature="get_eeglab")
    assert module.__name__ == "oct2py"


def test_dockerfile_installs_bids_and_mne_for_the_entrypoint() -> None:
    dockerfile = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    entry = (REPO_ROOT / "scripts" / "bids_minimal_preproc.py").read_text(encoding="utf-8")
    assert "--jobs" in entry and "4GB" in entry
    assert ".[pybids,mne]" in dockerfile or '".[pybids,mne]"' in dockerfile
