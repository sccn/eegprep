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
MOVED = ("oct2py", "psutil", "pyedflib")
STILL_REQUIRED = ("numpy", "scipy", "mne", "matplotlib", "pybids", "python-picard", "h5py")


def _project() -> dict:
    return tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]


def _requirement_names(requirements: list[str]) -> set[str]:
    names = set()
    for requirement in requirements:
        names.add(requirement.split(";", 1)[0].split(">", 1)[0].split("=", 1)[0].split("<", 1)[0].strip().lower())
    return names


def test_base_install_keeps_the_scientific_stack_and_moves_three_libraries() -> None:
    project = _project()
    base = _requirement_names(project["dependencies"])
    extras = project["optional-dependencies"]

    assert set(STILL_REQUIRED) <= base
    assert {"oct2py", "psutil", "pyedflib", "neo"}.isdisjoint(base)
    assert _requirement_names(extras["eeglab"]) == {"oct2py"}
    assert _requirement_names(extras["sys"]) == {"psutil"}
    assert _requirement_names(extras["edf"]) == {"pyedflib"}
    assert set(extras["full"]) == {'eegprep[eeglab]', 'eegprep[sys]', 'eegprep[edf]'}
    assert {"eegprep[eeglab]", "eegprep[sys]", "eegprep[edf]"} <= set(extras["all"])
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
        from eegprep.functions.miscfunc.misc import num_jobs_from_reservation
        from eegprep.functions.popfunc.pop_writeeeg import pop_writeeeg
        from eegprep.plugins.clean_rawdata.asr_process import asr_process
        from eegprep.plugins.clean_rawdata.asr_process_r import asr_process_r

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
        edf_dir = Path({str(output / "edf")!r})
        calls = {{
            "pop_writeeeg": call("pop_writeeeg", lambda: pop_writeeeg(eeg, edf_dir / "out.edf")),
            "asr_process": call("asr_process", lambda: asr_process(np.zeros((2, 8)), 100.0, {{}}, max_mem=None)),
            "asr_process_r": call("asr_process_r", lambda: asr_process_r(np.zeros((2, 8)), 100.0, {{}}, max_mem=None)),
            "num_jobs_from_reservation": call("num_jobs_from_reservation", lambda: num_jobs_from_reservation("4GB")),
            "get_eeglab": call("get_eeglab", lambda: get_eeglab("oct")),
        }}
        explicit = {{}}
        for name, fn in (
            ("asr_process", lambda: asr_process(np.zeros((2, 8)), 100.0, {{}}, max_mem=64)),
            ("asr_process_r", lambda: asr_process_r(np.zeros((2, 8)), 100.0, {{}}, max_mem=64)),
        ):
            try:
                fn()
                explicit[name] = "returned"
            except ImportError as exc:
                explicit[name] = str(exc)
            except Exception as exc:
                explicit[name] = type(exc).__name__
        print(json.dumps({{
            "import_failures": failures,
            "calls": calls,
            "explicit_max_mem": explicit,
            "cpu_jobs": num_jobs_from_reservation("1CPU"),
            "gdf_bytes": gdf.stat().st_size,
            "edf_dir": edf_dir.exists(),
        }}))
        """,
        tmp_path,
        env={"EEGPREP_EEGLAB_ROOT": str(eeglab_root)},
    )

    assert result["import_failures"] == {}
    assert result["gdf_bytes"] > 0
    assert result["edf_dir"] is False
    assert result["cpu_jobs"] >= 1
    expected = {
        "pop_writeeeg": ("pyedflib", "eegprep[edf]", "pop_writeeeg"),
        "asr_process": ("psutil", "eegprep[sys]", "asr_process"),
        "asr_process_r": ("psutil", "eegprep[sys]", "asr_process_r"),
        "num_jobs_from_reservation": ("psutil", "eegprep[sys]", "num_jobs_from_reservation"),
        "get_eeglab": ("oct2py", "eegprep[eeglab]", "get_eeglab"),
    }
    for name, (library, extra, feature) in expected.items():
        call = result["calls"][name]
        assert call["wrapped"] is True, call
        assert call["chained"] is True, call
        assert library in call["message"]
        assert feature in call["message"]
        assert f'pip install "{extra}"' in call["message"]
        assert 'pip install "eegprep[full]"' in call["message"]
    for name, outcome in result["explicit_max_mem"].items():
        assert "psutil" not in outcome, f"{name} with max_mem set still needed psutil: {outcome}"


def test_import_optional_dependency_returns_an_installed_library() -> None:
    pytest.importorskip("pyedflib")
    module = import_optional_dependency("pyedflib", feature="pop_writeeeg")
    assert module.__name__ == "pyedflib"


def test_dockerfile_installs_the_sys_extra_for_ram_reservations() -> None:
    dockerfile = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    entry = (REPO_ROOT / "scripts" / "bids_minimal_preproc.py").read_text(encoding="utf-8")
    assert "--jobs" in entry and "4GB" in entry
    assert ".[sys]" in dockerfile or "eegprep[sys]" in dockerfile
