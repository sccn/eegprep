"""Libraries that moved to extras fail with an install command, not ModuleNotFoundError.

The base install must also work where a package has no build, as in Pyodide, which has neither
``psutil`` nor ``pyedflib``. ``sys.modules[name] = None`` makes the import statement raise
``ImportError``, the way an environment without the package behaves. Checks that must prove an
import-time claim run in a subprocess because the parent process has already imported the packages.
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

import numpy as np
import pytest

from eegprep.extras import OPTIONAL_LIBRARIES, PUBLIC_EXTRAS, import_optional_dependency
from eegprep.functions.miscfunc.misc import num_jobs_from_reservation
from eegprep.functions.popfunc.pop_fileio import pop_fileio

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src" / "eegprep"
SAMPLE_SET = REPO_ROOT / "sample_data" / "eeglab_data.set"
SAMPLE_ICA_SET = REPO_ROOT / "sample_data" / "eeglab_data_with_ica_tmp.set"
MOVED = ("oct2py", "mne", "bids", "picard")
STILL_REQUIRED = ("numpy", "scipy", "matplotlib", "psutil", "pyedflib", "h5py")
# Packages a browser install does not have: the extras, and the two base packages that carry
# the emscripten marker.
OPTIONAL_PACKAGES = ["psutil", "oct2py", "pyedflib", "onnxruntime", "torch", "onnx"]
EMSCRIPTEN_MARKED = ("psutil", "pyedflib")

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
    assert _requirement_names(extras["eeglab"]) == {"oct2py"}
    assert _requirement_names(extras["mne"]) == {"mne", "defusedxml"}
    assert _requirement_names(extras["pybids"]) == {"pybids"}
    assert _requirement_names(extras["picard"]) == {"python-picard"}
    assert set(extras["full"]) == {"eegprep[eeglab]", "eegprep[mne]", "eegprep[pybids]", "eegprep[picard]"}
    assert {"eegprep[eeglab]", "eegprep[mne]", "eegprep[pybids]", "eegprep[picard]"} <= set(extras["all"])
    assert "eegprep[torch]" in extras["all"]
    assert _requirement_names(extras["iclabel"]) == {"onnxruntime"}
    assert "eegprep[iclabel]" in extras["all"]
    assert "eegprep[iclabel]" not in extras["full"]
    assert {"onnxruntime", "torch", "onnx"}.isdisjoint(base)


def test_psutil_and_pyedflib_stay_in_the_base_but_not_under_emscripten() -> None:
    """Pyodide has no build of either, so the base marks them out of Emscripten and micropip can resolve it."""
    requirements = {_requirement_names([item]).pop(): item for item in _project()["dependencies"]}

    for name in EMSCRIPTEN_MARKED:
        marker = requirements[name].partition(";")[2].replace(" ", "")
        assert marker == "sys_platform!='emscripten'", (name, requirements[name])


def test_mff_import_names_the_xml_dependency_and_its_extra(monkeypatch, tmp_path):
    pytest.importorskip('mne')
    folder = tmp_path / 'recording.mff'
    folder.mkdir()
    monkeypatch.setitem(sys.modules, 'defusedxml', None)

    with pytest.raises(ImportError, match='pop_fileio needs defusedxml') as caught:
        pop_fileio(folder)
    assert 'pip install "eegprep[mne]"' in str(caught.value)
    assert 'pip install "eegprep[full]"' in str(caught.value)
    assert isinstance(caught.value.__cause__, ImportError)
    assert caught.value.__cause__.name == 'defusedxml'


def _catches_import_error(handler: ast.ExceptHandler) -> bool:
    if handler.type is None:
        return True
    caught = ast.unparse(handler.type)
    return any(name in caught for name in ("ImportError", "ModuleNotFoundError", "Exception"))


def _is_type_checking(node: ast.AST) -> bool:
    return isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING"


def _unguarded_imports(path: Path) -> set[str]:
    """Top-level modules imported outside a function and outside an ImportError handler."""
    found: set[str] = set()

    def visit(node: ast.AST, guarded: bool) -> None:
        for child in ast.iter_child_nodes(node):
            child_guarded = guarded or isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
            if (
                isinstance(node, ast.Try)
                and child in node.body
                and any(_catches_import_error(h) for h in node.handlers)
            ):
                child_guarded = True
            if isinstance(node, ast.If) and _is_type_checking(node) and child in node.body:
                child_guarded = True
            if not child_guarded:
                if isinstance(child, ast.Import):
                    found.update(alias.name.split(".", 1)[0] for alias in child.names)
                elif isinstance(child, ast.ImportFrom) and child.level == 0 and child.module:
                    found.add(child.module.split(".", 1)[0])
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
        from eegprep.functions.popfunc.pop_resample import pop_resample
        from eegprep.functions.popfunc.pop_writeeeg import pop_writeeeg
        from eegprep.plugins.EEG_BIDS.bids_list_eeg_files import bids_list_eeg_files
        from eegprep.plugins.EEG_BIDS.pop_importbids import pop_importbids
        from eegprep.plugins.ICLabel.iclabel import iclabel
        from eegprep.plugins.clean_rawdata.private.ransac import calc_projector

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
            "pop_resample": call("pop_resample", lambda: pop_resample(eeg, 100, engine="octave")),
            "iclabel": call("iclabel", lambda: iclabel(eeg, engine="octave")),
            "clean_channels": call(
                "clean_channels",
                lambda: calc_projector(np.zeros((4, 3)), 1, 2, subroutine="octave"),
            ),
            "pop_importbids": call("pop_importbids", lambda: pop_importbids(".")),
            "pop_importbids_file": call(
                "pop_importbids_file",
                lambda: pop_importbids(
                    {str(REPO_ROOT / "sample_data" / "eeglab_data.set")!r},
                    bidsmetadata=False,
                    bidschanloc=False,
                    bidsevent=False,
                ),
            ),
        }}
        print(json.dumps({{
            "import_failures": failures,
            "calls": calls,
            "cpu_jobs": num_jobs_from_reservation("1CPU"),
            "memory_jobs": num_jobs_from_reservation("1MB"),
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
        "pop_resample": ("oct2py", "eegprep[eeglab]", "pop_resample"),
        "iclabel": ("oct2py", "eegprep[eeglab]", "iclabel"),
        "clean_channels": ("oct2py", "eegprep[eeglab]", "clean_channels"),
        "pop_importbids": ("bids", "eegprep[pybids]", "pop_importbids"),
    }
    for name, (library, extra, feature) in expected.items():
        call = result["calls"][name]
        assert call["wrapped"] is True, call
        assert call["chained"] is True, call
        assert library in call["message"]
        assert feature in call["message"]
        assert f'pip install "{extra}"' in call["message"]
        assert 'pip install "eegprep[full]"' in call["message"]
    assert result["calls"]["pop_importbids_file"]["message"] == "no ImportError"


_INSTEAD = {
    "clean_channels": "sphericalSplineInterpolate",
    "eeg_picard": "icatype is 'picard'",
    "eeglab_clean_artifacts": "clean_artifacts is the Python cleaner",
    "iclabel": 'pip install "eegprep[iclabel]"',
    "iclabel_async": 'pip install "eegprep[iclabel]"',
    "picard": "icatype is 'picard'",
    "pop_fileio": "do not need MNE",
    "pop_iclabel": 'pip install "eegprep[iclabel]"',
    "pop_iclabel_async": 'pip install "eegprep[iclabel]"',
    "pop_resample": "Leave engine unset to resample in Python.",
    "pop_runica": "Leave icatype unset to use runica",
    "resample": "Leave engine unset to resample in Python.",
}


def test_each_missing_extra_names_the_function_the_extra_and_the_install_command(monkeypatch) -> None:
    def blocked(name: str):
        root = name.split(".", 1)[0]
        raise ModuleNotFoundError(f"No module named {root!r}", name=root)

    monkeypatch.setattr("eegprep.extras.importlib.import_module", blocked)
    library_for: dict[str, str] = {}
    for library, extra in OPTIONAL_LIBRARIES.items():
        library_for.setdefault(extra, library)
    for feature, (extra, _detail) in PUBLIC_EXTRAS.items():
        with pytest.raises(ImportError) as caught:
            import_optional_dependency(library_for[extra], feature=feature)
        message = str(caught.value)
        assert message.startswith(f"{feature} needs "), message
        assert extra in message
        assert f'pip install "eegprep[{extra}]"' in message
        assert 'pip install "eegprep[full]"' in message
        note = _INSTEAD.get(feature)
        if note:
            assert note in message, message
        assert isinstance(caught.value.__cause__, ImportError)


@pytest.mark.parametrize('library', ['mne', 'defusedxml'])
def test_a_broken_dependency_is_not_reported_as_a_missing_extra(monkeypatch, library) -> None:
    def broken(name: str):
        raise ImportError("No module named 'numpy'", name="numpy")

    monkeypatch.setattr("eegprep.extras.importlib.import_module", broken)
    with pytest.raises(ImportError, match="numpy") as caught:
        import_optional_dependency(library, feature="pop_fileio")
    assert 'eegprep[mne]' not in str(caught.value)


def test_missing_onnxruntime_and_matlab_name_the_function_and_the_install(monkeypatch, tmp_path: Path) -> None:
    from eegprep.functions.adminfunc.eeglabcompat import get_eeglab
    from eegprep.plugins.ICLabel.iclabel import iclabel

    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    with pytest.raises(ImportError) as onnx_error:
        iclabel({}, feature="pop_iclabel")
    onnx_message = str(onnx_error.value)
    assert onnx_message.startswith("pop_iclabel needs ONNX Runtime")
    assert 'pip install "eegprep[iclabel]"' in onnx_message
    assert 'pip install "eegprep[all]"' in onnx_message
    assert "engine='octave'" in onnx_message
    assert 'pip install "eegprep[eeglab]"' in onnx_message
    assert isinstance(onnx_error.value.__cause__, ImportError)

    checkout = tmp_path / "eeglab"
    checkout.mkdir()
    (checkout / "eeglab.m").write_text("", encoding="utf-8")
    monkeypatch.setenv("EEGPREP_EEGLAB_ROOT", str(checkout))

    def missing_matlab(name: str):
        raise ImportError("No module named 'matlab'", name="matlab")

    monkeypatch.setattr("eegprep.functions.adminfunc.eeglabcompat.importlib.import_module", missing_matlab)
    with pytest.raises(ImportError) as matlab_error:
        get_eeglab("mat", feature="pop_resample", _cache={})
    matlab_message = str(matlab_error.value)
    assert matlab_message.startswith("pop_resample needs the MATLAB Engine for Python, which is not installed.")
    assert (
        "Install it from your MATLAB installation with: pip install <matlabroot>/extern/engines/python."
        in matlab_message
    )
    assert 'To use Octave instead, run: pip install "eegprep[eeglab]".' in matlab_message
    assert "Leave engine unset to resample in Python." in matlab_message


def test_import_optional_dependency_returns_an_installed_library() -> None:
    pytest.importorskip("oct2py")
    module = import_optional_dependency("oct2py", feature="get_eeglab")
    assert module.__name__ == "oct2py"


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _requirement_name(requirement: str) -> str:
    name = re.match(r"[A-Za-z0-9_.-]+", requirement)
    assert name is not None, f"unparseable requirement: {requirement!r}"
    return _normalize(name.group(0))


def _declared_distributions() -> tuple[set[str], set[str]]:
    """Return (base dependencies, extra-only dependencies) from pyproject.toml, normalized."""
    project = _project()
    base = {_requirement_name(requirement) for requirement in project["dependencies"]}
    extras = {
        _requirement_name(requirement)
        for requirements in project["optional-dependencies"].values()
        for requirement in requirements
    }
    return base, extras - base - {"eegprep"}


def _scan_imports(path: Path) -> list[tuple[str, bool]]:
    """Return (top-level module, unguarded-at-module-level) for each absolute import in a file.

    An import is guarded when it sits inside a function body, in the body of a ``try`` that
    handles ``ImportError``, or under ``if TYPE_CHECKING``; each defers or tolerates a missing
    optional package.
    """
    found: list[tuple[str, bool]] = []

    def visit(node: ast.AST, guarded: bool) -> None:
        for child in ast.iter_child_nodes(node):
            child_guarded = guarded or isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
            if (
                isinstance(node, ast.Try)
                and child in node.body
                and any(_catches_import_error(h) for h in node.handlers)
            ):
                child_guarded = True
            if isinstance(node, ast.If) and _is_type_checking(node) and child in node.body:
                child_guarded = True
            if isinstance(child, ast.Import):
                found.extend((alias.name.split(".")[0], not child_guarded) for alias in child.names)
            elif isinstance(child, ast.ImportFrom) and child.level == 0 and child.module:
                found.append((child.module.split(".")[0], not child_guarded))
            visit(child, child_guarded)

    visit(ast.parse(path.read_text(encoding="utf-8")), False)
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
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen", "EEGPREP_SKIP_MATLAB": "1", **(env or {})},
        timeout=300,
        check=False,
    )


def _json_result(completed: subprocess.CompletedProcess) -> dict:
    assert completed.returncode == 0, completed.stderr[-3000:]
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_transitively_declared_imports_are_hard_requirements_of_their_owner() -> None:
    """nibabel and pandas are imported by BIDS paths but declared only through pybids, so it must require them."""
    pytest.importorskip("bids", reason="the base-install job does not have the pybids extra")
    for name, owner in TRANSITIVE_OF_DECLARED.items():
        requirements = importlib.metadata.requires(owner) or []
        assert any(_requirement_name(r) == name and "extra ==" not in r for r in requirements), (
            f"{owner} no longer hard-requires {name}; declare {name} in pyproject.toml"
        )


def test_every_third_party_import_is_declared() -> None:
    """Each non-stdlib import in the package maps to a base dependency or an extra of pyproject.toml."""
    base, extras = _declared_distributions()
    declared = base | extras
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


def test_emscripten_marked_base_packages_are_never_imported_unguarded() -> None:
    """psutil and pyedflib are base packages, but a browser install lacks them: no module may need them to import."""
    offenders: dict[str, set[str]] = {}
    for path in _package_modules():
        hit = {module for module, unguarded in _scan_imports(path) if unguarded and module in EMSCRIPTEN_MARKED}
        if hit:
            offenders[path.relative_to(SRC_ROOT).as_posix()] = hit
    assert not offenders, offenders


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


def test_iclabel_without_onnxruntime_names_the_iclabel_extra(tmp_path: Path) -> None:
    """Both ICLabel entry points name ``eegprep[iclabel]`` when onnxruntime is missing (blocked, so no skip).

    Only onnxruntime is blocked: a ``None`` entry for torch makes SciPy's array-type detection fail inside
    feature extraction, which a real environment without torch never does.
    """
    completed = _run_without_optional_extras(
        f"""
        import asyncio, json
        from eegprep import iclabel, pop_loadset
        from eegprep.plugins.ICLabel.iclabel import iclabel_async
        from eegprep.plugins.ICLabel.pop_iclabel import pop_iclabel, pop_iclabel_async

        EEG = pop_loadset({str(SAMPLE_ICA_SET)!r})
        calls = {{
            "sync": lambda: iclabel(EEG),
            "async": lambda: asyncio.run(iclabel_async(EEG)),
            "pop_sync": lambda: pop_iclabel(EEG, "default", gui=False),
            "pop_async": lambda: asyncio.run(pop_iclabel_async(EEG, "default", gui=False)),
        }}
        messages = {{}}
        for name, call in calls.items():
            try:
                call()
            except ImportError as exc:
                messages[name] = str(exc)
        print(json.dumps(messages))
        """,
        tmp_path,
        blocked=["onnxruntime"],
    )

    messages = _json_result(completed)
    assert set(messages) == {"sync", "async", "pop_sync", "pop_async"}
    assert all('pip install "eegprep[iclabel]"' in message for message in messages.values())
    assert messages["pop_sync"].startswith("pop_iclabel needs ONNX Runtime")
    assert messages["pop_async"].startswith("pop_iclabel_async needs ONNX Runtime")


def test_pop_writeeeg_gdf_works_without_pyedflib(tmp_path: Path) -> None:
    """GDF output has its own writer, so it must not need pyedflib, which a browser install lacks."""
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


def test_pop_writeeeg_edf_names_pyedflib_and_pyodide_without_it(tmp_path: Path) -> None:
    """EDF and BDF need pyedflib; the error names it, says where it is absent, and chains the cause."""
    completed = _run_without_optional_extras(
        f"""
        import json
        import numpy as np
        from eegprep.functions.popfunc.pop_writeeeg import pop_writeeeg

        EEG = {{"data": np.arange(40, dtype=float).reshape(2, 20), "srate": 100.0, "nbchan": 2, "pnts": 20,
               "chanlocs": [{{"labels": "Cz"}}, {{"labels": "Pz"}}], "event": []}}
        try:
            pop_writeeeg(EEG, {str(tmp_path / "edf" / "out.edf")!r})
        except ImportError as exc:
            print(json.dumps({{"message": str(exc), "chained": isinstance(exc.__cause__, ImportError)}}))
        """,
        tmp_path,
    )

    result = _json_result(completed)
    assert "requires pyedflib" in result["message"]
    assert "Pyodide" in result["message"]
    assert result["chained"] is True
    assert not (tmp_path / "edf").exists()


def test_pop_writeeeg_rejecting_the_data_creates_no_directory(tmp_path: Path) -> None:
    """Data the EDF writer rejects must fail before the output directory is created."""
    pytest.importorskip("pyedflib")
    from eegprep.functions.popfunc.pop_writeeeg import pop_writeeeg

    data = np.arange(40, dtype=float).reshape(2, 20)
    data[0, 3] = np.nan
    EEG = {"data": data, "srate": 100.0, "nbchan": 2, "pnts": 20, "chanlocs": [{"labels": "Cz"}, {"labels": "Pz"}]}
    output_dir = tmp_path / "not_yet_created"

    with pytest.raises(ValueError, match="NaN or Inf"):
        pop_writeeeg(EEG, output_dir / "out.edf")

    assert not output_dir.exists()


def test_num_jobs_from_reservation_names_psutil_and_pyodide(monkeypatch: pytest.MonkeyPatch) -> None:
    """A RAM reservation needs psutil; the error keeps the real reason and says where psutil is absent."""
    monkeypatch.setitem(sys.modules, "psutil", None)

    with pytest.raises(ImportError) as excinfo:
        num_jobs_from_reservation("4GB")

    assert "psutil is required to determine available system RAM" in str(excinfo.value)
    assert "Pyodide" in str(excinfo.value)
    assert "None in sys.modules" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, ImportError)


def test_asr_runs_with_an_explicit_budget_and_names_maxmem_without_psutil(tmp_path: Path) -> None:
    """ASR needs psutil only for its free-memory fallback; an explicit budget works in a browser install."""
    completed = _run_without_optional_extras(
        """
        import json
        import numpy as np
        from eegprep.plugins.clean_rawdata.asr_calibrate import asr_calibrate
        from eegprep.plugins.clean_rawdata.asr_process import asr_process

        rng = np.random.default_rng(0)
        srate = 100.0
        calibration = rng.normal(size=(4, 3000))
        state = asr_calibrate(calibration, srate, maxmem=64)
        data = rng.normal(size=(4, 500))
        cleaned, _ = asr_process(data, srate, state, max_mem=64)
        try:
            asr_process(data, srate, state)
            message = "no ImportError"
        except ImportError as exc:
            message = str(exc)
        print(json.dumps({"shape": list(cleaned.shape), "message": message}))
        """,
        tmp_path,
    )

    result = _json_result(completed)
    assert result["shape"] == [4, 500]
    assert "psutil is required to size the ASR memory budget" in result["message"]
    assert "maxmem" in result["message"]


def test_asr_too_low_budget_names_every_budget_argument_without_psutil(tmp_path: Path) -> None:
    """A budget too small for the lookahead buffers falls back to free memory, which needs psutil."""
    completed = _run_without_optional_extras(
        """
        import json
        import numpy as np
        from eegprep.plugins.clean_rawdata.asr_calibrate import asr_calibrate
        from eegprep.plugins.clean_rawdata.asr_calibrate_r import asr_calibrate_r
        from eegprep.plugins.clean_rawdata.asr_process import asr_process
        from eegprep.plugins.clean_rawdata.asr_process_r import asr_process_r

        rng = np.random.default_rng(0)
        srate = 100.0
        calibration = rng.normal(size=(4, 3000))
        data = rng.normal(size=(4, 500))
        euclidean = asr_calibrate(calibration, srate, maxmem=64)
        riemannian = asr_calibrate_r(calibration, srate, maxmem=64)
        calls = {
            "asr_process": lambda: asr_process(data, srate, euclidean, max_mem=0.001),
            "asr_process_r": lambda: asr_process_r(data, srate, riemannian, max_mem=0.001),
        }
        messages = {}
        for name, call in calls.items():
            try:
                call()
            except ImportError as exc:
                messages[name] = str(exc)
        print(json.dumps(messages))
        """,
        tmp_path,
    )

    messages = _json_result(completed)
    assert set(messages) == {"asr_process", "asr_process_r"}
    for message in messages.values():
        assert "psutil is required to size the ASR memory budget" in message
        assert "maxmem in clean_asr, MaxMem in clean_artifacts" in message


def test_dockerfile_installs_bids_and_mne_for_the_entrypoint() -> None:
    dockerfile = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    entry = (REPO_ROOT / "scripts" / "bids_minimal_preproc.py").read_text(encoding="utf-8")
    assert "--jobs" in entry and "4GB" in entry
    assert ".[pybids,mne]" in dockerfile or '".[pybids,mne]"' in dockerfile
