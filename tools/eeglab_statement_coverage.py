"""Freeze the approved MATLAB source scope and run native statement coverage."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
from pathlib import Path

from tools.eeglab_test_port_audit import discover_matlab_test_scenarios, validate_suite_checkout


# Importer implementations remain in scope. These are dependency internals,
# not exclusions based on what EEGPrep supports or what a test happens to call.
DEPENDENCIES = (
    "plugins/clean_rawdata/manopt/",
    "plugins/ICLabel/matconvnet/",
    "plugins/EEG-BIDS/JSONio/",
    "plugins/bids-matlab-tools8.0/JSONio/",
    "plugins/LIMO4.1.2/external/",
    "plugins/Fieldtrip-lite250523/",
)
REQUIRED_PLUGINS = (
    "clean_rawdata",
    "firfilt",
    "ICLabel",
    "dipfit",
    "EEG-BIDS",
    "LIMO4.1.2",
    "PICARD2.0",
    "BDFimport1.2",
    "Biosig3.8.5",
    "Fileio260210",
    "bva-io",
    "neuroscanio",
    "erpssimport1.03",
    "bids-matlab-tools8.0",
)


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _git(directory: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(directory), *args], text=True).strip()


def source_inventory(eeglab_root: Path) -> list[dict]:
    """List the fixed source universe, including never-executed MATLAB files."""
    missing = [name for name in REQUIRED_PLUGINS if not (eeglab_root / "plugins" / name).is_dir()]
    if missing:
        raise ValueError(f"Missing approved plugins: {', '.join(missing)}")
    records = []
    for path in sorted(eeglab_root.rglob("*")):
        relative = path.relative_to(eeglab_root)
        if not path.is_file() or any(part.startswith(".") for part in relative.parts):
            continue
        name = relative.as_posix()
        if len(relative.parts) != 1 and relative.parts[0] not in {"functions", "plugins"}:
            continue
        if path.suffix == ".m":
            category = "dependency" if name.startswith(DEPENDENCIES) else "statement_scope"
            if (
                relative.parts[0] == "plugins"
                and relative.parts[1] not in REQUIRED_PLUGINS
                and category != "dependency"
            ):
                category = "outside_approved_plugin_scope"
        elif path.suffix.startswith(".mex") or path.suffix in {".p", ".dll", ".so", ".dylib"}:
            category = "native_or_opaque"
        else:
            continue
        family = "/".join(relative.parts[:2]) if len(relative.parts) > 1 else "root"
        records.append({"path": name, "family": family, "category": category, "sha256": _sha256(path)})
    return records


def freeze_manifest(suite_root: Path) -> dict:
    """Freeze pins, all source hashes, and native test source provenance."""
    validate_suite_checkout(suite_root)
    eeglab_root = suite_root / "eeglab"
    components = []
    for directory in [eeglab_root, *sorted((eeglab_root / "plugins").iterdir())]:
        if not directory.is_dir() or directory.name.startswith("."):
            continue
        components.append(
            {
                "path": directory.relative_to(eeglab_root).as_posix(),
                "git_commit": _git(directory, "rev-parse", "HEAD") if (directory / ".git").exists() else None,
                "provenance": "git checkout"
                if (directory / ".git").exists()
                else "installed directory; content hashes below",
            }
        )
    tests = sorted({item.source for item in discover_matlab_test_scenarios(suite_root)})
    return {
        "schema_version": 1,
        "metric": "statement",
        "suite_root": str(suite_root),
        "suite_commit": _git(suite_root, "rev-parse", "HEAD"),
        "eeglab_root": str(eeglab_root),
        "components": components,
        "sources": source_inventory(eeglab_root),
        "test_files": tests,
        "test_sources": [
            {"path": path.relative_to(suite_root).as_posix(), "sha256": _sha256(path)}
            for path in sorted(suite_root.rglob("*.m"))
            if not path.is_relative_to(eeglab_root)
            and not any(part.startswith(".") for part in path.relative_to(suite_root).parts)
        ],
        "worker_coverage": "Client coverage only; parallel-worker statement hits have not been independently verified.",
    }


def verify_manifest(manifest: dict, runtime_root: Path | None = None) -> None:
    """Reject source edits, additions, and removals after denominator freezing."""
    if source_inventory(runtime_root or Path(manifest["eeglab_root"])) != manifest["sources"]:
        raise ValueError("MATLAB source inventory changed after freeze; create and audit a new manifest")


def _baseline_record(directory: Path, manifest: dict, runtime_root: Path) -> dict:
    directory = directory.resolve()
    config = json.loads((directory / "run.json").read_text())
    report = json.loads((directory / "report.json").read_text())
    for field in ("metric", "sources", "test_files", "test_sources"):
        if config[field] != manifest[field]:
            raise ValueError(f"Coverage baseline has a different frozen {field}")
    if Path(config["eeglab_root"]).resolve() != runtime_root:
        raise ValueError("Coverage union requires the same verified runtime path")
    if not report["execution_complete"]:
        raise ValueError("Coverage baseline must be a completed measurement, not a timed-out batch")
    return {
        "directory": str(directory),
        "artifacts": {name: _sha256(directory / name) for name in ("run.json", "report.json", "coverage.mat")},
        "matlab_version": report["matlab_version"],
        "covered_statements": report["covered_statements"],
        "native_failures": any(item["failed"] or item["incomplete"] for item in report["tests"])
        or bool(report.get("baseline", {}).get("native_failures", False)),
    }


def _unmeasurable_approval(path: Path, manifest: dict) -> dict:
    approval = json.loads(path.read_text())
    sources = {item["path"]: item for item in manifest["sources"] if item["category"] == "statement_scope"}
    names = [item["path"] for item in approval["sources"]]
    if not names or len(set(names)) != len(names):
        raise ValueError("Unmeasurable-source approval requires a nonempty list of distinct source files")
    for item in approval["sources"]:
        if item["path"] not in sources or sources[item["path"]]["sha256"] != item["sha256"]:
            raise ValueError(f"Unmeasurable-source approval differs from frozen scope: {item['path']}")
    return {**approval, "approval_file": str(path.resolve()), "approval_sha256": _sha256(path)}


def run_native(
    manifest_path: Path,
    test_root: Path,
    selected: list[str],
    output: Path,
    matlab: str,
    timeout: float,
    support_paths: list[str],
    runtime_root: Path,
    additional_tests: list[Path] | None = None,
    baseline_run: Path | None = None,
    approved_unmeasurable: Path | None = None,
) -> int:
    """Run one explicit native suite batch with a finite wall-clock bound."""
    test_root, runtime_root, output = test_root.resolve(), runtime_root.resolve(), output.resolve()
    manifest = json.loads(manifest_path.read_text())
    verify_manifest(manifest)
    reference_root = Path(manifest["suite_root"]).resolve()
    if test_root.is_relative_to(reference_root) or runtime_root.is_relative_to(reference_root):
        raise ValueError("Use a writable scratch test-suite copy, not the pinned reference checkout")
    if output.is_relative_to(reference_root):
        raise ValueError("Keep coverage artifacts outside the pinned reference checkout")
    if (
        list(test_root.glob("*.prj"))
        or (test_root / "resources/project").exists()
        or (test_root / ".SimulinkProject").exists()
    ):
        raise ValueError("Use projectless scratch tests: omit project metadata, not scientific test sources or data")
    verify_manifest(manifest, runtime_root)
    nested_runtime = test_root / "eeglab"
    if nested_runtime.exists() and nested_runtime.resolve() != runtime_root.resolve():
        raise ValueError("Scratch tests/eeglab must be the verified runtime root, not a different EEGLAB tree")
    additional_tests = [path.resolve() for path in additional_tests or []]
    if (not selected and not additional_tests) or set(selected) - set(manifest["test_files"]):
        raise ValueError("Select at least one inventoried or additional native test file")
    additional_names = [path.name for path in additional_tests]
    if len(set(additional_names)) != len(additional_names):
        raise ValueError("Additional native tests must have distinct filenames")
    for path in additional_tests:
        if not path.is_file() or path.suffix != ".m":
            raise ValueError(f"Additional native test must be an existing .m file: {path}")
        if path.is_relative_to(reference_root) or path.is_relative_to(runtime_root):
            raise ValueError("Additional tests must be separate from the pinned reference and runtime source")
    for item in manifest["test_sources"]:
        copied = test_root / item["path"]
        if not copied.is_file() or not copied.resolve().is_relative_to(test_root) or _sha256(copied) != item["sha256"]:
            raise ValueError(f"Scratch test source is missing or differs from the frozen source: {copied}")
    baseline = _baseline_record(baseline_run, manifest, runtime_root) if baseline_run else {}
    approval = _unmeasurable_approval(approved_unmeasurable, manifest) if approved_unmeasurable else {}
    output.mkdir(parents=True, exist_ok=False)
    additions = []
    if additional_tests:
        (output / "additional_tests").mkdir()
    for path in additional_tests:
        # Copy the exact bytes before MATLAB starts; subsequent worktree edits
        # cannot change an in-flight test or its retained provenance.
        content = path.read_bytes()
        snapshot = Path("additional_tests") / path.name
        (output / snapshot).write_bytes(content)
        additions.append(
            {"source": str(path), "snapshot": snapshot.as_posix(), "sha256": hashlib.sha256(content).hexdigest()}
        )
    home_options = Path.home() / "eeg_options.m"
    home_hash = _sha256(home_options) if home_options.exists() else None
    runner = Path(__file__).with_suffix(".m")
    config = {
        **manifest,
        "test_root": str(test_root),
        "selected": selected,
        "additional_test_sources": additions,
        "baseline": baseline,
        "unmeasurable_approval": approval,
        "support_paths": support_paths,
        "output": str(output),
        "reference_eeglab_root": manifest["eeglab_root"],
        "eeglab_root": str(runtime_root),
        "home_options": str(home_options) if home_hash else "",
        "home_options_sha256": home_hash,
        "runner_sha256": _sha256(runner),
    }
    config_path = output / "run.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    # Freeze the exact native runner too: editing its worktree during a long
    # run must not change the code producing the retained measurement.
    shutil.copyfile(runner, output / runner.name)
    tool_dir = str(output).replace("'", "''")
    quoted_config = str(config_path).replace("'", "''")
    expression = f"addpath('{tool_dir}'); eeglab_statement_coverage('{quoted_config}')"
    with (output / "matlab.log").open("w") as log:
        process = subprocess.Popen(
            [matlab, "-batch", expression], stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            code = process.wait(timeout=timeout)
            status = "finished" if code == 0 else "failed"
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            code, status = 124, "timeout; selected native batch incomplete"
    home_unchanged = (_sha256(home_options) if home_options.exists() else None) == home_hash
    (output / "driver.json").write_text(
        json.dumps(
            {
                "process_status": status,
                "exit_code": code,
                "timeout_seconds": timeout,
                "home_options_unchanged": home_unchanged,
            },
            indent=2,
        )
        + "\n"
    )
    verify_manifest(manifest)
    verify_manifest(manifest, runtime_root)
    for item in additions:
        if _sha256(output / item["snapshot"]) != item["sha256"]:
            raise ValueError(f"Additional native test snapshot changed during execution: {item['snapshot']}")
    for name, digest in baseline.get("artifacts", {}).items():
        if _sha256(Path(baseline["directory"]) / name) != digest:
            raise ValueError(f"Coverage baseline changed during execution: {name}")
    if not home_unchanged:
        raise ValueError("Native tests changed home eeg_options.m; preserve evidence and investigate")
    return code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--suite-root", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("run")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--test-root", type=Path, required=True)
    run.add_argument("--runtime-root", type=Path, required=True)
    run.add_argument("--test-file", action="append", default=[])
    run.add_argument("--additional-test-file", type=Path, action="append", default=[])
    run.add_argument("--baseline-run", type=Path)
    run.add_argument("--approved-unmeasurable", type=Path)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--matlab", default="matlab")
    run.add_argument("--timeout", type=float, required=True)
    run.add_argument("--support-path", action="append", default=[])
    args = parser.parse_args()
    if args.command == "freeze":
        manifest = freeze_manifest(args.suite_root.resolve())
        with args.output.open("x") as stream:
            json.dump(manifest, stream, indent=2)
            stream.write("\n")
        return 0
    return run_native(
        args.manifest.resolve(),
        args.test_root,
        args.test_file,
        args.output,
        args.matlab,
        args.timeout,
        args.support_path,
        args.runtime_root,
        args.additional_test_file,
        args.baseline_run,
        args.approved_unmeasurable,
    )


if __name__ == "__main__":
    raise SystemExit(main())
