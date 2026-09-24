"""Audit pytest ports against the pinned current EEGLAB test suite."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Sequence


EEGLAB_TESTS_REPOSITORY = "https://github.com/sccn/eeglab_tests.git"
EEGLAB_TESTS_COMMIT = "ff605546f3f70868916fb8d49c007472b3257b50"
EEGLAB_TESTS_EEGLAB_COMMIT = "8ac485f654d6bbb1a6acb8dc9ef3f2eaf3d409ba"
STALE_EEGLAB_TESTS_REPOSITORY = "https://github.com/sccn/eeglab-testcases.git"
REFERENCE_ATTRIBUTE = "__eeglab_test_references__"
PROVENANCE_OPTION = "--eeglab-provenance-output"
LIMO_TEST_NAMES = frozenset({"limo_test1", "limo_test2"})
# This scientific validation workflow is not called by a wrapper. Its helper
# limo_test_glmboot is part of the workflow, not a separate executable test.
STANDALONE_WORKFLOWS = ("unittesting_limo/limo_zIRLS_validation_4_Arno.m",)

_FUNCTION_RE = re.compile(
    r"^\s*function\s+(?:(?:\[[^\]\n]+\]|[A-Za-z]\w*)\s*=\s*)?(?P<name>[A-Za-z]\w*)",
    re.MULTILINE,
)
_CLASS_BLOCK_RE = re.compile(
    r"^\s*(?P<kind>methods|properties|events|enumeration)\b\s*(?:\((?P<attributes>[^)]*)\))?",
    re.MULTILINE,
)
_TEST_ATTRIBUTE_RE = re.compile(r"(?:^|,)\s*Test\s*(?:,|$)")


class AuditInputError(RuntimeError):
    """Raised when an audit input cannot identify the pinned source of truth."""


@dataclass(frozen=True, order=True)
class MatlabTestScenario:
    """One source method/workflow, not an expanded parameter case or assertion."""

    source: str
    test: str

    def as_text(self) -> str:
        return f"{self.source}::{self.test}"


@dataclass(frozen=True)
class CollectedReference:
    """One provenance reference attached to a collected pytest test."""

    source: str
    test: str
    suite_commit: str
    eeglab_commit: str
    nodeid: str


@dataclass(frozen=True)
class AuditReport:
    """Completeness result for one suite checkout and EEGPrep checkout."""

    suite_commit: str
    expected: tuple[MatlabTestScenario, ...]
    covered: tuple[MatlabTestScenario, ...]
    missing: tuple[MatlabTestScenario, ...]
    invalid_references: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.missing and not self.invalid_references

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "suite_commit": self.suite_commit,
            "expected_count": len(self.expected),
            "covered_count": len(self.covered),
            "missing_count": len(self.missing),
            "missing": [scenario.as_text() for scenario in self.missing],
            "invalid_references": list(self.invalid_references),
        }


def discover_matlab_test_scenarios(suite_root: Path) -> set[MatlabTestScenario]:
    """Discover test definitions and unwrapped workflows in the reference suite.

    Class Test attributes, rather than file/method names, identify class tests.
    Parameter expansions and statements inside legacy workflows must still be
    checked when porting; a provenance match does not establish faithful behavior.
    The EEGLAB submodule is reference code, not part of the test-suite inventory.
    """
    scenarios: set[MatlabTestScenario] = set()
    for path in sorted(suite_root.rglob("*.m")):
        source = path.relative_to(suite_root).as_posix()
        if source.startswith("eeglab/") or any(part.startswith(".") for part in path.relative_to(suite_root).parts):
            continue
        text = _matlab_source(path)
        if re.search(r"^\s*classdef\b", text, re.MULTILINE):
            blocks = list(_CLASS_BLOCK_RE.finditer(text))
            for index, block in enumerate(blocks):
                if block["kind"] != "methods" or not _TEST_ATTRIBUTE_RE.search(block["attributes"] or ""):
                    continue
                end = blocks[index + 1].start() if index + 1 < len(blocks) else len(text)
                for function in _FUNCTION_RE.finditer(text, block.end(), end):
                    scenarios.add(MatlabTestScenario(source, function["name"]))
        elif re.search(r"\bfunctiontests\s*\(\s*localfunctions\s*\)", text):
            names = [match["name"] for match in _FUNCTION_RE.finditer(text)]
            for name in names[1:]:
                if name.lower().startswith("test") or name.lower().endswith("test") or name in LIMO_TEST_NAMES:
                    scenarios.add(MatlabTestScenario(source, name))
        elif source in STANDALONE_WORKFLOWS:
            scenarios.add(MatlabTestScenario(source, path.stem))
    return scenarios


def validate_suite_checkout(
    suite_root: Path,
    *,
    expected_suite_commit: str = EEGLAB_TESTS_COMMIT,
    expected_eeglab_commit: str | None = EEGLAB_TESTS_EEGLAB_COMMIT,
    expected_repository: str | None = EEGLAB_TESTS_REPOSITORY,
) -> str:
    """Validate that ``suite_root`` is the pinned, current source checkout."""
    if not suite_root.is_dir():
        raise AuditInputError(f"EEGLAB test checkout does not exist: {suite_root}")

    actual_commit = _git_output(suite_root, "rev-parse", "HEAD")
    if actual_commit != expected_suite_commit:
        raise AuditInputError(
            f"EEGLAB test checkout is at {actual_commit}; expected pinned commit {expected_suite_commit}"
        )

    if expected_repository is not None:
        actual_repository = _git_output(suite_root, "remote", "get-url", "origin")
        if _normalized_repository(actual_repository) != _normalized_repository(expected_repository):
            stale_note = (
                " (the eeglab-testcases repository is stale)"
                if _normalized_repository(actual_repository) == _normalized_repository(STALE_EEGLAB_TESTS_REPOSITORY)
                else ""
            )
            raise AuditInputError(
                f"EEGLAB test checkout origin is {actual_repository!r}, not {expected_repository!r}{stale_note}"
            )

    if expected_eeglab_commit is not None:
        eeglab_root = suite_root / "eeglab"
        actual_eeglab_commit = _git_output(eeglab_root, "rev-parse", "HEAD")
        if actual_eeglab_commit != expected_eeglab_commit:
            raise AuditInputError(f"EEGLAB submodule is at {actual_eeglab_commit}; expected {expected_eeglab_commit}")
    return actual_commit


def collect_pytest_references(repo_root: Path) -> tuple[CollectedReference, ...]:
    """Collect provenance from pytest items without executing their test bodies."""
    tests_root = repo_root / "tests"
    if not tests_root.is_dir():
        raise AuditInputError(f"pytest test directory does not exist: {tests_root}")

    with tempfile.TemporaryDirectory(prefix="eegprep-provenance-") as temporary_dir:
        output_path = Path(temporary_dir) / "references.json"
        environment = os.environ.copy()
        tool_root = Path(__file__).resolve().parents[1]
        python_paths = [str(repo_root / "src"), str(repo_root), str(tool_root)]
        if environment.get("PYTHONPATH"):
            python_paths.append(environment["PYTHONPATH"])
        environment.update(
            {
                "EEGPREP_SKIP_MATLAB": "1",
                "MPLBACKEND": "Agg",
                "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
                "PYTHONPATH": os.pathsep.join(python_paths),
                "QT_QPA_PLATFORM": "offscreen",
            }
        )
        command = [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "--eeglab-backend=python",
            "--quiet",
            "--disable-warnings",
            "-p",
            "tools.eeglab_test_port_audit",
            f"{PROVENANCE_OPTION}={output_path}",
            str(tests_root),
        ]
        completed = subprocess.run(
            command,
            cwd=repo_root,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            details = (completed.stderr or completed.stdout).strip()
            raise AuditInputError(f"pytest collection failed ({completed.returncode}): {details}")
        if not output_path.is_file():
            raise AuditInputError("pytest collection did not emit EEGLAB provenance")
        payload = json.loads(output_path.read_text(encoding="utf-8"))

    return tuple(CollectedReference(**entry) for entry in payload)


def compare_test_ports(
    suite_root: Path,
    expected: set[MatlabTestScenario],
    references: Sequence[CollectedReference],
    *,
    suite_commit: str,
    expected_eeglab_commit: str | None = EEGLAB_TESTS_EEGLAB_COMMIT,
) -> AuditReport:
    """Compare collected provenance with source scenarios and report exact gaps."""
    wrapper_lookup = _leaf_wrapper_lookup(expected)
    covered: set[MatlabTestScenario] = set()
    invalid: set[str] = set()

    for reference in references:
        label = f"{reference.nodeid}: {reference.source}::{reference.test}"
        if reference.suite_commit != suite_commit:
            invalid.add(f"{label} pins suite commit {reference.suite_commit}, expected {suite_commit}")
            continue
        if expected_eeglab_commit is not None and reference.eeglab_commit != expected_eeglab_commit:
            invalid.add(f"{label} pins EEGLAB commit {reference.eeglab_commit}, expected {expected_eeglab_commit}")
            continue

        source = PurePosixPath(reference.source)
        if source.is_absolute() or ".." in source.parts or source.suffix != ".m":
            invalid.add(f"{label} is not a relative MATLAB source path")
            continue
        if not (suite_root / Path(*source.parts)).is_file():
            invalid.add(f"{label} references a source absent from the pinned suite")
            continue

        scenario = MatlabTestScenario(source.as_posix(), reference.test)
        if scenario in expected:
            covered.add(scenario)
            continue

        leaf_key = (source.parent.as_posix(), source.stem, reference.test)
        candidates = wrapper_lookup.get(leaf_key, ())
        if len(candidates) == 1:
            covered.add(candidates[0])
        elif len(candidates) > 1:
            invalid.add(f"{label} ambiguously maps to multiple wrapper methods")
        elif reference.test in _matlab_function_names(suite_root / Path(*source.parts)):
            # Supporting helper references are valid provenance, but cannot
            # replace any of the independently discovered test definitions.
            continue
        else:
            invalid.add(f"{label} does not identify a MATLAB method in the pinned suite")

    missing = expected - covered
    return AuditReport(
        suite_commit=suite_commit,
        expected=tuple(sorted(expected)),
        covered=tuple(sorted(covered)),
        missing=tuple(sorted(missing)),
        invalid_references=tuple(sorted(invalid)),
    )


def audit_test_ports(
    suite_root: Path,
    repo_root: Path,
    *,
    expected_suite_commit: str = EEGLAB_TESTS_COMMIT,
    expected_eeglab_commit: str | None = EEGLAB_TESTS_EEGLAB_COMMIT,
    expected_repository: str | None = EEGLAB_TESTS_REPOSITORY,
) -> AuditReport:
    """Run the complete source-discovery and pytest-provenance audit."""
    suite_commit = validate_suite_checkout(
        suite_root,
        expected_suite_commit=expected_suite_commit,
        expected_eeglab_commit=expected_eeglab_commit,
        expected_repository=expected_repository,
    )
    expected = discover_matlab_test_scenarios(suite_root)
    references = collect_pytest_references(repo_root)
    return compare_test_ports(
        suite_root,
        expected,
        references,
        suite_commit=suite_commit,
        expected_eeglab_commit=expected_eeglab_commit,
    )


def format_report(report: AuditReport) -> str:
    """Format a concise human-readable audit report."""
    state = "PASS" if report.ok else "FAIL"
    lines = [
        f"EEGLAB test-port audit: {state}",
        f"Suite commit: {report.suite_commit}",
        f"Expected: {len(report.expected)}; covered: {len(report.covered)}; missing: {len(report.missing)}",
        "Counts describe source definitions only, not parameter cases, faithful ports, or MATLAB validation.",
    ]
    if report.missing:
        lines.append("Missing scenarios:")
        lines.extend(f"  {scenario.as_text()}" for scenario in report.missing)
    if report.invalid_references:
        lines.append("Invalid or stale provenance:")
        lines.extend(f"  {message}" for message in report.invalid_references)
    return "\n".join(lines)


def pytest_addoption(parser: Any) -> None:
    """Register the private output used by the collection subprocess."""
    parser.addoption(PROVENANCE_OPTION, action="store", default=None)


def pytest_collection_finish(session: Any) -> None:
    """Serialize decorator provenance from collected pytest items."""
    output = session.config.getoption(PROVENANCE_OPTION)
    if output is None:
        return

    records: list[dict[str, str]] = []
    for item in session.items:
        test_object = getattr(item, "obj", None)
        for reference in getattr(test_object, REFERENCE_ATTRIBUTE, ()):
            records.append(
                {
                    "source": str(reference.source),
                    "test": str(reference.test),
                    "suite_commit": str(reference.suite_commit),
                    "eeglab_commit": str(reference.eeglab_commit),
                    "nodeid": str(item.nodeid),
                }
            )
    Path(output).write_text(json.dumps(records, sort_keys=True), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the audit CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite_checkout", type=Path, help="checkout of sccn/eeglab_tests")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--json", action="store_true", help="emit a machine-readable report")
    arguments = parser.parse_args(argv)

    try:
        report = audit_test_ports(arguments.suite_checkout.resolve(), arguments.repo_root.resolve())
    except AuditInputError as error:
        if arguments.json:
            print(json.dumps({"ok": False, "input_error": str(error)}, indent=2, sort_keys=True))
        else:
            print(f"EEGLAB test-port audit input error: {error}", file=sys.stderr)
        return 2

    if arguments.json:
        print(json.dumps(report.to_jsonable(), indent=2, sort_keys=True))
    else:
        print(format_report(report))
    return 0 if report.ok else 1


def _matlab_function_names(path: Path) -> list[str]:
    return [match.group("name") for match in _FUNCTION_RE.finditer(_matlab_source(path))]


def _matlab_source(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    text = re.sub(r"^\s*%\{\s*$.*?^\s*%\}\s*$", "", text, flags=re.MULTILINE | re.DOTALL)
    return re.sub(r"^\s*%[^\n]*", "", text, flags=re.MULTILINE)


def _leaf_wrapper_lookup(
    expected: set[MatlabTestScenario],
) -> dict[tuple[str, str, str], tuple[MatlabTestScenario, ...]]:
    mutable: dict[tuple[str, str, str], list[MatlabTestScenario]] = {}
    for scenario in expected:
        source = PurePosixPath(scenario.source)
        if not source.name.endswith("wrapperTest.m") or not scenario.test.startswith("test_"):
            continue
        leaf_stem = scenario.test.removeprefix("test_")
        key = (source.parent.as_posix(), leaf_stem, scenario.test)
        mutable.setdefault(key, []).append(scenario)
    return {key: tuple(sorted(value)) for key, value in mutable.items()}


def _git_output(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        details = (completed.stderr or completed.stdout).strip()
        raise AuditInputError(f"cannot inspect git checkout at {repository}: {details}")
    return completed.stdout.strip()


def _normalized_repository(repository: str) -> str:
    normalized = repository.strip().removesuffix(".git").rstrip("/")
    if normalized.startswith("git@github.com:"):
        normalized = "https://github.com/" + normalized.removeprefix("git@github.com:")
    return normalized.lower()


if __name__ == "__main__":
    raise SystemExit(main())
