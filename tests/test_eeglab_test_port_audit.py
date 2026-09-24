from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path

import pytest

from tools.eeglab_test_port_audit import (
    AuditInputError,
    CollectedReference,
    MatlabTestScenario,
    audit_test_ports,
    compare_test_ports,
    discover_matlab_test_scenarios,
    format_report,
    validate_suite_checkout,
)


def test_discovers_wrapper_regression_and_nonstandard_limo_methods(tmp_path: Path) -> None:
    suite_root = _source_fixture(tmp_path)

    scenarios = discover_matlab_test_scenarios(suite_root)

    assert scenarios == {
        MatlabTestScenario("regression_tests/t_regression.m", "testRegression"),
        MatlabTestScenario("unittesting_limo/limo_wrapperTest.m", "limo_test1"),
        MatlabTestScenario("unittesting_limo/limo_wrapperTest.m", "limo_test2"),
        MatlabTestScenario("unittesting_miscfunc/example/example_wrapperTest.m", "test_alpha"),
        MatlabTestScenario("unittesting_miscfunc/example/example_wrapperTest.m", "test_beta"),
    }


def test_discovers_class_test_attributes_not_names_and_excludes_helpers(tmp_path: Path) -> None:
    source = tmp_path / "unittesting_statistics/statcond/statcondTest.m"
    source.parent.mkdir(parents=True)
    source.write_text(
        textwrap.dedent(
            """\
            classdef statcondTest < matlab.unittest.TestCase
                methods(TestClassSetup)
                    function testSetup(testCase)
                    end
                end
                methods(Test, TestTags = {'VectorReference'})
                    function paired1Anova(testCase, data)
                    end
                end
                methods(Test, ParameterCombination = 'exhaustive')
                    function shuffleAndPermutation(testCase, method, data)
                    end
                end
                methods(TestMethodTeardown)
                    function testTeardown(testCase)
                    end
                end
                methods(Static)
                    function testNamedHelper()
                    end
                end
            end
            """
        ),
        encoding="utf-8",
    )

    assert discover_matlab_test_scenarios(tmp_path) == {
        MatlabTestScenario("unittesting_statistics/statcond/statcondTest.m", "paired1Anova"),
        MatlabTestScenario("unittesting_statistics/statcond/statcondTest.m", "shuffleAndPermutation"),
    }


def test_discovers_function_suites_outside_wrapper_and_regression_patterns(tmp_path: Path) -> None:
    source = tmp_path / "newTests.m"
    source.write_text(
        textwrap.dedent(
            """\
            function tests = newTests
            tests = functiontests(localfunctions);
            function testBefore(~)
            function afterTest(~)
            function TestUppercase(~)
            function setupOnce(~)
            function helper(~)
            """
        ),
        encoding="utf-8",
    )

    assert discover_matlab_test_scenarios(tmp_path) == {
        MatlabTestScenario("newTests.m", "testBefore"),
        MatlabTestScenario("newTests.m", "afterTest"),
        MatlabTestScenario("newTests.m", "TestUppercase"),
    }


def test_discovers_unwrapped_limo_workflow_without_counting_called_helper(tmp_path: Path) -> None:
    limo_root = tmp_path / "unittesting_limo"
    limo_root.mkdir()
    (limo_root / "limo_zIRLS_validation_4_Arno.m").write_text(
        "clear variables\nresults = limo_test_glmboot(chanlocs, H0iw);\n", encoding="utf-8"
    )
    (limo_root / "limo_test_glmboot.m").write_text(
        "function results = limo_test_glmboot(chanlocs, varargin)\n", encoding="utf-8"
    )

    expected = discover_matlab_test_scenarios(tmp_path)
    workflow = MatlabTestScenario("unittesting_limo/limo_zIRLS_validation_4_Arno.m", "limo_zIRLS_validation_4_Arno")
    assert expected == {workflow}
    report = compare_test_ports(
        tmp_path,
        expected,
        [
            CollectedReference(workflow.source, workflow.test, "suite", "eeglab", "tests/test_limo.py::test_irls"),
            CollectedReference(
                "unittesting_limo/limo_test_glmboot.m",
                "limo_test_glmboot",
                "suite",
                "eeglab",
                "tests/test_limo.py::test_irls",
            ),
        ],
        suite_commit="suite",
        expected_eeglab_commit="eeglab",
    )
    assert report.ok
    assert report.covered == (workflow,)


def test_discovery_ignores_commented_definitions_and_reference_checkout(tmp_path: Path) -> None:
    suite = _source_fixture(tmp_path)
    source = suite / "commentedTest.m"
    source.write_text(
        "%{\nfunction tests = commentedTest\ntests = functiontests(localfunctions);\nfunction testFake(~)\n%}\n",
        encoding="utf-8",
    )
    for directory in ("eeglab", ".cache"):
        nested = suite / directory
        nested.mkdir()
        (nested / "internal_wrapperTest.m").write_text(
            "function tests = internal_wrapperTest\ntests = functiontests(localfunctions);\nfunction testFake(~)\n",
            encoding="utf-8",
        )

    assert len(discover_matlab_test_scenarios(suite)) == 5


def test_audit_collects_pytest_provenance_normalizes_leaf_and_reports_exact_gaps(tmp_path: Path) -> None:
    suite_root = _source_fixture(tmp_path / "suite")
    suite_commit = _initialize_git_checkout(suite_root)
    repo_root = tmp_path / "eegprep"
    _write_pytest_fixture(repo_root, suite_commit)

    report = audit_test_ports(
        suite_root,
        repo_root,
        expected_suite_commit=suite_commit,
        expected_eeglab_commit=None,
    )

    assert len(report.expected) == 5
    assert set(report.covered) == {
        MatlabTestScenario("regression_tests/t_regression.m", "testRegression"),
        MatlabTestScenario("unittesting_miscfunc/example/example_wrapperTest.m", "test_alpha"),
        MatlabTestScenario("unittesting_miscfunc/example/example_wrapperTest.m", "test_beta"),
    }
    assert set(report.missing) == {
        MatlabTestScenario("unittesting_limo/limo_wrapperTest.m", "limo_test1"),
        MatlabTestScenario("unittesting_limo/limo_wrapperTest.m", "limo_test2"),
    }
    assert report.invalid_references == ()
    assert "Expected: 5; covered: 3; missing: 2" in format_report(report)
    assert "unittesting_limo/limo_wrapperTest.m::limo_test1" in format_report(report)


def test_suite_validation_rejects_wrong_commit(tmp_path: Path) -> None:
    suite_root = _source_fixture(tmp_path)
    actual_commit = _initialize_git_checkout(suite_root)

    with pytest.raises(AuditInputError, match=f"at {actual_commit}; expected pinned commit deadbeef"):
        validate_suite_checkout(
            suite_root,
            expected_suite_commit="deadbeef",
            expected_eeglab_commit=None,
        )


def test_comparison_rejects_missing_stale_and_wrong_commit_provenance(tmp_path: Path) -> None:
    suite_root = _source_fixture(tmp_path)
    supplemental = suite_root / "supplemental.m"
    supplemental.write_text("function supplementalCase(~)\n", encoding="utf-8")
    expected = discover_matlab_test_scenarios(suite_root)
    references = (
        CollectedReference(
            source="supplemental.m",
            test="supplementalCase",
            suite_commit="current",
            eeglab_commit="eeglab",
            nodeid="tests/test_ports.py::test_supplemental",
        ),
        CollectedReference(
            source="unittesting_miscfunc/example/removed.m",
            test="test_alpha",
            suite_commit="current",
            eeglab_commit="eeglab",
            nodeid="tests/test_ports.py::test_removed",
        ),
        CollectedReference(
            source="unittesting_miscfunc/example/example_wrapperTest.m",
            test="test_beta",
            suite_commit="stale",
            eeglab_commit="eeglab",
            nodeid="tests/test_ports.py::test_stale_commit",
        ),
    )

    report = compare_test_ports(
        suite_root,
        expected,
        references,
        suite_commit="current",
        expected_eeglab_commit="eeglab",
    )

    messages = "\n".join(report.invalid_references)
    assert "references a source absent from the pinned suite" in messages
    assert "pins suite commit stale, expected current" in messages
    assert not report.ok


def _source_fixture(root: Path) -> Path:
    wrapper_root = root / "unittesting_miscfunc/example"
    wrapper_root.mkdir(parents=True)
    (wrapper_root / "example_wrapperTest.m").write_text(
        textwrap.dedent(
            """\
            function tests = example_wrapperTest
            tests = functiontests(localfunctions);

            function test_alpha(~)
            alpha

            function test_beta(~)
            beta

            function helper_not_a_case(~)
            """
        ),
        encoding="utf-8",
    )
    (wrapper_root / "alpha.m").write_text("function alpha()\n", encoding="utf-8")
    (wrapper_root / "beta.m").write_text("function beta()\n", encoding="utf-8")

    regression_root = root / "regression_tests"
    regression_root.mkdir()
    (regression_root / "t_regression.m").write_text(
        textwrap.dedent(
            """\
            classdef t_regression
                methods (Test)
                    function testRegression(~)
                    end
                end
                methods
                    function helper(~)
                    end
                end
            end
            """
        ),
        encoding="utf-8",
    )

    limo_root = root / "unittesting_limo"
    limo_root.mkdir()
    (limo_root / "limo_wrapperTest.m").write_text(
        textwrap.dedent(
            """\
            function tests = limo_wrapperTest
            tests = functiontests(localfunctions);

            function limo_test1(~)

            function limo_test2(~)
            """
        ),
        encoding="utf-8",
    )
    return root


def _initialize_git_checkout(suite_root: Path) -> str:
    _git(suite_root, "init")
    _git(suite_root, "config", "user.email", "tests@example.com")
    _git(suite_root, "config", "user.name", "EEGPrep Tests")
    _git(suite_root, "remote", "add", "origin", "https://github.com/sccn/eeglab_tests.git")
    _git(suite_root, "add", ".")
    _git(suite_root, "commit", "-m", "fixture")
    return _git(suite_root, "rev-parse", "HEAD")


def _write_pytest_fixture(repo_root: Path, suite_commit: str) -> None:
    package_root = repo_root / "src/eegprep"
    package_root.mkdir(parents=True)
    (package_root / "__init__.py").write_text('AUDIT_SENTINEL = "requested checkout"\n', encoding="utf-8")
    tests_root = repo_root / "tests"
    tests_root.mkdir(parents=True)
    (tests_root / "test_ports.py").write_text(
        textwrap.dedent(
            f"""\
            from dataclasses import dataclass
            import eegprep

            assert eegprep.AUDIT_SENTINEL == "requested checkout"

            @dataclass(frozen=True)
            class Reference:
                source: str
                test: str
                suite_commit: str = {suite_commit!r}
                eeglab_commit: str = "fixture-eeglab"

            def eeglab_test(source, test):
                def decorate(function):
                    function.__eeglab_test_references__ = (Reference(source, test),)
                    return function
                return decorate

            @eeglab_test("unittesting_miscfunc/example/alpha.m", "test_alpha")
            def test_leaf_provenance():
                pass

            @eeglab_test("unittesting_miscfunc/example/example_wrapperTest.m", "test_beta")
            def test_wrapper_provenance():
                pass

            @eeglab_test("regression_tests/t_regression.m", "testRegression")
            def test_regression_provenance():
                pass
            """
        ),
        encoding="utf-8",
    )


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()
