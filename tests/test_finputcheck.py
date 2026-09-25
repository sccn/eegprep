from __future__ import annotations

import math

import numpy as np
import pytest

from eegprep.functions.guifunc.finputcheck import finputcheck
from tests.eeglab_tests import MATLAB_TEST_EPSILON, eeglab_test, load_matlab_test_fixture


RULES = (
    ("the2test", "string", [], "bar"),
    ("key1", "integer", [1, 5], 2),
    ("p3rcent", "real", [0, 1], 1 / math.sqrt(2)),
    ("forth", "cell", [], []),
)
FINPUTCHECK_WRAPPER = "unittesting_guifunc/finputcheck/guifunc_finputcheck_wrapperTest.m"


# The two upstream fail_* bodies are entirely commented out, so these Python
# argument-validation checks are supplemental regressions, not source ports.
def test_finputcheck_requires_arguments_and_field_rules():
    with pytest.raises(TypeError):
        finputcheck()


def test_finputcheck_reports_an_incomplete_key_value_sequence():
    result = finputcheck(["key1", 3, "the2test"], RULES)

    assert result == "error: bad 'key', 'val' sequence"


def test_finputcheck_uses_defaults_for_an_empty_argument_list():
    rules = (*RULES[:3], ("forth", "cell", [], ["test", 4]))

    result = finputcheck([], rules)

    assert result == {
        "the2test": "bar",
        "key1": 2,
        "p3rcent": 1 / math.sqrt(2),
        "forth": ["test", 4],
    }


def test_finputcheck_validates_all_supported_rule_values():
    arguments = ["key1", 3, "the2test", "foo", "p3rcent", 0.4937, "forth", ["a", 2, "11b", "D"]]

    result = finputcheck(arguments, RULES)

    assert result == {"key1": 3, "the2test": "foo", "p3rcent": 0.4937, "forth": ["a", 2, "11b", "D"]}


def test_finputcheck_accepts_a_value_matching_any_declared_type():
    rules = (
        RULES[0],
        ("key1", ("integer", "string"), ([1, 5], ["test"]), 2),
        *RULES[2:],
    )
    arguments = ["key1", "test", "the2test", "foo", "p3rcent", 0.4937, "forth", ["a", 2, "11b", "D"]]

    result = finputcheck(arguments, rules)

    assert result == {"key1": "test", "the2test": "foo", "p3rcent": 0.4937, "forth": ["a", 2, "11b", "D"]}


def test_finputcheck_fills_only_omitted_values_from_defaults():
    rules = (*RULES[:3], ("forth", "cell", [], ["w", 8, "y", 2]))

    result = finputcheck(["key1", 3, "the2test", "foo"], rules)

    assert result == {
        "key1": 3,
        "the2test": "foo",
        "p3rcent": 1 / math.sqrt(2),
        "forth": ["w", 8, "y", 2],
    }


def test_finputcheck_accepts_declared_string_choices_case_insensitively():
    rules = (("the2test", "string", ["foo", "bar"], "baz"), *RULES[1:])
    arguments = ["key1", 3, "the2test", "foo", "p3rcent", 0.4937, "forth", ["a", 2, "11b", "D"]]

    result = finputcheck(arguments, rules)

    assert result == {"key1": 3, "the2test": "foo", "p3rcent": 0.4937, "forth": ["a", 2, "11b", "D"]}


def test_finputcheck_can_return_unrecognized_arguments_in_ignore_mode():
    arguments = [
        "key1",
        3,
        "the2test",
        "foo",
        "p3rcent",
        0.4937,
        "forth",
        ["a", 2, "11b", "D"],
        "invisible",
        "true",
    ]

    result, residual = finputcheck(
        arguments,
        RULES,
        "testfunction",
        "ignore",
        return_unrecognized=True,
    )

    assert result == {
        "key1": 3,
        "the2test": "foo",
        "p3rcent": 0.4937,
        "forth": ["a", 2, "11b", "D"],
        "invisible": "true",
    }
    assert residual == ["invisible", "true"]


def test_finputcheck_returns_eeglab_error_strings_for_invalid_values():
    assert finputcheck(["p3rcent", 2], RULES) == "error: value out of range for argument 'p3rcent'"
    assert finputcheck(["the2test", 2], RULES) == "error: argument 'the2test' must be a string"
    assert finputcheck(["unknown", 1], RULES) == "error: undefined argument 'unknown'"


def test_finputcheck_keeps_the_last_duplicate_value(caplog):
    caplog.set_level("INFO")

    result = finputcheck(["key1", 2, "key1", 3], RULES)

    assert result["key1"] == 3
    assert "keeping the last" in caplog.text


def _cell_row(*values):
    row = np.empty((1, len(values)), dtype=object)
    for index, value in enumerate(values):
        row[0, index] = value
    return row


def _source_finputcheck_case(case):
    empty = np.empty((0, 0))
    forth = _cell_row("a", np.array([[2.0]]), "11b", "D")
    correct = {"key1": np.array([[3.0]]), "the2test": "foo", "p3rcent": np.array([[0.4937]]), "forth": forth}
    rules = np.empty((4, 4), dtype=object)
    rules[0] = ["the2test", "string", empty, "bar"]
    rules[1] = ["key1", "integer", np.array([[1.0, 5.0]]), 2.0]
    rules[2] = ["p3rcent", "real", np.array([[0.0, 1.0]]), 1 / math.sqrt(2)]
    rules[3] = ["forth", "cell", empty, empty]
    arguments = _cell_row("key1", 3.0, "the2test", "foo", "p3rcent", 0.4937, "forth", forth)
    if case == "empty":
        arguments = np.empty((0, 0), dtype=object)
        rules[3, 3] = _cell_row("test", np.array([[4.0]]))
        correct = {
            "key1": np.array([[2.0]]),
            "the2test": "bar",
            "p3rcent": np.array([[1 / math.sqrt(2)]]),
            "forth": rules[3, 3],
        }
    elif case == "multiple_types":
        rules[1, 1] = np.array([["integer"], ["string"]], dtype=object)
        rules[1, 2] = _cell_row(np.array([[1.0, 5.0]]), "test").T
        arguments[0, 1] = "test"
        correct["key1"] = "test"
    elif case == "standard":
        arguments = _cell_row("key1", 3.0, "the2test", "foo")
        rules[3, 3] = _cell_row("w", np.array([[8.0]]), "y", np.array([[2.0]]))
        correct["p3rcent"] = np.array([[1 / math.sqrt(2)]])
        correct["forth"] = rules[3, 3]
    elif case == "strings":
        rules[0, 2] = _cell_row("foo", "bar")
        rules[0, 3] = "baz"
        correct["the2test"] = "bar"
    elif case == "unknown":
        arguments = np.concatenate((arguments, _cell_row("invisible", "true")), axis=1)
        correct["invisible"] = "true"
    return arguments, rules, correct


def _source_near_struct(actual, expected):
    if not isinstance(actual, dict) or set(actual) != set(expected):
        return False
    for name, value in expected.items():
        result = actual[name]
        if isinstance(value, str):
            if result != value:
                return False
        elif value.dtype == object:
            if not np.array_equal(result, value):
                return False
        else:
            result = np.asarray(result)
            if result.shape != value.shape or not np.all(
                (value - MATLAB_TEST_EPSILON <= result) & (result <= value + MATLAB_TEST_EPSILON)
            ):
                return False
    return True


@eeglab_test(FINPUTCHECK_WRAPPER, "test_pass_empty")
@eeglab_test(FINPUTCHECK_WRAPPER, "test_pass_general")
@eeglab_test(FINPUTCHECK_WRAPPER, "test_pass_multiple_types")
@eeglab_test(FINPUTCHECK_WRAPPER, "test_pass_standard")
@eeglab_test(FINPUTCHECK_WRAPPER, "test_pass_strings")
@eeglab_test(FINPUTCHECK_WRAPPER, "test_pass_unknown")
def test_reference_finputcheck_original_wrapper_cases(eeglab_backend):
    for case in ("empty", "general", "multiple_types", "standard", "strings", "unknown"):
        arguments, rules, correct = _source_finputcheck_case(case)
        if case == "unknown":
            result, residual = eeglab_backend("finputcheck", arguments, rules, "testfunction", "ignore", nargout=2)
            np.testing.assert_equal(residual, _cell_row("invisible", "true"))
        else:
            result = eeglab_backend("finputcheck", arguments, rules)
        assert _source_near_struct(result, correct) == (case != "strings")


def _assert_finputcheck_fixture(actual, expected, atol, rtol):
    if expected.dtype.names:
        assert set(actual) == set(expected.dtype.names)
        for name in expected.dtype.names:
            _assert_finputcheck_fixture(actual[name], expected[name][0, 0], atol, rtol)
    elif expected.dtype == object:
        assert actual.shape == expected.shape
        for result, reference in zip(actual.flat, expected.flat, strict=True):
            _assert_finputcheck_fixture(result, reference, atol, rtol)
    elif expected.dtype.kind in "US":
        assert actual == expected.item()
    else:
        assert actual.shape == expected.shape
        assert actual.dtype == expected.dtype
        np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)


@eeglab_test("regression_tests/t_finputcheck.m", "test_1")
@eeglab_test("regression_tests/t_finputcheck.m", "test_2")
@eeglab_test("regression_tests/t_finputcheck.m", "test_3")
@eeglab_test("regression_tests/t_finputcheck.m", "test_4")
@eeglab_test("regression_tests/t_finputcheck.m", "test_5")
@eeglab_test("regression_tests/t_finputcheck.m", "test_6")
def test_reference_finputcheck_original_regression_fixture(eeglab_backend, eeglab_suite_root):
    fixture = load_matlab_test_fixture(eeglab_suite_root / "regression_tests/t_finputcheck.mat")
    for index in range(6):
        arguments = fixture["inputs"][0, index][0]
        result = eeglab_backend("finputcheck", *arguments, nargout=2 if index == 5 else 1)
        if index == 5:
            result = _cell_row(*result)
        reference = fixture[f"test_{index + 1}"][0, 0]
        _assert_finputcheck_fixture(result, reference["value"], reference["absTol"].item(), reference["relTol"].item())
