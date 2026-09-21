from __future__ import annotations

import math

import pytest

from eegprep.functions.guifunc.finputcheck import finputcheck
from tests.eeglab_tests import eeglab_test


RULES = (
    ("the2test", "string", [], "bar"),
    ("key1", "integer", [1, 5], 2),
    ("p3rcent", "real", [0, 1], 1 / math.sqrt(2)),
    ("forth", "cell", [], []),
)
FINPUTCHECK_WRAPPER = "unittesting_guifunc/finputcheck/guifunc_finputcheck_wrapperTest.m"


@eeglab_test(FINPUTCHECK_WRAPPER, "test_fail_no_arg")
@eeglab_test("unittesting_guifunc/finputcheck/fail_no_arg.m", "test_fail_no_arg")
def test_finputcheck_requires_arguments_and_field_rules():
    with pytest.raises(TypeError):
        finputcheck()


@eeglab_test(FINPUTCHECK_WRAPPER, "test_fail_no_key_val")
@eeglab_test("unittesting_guifunc/finputcheck/fail_no_key_val.m", "test_fail_no_key_val")
def test_finputcheck_reports_an_incomplete_key_value_sequence():
    result = finputcheck(["key1", 3, "the2test"], RULES)

    assert result == "error: bad 'key', 'val' sequence"


@eeglab_test(FINPUTCHECK_WRAPPER, "test_pass_empty")
@eeglab_test("regression_tests/t_finputcheck.m", "test_1")
@eeglab_test("unittesting_guifunc/finputcheck/pass_empty.m", "test_pass_empty")
def test_finputcheck_uses_defaults_for_an_empty_argument_list():
    rules = (*RULES[:3], ("forth", "cell", [], ["test", 4]))

    result = finputcheck([], rules)

    assert result == {
        "the2test": "bar",
        "key1": 2,
        "p3rcent": 1 / math.sqrt(2),
        "forth": ["test", 4],
    }


@eeglab_test(FINPUTCHECK_WRAPPER, "test_pass_general")
@eeglab_test("regression_tests/t_finputcheck.m", "test_2")
@eeglab_test("unittesting_guifunc/finputcheck/pass_general.m", "test_pass_general")
def test_finputcheck_validates_all_supported_rule_values():
    arguments = ["key1", 3, "the2test", "foo", "p3rcent", 0.4937, "forth", ["a", 2, "11b", "D"]]

    result = finputcheck(arguments, RULES)

    assert result == {"key1": 3, "the2test": "foo", "p3rcent": 0.4937, "forth": ["a", 2, "11b", "D"]}


@eeglab_test(FINPUTCHECK_WRAPPER, "test_pass_multiple_types")
@eeglab_test("regression_tests/t_finputcheck.m", "test_3")
@eeglab_test("unittesting_guifunc/finputcheck/pass_multiple_types.m", "test_pass_multiple_types")
def test_finputcheck_accepts_a_value_matching_any_declared_type():
    rules = (
        RULES[0],
        ("key1", ("integer", "string"), ([1, 5], ["test"]), 2),
        *RULES[2:],
    )
    arguments = ["key1", "test", "the2test", "foo", "p3rcent", 0.4937, "forth", ["a", 2, "11b", "D"]]

    result = finputcheck(arguments, rules)

    assert result == {"key1": "test", "the2test": "foo", "p3rcent": 0.4937, "forth": ["a", 2, "11b", "D"]}


@eeglab_test(FINPUTCHECK_WRAPPER, "test_pass_standard")
@eeglab_test("regression_tests/t_finputcheck.m", "test_4")
@eeglab_test("unittesting_guifunc/finputcheck/pass_standard.m", "test_pass_standard")
def test_finputcheck_fills_only_omitted_values_from_defaults():
    rules = (*RULES[:3], ("forth", "cell", [], ["w", 8, "y", 2]))

    result = finputcheck(["key1", 3, "the2test", "foo"], rules)

    assert result == {
        "key1": 3,
        "the2test": "foo",
        "p3rcent": 1 / math.sqrt(2),
        "forth": ["w", 8, "y", 2],
    }


@eeglab_test(FINPUTCHECK_WRAPPER, "test_pass_strings")
@eeglab_test("regression_tests/t_finputcheck.m", "test_5")
@eeglab_test("unittesting_guifunc/finputcheck/pass_strings.m", "test_pass_strings")
def test_finputcheck_accepts_declared_string_choices_case_insensitively():
    rules = (("the2test", "string", ["foo", "bar"], "baz"), *RULES[1:])
    arguments = ["key1", 3, "the2test", "foo", "p3rcent", 0.4937, "forth", ["a", 2, "11b", "D"]]

    result = finputcheck(arguments, rules)

    assert result == {"key1": 3, "the2test": "foo", "p3rcent": 0.4937, "forth": ["a", 2, "11b", "D"]}


@eeglab_test(FINPUTCHECK_WRAPPER, "test_pass_unknown")
@eeglab_test("regression_tests/t_finputcheck.m", "test_6")
@eeglab_test("unittesting_guifunc/finputcheck/pass_unknown.m", "test_pass_unknown")
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
