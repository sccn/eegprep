from __future__ import annotations

import ast

import numpy as np

from eegprep.functions.adminfunc.getkeyval import getkeyval
from tests.eeglab_tests import assert_matlab_near, eeglab_test


COMMAND = "testfunction('key', 'val', 'foo', 'bar', 'eeglab', 'test');"
ARRAY_COMMAND = "testfunction('key', [3 1 4 1 6], 'foo', 'bar', 'eeglab', 'test');"
GETKEYVAL_WRAPPER = "unittesting_adminfunc/getkeyval/adminfunc_getkeyval_wrapperTest.m"


def test_getkeyval_returns_a_named_string_value_without_quotes():
    assert getkeyval(COMMAND, "eeglab") == "test"


def test_getkeyval_returns_a_named_numeric_literal():
    command = "testfunction('key', 'val', 'foo', 13, 'eeglab', 'test');"

    assert getkeyval(command, "foo") == "13"


def test_getkeyval_returns_the_default_for_empty_or_missing_values():
    assert getkeyval(COMMAND, "DOES_NOT_EXIST", "", "standard value") == "standard value"
    assert getkeyval("", "eeglab", "", "standard value") == "standard value"


def test_getkeyval_reports_whether_a_named_key_is_present():
    assert getkeyval(COMMAND, "eeglab", "present") == 1
    assert getkeyval(COMMAND, "DOES_NOT_EXIST", "present") == 0


def test_getkeyval_can_return_the_full_key_value_fragment():
    assert getkeyval(COMMAND, "eeglab", "full") == "'eeglab', 'test'"


def test_getkeyval_full_mode_handles_a_key_named_inside_a_parent_cell():
    command = "testfunction({'bar', 'eeglab', 'test'}, 'key', 'val', 'foo');"

    assert getkeyval(command, "eeglab", "full") == "'eeglab', 'key'"


def test_getkeyval_selects_a_one_based_numeric_vector_element():
    assert getkeyval(ARRAY_COMMAND, "key", 3) == "4"


def test_getkeyval_uses_default_for_an_out_of_range_vector_element():
    assert getkeyval(ARRAY_COMMAND, "key", 7, "standard value") == "standard value"


def test_getkeyval_uses_the_first_available_requested_vector_element():
    assert getkeyval(ARRAY_COMMAND, "key", [3, 7]) == "4"


def test_getkeyval_accepts_a_one_based_argument_position():
    assert getkeyval(COMMAND, 3) == "'foo'"


def test_getkeyval_preserves_a_positional_matrix_expression():
    command = "testfunction('key', 'val', 'foo', [['foo'];['bar']], 'eeglab', 'test');"

    assert getkeyval(command, 4) == "[['foo'];['bar']]"


def test_getkeyval_uses_default_for_an_out_of_range_argument_position():
    assert getkeyval(COMMAND, 8, "", "standard value") == "standard value"


@eeglab_test(GETKEYVAL_WRAPPER, "test_fail_no_arg")
def test_reference_getkeyval_original_non_enforcing_no_argument_call(eeglab_backend):
    # The source catches every error and its wrapper ignores the returned
    # passed flag. Completing this test does not prove that getkeyval succeeded.
    try:
        eeglab_backend("getkeyval", nargout=0)
    except Exception:
        pass


@eeglab_test(GETKEYVAL_WRAPPER, "test_pass_array")
@eeglab_test(GETKEYVAL_WRAPPER, "test_pass_array_default")
@eeglab_test(GETKEYVAL_WRAPPER, "test_pass_default")
@eeglab_test(GETKEYVAL_WRAPPER, "test_pass_empty")
@eeglab_test(GETKEYVAL_WRAPPER, "test_pass_full")
@eeglab_test(GETKEYVAL_WRAPPER, "test_pass_full_parent")
@eeglab_test(GETKEYVAL_WRAPPER, "test_pass_general")
@eeglab_test(GETKEYVAL_WRAPPER, "test_pass_multi_array")
@eeglab_test(GETKEYVAL_WRAPPER, "test_pass_num_key")
@eeglab_test(GETKEYVAL_WRAPPER, "test_pass_num_key_default")
@eeglab_test(GETKEYVAL_WRAPPER, "test_pass_numeric")
def test_reference_getkeyval_original_string_cases(eeglab_backend):
    cases = (
        ((ARRAY_COMMAND, "key", 3.0), "4"),
        ((ARRAY_COMMAND, "key", 7.0, "standard value"), "standard value"),
        ((COMMAND, "DOES_NOT_EXIST", "", "standard value"), "standard value"),
        (("", "eeglab", "", "standard value"), "standard value"),
        ((COMMAND, "eeglab", "full"), "'eeglab', 'test'"),
        (("testfunction({'bar', 'eeglab', 'test'}, 'key', 'val', 'foo');", "eeglab", "full"), "'eeglab',  'key'"),
        ((COMMAND, "eeglab", ""), "test"),
        ((ARRAY_COMMAND, "key", np.array([[3.0, 7.0]])), "4"),
        ((COMMAND, 3.0, ""), "'foo'"),
        ((COMMAND, 8.0, "", "standard value"), "standard value"),
        (("testfunction('key', 'val', 'foo', 13, 'eeglab', 'test');", "foo"), "13"),
    )
    for arguments, expected in cases:
        assert eeglab_backend("getkeyval", *arguments) == expected


@eeglab_test(GETKEYVAL_WRAPPER, "test_pass_present")
@eeglab_test(GETKEYVAL_WRAPPER, "test_pass_not_present")
def test_reference_getkeyval_original_presence_cases(eeglab_backend):
    assert_matlab_near(eeglab_backend("getkeyval", COMMAND, "eeglab", "present"), [[1.0]])
    assert_matlab_near(eeglab_backend("getkeyval", COMMAND, "DOES_NOT_EXIST", "present"), [[0.0]])


@eeglab_test(GETKEYVAL_WRAPPER, "test_pass_num_key_array")
def test_reference_getkeyval_original_evaluated_matrix(request, eeglab_backend):
    command = "testfunction('key', 'val', 'foo', [['foo'];['bar']], 'eeglab', 'test');"
    result = eeglab_backend("getkeyval", command, 4.0, "")
    expected = "[['foo'];['bar']]"
    if request.config.getoption("--eeglab-backend") == "matlab":
        np.testing.assert_array_equal(eeglab_backend("eval", result), eeglab_backend("eval", expected))
    else:
        np.testing.assert_array_equal(_character_matrix(result), _character_matrix(expected))


def _character_matrix(expression):
    # Only the quoted character rows in this source oracle need evaluation.
    rows = expression.strip()[1:-1].split(";")
    return np.array([list(ast.literal_eval(row.strip().removeprefix("[").removesuffix("]"))) for row in rows])
