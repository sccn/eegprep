from __future__ import annotations

import pytest

from eegprep.functions.adminfunc.getkeyval import getkeyval
from tests.eeglab_tests import eeglab_test


COMMAND = "testfunction('key', 'val', 'foo', 'bar', 'eeglab', 'test');"
ARRAY_COMMAND = "testfunction('key', [3 1 4 1 6], 'foo', 'bar', 'eeglab', 'test');"


@eeglab_test("unittesting_adminfunc/getkeyval/fail_no_arg.m", "test_fail_no_arg")
def test_getkeyval_requires_a_command_and_variable():
    with pytest.raises(TypeError):
        getkeyval()


@eeglab_test("unittesting_adminfunc/getkeyval/pass_general.m", "test_pass_general")
def test_getkeyval_returns_a_named_string_value_without_quotes():
    assert getkeyval(COMMAND, "eeglab") == "test"


@eeglab_test("unittesting_adminfunc/getkeyval/pass_numeric.m", "test_pass_numeric")
def test_getkeyval_returns_a_named_numeric_literal():
    command = "testfunction('key', 'val', 'foo', 13, 'eeglab', 'test');"

    assert getkeyval(command, "foo") == "13"


@eeglab_test("unittesting_adminfunc/getkeyval/pass_default.m", "test_pass_default")
@eeglab_test("unittesting_adminfunc/getkeyval/pass_empty.m", "test_pass_empty")
def test_getkeyval_returns_the_default_for_empty_or_missing_values():
    assert getkeyval(COMMAND, "DOES_NOT_EXIST", "", "standard value") == "standard value"
    assert getkeyval("", "eeglab", "", "standard value") == "standard value"


@eeglab_test("unittesting_adminfunc/getkeyval/pass_present.m", "test_pass_present")
@eeglab_test("unittesting_adminfunc/getkeyval/pass_not_present.m", "test_pass_not_present")
def test_getkeyval_reports_whether_a_named_key_is_present():
    assert getkeyval(COMMAND, "eeglab", "present") == 1
    assert getkeyval(COMMAND, "DOES_NOT_EXIST", "present") == 0


@eeglab_test("unittesting_adminfunc/getkeyval/pass_full.m", "test_pass_full")
def test_getkeyval_can_return_the_full_key_value_fragment():
    assert getkeyval(COMMAND, "eeglab", "full") == "'eeglab', 'test'"


@eeglab_test("unittesting_adminfunc/getkeyval/pass_full_parent.m", "test_pass_full_parent")
def test_getkeyval_full_mode_handles_a_key_named_inside_a_parent_cell():
    command = "testfunction({'bar', 'eeglab', 'test'}, 'key', 'val', 'foo');"

    assert getkeyval(command, "eeglab", "full") == "'eeglab', 'key'"


@eeglab_test("unittesting_adminfunc/getkeyval/pass_array.m", "test_pass_array")
def test_getkeyval_selects_a_one_based_numeric_vector_element():
    assert getkeyval(ARRAY_COMMAND, "key", 3) == "4"


@eeglab_test("unittesting_adminfunc/getkeyval/pass_array_default.m", "test_pass_array_default")
def test_getkeyval_uses_default_for_an_out_of_range_vector_element():
    assert getkeyval(ARRAY_COMMAND, "key", 7, "standard value") == "standard value"


@eeglab_test("unittesting_adminfunc/getkeyval/pass_multi_array.m", "test_pass_multi_array")
def test_getkeyval_uses_the_first_available_requested_vector_element():
    assert getkeyval(ARRAY_COMMAND, "key", [3, 7]) == "4"


@eeglab_test("unittesting_adminfunc/getkeyval/pass_num_key.m", "test_pass_num_key")
def test_getkeyval_accepts_a_one_based_argument_position():
    assert getkeyval(COMMAND, 3) == "'foo'"


@eeglab_test("unittesting_adminfunc/getkeyval/pass_num_key_array.m", "test_pass_num_key_array")
def test_getkeyval_preserves_a_positional_matrix_expression():
    command = "testfunction('key', 'val', 'foo', [['foo'];['bar']], 'eeglab', 'test');"

    assert getkeyval(command, 4) == "[['foo'];['bar']]"


@eeglab_test("unittesting_adminfunc/getkeyval/pass_num_key_default.m", "test_pass_num_key_default")
def test_getkeyval_uses_default_for_an_out_of_range_argument_position():
    assert getkeyval(COMMAND, 8, "", "standard value") == "standard value"
