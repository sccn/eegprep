from __future__ import annotations

import numpy as np

from eegprep.functions.adminfunc.vararg2str import vararg2str
from tests.eeglab_tests import eeglab_test


@eeglab_test("unittesting_adminfunc/vararg2str/test_vararg2str.m", "test_test_vararg2str")
def test_reference_vararg2str_all_eight_original_calls(eeglab_backend):
    arguments = np.empty((1, 3), dtype=object)
    arguments[0] = ["arg1", 2.0, {"a": "a", "b": 3.0}]
    eeglab_backend("vararg2str", arguments)
    for names in (["key1", "key2", "key3"], ["key1", "", "key2"]):
        eeglab_backend("vararg2str", arguments, np.array([names], dtype=object))
    eeglab_backend("vararg2str", arguments, np.array([["", "key2", ""]], dtype=object), np.array([[1.0, 3.0]]))
    for names, conversion in (
        (["", "", ""], [0.0, 0.0, 0.0]),
        (["", "", ""], [1.0, 1.0, 1.0]),
        (["key1", "", ""], [1.0, 1.0, 1.0]),
        (["key1", "", ""], [0.0, 0.0, 0.0]),
    ):
        eeglab_backend(
            "vararg2str", arguments, np.array([names], dtype=object), np.empty((0, 0)), np.array([conversion])
        )


def test_vararg2str_formats_values_names_and_raw_string_arguments():
    arguments = ["arg1", 2, {"a": "a", "b": 3}]

    assert vararg2str(arguments) == "'arg1',2,struct('a','a','b',3)"
    assert vararg2str(arguments, ["key1", "key2", "key3"]) == "key1,key2,key3"
    assert vararg2str(arguments, ["key1", "", "key2"]) == "key1,2,key2"
    assert vararg2str(arguments, ["", "key2", ""], [1, 3]) == "'arg1',key2,struct('a','a','b',3)"
    assert vararg2str(arguments, nostrconv=[1, 1, 1]) == "arg1,2,struct('a','a','b',3)"
