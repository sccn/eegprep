from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from eegprep.functions.adminfunc.gethelpvar import gethelpvar
from tests.eeglab_tests import eeglab_test
from tests.eeglab_tests.assertions import assert_matlab_struct_near


_GETHELPVAR_WRAPPER = "unittesting_adminfunc/gethelpvar/adminfunc_gethelpvar_wrapperTest.m"
_REFERENCE_HELP_NAMES = np.array([["functions", "silent", "report"]], dtype=object)


def _reference_help(eeglab_backend, eeglab_suite_root, expected, *variables):
    descriptions, names = eeglab_backend(
        "gethelpvar", str(eeglab_suite_root / "unittesting_adminfunc/gethelpvar/pass_general.m"), *variables, nargout=2
    )
    assert_matlab_struct_near(expected, descriptions)
    assert_matlab_struct_near(_REFERENCE_HELP_NAMES, names)


@eeglab_test(_GETHELPVAR_WRAPPER, "test_pass_all")
def test_reference_gethelpvar_all(eeglab_backend, eeglab_suite_root):
    _reference_help(
        eeglab_backend,
        eeglab_suite_root,
        np.array(
            [
                [
                    "function names to test (without extension '.m')",
                    "'noninteractive' for silent run\nwithout user interaction",
                    "new report in ./reports directory",
                ]
            ],
            dtype=object,
        ),
    )


@eeglab_test(_GETHELPVAR_WRAPPER, "test_pass_general")
def test_reference_gethelpvar_general(eeglab_backend, eeglab_suite_root):
    _reference_help(
        eeglab_backend,
        eeglab_suite_root,
        np.array([["function names to test (without extension '.m')"]], dtype=object),
        "functions",
    )


@eeglab_test(_GETHELPVAR_WRAPPER, "test_pass_no_var")
def test_reference_gethelpvar_no_var(eeglab_backend, eeglab_suite_root):
    expected = np.empty((1, 2), dtype=object)
    expected[0, 0] = "new report in ./reports directory"
    expected[0, 1] = np.empty((0,), dtype="<U1")
    _reference_help(eeglab_backend, eeglab_suite_root, expected, np.array([["report", "DOES_NO_EXIST"]], dtype=object))


@eeglab_test(_GETHELPVAR_WRAPPER, "test_pass_some")
def test_reference_gethelpvar_some(eeglab_backend, eeglab_suite_root):
    _reference_help(
        eeglab_backend,
        eeglab_suite_root,
        np.array(
            [
                [
                    "new report in ./reports directory",
                    "'noninteractive' for silent run\nwithout user interaction",
                ]
            ],
            dtype=object,
        ),
        np.array([["report", "silent"]], dtype=object),
    )


HELP_HEADER = """% test_eeglab() - tests eeglab
%
% Usage:
%  >> test_eeglab(functions, silent)
%
% Inputs:
%   functions  - function names to test (without extension '.m')
%   silent     - 'noninteractive' for silent run
%                without user interaction
%
% Outputs:
%   report     - new report in ./reports directory
%
% Notes: see eeglab/functions directory for function names
function pass_general()
"""


@pytest.fixture
def help_file(tmp_path: Path) -> Path:
    filename = tmp_path / "pass_general.m"
    filename.write_text(HELP_HEADER, encoding="utf-8")
    return filename


def test_gethelpvar_returns_all_documented_variables(help_file: Path):
    descriptions, names = gethelpvar(help_file)

    assert names == ["functions", "silent", "report"]
    assert descriptions == [
        "function names to test (without extension '.m')",
        "'noninteractive' for silent run\nwithout user interaction",
        "new report in ./reports directory",
    ]


def test_gethelpvar_returns_one_requested_description_and_all_names(help_file: Path):
    descriptions, names = gethelpvar(help_file, "functions")

    assert descriptions == ["function names to test (without extension '.m')"]
    assert names == ["functions", "silent", "report"]


def test_gethelpvar_returns_empty_text_for_an_unknown_variable(help_file: Path, caplog):
    descriptions, names = gethelpvar(help_file, ["report", "DOES_NOT_EXIST"])

    assert descriptions == ["new report in ./reports directory", ""]
    assert names == ["functions", "silent", "report"]
    assert "DOES_NOT_EXIST" in caplog.text


def test_gethelpvar_preserves_requested_variable_order(help_file: Path):
    descriptions, names = gethelpvar(help_file, ["report", "silent"])

    assert descriptions == [
        "new report in ./reports directory",
        "'noninteractive' for silent run\nwithout user interaction",
    ]
    assert names == ["functions", "silent", "report"]
