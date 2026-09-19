from __future__ import annotations

from pathlib import Path

import pytest

from eegprep.functions.adminfunc.gethelpvar import gethelpvar
from tests.eeglab_tests import eeglab_test


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


@eeglab_test("unittesting_adminfunc/gethelpvar/fail_no_arg.m", "test_fail_no_arg")
def test_gethelpvar_requires_a_filename():
    with pytest.raises(TypeError):
        gethelpvar()


@eeglab_test("unittesting_adminfunc/gethelpvar/fail_no_file.m", "test_fail_no_file")
def test_gethelpvar_rejects_a_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        gethelpvar(tmp_path / "DOES_NOT_EXIST.m", ["report"])


@eeglab_test("unittesting_adminfunc/gethelpvar/pass_all.m", "test_pass_all")
def test_gethelpvar_returns_all_documented_variables(help_file: Path):
    descriptions, names = gethelpvar(help_file)

    assert names == ["functions", "silent", "report"]
    assert descriptions == [
        "function names to test (without extension '.m')",
        "'noninteractive' for silent run\nwithout user interaction",
        "new report in ./reports directory",
    ]


@eeglab_test("unittesting_adminfunc/gethelpvar/pass_general.m", "test_pass_general")
def test_gethelpvar_returns_one_requested_description_and_all_names(help_file: Path):
    descriptions, names = gethelpvar(help_file, "functions")

    assert descriptions == ["function names to test (without extension '.m')"]
    assert names == ["functions", "silent", "report"]


@eeglab_test("unittesting_adminfunc/gethelpvar/pass_no_var.m", "test_pass_no_var")
def test_gethelpvar_returns_empty_text_for_an_unknown_variable(help_file: Path, caplog):
    descriptions, names = gethelpvar(help_file, ["report", "DOES_NOT_EXIST"])

    assert descriptions == ["new report in ./reports directory", ""]
    assert names == ["functions", "silent", "report"]
    assert "DOES_NOT_EXIST" in caplog.text


@eeglab_test("unittesting_adminfunc/gethelpvar/pass_some.m", "test_pass_some")
def test_gethelpvar_preserves_requested_variable_order(help_file: Path):
    descriptions, names = gethelpvar(help_file, ["report", "silent"])

    assert descriptions == [
        "new report in ./reports directory",
        "'noninteractive' for silent run\nwithout user interaction",
    ]
    assert names == ["functions", "silent", "report"]
