"""Expected failures for --eeglab-backend=python.

Each entry names one gap. ``raises`` is the stable exception type: a different
exception fails the job, and strict mode fails the job when the test passes.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ExpectedFailure:
    reason: str
    raises: type[BaseException] | tuple[type[BaseException], ...]


def _gap(reason: str, raises: type[BaseException] | tuple[type[BaseException], ...]) -> ExpectedFailure:
    return ExpectedFailure(f"{reason} (#430)", raises)


PYTHON_BACKEND_XFAILS = {
    "tests/test_eeg_store.py::test_reference_eeg_store_original_multiple_case": _gap(
        "an empty ALLEEG slot still does not reproduce the MATLAB data cell after the datasets are stored",
        AssertionError,
    ),
    "tests/test_finputcheck.py::test_reference_finputcheck_original_regression_fixture": _gap(
        "the regression fixture still passes nested size-1 character cells that are not Python strings",
        TypeError,
    ),
    "tests/test_gethelpvar.py::test_reference_gethelpvar_no_var": _gap(
        "a request that names a missing variable still does not return the MATLAB cell shape",
        AssertionError,
    ),
    "tests/test_pop_reref.py::test_reference_reref": _gap(
        "reref.m refloc values are still not location records the low-level call can append",
        TypeError,
    ),
    "tests/test_popfunc_dataset_eeglab_tests.py::test_reference_importevent_original_named_workspace_variables": _gap(
        "MATLAB resolves the event array with evalin; the Python harness has no caller workspace",
        ValueError,
    ),
    "tests/test_popfunc_dataset_eeglab_tests.py::test_reference_newset_original_eight_datasets_and_saved_files": _gap(
        "pop_saveset still compares a structured field with a Python value while saving the new dataset",
        TypeError,
    ),
    "tests/test_phase1b_file_edit_pop_functions.py::test_reference_copyset_original_set_out": _gap(
        "empty EEG.etc and EEG.roi dicts now compare equal, but another copied field still disagrees",
        AssertionError,
    ),
    "tests/test_pop_editeventfield_current.py::test_reference_description_missing_field": _gap(
        "eventdescription is still not the MATLAB character column when the named field is absent",
        AssertionError,
    ),
    "tests/test_pop_editeventfield_current.py::test_reference_field_description": _gap(
        "eventdescription is still not the MATLAB character column for a field description",
        AssertionError,
    ),
    "tests/test_pop_rmbase.py::test_reference_pop_rmbase_original_baselines_and_complete_commands": _gap(
        "the history command expands channel ranges and spaces arguments differently from pop_rmbase.m",
        AssertionError,
    ),
    "tests/test_rejection_workflows.py::test_reference_pop_rejchan": _gap(
        "pop_rejchan still rejects the original channel selection after the dataset fields are read",
        ValueError,
    ),
    "tests/test_rejection_workflows.py::test_reference_pop_rejtrend": _gap(
        "pop_rejtrend still rejects the original window length after the dataset fields are read",
        ValueError,
    ),
    "tests/test_study_workflows_eeglab_tests.py::test_reference_pop_clust[eeglab_sample_study0]": _gap(
        "the sample STUDY cluster index length still does not match the selected parent components",
        ValueError,
    ),
    "tests/test_study_workflows_eeglab_tests.py::test_reference_std_selectdesign[eeglab_sample_study0]": _gap(
        "STUDY.cluster fields such as allinds stay Python lists, so the design workflow cannot index them as structs",
        AttributeError,
    ),
}

PYTHON_BACKEND_SUBTEST_XFAILS: dict[str, dict[str, ExpectedFailure]] = {}
