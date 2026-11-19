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
    "tests/test_gethelpvar.py::test_reference_gethelpvar_no_var": _gap(
        "a request that names a missing variable still does not return the MATLAB cell shape",
        AssertionError,
    ),
    "tests/test_popfunc_dataset_eeglab_tests.py::test_reference_importevent_original_named_workspace_variables": _gap(
        "MATLAB resolves the event array with evalin; the Python harness has no caller workspace",
        ValueError,
    ),
    "tests/test_pop_editeventfield_current.py::test_reference_description_missing_field": _gap(
        "eventdescription is still not the MATLAB character column when the named field is absent",
        AssertionError,
    ),
    "tests/test_pop_editeventfield_current.py::test_reference_field_description": _gap(
        "eventdescription is still not the MATLAB character column for a field description",
        AssertionError,
    ),
    "tests/test_study_workflows_eeglab_tests.py::test_reference_std_selectdesign[eeglab_sample_study0]": _gap(
        "STUDY.cluster fields such as allinds stay Python lists, and the design workflow calls .size on them",
        AttributeError,
    ),
}

PYTHON_BACKEND_SUBTEST_XFAILS: dict[str, dict[str, ExpectedFailure]] = {}
