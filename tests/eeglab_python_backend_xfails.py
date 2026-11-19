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


STRUCT = "MATLAB struct and cell inputs arrive as lists, object arrays, or structured arrays"
SHAPE = "the result shape does not follow the MATLAB row, column, or empty-array contract"
INDEX = "an EEGLAB index or dimension is 1-based"
NAMEVALUE = "MATLAB name/value arguments are not accepted"
OUTPUTS = "the function does not return the MATLAB output count"
STUDY = "the sample STUDY loads, but design values are still nested MATLAB cells"
RMART = (
    "rmart.m:94 returns when rem(length(data), nchans) == 0, inverting the "
    "divisibility test, so the documented regression never runs; EEGPrep keeps "
    "the reachable correction. This is a permanent EEGLAB difference"
)

PYTHON_BACKEND_XFAILS = {
    "tests/test_core_event_eeglab_tests.py::test_reference_matchchans_original_complete_locations": _gap(
        "eeg_matchchans.m clears bglocs(1).bigchan on every iteration, so a match to the first large channel loses its index; EEGPrep keeps every match",
        AssertionError,
    ),
    "tests/test_core_event_eeglab_tests.py::test_reference_mergechan_uses_original_montages": _gap(SHAPE, TypeError),
    "tests/test_core_event_eeglab_tests.py::test_reference_mergelocs_uses_original_montages": _gap(SHAPE, TypeError),
    "tests/test_eeg_interp.py::test_reference_interp_original_methods_and_shuffled_montage": _gap(STRUCT, TypeError),
    "tests/test_eeg_options.py::test_reference_eeg_readoptions_backup": _gap(STRUCT, ValueError),
    "tests/test_eeg_retrieve.py::test_reference_eeg_retrieve_original_multiple_case": _gap(SHAPE, TypeError),
    "tests/test_eeg_store.py::test_reference_eeg_store_original_general_case": _gap(SHAPE, TypeError),
    "tests/test_eeg_store.py::test_reference_eeg_store_original_multiple_case": _gap(SHAPE, TypeError),
    "tests/test_eeg_store.py::test_reference_eeg_store_original_multiple_new_case": _gap(SHAPE, AttributeError),
    "tests/test_eeg_store.py::test_reference_eeg_store_original_new_case": _gap(SHAPE, TypeError),
    "tests/test_eegh.py::test_reference_eegh_add_multiple_history": _gap(STRUCT, TypeError),
    "tests/test_eegh.py::test_reference_eegh_add_multiple_history_default_empty": _gap(STRUCT, TypeError),
    "tests/test_finputcheck.py::test_reference_finputcheck_original_regression_fixture": _gap(STRUCT, TypeError),
    "tests/test_gethelpvar.py::test_reference_gethelpvar_no_var": _gap(STRUCT, TypeError),
    "tests/test_gethelpvar.py::test_reference_gethelpvar_some": _gap(STRUCT, TypeError),
    "tests/test_legacy_signal_io_eeglab_tests.py::test_upstream_readneurodat_original_file": _gap(STRUCT, TypeError),
    "tests/test_misc_data_event_eeglab_tests.py::test_reference_biosig_all_fields": _gap(STRUCT, AssertionError),
    "tests/test_misc_data_event_eeglab_tests.py::test_reference_biosig_partial_fields": _gap(STRUCT, AssertionError),
    "tests/test_phase1b_file_edit_pop_functions.py::test_reference_selectevent_dataset_array": _gap(
        SHAPE, AttributeError
    ),
    "tests/test_phase1b_file_edit_pop_functions.py::test_reference_selectevent_delete_unselected_events": _gap(
        SHAPE, IndexError
    ),
    "tests/test_phase1b_file_edit_pop_functions.py::test_reference_selectevent_inverse_epochs": _gap(
        SHAPE, AssertionError
    ),
    "tests/test_phase1b_file_edit_pop_functions.py::test_reference_selectevent_retains_matching_epochs": _gap(
        SHAPE, AssertionError
    ),
    "tests/test_phase2_file_channel_location.py::test_reference_convertlocs_cart2all": _gap(SHAPE, AssertionError),
    "tests/test_phase2_file_channel_location.py::test_reference_pop_chancenter_nine_recorded_location_calls": _gap(
        STRUCT, TypeError
    ),
    "tests/test_phase2_file_channel_location.py::test_reference_readeetraklocs_original_file": _gap(
        SHAPE, AttributeError
    ),
    "tests/test_phase2_file_channel_location.py::test_reference_readegilocs_all_original_channel_counts": _gap(
        SHAPE, AttributeError
    ),
    "tests/test_pop_interp.py::test_reference_pop_interp_original_locations_and_recording": _gap(STRUCT, TypeError),
    "tests/test_pop_reref.py::test_reference_pop_reref_original_bugzilla_270_input": _gap(STRUCT, TypeError),
    "tests/test_pop_reref.py::test_reference_reref": _gap(STRUCT, TypeError),
    "tests/test_pop_resample_python.py::test_reference_pop_resample_original_half_sample_boundary_workflow": _gap(
        STRUCT, KeyError
    ),
    "tests/test_popfunc_dataset_eeglab_tests.py::test_reference_chanedit_original_locations": _gap(STRUCT, TypeError),
    "tests/test_popfunc_dataset_eeglab_tests.py::test_reference_importevent_original_named_workspace_variables": _gap(
        STRUCT, ValueError
    ),
    "tests/test_popfunc_dataset_eeglab_tests.py::test_reference_mergeset_all_eight_recorded_data_calls": _gap(
        STRUCT, TypeError
    ),
    "tests/test_popfunc_dataset_eeglab_tests.py::test_reference_newset_original_eight_datasets_and_saved_files": _gap(
        SHAPE, AttributeError
    ),
    "tests/test_popfunc_dataset_eeglab_tests.py::test_reference_selectevent_original_overlapping_epoch_workflow": _gap(
        STRUCT, AttributeError
    ),
    "tests/test_miscfunc_numerical_eeglab_tests.py::test_reference_hungarian": _gap(
        "several assignments share the minimum cost and the selected one differs",
        AssertionError,
    ),
    "tests/test_eeg_lat2point.py::test_eeg_lat2point_current_suite_cell_epoch_case": _gap(INDEX, IndexError),
    "tests/test_eeg_lat2point.py::test_eeg_lat2point_current_suite_cell_latency_case": _gap(INDEX, IndexError),
    "tests/test_eeg_lat2point.py::test_eeg_lat2point_current_suite_clamps_out_of_range_latency": _gap(
        INDEX, IndexError
    ),
    "tests/test_eeg_lat2point.py::test_eeg_lat2point_current_suite_default_time_unit": _gap(INDEX, IndexError),
    "tests/test_eeg_lat2point.py::test_eeg_lat2point_current_suite_general_case": _gap(INDEX, IndexError),
    "tests/test_eeg_lat2point.py::test_eeg_lat2point_current_suite_single_epoch_broadcast": _gap(INDEX, IndexError),
    "tests/test_eeg_retrieve.py::test_reference_eeg_retrieve_original_general_case": _gap(SHAPE, AssertionError),
    "tests/test_misc_data_event_eeglab_tests.py::test_reference_rmart": _gap(RMART, ValueError),
    "tests/test_miscfunc_numerical_eeglab_tests.py::test_reference_gauss": _gap(INDEX, IndexError),
    "tests/test_pop_adjustevents.py::test_reference_adjustevents_continuous": _gap(INDEX, IndexError),
    "tests/test_pop_adjustevents.py::test_reference_adjustevents_epoched": _gap(INDEX, IndexError),
    "tests/test_phase1b_file_edit_pop_functions.py::test_reference_copyset_original_set_out": _gap(
        NAMEVALUE, ValueError
    ),
    "tests/test_pop_delset.py::test_reference_delset_original_datasets": _gap(NAMEVALUE, ValueError),
    "tests/test_pop_editeventfield_current.py::test_reference_add_field_file": _gap(NAMEVALUE, ValueError),
    "tests/test_pop_editeventfield_current.py::test_reference_add_field_file_indices": _gap(NAMEVALUE, ValueError),
    "tests/test_pop_editeventfield_current.py::test_reference_add_field_vector": _gap(NAMEVALUE, ValueError),
    "tests/test_pop_editeventfield_current.py::test_reference_delimiters": _gap(NAMEVALUE, ValueError),
    "tests/test_pop_editeventfield_current.py::test_reference_delold_invalid": _gap(NAMEVALUE, ValueError),
    "tests/test_pop_editeventfield_current.py::test_reference_delold_yes": _gap(NAMEVALUE, ValueError),
    "tests/test_pop_editeventfield_current.py::test_reference_description_missing_field": _gap(NAMEVALUE, ValueError),
    "tests/test_pop_editeventfield_current.py::test_reference_field_description": _gap(NAMEVALUE, ValueError),
    "tests/test_pop_editeventfield_current.py::test_reference_invalid_rename": _gap(NAMEVALUE, ValueError),
    "tests/test_pop_editeventfield_current.py::test_reference_latency_timeunit": _gap(NAMEVALUE, ValueError),
    "tests/test_pop_editeventfield_current.py::test_reference_modify_field": _gap(NAMEVALUE, ValueError),
    "tests/test_pop_editeventfield_current.py::test_reference_modify_field_indices": _gap(NAMEVALUE, ValueError),
    "tests/test_pop_editeventfield_current.py::test_reference_remove_missing_field": _gap(NAMEVALUE, ValueError),
    "tests/test_pop_editeventfield_current.py::test_reference_rename_field": _gap(NAMEVALUE, ValueError),
    "tests/test_pop_editeventfield_current.py::test_reference_rename_missing_field": _gap(NAMEVALUE, ValueError),
    "tests/test_pop_editeventfield_current.py::test_reference_skip_lines": _gap(NAMEVALUE, ValueError),
    "tests/test_sample_data_pop_functions.py::test_reference_expica_original_recording_and_ascii_outputs": _gap(
        NAMEVALUE, ValueError
    ),
    "tests/test_bdf_records_eeglab_tests.py::test_upstream_openbdf_original_channel_types": _gap(
        OUTPUTS, AssertionError
    ),
    "tests/test_bdf_records_eeglab_tests.py::test_upstream_openbdf_original_general_header": _gap(
        OUTPUTS, AssertionError
    ),
    "tests/test_bdf_records_eeglab_tests.py::test_upstream_openbdf_original_invalid_digital_limits": _gap(
        OUTPUTS, AssertionError
    ),
    "tests/test_bdf_records_eeglab_tests.py::test_upstream_openbdf_original_invalid_physical_limits": _gap(
        OUTPUTS, AssertionError
    ),
    "tests/test_bdf_records_eeglab_tests.py::test_upstream_openbdf_original_reversed_digital_limits": _gap(
        OUTPUTS, AssertionError
    ),
    "tests/test_bdf_records_eeglab_tests.py::test_upstream_openbdf_original_reversed_physical_limits": _gap(
        OUTPUTS, AssertionError
    ),
    "tests/test_miscfunc_decomposition_eeglab_tests.py::test_reference_varsort": _gap(OUTPUTS, AssertionError),
    "tests/test_miscfunc_decomposition_eeglab_tests.py::test_reference_zica": _gap(OUTPUTS, AssertionError),
    "tests/test_phase2_file_channel_location.py::test_reference_readelp_five_outputs": _gap(OUTPUTS, AssertionError),
    "tests/test_pop_rmbase.py::test_reference_pop_rmbase_original_baselines_and_complete_commands": _gap(
        OUTPUTS, AssertionError
    ),
    "tests/test_pop_rmbase.py::test_reference_rmbase": _gap(OUTPUTS, AssertionError),
    "tests/test_rejection_workflows.py::test_reference_pop_rejchan": _gap(OUTPUTS, AssertionError),
    "tests/test_rejection_workflows.py::test_reference_pop_rejtrend": _gap(OUTPUTS, AssertionError),
    "tests/test_popfunc_dataset_eeglab_tests.py::test_reference_importevent_original_text_file": _gap(
        "importevent returns a list of event dicts rather than a MATLAB struct array",
        AttributeError,
    ),
    "tests/test_study_workflows_eeglab_tests.py::test_reference_pop_clust[eeglab_sample_study0]": _gap(
        "the sample STUDY loads, but preclustdata is a numeric array used as a condition",
        ValueError,
    ),
    "tests/test_study_workflows_eeglab_tests.py::test_reference_std_makedesign[eeglab_sample_study0]": _gap(
        STUDY, TypeError
    ),
    "tests/test_study_workflows_eeglab_tests.py::test_reference_std_selectdesign[eeglab_sample_study0]": _gap(
        "the sample STUDY loads, but design is a list rather than a MATLAB struct array",
        AttributeError,
    ),
}

PYTHON_BACKEND_SUBTEST_XFAILS = {
    "tests/test_epoch.py::test_reference_epoch_original_five_output_contracts": {
        "(case='no_srate')": _gap(STRUCT, ValueError),
        "(case='general')": _gap(NAMEVALUE, ValueError),
        "(case='boundary')": _gap(NAMEVALUE, ValueError),
        "(case='valuelim')": _gap(NAMEVALUE, ValueError),
        "(case='allevents')": _gap(NAMEVALUE, ValueError),
    },
    "tests/test_sigprocfunc_plotting_eeglab_tests.py::test_reference_forcelocs_complete_original_coordinates": {
        "(source='x')": _gap(SHAPE, AssertionError),
        "(source='y')": _gap(SHAPE, AssertionError),
    },
}
