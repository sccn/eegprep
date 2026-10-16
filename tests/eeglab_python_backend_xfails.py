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


STATCOND_ZERO = (
    "stored statcond ANOVA fixtures use zero tolerance. "
    "Cases 3-6 and 15-30 are single because EEGLAB accumulates in single "
    "(anova1rm_cell.m:67-68, anova1_cell.m:98/107/116, anova2_cell.m:98-99, "
    "anova2rm_cell.m:83-131) and then evaluates single 1-fcdf. "
    "F errors reach absolute 6.1e-2 and relative 5.9e-3 from cancellation in "
    "ssA/ssAS (anova1rm_cell.m:97-104), and p saturates to 0 below about 6e-8 "
    "in single. eegprep computes these statistics in double"
)
STRUCT = "MATLAB struct and cell inputs arrive as lists, object arrays, or structured arrays"
SHAPE = "the result shape does not follow the MATLAB row, column, or empty-array contract"
INDEX = "an EEGLAB index or dimension is 1-based"
NAMEVALUE = "MATLAB name/value arguments are not accepted"
OUTPUTS = "the function does not return the MATLAB output count"
TRUTH = "a MATLAB array is used as a logical condition"
POSITIONAL = "the function does not accept MATLAB positional arguments beyond its Python signature"
STUDY = "the sample STUDY fixture is not valid UTF-8"
VALUES = "numeric or text values do not match the MATLAB result"
EVENT_EDIT = "an event-field edit is passed as a MATLAB cell rather than an index, field, and value"
RMART = (
    "rmart.m:94 returns when rem(length(data), nchans) == 0, inverting the "
    "divisibility test, so the documented regression never runs; EEGPrep keeps "
    "the reachable correction. This is a permanent EEGLAB difference"
)

PYTHON_BACKEND_XFAILS = {
    "tests/test_statcond_eeglab_tests.py::test_reference_statcond_regression_03": _gap(STATCOND_ZERO, AssertionError),
    "tests/test_statcond_eeglab_tests.py::test_reference_statcond_regression_04": _gap(STATCOND_ZERO, AssertionError),
    "tests/test_statcond_eeglab_tests.py::test_reference_statcond_regression_05": _gap(STATCOND_ZERO, AssertionError),
    "tests/test_statcond_eeglab_tests.py::test_reference_statcond_regression_06": _gap(STATCOND_ZERO, AssertionError),
    "tests/test_statcond_eeglab_tests.py::test_reference_statcond_regression_15": _gap(STATCOND_ZERO, AssertionError),
    "tests/test_statcond_eeglab_tests.py::test_reference_statcond_regression_16": _gap(STATCOND_ZERO, AssertionError),
    "tests/test_statcond_eeglab_tests.py::test_reference_statcond_regression_17": _gap(STATCOND_ZERO, AssertionError),
    "tests/test_statcond_eeglab_tests.py::test_reference_statcond_regression_18": _gap(STATCOND_ZERO, AssertionError),
    "tests/test_statcond_eeglab_tests.py::test_reference_statcond_regression_19": _gap(STATCOND_ZERO, AssertionError),
    "tests/test_statcond_eeglab_tests.py::test_reference_statcond_regression_20": _gap(STATCOND_ZERO, AssertionError),
    "tests/test_statcond_eeglab_tests.py::test_reference_statcond_regression_21": _gap(STATCOND_ZERO, AssertionError),
    "tests/test_statcond_eeglab_tests.py::test_reference_statcond_regression_22": _gap(STATCOND_ZERO, AssertionError),
    "tests/test_statcond_eeglab_tests.py::test_reference_statcond_regression_23": _gap(STATCOND_ZERO, AssertionError),
    "tests/test_statcond_eeglab_tests.py::test_reference_statcond_regression_24": _gap(STATCOND_ZERO, AssertionError),
    "tests/test_statcond_eeglab_tests.py::test_reference_statcond_regression_25": _gap(STATCOND_ZERO, AssertionError),
    "tests/test_statcond_eeglab_tests.py::test_reference_statcond_regression_26": _gap(STATCOND_ZERO, AssertionError),
    "tests/test_statcond_eeglab_tests.py::test_reference_statcond_regression_27": _gap(STATCOND_ZERO, AssertionError),
    "tests/test_statcond_eeglab_tests.py::test_reference_statcond_regression_28": _gap(STATCOND_ZERO, AssertionError),
    "tests/test_statcond_eeglab_tests.py::test_reference_statcond_regression_29": _gap(STATCOND_ZERO, AssertionError),
    "tests/test_statcond_eeglab_tests.py::test_reference_statcond_regression_30": _gap(STATCOND_ZERO, AssertionError),
    "tests/test_binary_eeglab_tests.py::test_upstream_chancoresp_clear": _gap(STRUCT, AttributeError),
    "tests/test_binary_eeglab_tests.py::test_upstream_chancoresp_explicit_pairs": _gap(STRUCT, AttributeError),
    "tests/test_binary_eeglab_tests.py::test_upstream_chancoresp_fiducials": _gap(STRUCT, AttributeError),
    "tests/test_binary_eeglab_tests.py::test_upstream_chancoresp_invalid_fiducials": _gap(STRUCT, AttributeError),
    "tests/test_binary_eeglab_tests.py::test_upstream_chancoresp_labels": _gap(STRUCT, AttributeError),
    "tests/test_binary_eeglab_tests.py::test_upstream_chancoresp_none": _gap(STRUCT, AttributeError),
    "tests/test_binary_eeglab_tests.py::test_upstream_chancoresp_pair": _gap(STRUCT, AttributeError),
    "tests/test_binary_eeglab_tests.py::test_upstream_chancoresp_unpair": _gap(STRUCT, AttributeError),
    "tests/test_binary_eeglab_tests.py::test_upstream_chanevent_original_recording_and_33_calls": _gap(
        STRUCT, TypeError
    ),
    "tests/test_core_event_eeglab_tests.py::test_reference_chaninds_uses_original_epoched_recording": _gap(
        STRUCT, ValueError
    ),
    "tests/test_core_event_eeglab_tests.py::test_reference_eegrej_boundary_events_follow_original_regression": _gap(
        STRUCT, AttributeError
    ),
    "tests/test_core_event_eeglab_tests.py::test_reference_eventtypes_preserves_inverted_source_condition": _gap(
        STRUCT, ValueError
    ),
    "tests/test_core_event_eeglab_tests.py::test_reference_getchanlist_no_match": _gap(STRUCT, AttributeError),
    "tests/test_core_event_eeglab_tests.py::test_reference_getepochevent_default_latency": _gap(STRUCT, ValueError),
    "tests/test_core_event_eeglab_tests.py::test_reference_getepochevent_default_time_window": _gap(STRUCT, ValueError),
    "tests/test_core_event_eeglab_tests.py::test_reference_getepochevent_duration_old_and_new_calls": _gap(
        STRUCT, ValueError
    ),
    "tests/test_core_event_eeglab_tests.py::test_reference_getepochevent_four_argument_window": _gap(
        STRUCT, ValueError
    ),
    "tests/test_core_event_eeglab_tests.py::test_reference_getepochevent_without_epoch_field": _gap(STRUCT, ValueError),
    "tests/test_core_event_eeglab_tests.py::test_reference_insertbound_original_epoched_events": _gap(
        STRUCT, ValueError
    ),
    "tests/test_core_event_eeglab_tests.py::test_reference_matchchans_original_complete_locations": _gap(
        STRUCT, TypeError
    ),
    "tests/test_core_event_eeglab_tests.py::test_reference_mergechan_uses_original_montages": _gap(STRUCT, TypeError),
    "tests/test_core_event_eeglab_tests.py::test_reference_mergelocs_uses_original_montages": _gap(STRUCT, TypeError),
    "tests/test_core_event_eeglab_tests.py::test_reference_urlatency_boundary_durations": _gap(STRUCT, ValueError),
    "tests/test_eeg_decodechan.py::test_reference_decodechan_original_channel_names": _gap(STRUCT, ValueError),
    "tests/test_eeg_interp.py::test_reference_interp_original_methods_and_shuffled_montage": _gap(STRUCT, TypeError),
    "tests/test_eeg_options.py::test_reference_eeg_readoptions_backup": _gap(STRUCT, ValueError),
    "tests/test_eeg_retrieve.py::test_reference_eeg_retrieve_original_multiple_case": _gap(STRUCT, TypeError),
    "tests/test_eeg_store.py::test_reference_eeg_store_original_general_case": _gap(STRUCT, TypeError),
    "tests/test_eeg_store.py::test_reference_eeg_store_original_multiple_case": _gap(STRUCT, TypeError),
    "tests/test_eeg_store.py::test_reference_eeg_store_original_multiple_new_case": _gap(STRUCT, TypeError),
    "tests/test_eeg_store.py::test_reference_eeg_store_original_new_case": _gap(STRUCT, TypeError),
    "tests/test_eegh.py::test_reference_eegh_add_multiple_history": _gap(STRUCT, TypeError),
    "tests/test_eegh.py::test_reference_eegh_add_multiple_history_default_empty": _gap(STRUCT, TypeError),
    "tests/test_finputcheck.py::test_reference_finputcheck_original_regression_fixture": _gap(STRUCT, TypeError),
    "tests/test_gethelpvar.py::test_reference_gethelpvar_no_var": _gap(STRUCT, TypeError),
    "tests/test_gethelpvar.py::test_reference_gethelpvar_some": _gap(STRUCT, TypeError),
    "tests/test_legacy_signal_io_eeglab_tests.py::test_upstream_readneurodat_original_file": _gap(STRUCT, TypeError),
    "tests/test_misc_data_event_eeglab_tests.py::test_reference_biosig_all_fields": _gap(STRUCT, TypeError),
    "tests/test_misc_data_event_eeglab_tests.py::test_reference_biosig_partial_fields": _gap(STRUCT, TypeError),
    "tests/test_misc_data_event_eeglab_tests.py::test_reference_unique_cell_string_original_mixed_cells": _gap(
        STRUCT, AttributeError
    ),
    "tests/test_phase1b_file_edit_pop_functions.py::test_reference_selectevent_dataset_array": _gap(
        STRUCT, AttributeError
    ),
    "tests/test_phase1b_file_edit_pop_functions.py::test_reference_selectevent_delete_unselected_events": _gap(
        STRUCT, AttributeError
    ),
    "tests/test_phase1b_file_edit_pop_functions.py::test_reference_selectevent_empty_selection_allowed": _gap(
        STRUCT, AttributeError
    ),
    "tests/test_phase1b_file_edit_pop_functions.py::test_reference_selectevent_empty_selection_error": _gap(
        STRUCT, AttributeError
    ),
    "tests/test_phase1b_file_edit_pop_functions.py::test_reference_selectevent_explicit_error_nonempty_selection": _gap(
        STRUCT, AttributeError
    ),
    "tests/test_phase1b_file_edit_pop_functions.py::test_reference_selectevent_inverse_epochs": _gap(
        STRUCT, AttributeError
    ),
    "tests/test_phase1b_file_edit_pop_functions.py::test_reference_selectevent_keep_epochs_delete_events": _gap(
        STRUCT, AttributeError
    ),
    "tests/test_phase1b_file_edit_pop_functions.py::test_reference_selectevent_retains_matching_epochs": _gap(
        STRUCT, AttributeError
    ),
    "tests/test_phase2_file_channel_location.py::test_reference_convertlocs_cart2all": _gap(STRUCT, AttributeError),
    "tests/test_phase2_file_channel_location.py::test_reference_pop_chancenter_nine_recorded_location_calls": _gap(
        STRUCT, TypeError
    ),
    "tests/test_phase2_file_channel_location.py::test_reference_readeetraklocs_original_file": _gap(
        STRUCT, AttributeError
    ),
    "tests/test_phase2_file_channel_location.py::test_reference_readegilocs_all_original_channel_counts": _gap(
        STRUCT, AttributeError
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
        STRUCT, AttributeError
    ),
    "tests/test_popfunc_dataset_eeglab_tests.py::test_reference_selectevent_original_overlapping_epoch_workflow": _gap(
        STRUCT, AttributeError
    ),
    "tests/test_sigproc_eegrej.py::test_reference_eegrej_original_boundary_reconstruction": _gap(STRUCT, TypeError),
    "tests/test_sigprocfunc_numerical_eeglab_tests.py::test_reference_mattocell": _gap(STRUCT, AttributeError),
    "tests/test_bdf_records_eeglab_tests.py::test_upstream_readbdf_original_record_selection": _gap(
        SHAPE, AssertionError
    ),
    "tests/test_cart2topo.py::test_reference_cart2topo_original_coordinate_matrices": _gap(SHAPE, AssertionError),
    "tests/test_core_event_eeglab_tests.py::test_reference_eegrej_preserves_metadata_and_recorded_data_cases": _gap(
        SHAPE, AssertionError
    ),
    "tests/test_core_event_eeglab_tests.py::test_reference_getchanlist_case_insensitive_type": _gap(
        SHAPE, AssertionError
    ),
    "tests/test_core_event_eeglab_tests.py::test_reference_getchanlist_default_all_channels": _gap(
        SHAPE, AssertionError
    ),
    "tests/test_core_event_eeglab_tests.py::test_reference_getchanlist_multiple_types_and_missing_type": _gap(
        SHAPE, AssertionError
    ),
    "tests/test_core_event_eeglab_tests.py::test_reference_urlatency_missing_duration": _gap(SHAPE, AssertionError),
    "tests/test_eeg_ms2f.py::test_reference_eeg_ms2f_original_dataset_and_latencies": _gap(SHAPE, AssertionError),
    "tests/test_getkeyval.py::test_reference_getkeyval_original_presence_cases": _gap(SHAPE, AssertionError),
    "tests/test_legacy_signal_io_eeglab_tests.py::test_reference_movav": _gap(SHAPE, AssertionError),
    "tests/test_legacy_signal_io_eeglab_tests.py::test_reference_parsetxt": _gap(SHAPE, AssertionError),
    "tests/test_legacy_signal_io_eeglab_tests.py::test_upstream_loadtxt_force": _gap(SHAPE, AssertionError),
    "tests/test_miscfunc_numerical_eeglab_tests.py::test_reference_covary": _gap(SHAPE, AssertionError),
    "tests/test_miscfunc_numerical_eeglab_tests.py::test_reference_datlim": _gap(SHAPE, AssertionError),
    "tests/test_miscfunc_numerical_eeglab_tests.py::test_reference_hungarian": ExpectedFailure(
        "boundary converter (#430 PR-A)", AssertionError
    ),
    "tests/test_miscfunc_numerical_eeglab_tests.py::test_reference_nan_std": _gap(SHAPE, AssertionError),
    "tests/test_miscfunc_numerical_eeglab_tests.py::test_reference_pcsquash": _gap(SHAPE, AssertionError),
    "tests/test_miscfunc_numerical_eeglab_tests.py::test_reference_perminv": _gap(SHAPE, ValueError),
    "tests/test_miscfunc_numerical_eeglab_tests.py::test_reference_uniquef": _gap(SHAPE, AssertionError),
    "tests/test_mmo_eeglab_tests.py::test_upstream_mmo_original_assignments[case-6]": _gap(SHAPE, AssertionError),
    "tests/test_mmo_eeglab_tests.py::test_upstream_mmo_original_deletions[case-3]": _gap(SHAPE, AssertionError),
    "tests/test_mmo_eeglab_tests.py::test_upstream_mmo_original_deletions[case-5]": _gap(SHAPE, AssertionError),
    "tests/test_mmo_eeglab_tests.py::test_upstream_mmo_original_preprocessing[continuous_filter]": _gap(
        SHAPE, AssertionError
    ),
    "tests/test_mmo_eeglab_tests.py::test_upstream_mmo_original_preprocessing[continuous_filter_after_rejection]": _gap(
        SHAPE, AssertionError
    ),
    "tests/test_mmo_eeglab_tests.py::test_upstream_mmo_original_preprocessing[epoched_filter]": _gap(
        SHAPE, AssertionError
    ),
    "tests/test_mmo_eeglab_tests.py::test_upstream_mmo_original_transposed_assignments[case1]": _gap(
        SHAPE, AssertionError
    ),
    "tests/test_numdim.py::test_reference_numdim": _gap(SHAPE, AssertionError),
    "tests/test_phase2_file_channel_location.py::test_reference_chancenter_2d": _gap(SHAPE, AssertionError),
    "tests/test_phase2_file_channel_location.py::test_reference_chancenter_all_negative": _gap(SHAPE, AssertionError),
    "tests/test_phase2_file_channel_location.py::test_reference_chancenter_optimize": _gap(SHAPE, AssertionError),
    "tests/test_phase2_file_channel_location.py::test_reference_chancenter_scalar": _gap(SHAPE, AssertionError),
    "tests/test_phase2_file_channel_location.py::test_reference_chancenter_zero_radius": _gap(SHAPE, AssertionError),
    "tests/test_phase4_timefreq_statistics.py::test_reference_angtimewarp": _gap(SHAPE, AssertionError),
    "tests/test_phase7_long_tail_helpers.py::test_reference_eegthresh": _gap(SHAPE, AssertionError),
    "tests/test_phase7_long_tail_helpers.py::test_reference_eegthresh_all_rejected": _gap(SHAPE, AssertionError),
    "tests/test_phase7_long_tail_helpers.py::test_reference_eegthresh_none_rejected": _gap(SHAPE, AssertionError),
    "tests/test_phase7_long_tail_helpers.py::test_reference_eegthresh_one_electrode": _gap(SHAPE, AssertionError),
    "tests/test_phase7_long_tail_helpers.py::test_reference_entropy_rej_vectors": _gap(SHAPE, AssertionError),
    "tests/test_phase7_long_tail_helpers.py::test_reference_kurt": _gap(SHAPE, AssertionError),
    "tests/test_phase7_long_tail_helpers.py::test_reference_realproba": _gap(SHAPE, AssertionError),
    "tests/test_rejection_workflows.py::test_reference_rejsuperpose_all": _gap(SHAPE, AssertionError),
    "tests/test_rejection_workflows.py::test_reference_rejsuperpose_all_ica": _gap(SHAPE, AssertionError),
    "tests/test_rejection_workflows.py::test_reference_rejsuperpose_empty": _gap(SHAPE, AssertionError),
    "tests/test_rejection_workflows.py::test_reference_rejsuperpose_general": _gap(SHAPE, AssertionError),
    "tests/test_rejection_workflows.py::test_reference_rejsuperpose_zero": _gap(SHAPE, AssertionError),
    "tests/test_sigproc_eegrej.py::test_reference_eegrej_explicit_and_omitted_events": _gap(SHAPE, AssertionError),
    "tests/test_sigproc_eegrej.py::test_reference_eegrej_overlapping_regions": _gap(SHAPE, AssertionError),
    "tests/test_sigprocfunc_ica_legacy_eeglab_tests.py::test_reference_kmeanscluster": _gap(SHAPE, AssertionError),
    "tests/test_sigprocfunc_numerical_eeglab_tests.py::test_reference_matsel": _gap(SHAPE, AssertionError),
    "tests/test_sigprocfunc_numerical_eeglab_tests.py::test_reference_nan_mean": _gap(SHAPE, AssertionError),
    "tests/test_sigprocfunc_numerical_eeglab_tests.py::test_reference_quantile": _gap(SHAPE, AssertionError),
    "tests/test_statistics_package.py::test_reference_concatdata_original_arrays": _gap(SHAPE, AssertionError),
    "tests/test_eeg_lat2point.py::test_eeg_lat2point_current_suite_cell_epoch_case": _gap(INDEX, IndexError),
    "tests/test_eeg_lat2point.py::test_eeg_lat2point_current_suite_cell_latency_case": _gap(INDEX, IndexError),
    "tests/test_eeg_lat2point.py::test_eeg_lat2point_current_suite_clamps_out_of_range_latency": _gap(
        INDEX, IndexError
    ),
    "tests/test_eeg_lat2point.py::test_eeg_lat2point_current_suite_default_time_unit": _gap(INDEX, IndexError),
    "tests/test_eeg_lat2point.py::test_eeg_lat2point_current_suite_general_case": _gap(INDEX, IndexError),
    "tests/test_eeg_lat2point.py::test_eeg_lat2point_current_suite_single_epoch_broadcast": _gap(INDEX, IndexError),
    "tests/test_eeg_retrieve.py::test_reference_eeg_retrieve_original_general_case": _gap(INDEX, IndexError),
    "tests/test_misc_data_event_eeglab_tests.py::test_reference_rmart": _gap(RMART, ValueError),
    "tests/test_miscfunc_numerical_eeglab_tests.py::test_reference_gauss": _gap(INDEX, IndexError),
    "tests/test_mmo_eeglab_tests.py::test_upstream_mmo_original_assignments[case-10]": _gap(INDEX, IndexError),
    "tests/test_mmo_eeglab_tests.py::test_upstream_mmo_original_assignments[case-11]": _gap(INDEX, IndexError),
    "tests/test_mmo_eeglab_tests.py::test_upstream_mmo_original_assignments[case-12]": _gap(INDEX, IndexError),
    "tests/test_mmo_eeglab_tests.py::test_upstream_mmo_original_assignments[case-13]": _gap(INDEX, IndexError),
    "tests/test_mmo_eeglab_tests.py::test_upstream_mmo_original_assignments[case-14]": _gap(INDEX, IndexError),
    "tests/test_mmo_eeglab_tests.py::test_upstream_mmo_original_assignments[case-15]": _gap(INDEX, IndexError),
    "tests/test_mmo_eeglab_tests.py::test_upstream_mmo_original_assignments[case-3]": _gap(INDEX, IndexError),
    "tests/test_mmo_eeglab_tests.py::test_upstream_mmo_original_assignments[case-5]": _gap(INDEX, IndexError),
    "tests/test_mmo_eeglab_tests.py::test_upstream_mmo_original_assignments[case-7]": _gap(INDEX, IndexError),
    "tests/test_mmo_eeglab_tests.py::test_upstream_mmo_original_assignments[case-8]": _gap(INDEX, IndexError),
    "tests/test_mmo_eeglab_tests.py::test_upstream_mmo_original_assignments[case-9]": _gap(INDEX, IndexError),
    "tests/test_mmo_eeglab_tests.py::test_upstream_mmo_original_copy_on_write_diagnostics": _gap(INDEX, IndexError),
    "tests/test_mmo_eeglab_tests.py::test_upstream_mmo_original_deletions[case-4]": _gap(INDEX, IndexError),
    "tests/test_mmo_eeglab_tests.py::test_upstream_mmo_original_returned_workspace": _gap(INDEX, IndexError),
    "tests/test_mmo_eeglab_tests.py::test_upstream_mmo_original_transposed_assignments[case0]": _gap(INDEX, IndexError),
    "tests/test_mmo_eeglab_tests.py::test_upstream_mmo_original_transposed_assignments[case2]": _gap(INDEX, IndexError),
    "tests/test_mmo_eeglab_tests.py::test_upstream_mmo_original_transposed_assignments[case3]": _gap(INDEX, IndexError),
    "tests/test_mmo_eeglab_tests.py::test_upstream_mmo_original_transposed_assignments[case4]": _gap(INDEX, IndexError),
    "tests/test_mmo_eeglab_tests.py::test_upstream_mmo_original_transposed_assignments[case5]": _gap(INDEX, IndexError),
    "tests/test_mmo_eeglab_tests.py::test_upstream_mmo_original_transposed_assignments[case6]": _gap(INDEX, IndexError),
    "tests/test_mmo_eeglab_tests.py::test_upstream_mmo_original_transposed_assignments[case7]": _gap(INDEX, IndexError),
    "tests/test_mmo_eeglab_tests.py::test_upstream_mmo_original_transposed_assignments[case8]": _gap(INDEX, IndexError),
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
    "tests/test_phase2_file_channel_location.py::test_reference_pop_chancenter_empty_center": _gap(
        OUTPUTS, AssertionError
    ),
    "tests/test_phase2_file_channel_location.py::test_reference_pop_chancenter_no_omitchans": _gap(
        OUTPUTS, AssertionError
    ),
    "tests/test_phase2_file_channel_location.py::test_reference_pop_chancenter_with_omitchans": _gap(
        OUTPUTS, AssertionError
    ),
    "tests/test_phase2_file_channel_location.py::test_reference_readelp_five_outputs": _gap(OUTPUTS, AssertionError),
    "tests/test_pop_comments.py::test_reference_comments_concatenation": _gap(OUTPUTS, AssertionError),
    "tests/test_pop_comments.py::test_reference_comments_replacement": _gap(OUTPUTS, AssertionError),
    "tests/test_pop_rmbase.py::test_reference_pop_rmbase_original_baselines_and_complete_commands": _gap(
        OUTPUTS, AssertionError
    ),
    "tests/test_pop_rmbase.py::test_reference_rmbase": _gap(OUTPUTS, AssertionError),
    "tests/test_rejection_workflows.py::test_reference_pop_rejchan": _gap(OUTPUTS, AssertionError),
    "tests/test_rejection_workflows.py::test_reference_pop_rejtrend": _gap(OUTPUTS, AssertionError),
    "tests/test_binary_eeglab_tests.py::test_upstream_chancoresp_original_39_calls": _gap(TRUTH, ValueError),
    "tests/test_clean_rawdata_eeglab_tests.py::test_reference_clean_rawdata_channels_with_locations": _gap(
        TRUTH, ValueError
    ),
    "tests/test_clean_rawdata_eeglab_tests.py::test_reference_clean_rawdata_filtering": _gap(TRUTH, ValueError),
    "tests/test_clean_rawdata_eeglab_tests.py::test_reference_clean_rawdata_final_rejection": _gap(TRUTH, ValueError),
    "tests/test_clean_rawdata_eeglab_tests.py::test_reference_clean_rawdata_rejection": _gap(TRUTH, ValueError),
    "tests/test_eegh.py::test_reference_eegh_empty_command": _gap(TRUTH, ValueError),
    "tests/test_misc_data_event_eeglab_tests.py::test_reference_shortread_original_two_million_samples": _gap(
        TRUTH, ValueError
    ),
    "tests/test_phase1b_file_edit_pop_functions.py::test_reference_copyset_recorded_dataset_calls": _gap(
        TRUTH, ValueError
    ),
    "tests/test_phase2_file_channel_location.py::test_reference_floatread_offset": _gap(TRUTH, ValueError),
    "tests/test_phase2_file_channel_location.py::test_reference_readlocs_bugzilla_339": _gap(TRUTH, ValueError),
    "tests/test_popfunc_dataset_eeglab_tests.py::test_reference_importevent_original_text_file": _gap(
        TRUTH, ValueError
    ),
    "tests/test_sample_data_pop_functions.py::test_reference_saveh_original_global_and_recording_history": _gap(
        TRUTH, ValueError
    ),
    "tests/test_sample_data_pop_functions.py::test_reference_select_original_time_trials_and_channel": _gap(
        TRUTH, ValueError
    ),
    "tests/test_vararg2str.py::test_reference_vararg2str_all_eight_original_calls": _gap(TRUTH, ValueError),
    "tests/test_binary_eeglab_tests.py::test_upstream_pop_biosig_original_bdf_blockrange": _gap(POSITIONAL, TypeError),
    "tests/test_eegobj.py::test_reference_eegobj_collection_operations": _gap(POSITIONAL, TypeError),
    "tests/test_miscfunc_decomposition_eeglab_tests.py::test_reference_make_timewarp": _gap(POSITIONAL, TypeError),
    "tests/test_miscfunc_decomposition_eeglab_tests.py::test_reference_runicalowmem": _gap(POSITIONAL, TypeError),
    "tests/test_phase4_timefreq_statistics.py::test_reference_newtimef_original_baseline_formulas": _gap(
        POSITIONAL, TypeError
    ),
    "tests/test_runica.py::test_reference_runica_extended": _gap(POSITIONAL, TypeError),
    "tests/test_runica.py::test_reference_runica_extended_noise": _gap(POSITIONAL, TypeError),
    "tests/test_runica.py::test_reference_runica_general_smoke": _gap(POSITIONAL, TypeError),
    "tests/test_runica.py::test_reference_runica_pca": _gap(POSITIONAL, TypeError),
    "tests/test_sample_data_pop_functions.py::test_upstream_pop_fileio_original_sample_options": _gap(
        POSITIONAL, TypeError
    ),
    "tests/test_study_params_eeglab_tests.py::test_reference_pop_dipparams[eeglab_sample_study0]": _gap(
        STUDY, UnicodeDecodeError
    ),
    "tests/test_study_params_eeglab_tests.py::test_reference_pop_erpimparams[eeglab_sample_study0]": _gap(
        STUDY, UnicodeDecodeError
    ),
    "tests/test_study_params_eeglab_tests.py::test_reference_pop_erpparams[eeglab_sample_study0]": _gap(
        STUDY, UnicodeDecodeError
    ),
    "tests/test_study_params_eeglab_tests.py::test_reference_pop_erspparams[eeglab_sample_study0]": _gap(
        STUDY, UnicodeDecodeError
    ),
    "tests/test_study_params_eeglab_tests.py::test_reference_pop_specparams[eeglab_sample_study0]": _gap(
        STUDY, UnicodeDecodeError
    ),
    "tests/test_study_params_eeglab_tests.py::test_reference_pop_statparams[eeglab_sample_study0]": _gap(
        STUDY, UnicodeDecodeError
    ),
    "tests/test_study_workflows_eeglab_tests.py::test_reference_pop_clust[eeglab_sample_study0]": _gap(
        STUDY, UnicodeDecodeError
    ),
    "tests/test_study_workflows_eeglab_tests.py::test_reference_std_makedesign[eeglab_sample_study0]": _gap(
        STUDY, UnicodeDecodeError
    ),
    "tests/test_study_workflows_eeglab_tests.py::test_reference_std_selectdesign[eeglab_sample_study0]": _gap(
        STUDY, UnicodeDecodeError
    ),
    "tests/test_finputcheck.py::test_reference_finputcheck_original_wrapper_cases": _gap(SHAPE, AssertionError),
    "tests/test_pop_editeventvals_current.py::test_reference_editeventvals_original_recording": _gap(
        EVENT_EDIT, ValueError
    ),
}

PYTHON_BACKEND_SUBTEST_XFAILS = {
    "tests/test_eegplot.py::test_reference_eegplot2trial_original_marks": {
        "(source='general')": _gap(SHAPE, AssertionError),
        "(source='multiple')": _gap(SHAPE, AssertionError),
        "(source='multiple_color')": _gap(SHAPE, AssertionError),
        "(source='multiple_colorin')": _gap(SHAPE, AssertionError),
        "(source='multiple_colorout')": _gap(SHAPE, AssertionError),
    },
    "tests/test_epoch.py::test_reference_epoch_original_five_output_contracts": {
        "(case='no_srate')": _gap(STRUCT, ValueError),
        "(case='general')": _gap(NAMEVALUE, ValueError),
        "(case='boundary')": _gap(NAMEVALUE, ValueError),
        "(case='valuelim')": _gap(NAMEVALUE, ValueError),
        "(case='allevents')": _gap(NAMEVALUE, ValueError),
    },
    "tests/test_sigprocfunc_plotting_eeglab_tests.py::test_reference_forcelocs_complete_original_coordinates": {
        "(source='x')": _gap(STRUCT, AttributeError),
        "(source='y')": _gap(STRUCT, AttributeError),
    },
}
