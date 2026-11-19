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
OUTPUTS = "the function does not return the MATLAB output count"
EVENT_CONTAINER = (
    "EEG.event stays a list of dicts and EEG.eventdescription stays a list, "
    "so MATLAB struct-array and character-array indexing does not apply"
)
EMPTY_STRUCT = (
    "near.m returns false for two empty structs because its field loop never "
    "runs, and EEG.etc and EEG.roi are empty dicts"
)
LOADED_EVENTS = "pop_loadset events are an object array of dicts, so field indexing fails before pop_adjustevents runs"
STUDY = "the sample STUDY loads, but design values are still nested MATLAB cells"
RMART = (
    "rmart.m:94 returns when rem(length(data), nchans) == 0, inverting the "
    "divisibility test, so the documented regression never runs; EEGPrep keeps "
    "the reachable correction. This is a permanent EEGLAB difference"
)

PYTHON_BACKEND_XFAILS = {
    "tests/test_binary_eeglab_tests.py::test_upstream_chancoresp_clear": _gap(OUTPUTS, AssertionError),
    "tests/test_binary_eeglab_tests.py::test_upstream_chancoresp_explicit_pairs": _gap(SHAPE, AssertionError),
    "tests/test_binary_eeglab_tests.py::test_upstream_chancoresp_fiducials": _gap(SHAPE, AssertionError),
    "tests/test_binary_eeglab_tests.py::test_upstream_chancoresp_invalid_fiducials": _gap(SHAPE, AssertionError),
    "tests/test_binary_eeglab_tests.py::test_upstream_chancoresp_labels": _gap(SHAPE, AssertionError),
    "tests/test_binary_eeglab_tests.py::test_upstream_chancoresp_none": _gap(SHAPE, AssertionError),
    "tests/test_binary_eeglab_tests.py::test_upstream_chancoresp_pair": _gap(SHAPE, AssertionError),
    "tests/test_binary_eeglab_tests.py::test_upstream_chancoresp_unpair": _gap(SHAPE, AssertionError),
    "tests/test_core_event_eeglab_tests.py::test_reference_getchanlist_no_match": _gap(SHAPE, AttributeError),
    "tests/test_core_event_eeglab_tests.py::test_reference_getepochevent_default_latency": _gap(SHAPE, AssertionError),
    "tests/test_core_event_eeglab_tests.py::test_reference_getepochevent_default_time_window": _gap(
        SHAPE, AssertionError
    ),
    "tests/test_core_event_eeglab_tests.py::test_reference_getepochevent_duration_old_and_new_calls": _gap(
        SHAPE, AssertionError
    ),
    "tests/test_core_event_eeglab_tests.py::test_reference_getepochevent_four_argument_window": _gap(
        SHAPE, AssertionError
    ),
    "tests/test_core_event_eeglab_tests.py::test_reference_getepochevent_without_epoch_field": _gap(
        SHAPE, AssertionError
    ),
    "tests/test_core_event_eeglab_tests.py::test_reference_matchchans_original_complete_locations": _gap(
        SHAPE, AssertionError
    ),
    "tests/test_core_event_eeglab_tests.py::test_reference_mergechan_uses_original_montages": _gap(SHAPE, TypeError),
    "tests/test_core_event_eeglab_tests.py::test_reference_mergelocs_uses_original_montages": _gap(SHAPE, TypeError),
    "tests/test_core_event_eeglab_tests.py::test_reference_urlatency_boundary_durations": _gap(SHAPE, AssertionError),
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
    "tests/test_sigprocfunc_numerical_eeglab_tests.py::test_reference_mattocell": _gap(SHAPE, AttributeError),
    "tests/test_bdf_records_eeglab_tests.py::test_upstream_readbdf_original_record_selection": _gap(
        SHAPE, AssertionError
    ),
    "tests/test_cart2topo.py::test_reference_cart2topo_original_coordinate_matrices": _gap(SHAPE, AssertionError),
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
    "tests/test_eeg_retrieve.py::test_reference_eeg_retrieve_original_general_case": _gap(SHAPE, AssertionError),
    "tests/test_misc_data_event_eeglab_tests.py::test_reference_rmart": _gap(RMART, ValueError),
    "tests/test_pop_adjustevents.py::test_reference_adjustevents_continuous": _gap(LOADED_EVENTS, IndexError),
    "tests/test_pop_adjustevents.py::test_reference_adjustevents_epoched": _gap(LOADED_EVENTS, IndexError),
    "tests/test_phase1b_file_edit_pop_functions.py::test_reference_copyset_original_set_out": _gap(
        EMPTY_STRUCT, AssertionError
    ),
    "tests/test_pop_editeventfield_current.py::test_reference_add_field_file": _gap(EVENT_CONTAINER, IndexError),
    "tests/test_pop_editeventfield_current.py::test_reference_add_field_file_indices": _gap(
        EVENT_CONTAINER, IndexError
    ),
    "tests/test_pop_editeventfield_current.py::test_reference_add_field_vector": _gap(EVENT_CONTAINER, IndexError),
    "tests/test_pop_editeventfield_current.py::test_reference_delimiters": _gap(EVENT_CONTAINER, IndexError),
    "tests/test_pop_editeventfield_current.py::test_reference_delold_yes": _gap(EVENT_CONTAINER, AttributeError),
    "tests/test_pop_editeventfield_current.py::test_reference_description_missing_field": _gap(
        EVENT_CONTAINER, AssertionError
    ),
    "tests/test_pop_editeventfield_current.py::test_reference_field_description": _gap(EVENT_CONTAINER, TypeError),
    "tests/test_pop_editeventfield_current.py::test_reference_invalid_rename": _gap(EVENT_CONTAINER, TypeError),
    "tests/test_pop_editeventfield_current.py::test_reference_latency_timeunit": _gap(EVENT_CONTAINER, TypeError),
    "tests/test_pop_editeventfield_current.py::test_reference_modify_field": _gap(EVENT_CONTAINER, IndexError),
    "tests/test_pop_editeventfield_current.py::test_reference_modify_field_indices": _gap(EVENT_CONTAINER, IndexError),
    "tests/test_pop_editeventfield_current.py::test_reference_rename_field": _gap(EVENT_CONTAINER, TypeError),
    "tests/test_pop_editeventfield_current.py::test_reference_rename_missing_field": _gap(EVENT_CONTAINER, TypeError),
    "tests/test_pop_editeventfield_current.py::test_reference_skip_lines": _gap(EVENT_CONTAINER, IndexError),
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
    "tests/test_finputcheck.py::test_reference_finputcheck_original_wrapper_cases": _gap(SHAPE, AssertionError),
}

PYTHON_BACKEND_SUBTEST_XFAILS = {
    "tests/test_eegplot.py::test_reference_eegplot2trial_original_marks": {
        "(source='general')": _gap(SHAPE, AssertionError),
        "(source='multiple')": _gap(SHAPE, AssertionError),
        "(source='multiple_color')": _gap(SHAPE, AssertionError),
        "(source='multiple_colorin')": _gap(SHAPE, AssertionError),
        "(source='multiple_colorout')": _gap(SHAPE, AssertionError),
    },
    "tests/test_sigprocfunc_plotting_eeglab_tests.py::test_reference_forcelocs_complete_original_coordinates": {
        "(source='x')": _gap(SHAPE, AssertionError),
        "(source='y')": _gap(SHAPE, AssertionError),
    },
}
