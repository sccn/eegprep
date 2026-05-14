"""Explicit placeholders for EEGLAB main-window actions not yet ported.

TODO: replace each action family with the matching EEGPrep GUI flow and
remove it from this registry when the implementation is complete.
"""

from __future__ import annotations


PLACEHOLDER_ACTIONS = frozenset(
    {
        "bids_exporter",
        "eeg_rejsuperpose",
        "eeglab_update",
        "plugin_menu",
        "pop_autorej",
        "pop_biosig",
        "pop_chanedit",
        "pop_chanevent",
        "pop_chanplot",
        "pop_clust",
        "pop_clustedit",
        "pop_comments",
        "pop_comperp",
        "pop_copyset",
        "pop_dipfit_gridsearch",
        "pop_dipfit_headmodel",
        "pop_dipfit_loreta",
        "pop_dipfit_nonlinear",
        "pop_dipfit_settings",
        "pop_dipplot",
        "pop_editeventfield",
        "pop_editeventvals",
        "pop_editoptions",
        "pop_editset",
        "pop_eegfilt",
        "pop_eegfiltnew",
        "pop_eegplot",
        "pop_eegthresh",
        "pop_envtopo",
        "pop_epoch",
        "pop_erpimage",
        "pop_eventstat",
        "pop_expevents",
        "pop_expica",
        "pop_export",
        "pop_exportbids",
        "pop_eventinfo",
        "pop_fileio",
        "pop_fileio_brainvision",
        "pop_fileio_brainvision_mat",
        "pop_fileio_cnt",
        "pop_fileio_eeg",
        "pop_fileio_mff",
        "pop_firma",
        "pop_firpm",
        "pop_firws",
        "pop_headplot",
        "pop_icflag",
        "pop_importdata",
        "pop_importbids",
        "pop_importepoch",
        "pop_importerplab",
        "pop_importevent",
        "pop_importpres",
        "pop_jointprob",
        "pop_leadfield",
        "pop_loadstudy",
        "pop_mergeset",
        "pop_multifit",
        "pop_newcrossf",
        "pop_newtimef",
        "pop_plotdata",
        "pop_plottopo",
        "pop_preclust",
        "pop_precomp",
        "pop_participantinfo",
        "pop_prop",
        "pop_rejchan",
        "pop_rejcont",
        "pop_rejepoch",
        "pop_rejkurt",
        "pop_rejmenu",
        "pop_rejspec",
        "pop_rejtrend",
        "pop_rmbase",
        "pop_rmdat",
        "pop_runscript",
        "pop_saveh",
        "pop_savestudy",
        "pop_selectcomps",
        "pop_selectevent",
        "pop_signalstat",
        "pop_spectopo",
        "pop_study",
        "pop_studydesign",
        "pop_studyerp",
        "pop_studywizard",
        "pop_subcomp",
        "pop_taskinfo",
        "pop_timtopo",
        "pop_topoplot",
        "pop_viewprops",
        "pop_writeeeg",
        "select_multiple_datasets",
        "select_study_set",
        "topoplot",
        "validate_bids",
    }
)


def is_placeholder_action(action: str) -> bool:
    """Return whether a menu action has an explicit coming-soon placeholder."""
    return action.partition(":")[0] in PLACEHOLDER_ACTIONS


def placeholder_message(action: str) -> str:
    """Build the shared EEGLAB-style placeholder message for ``action``."""
    return (
        f"{action} is not yet available in EEGPrep.\n\n"
        "Track progress or request this workflow at https://github.com/sccn/eegprep/issues."
    )
