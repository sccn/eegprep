"""Explicit placeholders for EEGLAB main-window actions not yet ported.

Each placeholder carries either an implementation phase or an exclusion reason
so later phase agents can distinguish planned work from intentionally excluded
browser/external-plugin workflows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class PlaceholderMetadata:
    """Phase ownership or exclusion metadata for a placeholder action."""

    phase: str | None = None
    excluded_reason: str | None = None


def _phase_entries(phase: str, actions: tuple[str, ...]) -> dict[str, PlaceholderMetadata]:
    return {action: PlaceholderMetadata(phase=phase) for action in actions}


PLACEHOLDER_ACTION_METADATA: Mapping[str, PlaceholderMetadata] = {
    **_phase_entries(
        "1b",
        (
            "pop_chanedit",
            "pop_comments",
            "pop_copyset",
            "pop_editeventfield",
            "pop_editeventvals",
            "pop_editset",
            "pop_fileio_brainvision_mat",
            "pop_mergeset",
            "pop_rmdat",
            "pop_selectevent",
            "select_multiple_datasets",
        ),
    ),
    **_phase_entries(
        "2",
        (
            "pop_eegfilt",
            "pop_eegfiltnew",
            "pop_firma",
            "pop_firpm",
            "pop_firws",
            "pop_rmbase",
        ),
    ),
    **_phase_entries(
        "3",
        (
            "eeg_rejsuperpose",
            "pop_autorej",
            "pop_dipfit_gridsearch",
            "pop_dipfit_headmodel",
            "pop_dipfit_loreta",
            "pop_dipfit_nonlinear",
            "pop_dipfit_settings",
            "pop_dipplot",
            "pop_eegthresh",
            "pop_icflag",
            "pop_jointprob",
            "pop_leadfield",
            "pop_multifit",
            "pop_rejchan",
            "pop_rejcont",
            "pop_rejepoch",
            "pop_rejkurt",
            "pop_rejmenu",
            "pop_rejspec",
            "pop_rejtrend",
            "pop_selectcomps",
            "pop_subcomp",
            "pop_viewprops",
        ),
    ),
    **_phase_entries(
        "4",
        (
            "pop_chanplot",
            "pop_comperp",
            "pop_envtopo",
            "pop_erpimage",
            "pop_eventstat",
            "pop_headplot",
            "pop_newcrossf",
            "pop_newtimef",
            "pop_plotdata",
            "pop_plottopo",
            "pop_prop",
            "pop_signalstat",
            "pop_spectopo",
            "pop_timtopo",
            "pop_topoplot",
            "topoplot",
        ),
    ),
    **_phase_entries(
        "5",
        (
            "pop_clust",
            "pop_clustedit",
            "pop_preclust",
            "pop_precomp",
            "pop_studydesign",
            "select_study_set",
        ),
    ),
    **_phase_entries("6", ("eeglab_update",)),
    "pop_eegplot": PlaceholderMetadata(excluded_reason="eegbrowser"),
}
PLACEHOLDER_ACTIONS = frozenset(PLACEHOLDER_ACTION_METADATA)


def placeholder_metadata(action: str) -> PlaceholderMetadata | None:
    """Return placeholder metadata for ``action`` when registered."""
    return PLACEHOLDER_ACTION_METADATA.get(action.partition(":")[0])


def placeholder_inventory() -> Mapping[str, PlaceholderMetadata]:
    """Return the machine-readable placeholder inventory."""
    return PLACEHOLDER_ACTION_METADATA


def is_placeholder_action(action: str) -> bool:
    """Return whether a menu action has an explicit coming-soon placeholder."""
    return placeholder_metadata(action) is not None


def placeholder_message(action: str) -> str:
    """Build the shared EEGLAB-style placeholder message for ``action``."""
    metadata = placeholder_metadata(action)
    if metadata is not None and metadata.excluded_reason == "eegbrowser":
        return (
            f"{action} depends on EEGBrowser/eegplot-style scrolling browser workflows, "
            "which are outside the current EEGPrep parity scope."
        )
    return (
        f"{action} is not yet available in EEGPrep.\n\n"
        "Track progress or request this workflow at https://github.com/sccn/eegprep/issues."
    )
