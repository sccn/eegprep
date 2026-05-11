"""EEG-BIDS plugin menu placeholders."""

from __future__ import annotations

from eegprep.functions.guifunc.menu_spec import MenuItemSpec, menu_item


def eeg_bids_import_items() -> tuple[MenuItemSpec, ...]:
    """Return File > Import data additions from the EEGLAB EEG-BIDS/File-IO plugins."""
    return (
        menu_item("From BIDS folder structure", action="pop_importbids", separator=True, origin="plugin:eeg_bids"),
        menu_item("Import Magstim/EGI .mff file", action="pop_fileio_mff", separator=True, origin="plugin:fileio"),
        menu_item("From Neuroscan .CNT file", action="pop_fileio_cnt", separator=True, origin="plugin:fileio"),
        menu_item("From Neuroscan .EEG file", action="pop_fileio_eeg", origin="plugin:fileio"),
        menu_item(
            "From Brain Vis. Rec. .vhdr file",
            action="pop_fileio_brainvision",
            separator=True,
            origin="plugin:fileio",
        ),
        menu_item("From Brain Vis. Anal. Matlab file", action="pop_fileio_brainvision_mat", origin="plugin:fileio"),
    )


def eeg_bids_export_items() -> tuple[MenuItemSpec, ...]:
    """Return File > Export additions from the EEGLAB EEG-BIDS plugin."""
    return (
        menu_item(
            "To BIDS folder structure",
            action="pop_exportbids",
            userdata="startup:off;study:on",
            separator=True,
            origin="plugin:eeg_bids",
        ),
    )


def eeg_bids_tools_menu() -> MenuItemSpec:
    """Return the File > BIDS tools plugin submenu."""
    return menu_item(
        "BIDS tools",
        userdata="startup:on;study:on",
        separator=True,
        origin="plugin:eeg_bids",
        children=[
            menu_item("BIDS export wizard (from raw EEG to BIDS)", action="bids_exporter", origin="plugin:eeg_bids"),
            menu_item("Import BIDS folder to STUDY", action="pop_importbids", separator=True, origin="plugin:eeg_bids"),
            menu_item(
                "Export STUDY to BIDS folder",
                action="pop_exportbids",
                userdata="startup:off;study:on",
                origin="plugin:eeg_bids",
            ),
            menu_item(
                "Edit BIDS task info",
                action="pop_taskinfo",
                userdata="study:on",
                separator=True,
                origin="plugin:eeg_bids",
            ),
            menu_item(
                "Edit BIDS participant info",
                action="pop_participantinfo",
                userdata="study:on",
                origin="plugin:eeg_bids",
            ),
            menu_item("Edit BIDS event info", action="pop_eventinfo", userdata="study:on", origin="plugin:eeg_bids"),
            menu_item(
                "Validate BIDS dataset",
                action="validate_bids",
                userdata="startup:on;study:on",
                separator=True,
                origin="plugin:eeg_bids",
            ),
        ],
    )
