"""EEGLAB main-window menu tree for EEGPrep."""

from __future__ import annotations

from eegprep.functions.guifunc.menu_spec import MenuItemSpec, menu_item, visible_menu_items
from eegprep.plugins.EEG_BIDS.menu import (
    eeg_bids_export_items,
    eeg_bids_import_items,
    eeg_bids_tools_menu,
)
from eegprep.plugins.ICLabel.menu import iclabel_menu, viewprops_plot_menus
from eegprep.plugins.clean_rawdata.menu import clean_rawdata_menu
from eegprep.plugins.dipfit.menu import dipfit_menu
from eegprep.plugins.firfilt.menu import firfilt_filter_items

ON = "study:on"
ON_NO_STUDY = ""
ON_DATA = "startup:off"
ON_DATA_NO_ROI = "startup:off;roi:off"
ON_EPOCH = "startup:off;continuous:off"
ON_EPOCH_NO_ROI = "startup:off;continuous:off;roi:off"
ON_DATA_STUDY = "startup:off;study:on"
ON_DATA_STUDY_NO_ROI = "startup:off;study:on;roi:off"
ON_CHANNEL = "startup:off;chanloc:on"
ON_CHANNEL_NO_ROI = "startup:off;chanloc:on;roi:off"
ON_EPOCH_CHANNEL = "startup:off;continuous:off;chanloc:on"
ON_STUDY = "startup:off;epoch:off;continuous:off;study:on"
ON_STUDY_NO_ROI = "startup:off;epoch:off;continuous:off;study:on;roi:off"
OFF = "enable:off"


def eeglab_core_menus() -> tuple[MenuItemSpec, ...]:
    """Return the core EEGLAB menu tree before plugin insertion."""
    file_menu = menu_item(
        "File",
        userdata=ON,
        children=[
            menu_item(
                "Import data",
                userdata=ON_NO_STUDY,
                children=[
                    menu_item(
                        "Using EEGPrep functions and plugins",
                        tag="import data",
                        userdata=ON_NO_STUDY,
                        children=[
                            menu_item("(for more use menu File > Manage EEGPrep extensions)", userdata=OFF, enabled=False),
                            menu_item("From ASCII/float file or MATLAB array", action="pop_importdata", separator=True),
                            menu_item("From Biosemi BDF file (BIOSIG toolbox)", action="pop_biosig", separator=True),
                            menu_item("From EDF/EDF+/GDF files (BIOSIG toolbox)", action="pop_biosig"),
                        ],
                    ),
                    menu_item("Using the FILE-IO interface", action="pop_fileio", separator=True),
                    menu_item("Using the BIOSIG interface", action="pop_biosig"),
                    menu_item("Troubleshooting data formats...", action="help:troubleshooting_data_formats"),
                ],
            ),
            menu_item(
                "Import epoch info",
                tag="import epoch",
                userdata=ON_EPOCH,
                children=[menu_item("From MATLAB array or ASCII file", action="pop_importepoch")],
            ),
            menu_item(
                "Import event info",
                tag="import event",
                userdata=ON_DATA,
                children=[
                    menu_item("From MATLAB array or ASCII file", action="pop_importevent"),
                    menu_item("From data channel", action="pop_chanevent"),
                    menu_item("From Presentation .LOG file", action="pop_importpres"),
                    menu_item("From E-Prime ASCII (text) file", action="pop_importevent"),
                    menu_item("From ERPLAB text files", action="pop_importerplab"),
                ],
            ),
            menu_item(
                "Export",
                tag="export",
                userdata=ON_DATA_STUDY,
                children=[
                    menu_item("(for more use menu File > Manage EEGPrep extensions)", userdata=OFF, enabled=False),
                    menu_item("Data and ICA activity to text file", action="pop_export", separator=True),
                    menu_item("Weight matrix to text file", action="pop_expica:weights"),
                    menu_item("Inverse weight matrix to text file", action="pop_expica:inv"),
                    menu_item("Events to text file", action="pop_expevents"),
                    menu_item("Data to EDF/BDF/GDF file", action="pop_writeeeg", separator=True),
                ],
            ),
            menu_item("Load existing dataset", action="pop_loadset", userdata=ON_NO_STUDY, separator=True),
            menu_item("Resave current dataset(s)", action="pop_saveset:resave", userdata=ON_DATA_STUDY),
            menu_item("Save current dataset as", action="pop_saveset", userdata=ON_DATA),
            menu_item("Clear dataset(s)", action="pop_delset", userdata=ON_DATA),
            menu_item(
                "Create study",
                userdata=ON,
                separator=True,
                children=[
                    menu_item("Using all loaded datasets", action="pop_study", userdata=ON_DATA),
                    menu_item("Browse for datasets", action="pop_studywizard", userdata=ON),
                    menu_item("Simple ERP STUDY", action="pop_studyerp", userdata=ON),
                ],
            ),
            menu_item("Load existing study", action="pop_loadstudy", userdata=ON, separator=True),
            menu_item("Save current study", action="pop_savestudy:resave", userdata=ON_STUDY),
            menu_item("Save current study as", action="pop_savestudy", userdata=ON_STUDY),
            menu_item("Clear study / Clear all", action="clear_study", userdata=ON_DATA_STUDY),
            menu_item("Preferences", action="pop_editoptions", userdata=ON, separator=True),
            menu_item(
                "History scripts",
                userdata=ON,
                separator=True,
                children=[
                    menu_item("Save dataset history script", action="pop_saveh:dataset", userdata=ON_DATA),
                    menu_item("Save session history script", action="pop_saveh:session", userdata=ON_DATA_STUDY),
                    menu_item("Run script", action="pop_runscript", userdata=ON),
                ],
            ),
            menu_item("Manage EEGPrep extensions", action="plugin_menu", userdata=ON),
            menu_item("Quit", action="quit", userdata=ON, separator=True),
        ],
    )

    edit_menu = menu_item(
        "Edit",
        userdata=ON_DATA_STUDY,
        children=[
            menu_item("Dataset info", action="pop_editset", userdata=ON_DATA),
            menu_item("Event fields", action="pop_editeventfield", userdata=ON_DATA, visibility="allmenus"),
            menu_item("Event values", action="pop_editeventvals", userdata=ON_DATA),
            menu_item("Adjust event latencies", action="pop_adjustevents", userdata=ON_DATA, visibility="allmenus"),
            menu_item("About this dataset", action="pop_comments", userdata=ON_DATA),
            menu_item("Channel locations", action="pop_chanedit", userdata=ON_DATA_STUDY),
            menu_item("Select data", action="pop_select", userdata=ON_DATA_STUDY, separator=True),
            menu_item("Select data using events", action="pop_rmdat", userdata=ON_DATA_STUDY),
            menu_item("Select epochs or events", action="pop_selectevent", userdata=ON_DATA_STUDY),
            menu_item("Copy current dataset", action="pop_copyset", userdata=ON_DATA, separator=True),
            menu_item("Append datasets", action="pop_mergeset", userdata=ON_DATA),
            menu_item("Delete dataset(s) from memory", action="pop_delset", userdata=ON_DATA),
        ],
    )

    tools_menu = menu_item(
        "Tools",
        tag="tools",
        userdata=ON_DATA_STUDY,
        children=[
            menu_item('(Expand tool choices via "File > Preferences")', userdata=OFF, enabled=False, visibility="default"),
            menu_item("Change sampling rate", action="pop_resample", userdata=ON_DATA_STUDY, separator=True),
            menu_item(
                "Filter the data",
                tag="filter",
                userdata=ON_DATA_STUDY,
                children=[menu_item("Basic FIR filter (legacy)", action="pop_eegfilt", userdata=ON_DATA_STUDY)],
            ),
            menu_item("Re-reference the data", action="pop_reref", userdata=ON_DATA_STUDY),
            menu_item("Interpolate electrodes", action="pop_interp", userdata=ON_DATA),
            menu_item("Inspect/reject data by eye", action="pop_eegplot:data", userdata=ON_DATA, separator=True),
            menu_item("Automatic channel rejection", action="pop_rejchan", userdata=ON_DATA, visibility="allmenus"),
            menu_item("Automatic continuous rejection", action="pop_rejcont", userdata=ON_DATA, visibility="allmenus"),
            menu_item("Automatic epoch rejection", action="pop_autorej", userdata=ON_EPOCH, visibility="allmenus"),
            menu_item("Decompose data by ICA", action="pop_runica", userdata=ON_DATA_STUDY_NO_ROI, separator=True),
            menu_item(
                "Reject data epochs",
                userdata=ON_EPOCH,
                visibility="allmenus",
                children=[
                    menu_item("Reject data (all methods)", action="pop_rejmenu:data", userdata=ON_EPOCH),
                    menu_item("Reject by inspection", action="pop_eegplot:reject_data", userdata=ON_EPOCH),
                    menu_item("Reject extreme values", action="pop_eegthresh:data", userdata=ON_EPOCH),
                    menu_item("Reject by linear trend/variance", action="pop_rejtrend:data", userdata=ON_EPOCH),
                    menu_item("Reject by probability", action="pop_jointprob:data", userdata=ON_EPOCH),
                    menu_item("Reject by kurtosis", action="pop_rejkurt:data", userdata=ON_EPOCH),
                    menu_item("Reject by spectra", action="pop_rejspec:data", userdata=ON_EPOCH),
                    menu_item("Export marks to ICA reject", action="eeg_rejsuperpose:data_to_ica", userdata=ON_EPOCH, separator=True),
                    menu_item("Reject marked epochs", action="pop_rejepoch:data", userdata=ON_EPOCH, separator=True),
                ],
            ),
            menu_item("Inspect/label components by map", action="pop_selectcomps", userdata=ON_DATA, visibility="default"),
            menu_item(
                "Reject data using ICA",
                userdata=ON_DATA,
                visibility="allmenus",
                children=[
                    menu_item("Reject components by map", action="pop_selectcomps", userdata=ON_DATA),
                    menu_item("Reject data (all methods)", action="pop_rejmenu:ica", userdata=ON_EPOCH, separator=True),
                    menu_item("Reject by inspection", action="pop_eegplot:reject_ica", userdata=ON_EPOCH),
                    menu_item("Reject extreme values", action="pop_eegthresh:ica", userdata=ON_EPOCH),
                    menu_item("Reject by linear trend/variance", action="pop_rejtrend:ica", userdata=ON_EPOCH),
                    menu_item("Reject by probability", action="pop_jointprob:ica", userdata=ON_EPOCH),
                    menu_item("Reject by kurtosis", action="pop_rejkurt:ica", userdata=ON_EPOCH),
                    menu_item("Reject by spectra", action="pop_rejspec:ica", userdata=ON_EPOCH),
                    menu_item("Export marks to data reject", action="eeg_rejsuperpose:ica_to_data", userdata=ON_EPOCH, separator=True),
                    menu_item("Reject marked epochs", action="pop_rejepoch:ica", userdata=ON_EPOCH, separator=True),
                ],
            ),
            menu_item("Remove components from data", action="pop_subcomp", userdata=ON_DATA_STUDY),
            menu_item("Extract epochs", action="pop_epoch", userdata=ON_DATA_STUDY, separator=True),
            menu_item("Remove epoch baseline", action="pop_rmbase", userdata=ON_DATA_STUDY),
        ],
    )

    plot_menu = menu_item(
        "Plot",
        tag="plot",
        userdata=ON_DATA,
        children=[
            menu_item(
                "Channel locations",
                userdata=ON_CHANNEL,
                children=[
                    menu_item("By name", action="topoplot:labels", userdata=ON_CHANNEL),
                    menu_item("By number", action="topoplot:numbers", userdata=ON_CHANNEL),
                ],
            ),
            menu_item("Channel data (scroll)", action="pop_eegplot:channels", userdata=ON_DATA, separator=True),
            menu_item("Channel spectra and maps", action="pop_spectopo:channels", userdata=ON_DATA),
            menu_item("Channel properties", action="pop_prop:channels", userdata=ON_DATA),
            menu_item("Channel ERP image", action="pop_erpimage:channels", userdata=ON_EPOCH),
            menu_item(
                "Channel ERPs",
                userdata=ON_EPOCH,
                children=[
                    menu_item("With scalp maps", action="pop_timtopo"),
                    menu_item("In scalp/rect. array", action="pop_plottopo"),
                ],
            ),
            menu_item(
                "ERP map series",
                userdata=ON_EPOCH_CHANNEL,
                children=[
                    menu_item("In 2-D", action="pop_topoplot:erp"),
                    menu_item("In 3-D", action="pop_headplot:erp"),
                ],
            ),
            menu_item("Sum/Compare ERPs", action="pop_comperp:channels", userdata=ON_EPOCH, visibility="allmenus"),
            menu_item("Channel time-frequency", action="pop_newtimef:channels", visibility="default"),
            menu_item("Component activations (scroll)", action="pop_eegplot:components", userdata=ON_DATA, separator=True),
            menu_item("Component spectra and maps", action="pop_spectopo:components", userdata=ON_DATA_NO_ROI),
            menu_item(
                "Component maps",
                userdata=ON_CHANNEL_NO_ROI,
                children=[
                    menu_item("In 2-D", action="pop_topoplot:components"),
                    menu_item("In 3-D", action="pop_headplot:components"),
                ],
            ),
            menu_item("Component properties", action="pop_prop:components", userdata=ON_DATA),
            menu_item("Component ERP image", action="pop_erpimage:components", userdata=ON_EPOCH),
            menu_item(
                "Component ERPs",
                userdata=ON_EPOCH,
                children=[
                    menu_item("With component maps", action="pop_envtopo", userdata=ON_EPOCH_NO_ROI),
                    menu_item("With comp. maps (compare)", action="pop_envtopo:compare", userdata=ON_EPOCH_NO_ROI),
                    menu_item("In rectangular array", action="pop_plotdata:components", userdata=ON_EPOCH),
                ],
            ),
            menu_item("Sum/Compare comp. ERPs", action="pop_comperp:components", userdata=ON_EPOCH_NO_ROI, visibility="allmenus"),
            menu_item(
                "Data statistics",
                userdata=ON_DATA,
                separator=True,
                visibility="allmenus",
                children=[
                    menu_item("Channel statistics", action="pop_signalstat:channels"),
                    menu_item("Component statistics", action="pop_signalstat:components"),
                    menu_item("Event statistics", action="pop_eventstat"),
                ],
            ),
            menu_item(
                "Time-frequency transforms",
                userdata=ON_DATA,
                separator=True,
                visibility="allmenus",
                children=[
                    menu_item("Channel time-frequency", action="pop_newtimef:channels"),
                    menu_item("Channel cross-coherence", action="pop_newcrossf:channels"),
                    menu_item("Component time-frequency", action="pop_newtimef:components", separator=True),
                    menu_item("Component cross-coherence", action="pop_newcrossf:components"),
                ],
            ),
            menu_item("Component time-frequency", action="pop_newtimef:components", visibility="default"),
        ],
    )

    study_menu = menu_item(
        "Study",
        tag="study",
        userdata=ON_STUDY,
        children=[
            menu_item("Edit study info", action="pop_study:edit", userdata=ON_STUDY),
            menu_item("Select/Edit study design(s)", action="pop_studydesign", userdata=ON_STUDY),
            menu_item("Precompute channel measures", action="pop_precomp:channels", userdata=ON_STUDY, separator=True),
            menu_item("Plot channel measures", action="pop_chanplot", userdata=ON_STUDY),
            menu_item("Precompute component measures", action="pop_precomp:components", userdata=ON_STUDY, separator=True),
            menu_item(
                "PCA clustering (original)",
                userdata=ON_STUDY_NO_ROI,
                children=[
                    menu_item("Build preclustering array", action="pop_preclust", userdata=ON_STUDY_NO_ROI),
                    menu_item("Cluster components", action="pop_clust", userdata=ON_STUDY_NO_ROI),
                ],
            ),
            menu_item("Edit/plot component clusters", action="pop_clustedit", userdata=ON_STUDY),
        ],
    )

    datasets_menu = menu_item(
        "Datasets",
        userdata=ON_DATA_STUDY,
        children=[menu_item("Select multiple datasets", action="select_multiple_datasets", separator=True)],
    )
    help_menu = menu_item(
        "Help",
        userdata=ON,
        children=[
            menu_item("About EEGPrep", action="help:eegprep", userdata=ON),
            menu_item("Check for EEGPrep updates", action="updates", userdata=ON),
            menu_item("About EEGPrep help", action="help:eeg_helphelp", userdata=ON),
            menu_item("EEGPrep menus", action="help:eeg_helpmenu", userdata=ON, separator=True),
            menu_item(
                "EEGPrep functions",
                userdata=ON,
                children=[
                    menu_item("Admin. functions", action="help:eeg_helpadmin", userdata=ON),
                    menu_item("Interactive pop_ functions", action="help:eeg_helppop", userdata=ON),
                    menu_item("Signal processing functions", action="help:eeg_helpsigproc", userdata=ON),
                    menu_item("Group data (STUDY) functions", action="help:eeg_helpstudy", userdata=ON),
                    menu_item("Time-frequency functions", action="help:eeg_helptimefreq", userdata=ON),
                    menu_item("Statistical functions", action="help:eeg_helpstatistics", userdata=ON),
                    menu_item("Graphic interface builder functions", action="help:eeg_helpgui", userdata=ON),
                    menu_item("Misc. command line functions", action="help:eeg_helpmisc", userdata=ON),
                ],
            ),
            menu_item("EEGPrep license", action="license", userdata=ON),
            menu_item("EEGPrep tutorial", action="tutorial", userdata=ON, separator=True),
            # TODO: Replace this EEGLAB support alias with an EEGPrep-owned
            # contact channel once one is published.
            menu_item("Email the EEGPrep team", action="mailto:eeglab@sccn.ucsd.edu", userdata=ON),
            menu_item("Report an EEGPrep issue", action="issues", userdata=ON),
        ],
    )
    return (file_menu, edit_menu, tools_menu, plot_menu, study_menu, datasets_menu, help_menu)


def eeglab_plugin_menus() -> tuple[MenuItemSpec, ...]:
    """Return plugin menu additions represented in the sibling EEGLAB checkout."""
    return (
        clean_rawdata_menu(),
        iclabel_menu(),
        dipfit_menu(),
    )


def eeglab_menus(*, all_menus: bool = False, include_plugins: bool = True) -> tuple[MenuItemSpec, ...]:
    """Return the complete visible menu tree for the configured menu mode."""
    menus = list(eeglab_core_menus())
    if include_plugins:
        menus[0] = _insert_file_plugins(menus[0])
        menus[2] = _insert_tools_plugins(menus[2])
        menus[2] = _insert_firfilt(menus[2])
        menus[3] = _insert_plot_plugins(menus[3])
    return visible_menu_items(tuple(menus), all_menus=all_menus)


def menu_actions(items: tuple[MenuItemSpec, ...]) -> set[str]:
    """Return all action identifiers in a menu tree."""
    actions: set[str] = set()
    for item in items:
        if item.action:
            actions.add(item.action)
        actions.update(menu_actions(item.children))
    return actions


def _insert_tools_plugins(tools_menu: MenuItemSpec) -> MenuItemSpec:
    children = []
    clean_rawdata, iclabel, dipfit = eeglab_plugin_menus()
    for item in tools_menu.children:
        children.append(item)
        if item.action == "pop_eegplot:data":
            children.append(clean_rawdata)
        if item.action == "pop_selectcomps":
            children.append(iclabel)
        if item.action == "pop_rmbase":
            children.append(dipfit)
    return tools_menu.with_children(tuple(children))


def _insert_firfilt(tools_menu: MenuItemSpec) -> MenuItemSpec:
    children = []
    for item in tools_menu.children:
        if item.tag != "filter":
            children.append(item)
            continue
        filter_children = (
            *firfilt_filter_items(),
            *item.children,
        )
        children.append(item.with_children(filter_children))
    return tools_menu.with_children(tuple(children))


def _insert_plot_plugins(plot_menu: MenuItemSpec) -> MenuItemSpec:
    return plot_menu.with_children((*plot_menu.children, *viewprops_plot_menus()))


def _insert_file_plugins(file_menu: MenuItemSpec) -> MenuItemSpec:
    children: list[MenuItemSpec] = []
    for item in file_menu.children:
        if item.label == "Import data":
            children.append(_insert_import_plugins(item))
            continue
        if item.tag == "export":
            children.append(item.with_children((*item.children, *eeg_bids_export_items())))
            children.append(eeg_bids_tools_menu())
            continue
        children.append(item)
    return file_menu.with_children(tuple(children))


def _insert_import_plugins(import_menu: MenuItemSpec) -> MenuItemSpec:
    children: list[MenuItemSpec] = []
    for item in import_menu.children:
        if item.tag == "import data":
            children.append(item.with_children((*item.children, *eeg_bids_import_items())))
            continue
        children.append(item)
    return import_menu.with_children(tuple(children))
