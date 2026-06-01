"""Action dispatch for the EEGPrep EEGLAB-style main window."""

from __future__ import annotations

import logging
import webbrowser
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from eegprep.functions.guifunc.menu_placeholders import PLACEHOLDER_ACTIONS, is_placeholder_action, placeholder_message
from eegprep.functions.guifunc.pophelp import pophelp
from eegprep.functions.guifunc.session import EEGPrepSession, has_eeg_data
from eegprep.functions.popfunc._coming_soon import coming_soon


logger = logging.getLogger(__name__)

# Action handlers import user-facing pop/plugin modules lazily so launching the
# main GUI does not eagerly load heavier signal-processing and optional stacks.
IMPLEMENTED_ACTIONS = {
    "clear_study",
    "docs",
    "help",
    "issues",
    "license",
    "mailto",
    "eeg_rejsuperpose",
    "pop_adjustevents",
    "pop_autorej",
    "pop_biosig",
    "pop_chanevent",
    "pop_chanplot",
    "pop_clust",
    "pop_clustedit",
    "pop_clean_rawdata",
    "pop_chanedit",
    "pop_comperp",
    "pop_delset",
    "pop_dipfit_gridsearch",
    "pop_dipfit_headmodel",
    "pop_dipfit_loreta",
    "pop_dipfit_nonlinear",
    "pop_dipfit_settings",
    "pop_dipplot",
    "pop_comments",
    "pop_copyset",
    "pop_editset",
    "pop_editeventfield",
    "pop_editeventvals",
    "pop_editoptions",
    "pop_eegplot",
    "pop_eegfilt",
    "pop_eegfiltnew",
    "pop_envtopo",
    "pop_epoch",
    "pop_erpimage",
    "pop_eventstat",
    "pop_eventinfo",
    "pop_expevents",
    "pop_expica",
    "pop_export",
    "pop_exportbids",
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
    "pop_iclabel",
    "pop_jointprob",
    "pop_leadfield",
    "pop_importbids",
    "pop_importdata",
    "pop_importepoch",
    "pop_importerplab",
    "pop_importevent",
    "pop_importpres",
    "pop_interp",
    "pop_loadstudy",
    "pop_loadset",
    "pop_plotdata",
    "pop_plottopo",
    "pop_preclust",
    "pop_precomp",
    "pop_prop",
    "pop_runscript",
    "pop_saveh",
    "pop_savestudy",
    "pop_eegthresh",
    "pop_rejchan",
    "pop_rejcont",
    "pop_rejepoch",
    "pop_rejkurt",
    "pop_rejmenu",
    "pop_rejspec",
    "pop_rejtrend",
    "pop_reref",
    "pop_resample",
    "pop_rmbase",
    "pop_rmdat",
    "pop_runica",
    "pop_saveset",
    "pop_select",
    "pop_selectevent",
    "pop_selectcomps",
    "pop_spectopo",
    "pop_study",
    "pop_studydesign",
    "pop_studyerp",
    "pop_studywizard",
    "pop_subcomp",
    "pop_timtopo",
    "pop_taskinfo",
    "pop_topoplot",
    "pop_viewprops",
    "pop_participantinfo",
    "pop_mergeset",
    "pop_multifit",
    "pop_newcrossf",
    "pop_newtimef",
    "pop_writeeeg",
    "pop_signalstat",
    "bids_exporter",
    "plugin_menu",
    "quit",
    "retrieve_dataset",
    "select_multiple_datasets",
    "select_study_set",
    "topoplot",
    "tutorial",
    "updates",
    "validate_bids",
}

EEGPREP_REPO_URL = "https://github.com/sccn/eegprep"
EEGPREP_DOCS_URL = "https://sccn.github.io/eegprep/"
EEGPREP_SOURCE_URL = f"{EEGPREP_REPO_URL}/blob/develop"

_MULTIPLE_DATASET_ACTIONS = {
    "pop_clean_rawdata",
    "pop_chanedit",
    "pop_eegfilt",
    "pop_eegfiltnew",
    "pop_epoch",
    "pop_firma",
    "pop_firpm",
    "pop_firws",
    "pop_icflag",
    "pop_iclabel",
    "pop_reref",
    "pop_rmdat",
    "pop_resample",
    "pop_rmbase",
    "pop_runica",
    "pop_select",
    "pop_selectevent",
    "pop_eegthresh",
    "pop_jointprob",
    "pop_rejkurt",
    "pop_rejspec",
    "pop_rejtrend",
    "pop_subcomp",
    "pop_dipfit_settings",
}

_DIPFIT_ACTIONS = {
    "pop_dipfit_gridsearch",
    "pop_dipfit_headmodel",
    "pop_dipfit_loreta",
    "pop_dipfit_nonlinear",
    "pop_dipfit_settings",
    "pop_dipplot",
    "pop_leadfield",
    "pop_multifit",
}

_BROWSER_ACCEPT_POP_ACTIONS = {
    "pop_autorej",
    "pop_eegthresh",
    "pop_jointprob",
    "pop_rejcont",
    "pop_rejkurt",
    "pop_rejspec",
    "pop_rejtrend",
}


class MenuActionDispatcher:
    """Dispatch menu action identifiers to real functions or placeholders."""

    def __init__(
        self,
        session: EEGPrepSession,
        refresh: Callable[[], None] | None = None,
        *,
        native_file_dialogs: bool = True,
    ):
        self.session = session
        self.refresh = refresh
        self.native_file_dialogs = native_file_dialogs

    def dispatch_gui(self, action: str, parent: Any | None = None) -> None:
        """Run a menu action from Qt and show user-facing errors."""
        with self.session.gui_action(action):
            try:
                self.dispatch(action, parent)
            except Exception as exc:
                logger.exception("EEGPrep GUI menu action failed: %s", action)
                if parent is None:
                    raise
                self._warn(parent, str(exc))

    def dispatch(self, action: str, parent: Any | None = None) -> None:
        """Run a menu action."""
        base, _sep, variant = action.partition(":")
        if action_kind(action) == "placeholder":
            self.show_coming_soon(action, parent)
            return
        if base == "quit":
            if parent is not None:
                parent.close()
            return
        if base == "clear_study":
            self.session.clear_all()
            self._refresh()
            return
        if base == "plugin_menu":
            self._show_extension_manager(parent)
            return
        if base == "help":
            self._show_help(variant or "eegprep", parent)
            return
        if base == "docs":
            webbrowser.open(_docs_url(variant))
            return
        if base == "issues":
            webbrowser.open(f"{EEGPREP_REPO_URL}/issues")
            return
        if base == "license":
            webbrowser.open(f"{EEGPREP_SOURCE_URL}/LICENSE")
            return
        if base == "mailto":
            webbrowser.open(f"mailto:{variant}")
            return
        if base == "tutorial":
            webbrowser.open(_tutorial_url())
            return
        if base == "updates":
            webbrowser.open(f"{EEGPREP_REPO_URL}/releases")
            return
        if base == "pop_loadset":
            self._loadset(parent)
            return
        if base in {
            "pop_biosig",
            "pop_fileio",
            "pop_fileio_brainvision",
            "pop_fileio_brainvision_mat",
            "pop_fileio_cnt",
            "pop_fileio_eeg",
            "pop_fileio_mff",
            "pop_importbids",
            "pop_importdata",
        }:
            self._import_dataset(base, parent)
            return
        if base in {"pop_importepoch", "pop_importevent", "pop_chanevent", "pop_importpres", "pop_importerplab"}:
            self._import_current_dataset_metadata(base, parent)
            return
        if base in {"pop_export", "pop_expica", "pop_expevents", "pop_writeeeg", "pop_exportbids", "bids_exporter"}:
            self._export_current_dataset(base, variant, parent)
            return
        if base == "pop_saveset":
            self._saveset(parent, resave=variant == "resave")
            return
        if base == "pop_delset":
            self.session.delete_current()
            self._refresh()
            return
        if base == "retrieve_dataset":
            self._retrieve_dataset(int(variant))
            return
        if base == "select_multiple_datasets":
            self._select_multiple_datasets(parent)
            return
        if base == "select_study_set":
            self._select_study_set(parent)
            return
        if base == "pop_copyset":
            self._copy_current_dataset(parent)
            return
        if base == "pop_mergeset":
            self._merge_datasets(parent)
            return
        if base in {
            "pop_study",
            "pop_studydesign",
            "pop_studywizard",
            "pop_studyerp",
            "pop_loadstudy",
            "pop_savestudy",
            "pop_preclust",
            "pop_clust",
            "pop_clustedit",
            "pop_precomp",
        }:
            self._study_action(base, variant, parent)
            return
        if base == "pop_editoptions":
            self._edit_options(parent)
            return
        if base == "pop_saveh":
            self._save_history(variant, parent)
            return
        if base == "pop_runscript":
            self._run_script(parent)
            return
        if base in {"pop_taskinfo", "pop_participantinfo", "pop_eventinfo", "validate_bids"}:
            self._bids_tool_action(base, parent)
            return
        if base == "pop_adjustevents":
            self._run_pop_function("pop_adjustevents", parent)
            return
        if base == "pop_comments":
            self._run_pop_function("pop_comments", parent)
            return
        if base == "pop_editset":
            self._run_pop_function("pop_editset", parent)
            return
        if base == "pop_editeventfield":
            self._run_pop_function("pop_editeventfield", parent)
            return
        if base == "pop_editeventvals":
            self._run_pop_function("pop_editeventvals", parent)
            return
        if base == "pop_chanedit":
            self._run_pop_function("pop_chanedit", parent)
            return
        if base == "pop_clean_rawdata":
            self._run_pop_function("pop_clean_rawdata", parent)
            return
        if base == "pop_eegfilt":
            self._run_pop_function("pop_eegfilt", parent)
            return
        if base == "pop_eegfiltnew":
            self._run_pop_function("pop_eegfiltnew", parent)
            return
        if base == "pop_epoch":
            self._run_pop_function("pop_epoch", parent)
            return
        if base == "pop_firma":
            self._run_pop_function("pop_firma", parent)
            return
        if base == "pop_firpm":
            self._run_pop_function("pop_firpm", parent)
            return
        if base == "pop_firws":
            self._run_pop_function("pop_firws", parent)
            return
        if base == "pop_reref":
            self._run_pop_function("pop_reref", parent)
            return
        if base == "pop_interp":
            self._run_pop_function("pop_interp", parent)
            return
        if base == "pop_resample":
            self._run_pop_function("pop_resample", parent)
            return
        if base == "pop_rmbase":
            self._run_pop_function("pop_rmbase", parent)
            return
        if base == "pop_rmdat":
            self._run_pop_function("pop_rmdat", parent)
            return
        if base == "pop_runica":
            self._run_pop_function("pop_runica", parent)
            return
        if base == "pop_select":
            self._run_pop_function("pop_select", parent)
            return
        if base == "pop_selectevent":
            self._run_pop_function("pop_selectevent", parent)
            return
        if base == "pop_iclabel":
            self._run_pop_function("pop_iclabel", parent)
            return
        if base == "pop_icflag":
            self._run_pop_function("pop_icflag", parent)
            return
        if base == "pop_subcomp":
            self._run_pop_function("pop_subcomp", parent)
            return
        if base == "pop_eegplot":
            self._run_pop_function("pop_eegplot", parent, variant=variant)
            return
        if base in {
            "pop_autorej",
            "pop_eegthresh",
            "pop_jointprob",
            "pop_rejchan",
            "pop_rejcont",
            "pop_rejepoch",
            "pop_rejkurt",
            "pop_rejmenu",
            "pop_rejspec",
            "pop_rejtrend",
            "pop_selectcomps",
            "pop_viewprops",
        }:
            self._run_pop_function(base, parent, variant=variant)
            return
        if base == "eeg_rejsuperpose":
            self._run_rejsuperpose(variant, parent)
            return
        if base in _DIPFIT_ACTIONS:
            self._run_dipfit_function(base, parent)
            return
        if base == "topoplot":
            self._plot_channel_locations(variant, parent)
            return
        if base == "pop_topoplot":
            self._run_topoplot(variant, parent)
            return
        if base in {
            "pop_spectopo",
            "pop_prop",
            "pop_timtopo",
            "pop_plottopo",
            "pop_headplot",
            "pop_plotdata",
            "pop_erpimage",
            "pop_envtopo",
            "pop_comperp",
            "pop_newtimef",
            "pop_newcrossf",
            "pop_signalstat",
            "pop_eventstat",
        }:
            self._run_plot_function(base, variant, parent)
            return
        if base == "pop_chanplot":
            self._run_chanplot(parent)
            return
        self.show_coming_soon(action, parent)

    def show_coming_soon(self, action: str, parent: Any | None = None) -> None:
        """Show the shared placeholder dialog."""
        message = placeholder_message(action)
        qt_widgets = _qt_widgets()
        if qt_widgets is None:
            coming_soon(action)
            return
        qt_widgets.QMessageBox.information(parent, "EEGPrep", message)

    def _loadset(self, parent: Any | None) -> None:
        from eegprep.functions.popfunc.pop_loadset import pop_loadset

        qt_widgets = _require_qt_widgets()
        filename, _filter = qt_widgets.QFileDialog.getOpenFileName(
            parent,
            "Load existing dataset",
            "",
            "EEGPrep/EEGLAB datasets (*.set *.mat);;All files (*)",
            **self._file_dialog_kwargs(qt_widgets),
        )
        if not filename:
            return
        command = f"EEG = pop_loadset({filename!r});"
        eeg = pop_loadset(filename)
        self._store_current_from_gui(eeg, new=True, command=command)
        self._refresh()

    def _saveset(self, parent: Any | None, *, resave: bool = False) -> None:
        selection = self._current_selection_or_warn(parent, allow_multiple=resave)
        if selection is None:
            return
        datasets = selection if isinstance(selection, list) else [selection]
        filenames = [_existing_dataset_filename(eeg) if resave else "" for eeg in datasets]
        if resave and len(datasets) > 1 and not all(filenames):
            self._warn(parent, "Cannot resave multiple datasets until every selected dataset has a filename.")
            return
        filename = filenames[0] if len(datasets) == 1 else ""
        if len(datasets) == 1 and not filename:
            qt_widgets = _require_qt_widgets()
            filename, _filter = qt_widgets.QFileDialog.getSaveFileName(
                parent,
                "Save current dataset as",
                str(datasets[0].get("filename") or ""),
                "EEGPrep/EEGLAB datasets (*.set);;All files (*)",
                **self._file_dialog_kwargs(qt_widgets),
            )
            filenames = [filename]
        if len(datasets) == 1 and not filename:
            return
        from eegprep.functions.popfunc.pop_saveset import pop_saveset

        for eeg, filename in zip(datasets, filenames):
            pop_saveset(eeg, filename)
            _apply_save_metadata(eeg, filename)
        stored = datasets if isinstance(selection, list) else datasets[0]
        command = (
            "EEG = pop_saveset(EEG, 'savemode', 'resave');" if resave else f"EEG = pop_saveset(EEG, {filenames[0]!r});"
        )
        self._store_current_from_gui(stored, command=command, mark_saved=True)
        self._refresh()

    def _import_dataset(self, action: str, parent: Any | None) -> None:
        qt_widgets = _require_qt_widgets()
        if action == "pop_importbids":
            source = qt_widgets.QFileDialog.getExistingDirectory(
                parent,
                "Import BIDS folder structure",
                "",
                **self._file_dialog_kwargs(qt_widgets, directories=True),
            )
            if not source:
                return
            from eegprep.plugins.EEG_BIDS.pop_importbids import pop_importbids

            eeg_out, command = pop_importbids(source, return_com=True)
        else:
            filename = self._open_import_filename(action, parent)
            if not filename:
                return
            if action == "pop_importdata":
                srate = self._ask_float(parent, "Import data", "Data sampling rate (Hz)", 1.0)
                if srate is None:
                    return
                from eegprep.functions.popfunc.pop_importdata import pop_importdata

                eeg_out, command = pop_importdata("data", filename, "srate", srate, return_com=True)
            elif action == "pop_biosig":
                from eegprep.functions.popfunc.pop_biosig import pop_biosig

                eeg_out, command = pop_biosig(filename, return_com=True)
            elif action == "pop_fileio_brainvision_mat":
                from eegprep.functions.popfunc.pop_fileio_brainvision_mat import pop_fileio_brainvision_mat

                eeg_out, command = pop_fileio_brainvision_mat(filename, return_com=True)
            else:
                from eegprep.functions.popfunc.pop_fileio import pop_fileio

                eeg_out, command = pop_fileio(filename, return_com=True)
        self._store_current_from_gui(eeg_out, new=True, command=command)
        self._refresh()

    def _open_import_filename(self, action: str, parent: Any | None) -> str:
        qt_widgets = _require_qt_widgets()
        filters = {
            "pop_importdata": "Data arrays (*.txt *.csv *.tsv *.mat *.npy *.npz *.fdt);;All files (*)",
            "pop_biosig": "BIOSIG files (*.bdf *.edf *.gdf);;All files (*)",
            "pop_fileio_mff": "EGI MFF (*.mff);;All files (*)",
            "pop_fileio_cnt": "Neuroscan CNT (*.cnt);;All files (*)",
            "pop_fileio_eeg": "Neuroscan/BrainVision EEG (*.eeg);;All files (*)",
            "pop_fileio_brainvision": "BrainVision header (*.vhdr);;All files (*)",
            "pop_fileio_brainvision_mat": "BrainVision Analyzer MATLAB (*.mat);;All files (*)",
        }
        filename, _filter = qt_widgets.QFileDialog.getOpenFileName(
            parent,
            "Import data",
            "",
            filters.get(action, "EEG files (*.set *.edf *.bdf *.gdf *.vhdr *.mff *.cnt *.eeg);;All files (*)"),
            **self._file_dialog_kwargs(qt_widgets),
        )
        return filename

    def _import_current_dataset_metadata(self, action: str, parent: Any | None) -> None:
        selection = self._current_selection_or_warn(parent)
        if selection is None:
            return
        qt_widgets = _require_qt_widgets()
        if action == "pop_chanevent":
            channel, ok = qt_widgets.QInputDialog.getInt(parent, "Import event info", "Event channel index", 1, 1)
            if not ok:
                return
            from eegprep.functions.popfunc.pop_chanevent import pop_chanevent

            eeg_out, command = pop_chanevent(selection, channel, return_com=True)
        else:
            filename, _filter = qt_widgets.QFileDialog.getOpenFileName(
                parent,
                "Import event/epoch info",
                "",
                "Text tables (*.txt *.tsv *.csv *.log);;All files (*)",
                **self._file_dialog_kwargs(qt_widgets),
            )
            if not filename:
                return
            if action == "pop_importepoch":
                from eegprep.functions.popfunc.pop_importepoch import pop_importepoch

                eeg_out, command = pop_importepoch(selection, filename, return_com=True)
            elif action == "pop_importpres":
                from eegprep.functions.popfunc.pop_importpres import pop_importpres

                eeg_out, command = pop_importpres(selection, filename, return_com=True)
            elif action == "pop_importerplab":
                from eegprep.functions.popfunc.pop_importerplab import pop_importerplab

                eeg_out, command = pop_importerplab(selection, filename, return_com=True)
            else:
                from eegprep.functions.popfunc.pop_importevent import pop_importevent

                eeg_out, command = pop_importevent(selection, "event", filename, return_com=True)
        self._store_current_from_gui(eeg_out, command=command)
        self._refresh()

    def _export_current_dataset(self, action: str, variant: str, parent: Any | None) -> None:
        allow_multiple = action in {"pop_exportbids", "bids_exporter"}
        selection = self._current_selection_or_warn(parent, allow_multiple=allow_multiple)
        if selection is None:
            return
        qt_widgets = _require_qt_widgets()
        if action in {"pop_exportbids", "bids_exporter"}:
            directory = qt_widgets.QFileDialog.getExistingDirectory(
                parent,
                "Export to BIDS folder structure",
                "",
                **self._file_dialog_kwargs(qt_widgets, directories=True),
            )
            if not directory:
                return
            from eegprep.plugins.EEG_BIDS.pop_exportbids import pop_exportbids

            _output, command = pop_exportbids(selection, directory, return_com=True)
        else:
            filename, _filter = qt_widgets.QFileDialog.getSaveFileName(
                parent,
                "Export data",
                "",
                _export_filter(action),
                **self._file_dialog_kwargs(qt_widgets),
            )
            if not filename:
                return
            if action == "pop_export":
                from eegprep.functions.popfunc.pop_export import pop_export

                command = pop_export(selection, filename, "transpose", "on")
            elif action == "pop_expica":
                from eegprep.functions.popfunc.pop_expica import pop_expica

                command = pop_expica(selection, filename, variant or "weights")
            elif action == "pop_expevents":
                from eegprep.functions.popfunc.pop_expevents import pop_expevents

                command = pop_expevents(selection, filename)
            else:
                from eegprep.functions.popfunc.pop_writeeeg import pop_writeeeg

                command = pop_writeeeg(selection, filename)
        self._add_history_from_gui(command)
        self._refresh()

    def _study_action(self, action: str, variant: str, parent: Any | None) -> None:
        if action == "pop_loadstudy":
            qt_widgets = _require_qt_widgets()
            filename, _filter = qt_widgets.QFileDialog.getOpenFileName(
                parent,
                "Load existing study",
                "",
                "STUDY files (*.study *.json);;All files (*)",
                **self._file_dialog_kwargs(qt_widgets),
            )
            if not filename:
                return
            from eegprep.functions.studyfunc.pop_loadstudy import pop_loadstudy

            study, alleeg, command = pop_loadstudy(filename, return_com=True)
            self.session.echo_command(command)
            self.session.set_study(study, alleeg, command=command)
            self._refresh()
            return
        if action == "pop_savestudy":
            qt_widgets = _require_qt_widgets()
            if not self.session.STUDY:
                self._warn(parent, "No current study")
                return
            filename = _existing_study_filename(self.session.STUDY) if variant == "resave" else ""
            if not filename:
                filename, _filter = qt_widgets.QFileDialog.getSaveFileName(
                    parent,
                    "Save current study as",
                    "",
                    "STUDY files (*.study);;All files (*)",
                    **self._file_dialog_kwargs(qt_widgets),
                )
            if not filename:
                return
            from eegprep.functions.studyfunc.pop_savestudy import pop_savestudy

            study, command = pop_savestudy(
                self.session.STUDY, self.session.ALLEEG, filename, savemode=variant or None, return_com=True
            )
            self.session.echo_command(command)
            self.session.set_study(study, command=command)
            self._refresh()
            return
        if action == "pop_studydesign":
            if not self.session.STUDY:
                self._warn(parent, "Create or load a STUDY before editing designs")
                return
            from eegprep.functions.studyfunc.pop_studydesign import pop_studydesign

            study, alleeg, command = pop_studydesign(self.session.STUDY, self.session.ALLEEG, gui=True, return_com=True)
            if not command:
                return
            self.session.echo_command(command)
            self.session.set_study(study, alleeg, command=command)
            self._refresh()
            return
        if action == "pop_preclust":
            if not self.session.STUDY:
                self._warn(parent, "Create or load a STUDY before preclustering components")
                return
            from eegprep.functions.studyfunc.pop_preclust import pop_preclust

            study, alleeg, command = pop_preclust(self.session.STUDY, self.session.ALLEEG, gui=True, return_com=True)
            if not command:
                return
            self.session.echo_command(command)
            self.session.set_study(study, alleeg, command=command)
            self._refresh()
            return
        if action == "pop_precomp":
            if not self.session.STUDY:
                self._warn(parent, "Create or load a STUDY before precomputing measures")
                return
            from eegprep.functions.studyfunc.pop_precomp import pop_precomp

            target = "components" if variant == "components" else "channels"
            study, alleeg, command = pop_precomp(
                self.session.STUDY, self.session.ALLEEG, target, gui=True, return_com=True
            )
            if not command:
                return
            self.session.echo_command(command)
            self.session.set_study(study, alleeg, command=command)
            self._refresh()
            return
        if action == "pop_clust":
            if not self.session.STUDY:
                self._warn(parent, "Create or load a STUDY before clustering components")
                return
            from eegprep.functions.studyfunc.pop_clust import pop_clust

            study, command = pop_clust(self.session.STUDY, self.session.ALLEEG, gui=True, return_com=True)
            if not command:
                return
            self.session.echo_command(command)
            self.session.set_study(study, command=command)
            self._refresh()
            return
        if action == "pop_clustedit":
            if not self.session.STUDY:
                self._warn(parent, "Create or load a STUDY before editing clusters")
                return
            from eegprep.functions.studyfunc.pop_clustedit import pop_clustedit

            study, command, _figure = pop_clustedit(self.session.STUDY, self.session.ALLEEG, gui=True, return_com=True)
            if not command:
                return
            self.session.echo_command(command)
            self.session.set_study(study, command=command)
            self._refresh()
            return
        if action == "pop_studywizard":
            qt_widgets = _require_qt_widgets()
            filenames, _filter = qt_widgets.QFileDialog.getOpenFileNames(
                parent,
                "Browse for datasets",
                "",
                "EEGPrep/EEGLAB datasets (*.set *.mat);;All files (*)",
                **self._file_dialog_kwargs(qt_widgets),
            )
            if not filenames:
                return
            from eegprep.functions.studyfunc.pop_studywizard import pop_studywizard

            study, alleeg, command = pop_studywizard(filenames, return_com=True)
        elif action == "pop_studyerp":
            from eegprep.functions.studyfunc.pop_studyerp import pop_studyerp

            study, alleeg, command = pop_studyerp(self.session.ALLEEG, return_com=True)
        elif action == "pop_study" and variant == "edit":
            if not self.session.STUDY:
                self._warn(parent, "Create or load a STUDY before editing study info")
                return
            from eegprep.functions.studyfunc.pop_study import pop_study

            study, alleeg, command = pop_study(self.session.STUDY, self.session.ALLEEG, gui=True, return_com=True)
            if not command:
                return
        else:
            if not self.session.ALLEEG:
                self._warn(parent, "Load at least one dataset before creating a study")
                return
            from eegprep.functions.studyfunc.pop_study import pop_study

            study, alleeg, command = pop_study(None, self.session.ALLEEG, gui=True, return_com=True)
            if not command:
                return
        self.session.echo_command(command)
        self.session.set_study(study, alleeg, command=command)
        self._refresh()

    def _edit_options(self, parent: Any | None) -> None:
        from eegprep.functions.adminfunc.eeg_options import EEG_OPTIONS
        from eegprep.functions.adminfunc.pop_editoptions import pop_editoptions

        enabled = int(not bool(EEG_OPTIONS.get("option_allmenus", 0)))
        if parent is not None:
            qt_widgets = _require_qt_widgets()
            result = qt_widgets.QMessageBox.question(
                parent,
                "EEGPrep preferences",
                "Show advanced legacy menu items?",
                qt_widgets.QMessageBox.Yes | qt_widgets.QMessageBox.No | qt_widgets.QMessageBox.Cancel,
                qt_widgets.QMessageBox.Yes if enabled else qt_widgets.QMessageBox.No,
            )
            if result == qt_widgets.QMessageBox.Cancel:
                return
            enabled = int(result == qt_widgets.QMessageBox.Yes)
        command = pop_editoptions(option_allmenus=enabled)
        self._add_history_from_gui(command)
        self._info(parent, "Preferences updated. Reopen the main window to rebuild the menu mode.")
        self._refresh()

    def _save_history(self, variant: str, parent: Any | None) -> None:
        qt_widgets = _require_qt_widgets()
        filename, _filter = qt_widgets.QFileDialog.getSaveFileName(
            parent,
            "Save history script",
            "eegprephist.m",
            "MATLAB scripts (*.m);;All files (*)",
            **self._file_dialog_kwargs(qt_widgets),
        )
        if not filename:
            return
        from eegprep.functions.popfunc.pop_saveh import pop_saveh

        path = Path(filename)
        history = (
            self.session.EEG.get("history", "")
            if variant == "dataset" and isinstance(self.session.EEG, dict)
            else self.session.ALLCOM
        )
        command = pop_saveh(history, path.name, path.parent)
        self._add_history_from_gui(command)
        self._refresh()

    def _run_script(self, parent: Any | None) -> None:
        qt_widgets = _require_qt_widgets()
        filename, _filter = qt_widgets.QFileDialog.getOpenFileName(
            parent,
            "Run history script",
            "",
            "Scripts (*.py *.m *.txt);;All files (*)",
            **self._file_dialog_kwargs(qt_widgets),
        )
        if not filename:
            return
        from eegprep.functions.popfunc.pop_runscript import pop_runscript

        namespace = {
            "EEG": self.session.EEG,
            "ALLEEG": self.session.ALLEEG,
            "CURRENTSET": self.session.current_set_value(),
            "STUDY": self.session.STUDY,
        }
        command = pop_runscript(filename, namespace)
        self.session.EEG = namespace.get("EEG", self.session.EEG)
        self.session.ALLEEG = namespace.get("ALLEEG", self.session.ALLEEG)
        self.session.CURRENTSET = _currentset_list(namespace.get("CURRENTSET", self.session.current_set_value()))
        self.session.STUDY = namespace.get("STUDY", self.session.STUDY)
        self._add_history_from_gui(command)
        self._refresh()

    def _bids_tool_action(self, action: str, parent: Any | None) -> None:
        if action == "validate_bids":
            qt_widgets = _require_qt_widgets()
            directory = qt_widgets.QFileDialog.getExistingDirectory(
                parent,
                "Validate BIDS dataset",
                "",
                **self._file_dialog_kwargs(qt_widgets, directories=True),
            )
            if not directory:
                return
            from eegprep.plugins.EEG_BIDS.bids_tools import validate_bids

            report = validate_bids(directory)
            self._info(
                parent, f"BIDS validation complete: {len(report['errors'])} errors, {len(report['warnings'])} warnings."
            )
            return
        target = self.session.STUDY if self.session.CURRENTSTUDY == 1 and self.session.STUDY else self.session.EEG
        if isinstance(target, list):
            self._warn(parent, "Select one dataset or a study before editing BIDS metadata")
            return
        metadata = self._ask_metadata(parent, action)
        if metadata is None:
            return
        from eegprep.plugins.EEG_BIDS import bids_tools

        updated, command = getattr(bids_tools, action)(target, **metadata)
        if self.session.CURRENTSTUDY == 1 and self.session.STUDY:
            self.session.STUDY = updated
            self._add_history_from_gui(command)
        else:
            self._store_current_from_gui(updated, command=command)
        self._refresh()

    def _show_extension_manager(self, parent: Any | None) -> None:
        from eegprep.functions.adminfunc.plugin_menu import plugin_menu

        show_dialog = parent is not None and _qt_widgets() is not None
        plugin_menu(parent=parent if show_dialog else None, session=self.session, show=show_dialog)

    def _run_pop_function(self, name: str, parent: Any | None, *, variant: str = "") -> None:
        selection = self._current_selection_or_warn(parent, allow_multiple=name in _MULTIPLE_DATASET_ACTIONS)
        if selection is None:
            return
        if name == "pop_adjustevents":
            from eegprep.functions.popfunc.pop_adjustevents import pop_adjustevents

            out = pop_adjustevents(selection, return_com=True)
        elif name == "pop_comments":
            from eegprep.functions.popfunc.pop_comments import pop_comments

            title = _dataset_comments_title(selection)
            out = pop_comments(selection, title, return_com=True)
        elif name == "pop_editset":
            from eegprep.functions.popfunc.pop_editset import pop_editset

            out = pop_editset(selection, return_com=True)
        elif name == "pop_editeventfield":
            from eegprep.functions.popfunc.pop_editeventfield import pop_editeventfield

            out = pop_editeventfield(selection, return_com=True)
        elif name == "pop_editeventvals":
            from eegprep.functions.popfunc.pop_editeventvals import pop_editeventvals

            out = pop_editeventvals(selection, return_com=True)
        elif name == "pop_chanedit":
            from eegprep.functions.popfunc.pop_chanedit import pop_chanedit

            out = pop_chanedit(selection, return_com=True)
        elif name == "pop_clean_rawdata":
            from eegprep.plugins.clean_rawdata.pop_clean_rawdata import pop_clean_rawdata

            out = pop_clean_rawdata(selection, return_com=True)
        elif name == "pop_eegfilt":
            from eegprep.functions.popfunc.pop_eegfilt import pop_eegfilt

            out = pop_eegfilt(selection, return_com=True)
        elif name == "pop_eegfiltnew":
            from eegprep.plugins.firfilt.pop_eegfiltnew import pop_eegfiltnew

            out = pop_eegfiltnew(selection, return_com=True)
        elif name == "pop_epoch":
            from eegprep.functions.popfunc.pop_epoch import pop_epoch

            out = pop_epoch(selection, return_com=True)
        elif name == "pop_firma":
            from eegprep.plugins.firfilt.pop_firma import pop_firma

            out = pop_firma(selection, return_com=True)
        elif name == "pop_firpm":
            from eegprep.plugins.firfilt.pop_firpm import pop_firpm

            out = pop_firpm(selection, return_com=True)
        elif name == "pop_firws":
            from eegprep.plugins.firfilt.pop_firws import pop_firws

            out = pop_firws(selection, return_com=True)
        elif name == "pop_reref":
            from eegprep.functions.popfunc.pop_reref import pop_reref

            out = pop_reref(selection, return_com=True)
        elif name == "pop_interp":
            from eegprep.functions.popfunc.pop_interp import pop_interp

            out = pop_interp(selection, alleeg=self.session.ALLEEG, return_com=True)
        elif name == "pop_iclabel":
            from eegprep.plugins.ICLabel.pop_iclabel import pop_iclabel

            out = pop_iclabel(selection, return_com=True)
        elif name == "pop_icflag":
            from eegprep.plugins.ICLabel.pop_icflag import pop_icflag

            out = pop_icflag(selection, return_com=True)
        elif name == "pop_resample":
            from eegprep.functions.popfunc.pop_resample import pop_resample

            out = pop_resample(selection, return_com=True)
        elif name == "pop_rmbase":
            from eegprep.functions.popfunc.pop_rmbase import pop_rmbase

            out = pop_rmbase(selection, return_com=True)
        elif name == "pop_rmdat":
            from eegprep.functions.popfunc.pop_rmdat import pop_rmdat

            out = pop_rmdat(selection, return_com=True)
        elif name == "pop_runica":
            from eegprep.functions.popfunc.pop_runica import pop_runica

            out = pop_runica(selection, return_com=True)
        elif name == "pop_select":
            from eegprep.functions.popfunc.pop_select import pop_select

            out = pop_select(selection, return_com=True)
        elif name == "pop_selectevent":
            from eegprep.functions.popfunc.pop_selectevent import pop_selectevent

            out = pop_selectevent(selection, return_com=True)
        elif name == "pop_subcomp":
            from eegprep.functions.popfunc.pop_subcomp import pop_subcomp

            out = pop_subcomp(selection, return_com=True)
        elif name == "pop_eegplot":
            from eegprep.functions.popfunc.pop_eegplot import eegplot_accept_creates_dataset, pop_eegplot

            icacomp, superpose, reject = _pop_eegplot_variant_options(variant)
            target_index = list(self.session.CURRENTSET)
            recorded_commands: set[str] = set()

            def accept_eegplot(eeg_out: Any, command: str) -> None:
                with self.session.gui_action("pop_eegplot"):
                    if command:
                        store_new = eegplot_accept_creates_dataset(selection, eeg_out, reject)
                        store_command = "" if command in recorded_commands else command
                        recorded_commands.add(command)
                        store_index = None if store_new else target_index
                        self._store_current_from_gui(
                            eeg_out,
                            new=store_new,
                            command=store_command,
                            index=store_index,
                        )
                        self._refresh()

            out = pop_eegplot(
                selection,
                icacomp=icacomp,
                superpose=superpose,
                reject=reject,
                command_callback=accept_eegplot,
                return_com=True,
            )
            command = out[1] if isinstance(out, tuple) and len(out) > 1 else ""
            if command:
                self._add_history_from_gui(command)
                recorded_commands.add(command)
                self._refresh()
            return
        elif name in _BROWSER_ACCEPT_POP_ACTIONS:
            from eegprep.functions.popfunc.pop_eegplot import eegplot_accept_creates_dataset

            target_index = list(self.session.CURRENTSET)
            recorded_commands: set[str] = set()
            if isinstance(selection, list):
                out = self._run_browser_accept_pop_action(name, selection, variant, command_callback=None)
                command = out[1] if isinstance(out, tuple) and len(out) > 1 else ""
                eeg_out = out[0] if isinstance(out, tuple) and out else out
                if command:
                    self._store_current_from_gui(eeg_out, command=command, index=target_index)
                    self._refresh()
                return

            def accept_browser_result(eeg_out: Any, command: str) -> None:
                with self.session.gui_action(name):
                    if command:
                        store_new = eegplot_accept_creates_dataset(selection, eeg_out, reject=1)
                        store_command = "" if command in recorded_commands else command
                        recorded_commands.add(command)
                        store_index = None if store_new else target_index
                        self._store_current_from_gui(
                            eeg_out,
                            new=store_new,
                            command=store_command,
                            index=store_index,
                        )
                        self._refresh()

            out = self._run_browser_accept_pop_action(name, selection, variant, command_callback=accept_browser_result)
            command = out[1] if isinstance(out, tuple) and len(out) > 1 else ""
            eeg_out = out[0] if isinstance(out, tuple) and out else out
            if name == "pop_rejcont":
                return
            if command:
                self._store_current_from_gui(eeg_out, command=command)
                recorded_commands.add(command)
                self._refresh()
            return
        elif name == "pop_rejchan":
            from eegprep.functions.popfunc.pop_rejchan import pop_rejchan

            out = pop_rejchan(selection, return_com=True)
        elif name == "pop_rejepoch":
            from eegprep.functions.popfunc.pop_rejepoch import pop_rejepoch

            if isinstance(selection, list):
                self._warn(parent, "Select one dataset before rejecting marked epochs.")
                return
            marks = (selection.get("reject") or {}).get("rejglobal", [])
            out = pop_rejepoch(selection, marks, return_com=True)
        elif name == "pop_rejmenu":
            from eegprep.functions.popfunc.pop_rejmenu import pop_rejmenu

            out = pop_rejmenu(selection, _icacomp_from_variant(variant), return_com=True)
        elif name == "pop_selectcomps":
            from eegprep.functions.popfunc.pop_selectcomps import pop_selectcomps

            out = pop_selectcomps(selection, return_com=True)
        elif name == "pop_viewprops":
            from eegprep.plugins.ICLabel.pop_viewprops import pop_viewprops

            out = pop_viewprops(selection, typecomp=0 if variant == "components" else 1, return_com=True)
        else:
            self.show_coming_soon(name, parent)
            return
        if isinstance(out, tuple):
            eeg_out, command = out[0], out[1] if len(out) > 1 else ""
        else:
            eeg_out, command = out, ""
        if name == "pop_viewprops":
            self._add_history_from_gui(command)
            self._refresh()
            return
        if command:
            self._store_current_from_gui(eeg_out, command=command)
            self._refresh()

    def _run_browser_accept_pop_action(
        self,
        name: str,
        selection: Any,
        variant: str,
        *,
        command_callback: Any | None,
    ) -> Any:
        callback_kwargs = {"command_callback": command_callback} if command_callback is not None else {}
        if name == "pop_autorej":
            from eegprep.functions.popfunc.pop_autorej import pop_autorej

            return pop_autorej(selection, **callback_kwargs, return_com=True)
        if name == "pop_eegthresh":
            from eegprep.functions.popfunc.pop_eegthresh import pop_eegthresh

            return pop_eegthresh(selection, _icacomp_from_variant(variant), **callback_kwargs, return_com=True)
        if name == "pop_jointprob":
            from eegprep.functions.popfunc.pop_jointprob import pop_jointprob

            return pop_jointprob(selection, _icacomp_from_variant(variant), **callback_kwargs, return_com=True)
        if name == "pop_rejcont":
            from eegprep.functions.popfunc.pop_rejcont import pop_rejcont

            return pop_rejcont(selection, **callback_kwargs, return_com=True)
        if name == "pop_rejkurt":
            from eegprep.functions.popfunc.pop_rejkurt import pop_rejkurt

            return pop_rejkurt(selection, _icacomp_from_variant(variant), **callback_kwargs, return_com=True)
        if name == "pop_rejspec":
            from eegprep.functions.popfunc.pop_rejspec import pop_rejspec

            return pop_rejspec(selection, _icacomp_from_variant(variant), **callback_kwargs, return_com=True)
        from eegprep.functions.popfunc.pop_rejtrend import pop_rejtrend

        return pop_rejtrend(selection, _icacomp_from_variant(variant), **callback_kwargs, return_com=True)

    def _select_multiple_datasets(self, parent: Any | None) -> None:
        if not self.session.ALLEEG:
            self._warn(parent, "No datasets available")
            return
        from eegprep.functions.guifunc.select_multiple_datasets import select_multiple_datasets

        eeg_out, command = select_multiple_datasets(self.session, gui=True, return_com=True)
        if command:
            self.session.echo_command(command)
            self.session.add_history(command, notify=False)
            self.session.notify_changed()
            self._refresh()

    def _select_study_set(self, parent: Any | None) -> None:
        if not self.session.STUDY:
            self._warn(parent, "No current STUDY")
            return
        command = "CURRENTSTUDY = 1"
        self.session.echo_command(command)
        self.session.select_study(command=command)
        self._refresh()

    def _copy_current_dataset(self, parent: Any | None) -> None:
        if not self.session.CURRENTSET:
            self._warn(parent, "No current dataset")
            return
        from eegprep.functions.popfunc.pop_copyset import pop_copyset

        set_in = self.session.CURRENTSET[0]
        alleeg, eeg_out, current_set, command = pop_copyset(self.session.ALLEEG, set_in, gui=True, return_com=True)
        if not command:
            return
        self.session.ALLEEG = alleeg
        self.session.EEG = eeg_out
        self.session.CURRENTSET = _currentset_list(current_set)
        self._add_history_from_gui(command)
        self.session.notify_changed()
        self._refresh()

    def _merge_datasets(self, parent: Any | None) -> None:
        if len(self.session.ALLEEG) < 2:
            self._warn(parent, "Load at least two datasets before merging")
            return
        from eegprep.functions.popfunc.pop_mergeset import pop_mergeset

        selected = self.session.selected_dataset_indices()
        default_indices = selected if len(selected) >= 2 else None
        eeg_out, command = pop_mergeset(self.session.ALLEEG, default_indices, gui=True, return_com=True)
        if command:
            self._store_current_from_gui(eeg_out, new=True, command=command)
            self._refresh()

    def _run_rejsuperpose(self, variant: str, parent: Any | None) -> None:
        selection = self._current_selection_or_warn(parent)
        if selection is None:
            return
        from eegprep.functions.popfunc.eeg_rejsuperpose import eeg_rejsuperpose

        typerej = 0 if variant == "data_to_ica" else 1
        eeg_out, command = eeg_rejsuperpose(selection, typerej, 1, 1, 1, 1, 1, 1, 1, return_com=True)
        self._store_current_from_gui(eeg_out, command=command)
        self._refresh()

    def _run_dipfit_function(self, name: str, parent: Any | None) -> None:
        selection = self._current_selection_or_warn(parent, allow_multiple=name in _MULTIPLE_DATASET_ACTIONS)
        if selection is None:
            return
        if name == "pop_dipfit_settings":
            from eegprep.plugins.dipfit.pop_dipfit_settings import pop_dipfit_settings

            eeg_out, command = pop_dipfit_settings(selection, return_com=True)
            if command:
                self._store_current_from_gui(eeg_out, command=command)
                self._refresh()
            return
        if name == "pop_dipplot":
            from eegprep.plugins.dipfit.pop_dipplot import pop_dipplot

            figures, command = pop_dipplot(selection, return_com=True)
            for figure in figures:
                show = getattr(figure, "show", None)
                if callable(show):
                    show()
            self._add_history_from_gui(command)
            self._refresh()
            return
        if name == "pop_dipfit_headmodel":
            from eegprep.plugins.dipfit.pop_dipfit_headmodel import pop_dipfit_headmodel

            pop_dipfit_headmodel(selection, return_com=True)
            return
        if name == "pop_dipfit_gridsearch":
            from eegprep.plugins.dipfit.pop_dipfit_gridsearch import pop_dipfit_gridsearch

            pop_dipfit_gridsearch(selection, return_com=True)
            return
        if name == "pop_dipfit_nonlinear":
            from eegprep.plugins.dipfit.pop_dipfit_nonlinear import pop_dipfit_nonlinear

            pop_dipfit_nonlinear(selection, return_com=True)
            return
        if name == "pop_multifit":
            from eegprep.plugins.dipfit.pop_multifit import pop_multifit

            pop_multifit(selection, return_com=True)
            return
        if name == "pop_leadfield":
            from eegprep.plugins.dipfit.pop_leadfield import pop_leadfield

            pop_leadfield(selection, return_com=True)
            return
        if name == "pop_dipfit_loreta":
            from eegprep.plugins.dipfit.pop_dipfit_loreta import pop_dipfit_loreta

            pop_dipfit_loreta(selection, return_com=True)
            return
        self.show_coming_soon(name, parent)

    def _plot_channel_locations(self, variant: str, parent: Any | None) -> None:
        selection = self._current_selection_or_warn(parent)
        if selection is None:
            return
        from eegprep.functions.popfunc.pop_topoplot import plot_channel_locations

        _figure, command = plot_channel_locations(selection, mode=variant or "labels", return_com=True)
        self._add_history_from_gui(command)
        self._refresh()

    def _run_topoplot(self, variant: str, parent: Any | None) -> None:
        selection = self._current_selection_or_warn(parent)
        if selection is None:
            return
        from eegprep.functions.popfunc.pop_topoplot import pop_topoplot

        typeplot = 0 if variant == "components" else 1
        _figures, command = pop_topoplot(selection, typeplot=typeplot, return_com=True)
        self._add_history_from_gui(command)
        self._refresh()

    def _run_plot_function(self, name: str, variant: str, parent: Any | None) -> None:
        allow_multiple = name == "pop_comperp"
        selection = self._current_selection_or_warn(parent, allow_multiple=allow_multiple)
        if selection is None:
            return
        if name == "pop_spectopo":
            from eegprep.functions.popfunc.pop_spectopo import pop_spectopo

            _result, command = pop_spectopo(selection, dataflag=0 if variant == "components" else 1, return_com=True)
        elif name == "pop_prop":
            from eegprep.functions.popfunc.pop_prop import pop_prop

            _result, command = pop_prop(selection, typecomp=0 if variant == "components" else 1, return_com=True)
        elif name == "pop_timtopo":
            from eegprep.functions.popfunc.pop_timtopo import pop_timtopo

            _result, command = pop_timtopo(selection, return_com=True)
        elif name == "pop_plottopo":
            from eegprep.functions.popfunc.pop_plottopo import pop_plottopo

            _result, command = pop_plottopo(selection, return_com=True)
        elif name == "pop_headplot":
            from eegprep.functions.popfunc.pop_headplot import pop_headplot

            _result, command = pop_headplot(selection, typeplot=0 if variant == "components" else 1, return_com=True)
        elif name == "pop_plotdata":
            from eegprep.functions.popfunc.pop_plotdata import pop_plotdata

            _result, command = pop_plotdata(selection, return_com=True)
        elif name == "pop_erpimage":
            from eegprep.functions.popfunc.pop_erpimage import pop_erpimage

            _result, command = pop_erpimage(selection, typeplot=0 if variant == "components" else 1, return_com=True)
        elif name == "pop_envtopo":
            from eegprep.functions.popfunc.pop_envtopo import pop_envtopo

            _result, command = pop_envtopo(selection, return_com=True)
        elif name == "pop_comperp":
            from eegprep.functions.popfunc.pop_comperp import pop_comperp

            datasets = self.session.ALLEEG or (selection if isinstance(selection, list) else [selection])
            _result, command = pop_comperp(datasets, flag=0 if variant == "components" else 1, return_com=True)
        elif name == "pop_newtimef":
            from eegprep.functions.popfunc.pop_newtimef import pop_newtimef

            _result, command = pop_newtimef(selection, typeproc=0 if variant == "components" else 1, return_com=True)
        elif name == "pop_newcrossf":
            from eegprep.functions.popfunc.pop_newcrossf import pop_newcrossf

            _result, command = pop_newcrossf(selection, typeproc=0 if variant == "components" else 1, return_com=True)
        elif name == "pop_signalstat":
            from eegprep.functions.popfunc.pop_signalstat import pop_signalstat

            _result, command = pop_signalstat(selection, typeproc=0 if variant == "components" else 1, return_com=True)
        elif name == "pop_eventstat":
            from eegprep.functions.popfunc.pop_eventstat import pop_eventstat

            _result, command = pop_eventstat(selection, return_com=True)
        else:
            self.show_coming_soon(name, parent)
            return
        self._add_history_from_gui(command)
        self._refresh()

    def _run_chanplot(self, parent: Any | None) -> None:
        if not self.session.STUDY:
            self._warn(parent, "Select or create a STUDY before plotting channel measures.")
            return
        from eegprep.functions.studyfunc.pop_chanplot import pop_chanplot

        study, command, _figure = pop_chanplot(self.session.STUDY, self.session.ALLEEG, gui=True, return_com=True)
        if not command:
            return
        self.session.STUDY = study
        self._add_history_from_gui(command)
        self._refresh()

    def _store_current_from_gui(self, eeg: Any, **kwargs: Any) -> Any:
        command = kwargs.get("command")
        self.session.echo_command(command)
        return self.session.store_current(eeg, **kwargs)

    def _add_history_from_gui(self, command: str | None) -> None:
        self.session.echo_command(command)
        self.session.add_history(command)

    def _retrieve_dataset(self, index: int) -> None:
        was_study = self.session.CURRENTSTUDY == 1
        self.session.retrieve(index)
        command = f"[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, CURRENTSET, 'retrieve', {index});"
        if was_study:
            self.session.CURRENTSTUDY = 0
            command = f"CURRENTSTUDY = 0;{command}"
        self._add_history_from_gui(command)
        self._refresh()

    def _show_help(self, function_name: str, parent: Any | None) -> None:
        pophelp(function_name, parent=parent)

    def _current_selection_or_warn(
        self,
        parent: Any | None,
        *,
        allow_multiple: bool = False,
    ) -> dict[str, Any] | list[dict[str, Any]] | None:
        selection = self.session.current_eeg()
        if isinstance(selection, list):
            if not any(has_eeg_data(eeg) for eeg in selection):
                self._warn(parent, "No current dataset")
                return None
            if len(selection) > 1:
                if allow_multiple:
                    return selection
                self._warn(parent, "This action is not available for multiple selected datasets")
                return None
            return selection[0]
        if has_eeg_data(selection):
            return selection
        self._warn(parent, "No current dataset")
        return None

    def _warn(self, parent: Any | None, message: str) -> None:
        qt_widgets = _qt_widgets()
        if qt_widgets is not None:
            qt_widgets.QMessageBox.warning(parent, "EEGPrep", message)

    def _info(self, parent: Any | None, message: str) -> None:
        qt_widgets = _qt_widgets()
        if qt_widgets is not None:
            qt_widgets.QMessageBox.information(parent, "EEGPrep", message)

    def _ask_float(self, parent: Any | None, title: str, label: str, value: float) -> float | None:
        qt_widgets = _require_qt_widgets()
        result, ok = qt_widgets.QInputDialog.getDouble(parent, title, label, value, 0.0)
        return float(result) if ok else None

    def _ask_metadata(self, parent: Any | None, action: str) -> dict[str, str] | None:
        qt_widgets = _require_qt_widgets()
        text, ok = qt_widgets.QInputDialog.getMultiLineText(
            parent,
            "BIDS metadata",
            "Enter one key=value pair per line",
            _default_bids_metadata(action),
        )
        if not ok:
            return None
        metadata = {}
        for line in text.splitlines():
            if not line.strip():
                continue
            if "=" not in line:
                raise ValueError("BIDS metadata lines must use key=value syntax")
            key, value = line.split("=", 1)
            metadata[key.strip()] = value.strip()
        return metadata

    def _refresh(self) -> None:
        if self.refresh is not None:
            self.refresh()

    def _file_dialog_kwargs(self, qt_widgets: Any, *, directories: bool = False) -> dict[str, Any]:
        return _file_dialog_kwargs(
            qt_widgets,
            native_file_dialogs=self.native_file_dialogs,
            directories=directories,
        )


def _existing_dataset_filename(eeg: dict[str, Any]) -> str:
    filepath = str(eeg.get("filepath") or "")
    filename = str(eeg.get("filename") or "")
    if filepath and filename:
        return str(Path(filepath) / filename)
    return filename


def _existing_study_filename(study: dict[str, Any]) -> str:
    filepath = str(study.get("filepath") or "")
    filename = str(study.get("filename") or "")
    if filepath and filename:
        return str(Path(filepath) / filename)
    return filename


def _dataset_comments_title(eeg: dict[str, Any] | list[dict[str, Any]]) -> str:
    dataset = eeg[0] if isinstance(eeg, list) and eeg else eeg
    if not isinstance(dataset, dict):
        return ""
    setname = str(dataset.get("setname") or "").strip()
    return f"Comments of dataset: {setname}" if setname else "Comments of dataset"


def _export_filter(action: str) -> str:
    if action == "pop_writeeeg":
        return "EDF/BDF/GDF files (*.edf *.bdf *.gdf);;All files (*)"
    if action == "pop_expevents":
        return "Text tables (*.tsv *.txt);;All files (*)"
    return "Text files (*.txt *.tsv *.csv);;All files (*)"


def _default_bids_metadata(action: str) -> str:
    if action == "pop_taskinfo":
        return "TaskName=eeg"
    if action == "pop_participantinfo":
        return "participant_id=sub-01"
    return "trial_type=event"


def _icacomp_from_variant(variant: str) -> int:
    return 0 if variant == "ica" else 1


def _currentset_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, str):
        if value == "":
            return []
        current = int(value)
        return [current] if current > 0 else []
    if isinstance(value, bool):
        return [1] if value else []
    if isinstance(value, (int, float)):
        current = int(value)
        return [current] if current > 0 else []
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        return [int(item) for item in value if int(item) > 0]
    current = int(value)
    return [current] if current > 0 else []


def _pop_eegplot_variant_options(variant: str) -> tuple[int, int, int]:
    # Mirrors eeglab.m callbacks: cb_eegplot/cb_eegplotrej* omit optional args,
    # while Plot scroll actions explicitly call pop_eegplot(EEG, icacomp, 1, 1).
    if variant == "components":
        return 0, 1, 1
    if variant == "reject_ica":
        return 0, 0, 1
    if variant == "channels":
        return 1, 1, 1
    return 1, 0, 1


def _apply_save_metadata(eeg: dict[str, Any], filename: str) -> None:
    path = Path(filename)
    eeg["filename"] = path.name
    eeg["filepath"] = str(path.parent)
    eeg["saved"] = "yes"


def _docs_url(path: str = "") -> str:
    path = str(path or "").lstrip("/")
    return EEGPREP_DOCS_URL + path


def _tutorial_url() -> str:
    return _docs_url("user_guide/quickstart.html")


def action_kind(action: str) -> str:
    """Return ``implemented``, ``placeholder``, or ``unknown`` for an action id."""
    base = action.partition(":")[0]
    if action in PLACEHOLDER_ACTIONS:
        return "placeholder"
    if base in IMPLEMENTED_ACTIONS:
        return "implemented"
    if is_placeholder_action(action):
        return "placeholder"
    return "unknown"


def _file_dialog_kwargs(
    qt_widgets: Any, *, native_file_dialogs: bool = True, directories: bool = False
) -> dict[str, Any]:
    if native_file_dialogs:
        return {}
    # Native macOS file panels can close immediately under IPython's Qt input hook.
    options = _qt_enum_value(qt_widgets.QFileDialog, "Option", "DontUseNativeDialog")
    if directories:
        show_dirs = _qt_enum_value(qt_widgets.QFileDialog, "Option", "ShowDirsOnly")
        if options is None:
            options = show_dirs
        elif show_dirs is not None:
            options = options | show_dirs
    return {"options": options} if options is not None else {}


def _qt_enum_value(owner: Any, enum_name: str, value_name: str) -> Any | None:
    enum = getattr(owner, enum_name, None)
    value = getattr(enum, value_name, None) if enum is not None else None
    if value is not None:
        return value
    return getattr(owner, value_name, None)


def _qt_widgets() -> Any | None:
    try:
        from PySide6 import QtWidgets
    except ImportError:
        return None
    return QtWidgets


def _require_qt_widgets() -> Any:
    qt_widgets = _qt_widgets()
    if qt_widgets is None:
        raise RuntimeError(
            "PySide6 is required for EEGPrep GUI actions. Install it with "
            "`pip install -e .[gui]` or `pip install eegprep[gui]`."
        )
    return qt_widgets
