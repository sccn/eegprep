"""Action dispatch for the EEGPrep EEGLAB-style main window."""

from __future__ import annotations

import logging
import webbrowser
from collections.abc import Callable
from pathlib import Path
from typing import Any

from eegprep.functions.guifunc.menu_placeholders import is_placeholder_action, placeholder_message
from eegprep.functions.guifunc.pophelp import pophelp
from eegprep.functions.guifunc.session import EEGPrepSession, has_eeg_data
from eegprep.functions.popfunc._coming_soon import coming_soon


logger = logging.getLogger(__name__)

IMPLEMENTED_ACTIONS = {
    "clear_study",
    "docs",
    "help",
    "issues",
    "license",
    "mailto",
    "pop_adjustevents",
    "pop_biosig",
    "pop_chanevent",
    "pop_clean_rawdata",
    "pop_delset",
    "pop_editoptions",
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
    "pop_iclabel",
    "pop_importbids",
    "pop_importdata",
    "pop_importepoch",
    "pop_importerplab",
    "pop_importevent",
    "pop_importpres",
    "pop_interp",
    "pop_loadstudy",
    "pop_loadset",
    "pop_runscript",
    "pop_saveh",
    "pop_savestudy",
    "pop_reref",
    "pop_resample",
    "pop_runica",
    "pop_saveset",
    "pop_select",
    "pop_study",
    "pop_studyerp",
    "pop_studywizard",
    "pop_taskinfo",
    "pop_participantinfo",
    "pop_writeeeg",
    "bids_exporter",
    "plugin_menu",
    "quit",
    "retrieve_dataset",
    "tutorial",
    "updates",
    "validate_bids",
}

EEGPREP_REPO_URL = "https://github.com/sccn/eegprep"
EEGPREP_DOCS_URL = "https://sccn.github.io/eegprep/"
EEGPREP_SOURCE_URL = f"{EEGPREP_REPO_URL}/blob/develop"

HELP_TOPIC_LABELS = {
    "eegprep": "About EEGPrep",
    "eeg_helphelp": "About EEGPrep help",
    "eeg_helpmenu": "EEGPrep menus",
    "eeg_helpadmin": "Admin. functions",
    "eeg_helppop": "Interactive pop_ functions",
    "eeg_helpsigproc": "Signal processing functions",
    "eeg_helpstudy": "Group data (STUDY) functions",
    "eeg_helptimefreq": "Time-frequency functions",
    "eeg_helpstatistics": "Statistical functions",
    "eeg_helpgui": "Graphic interface builder functions",
    "eeg_helpmisc": "Misc. command line functions",
    "troubleshooting_data_formats": "Troubleshooting data formats",
}
HELP_DOC_PATHS = {
    "eegprep": "",
    "eeg_helphelp": "user_guide/index.html#getting-help",
    "eeg_helpadmin": "api/core.html",
    "eeg_helppop": "api/index.html",
    "eeg_helpsigproc": "api/signal_processing.html",
    "eeg_helpstatistics": "api/utils.html#statistical-utilities",
    "eeg_helpmisc": "api/utils.html",
    "troubleshooting_data_formats": "faq.html#what-data-formats-are-supported",
}
HELP_UNAVAILABLE_TOPICS = frozenset(set(HELP_TOPIC_LABELS) - set(HELP_DOC_PATHS))

_MULTIPLE_DATASET_ACTIONS = {
    "pop_clean_rawdata",
    "pop_iclabel",
    "pop_reref",
    "pop_resample",
    "pop_runica",
    "pop_select",
}


class MenuActionDispatcher:
    """Dispatch menu action identifiers to real functions or placeholders."""

    def __init__(self, session: EEGPrepSession, refresh: Callable[[], None] | None = None):
        self.session = session
        self.refresh = refresh

    def dispatch_gui(self, action: str, parent: Any | None = None) -> None:
        """Run a menu action from Qt and show user-facing errors."""
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
            self._show_help(variant or "eeglab", parent)
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
        if base in {"pop_study", "pop_studywizard", "pop_studyerp", "pop_loadstudy", "pop_savestudy"}:
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
        if base == "pop_clean_rawdata":
            self._run_pop_function("pop_clean_rawdata", parent)
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
        if base == "pop_runica":
            self._run_pop_function("pop_runica", parent)
            return
        if base == "pop_select":
            self._run_pop_function("pop_select", parent)
            return
        if base == "pop_iclabel":
            self._run_pop_function("pop_iclabel", parent)
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
        )
        if not filename:
            return
        eeg = pop_loadset(filename)
        self.session.store_current(eeg, new=True, command=f"EEG = pop_loadset({filename!r});")
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
            "EEG = pop_saveset(EEG, 'savemode', 'resave');"
            if resave
            else f"EEG = pop_saveset(EEG, {filenames[0]!r});"
        )
        self.session.store_current(stored, command=command, mark_saved=True)
        self._refresh()

    def _import_dataset(self, action: str, parent: Any | None) -> None:
        qt_widgets = _require_qt_widgets()
        if action == "pop_importbids":
            source = qt_widgets.QFileDialog.getExistingDirectory(parent, "Import BIDS folder structure", "")
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
            else:
                from eegprep.functions.popfunc.pop_fileio import pop_fileio

                eeg_out, command = pop_fileio(filename, return_com=True)
        self.session.store_current(eeg_out, new=True, command=command)
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
            "pop_fileio_brainvision_mat": "BrainVision MATLAB (*.mat);;All files (*)",
        }
        filename, _filter = qt_widgets.QFileDialog.getOpenFileName(
            parent,
            "Import data",
            "",
            filters.get(action, "EEG files (*.set *.edf *.bdf *.gdf *.vhdr *.mff *.cnt *.eeg);;All files (*)"),
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
        self.session.store_current(eeg_out, command=command)
        self._refresh()

    def _export_current_dataset(self, action: str, variant: str, parent: Any | None) -> None:
        allow_multiple = action in {"pop_exportbids", "bids_exporter"}
        selection = self._current_selection_or_warn(parent, allow_multiple=allow_multiple)
        if selection is None:
            return
        qt_widgets = _require_qt_widgets()
        if action in {"pop_exportbids", "bids_exporter"}:
            directory = qt_widgets.QFileDialog.getExistingDirectory(parent, "Export to BIDS folder structure", "")
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
        self.session.add_history(command)
        self._refresh()

    def _study_action(self, action: str, variant: str, parent: Any | None) -> None:
        qt_widgets = _require_qt_widgets()
        if action == "pop_loadstudy":
            filename, _filter = qt_widgets.QFileDialog.getOpenFileName(parent, "Load existing study", "", "STUDY files (*.study *.json);;All files (*)")
            if not filename:
                return
            from eegprep.functions.studyfunc.pop_loadstudy import pop_loadstudy

            study, alleeg, command = pop_loadstudy(filename)
            self.session.STUDY = study
            if alleeg:
                self.session.ALLEEG = alleeg
            self.session.CURRENTSTUDY = 1
            self.session.add_history(command)
            self._refresh()
            return
        if action == "pop_savestudy":
            if not self.session.STUDY:
                self._warn(parent, "No current study")
                return
            filename = _existing_study_filename(self.session.STUDY) if variant == "resave" else ""
            if not filename:
                filename, _filter = qt_widgets.QFileDialog.getSaveFileName(parent, "Save current study as", "", "STUDY files (*.study);;All files (*)")
            if not filename:
                return
            from eegprep.functions.studyfunc.pop_savestudy import pop_savestudy

            study, command = pop_savestudy(self.session.STUDY, self.session.EEG, filename, savemode=variant or None)
            self.session.STUDY = study
            self.session.add_history(command)
            self._refresh()
            return
        if action == "pop_studywizard":
            filenames, _filter = qt_widgets.QFileDialog.getOpenFileNames(parent, "Browse for datasets", "", "EEGPrep/EEGLAB datasets (*.set *.mat);;All files (*)")
            if not filenames:
                return
            from eegprep.functions.studyfunc.pop_studywizard import pop_studywizard

            study, alleeg, command = pop_studywizard(filenames)
        elif action == "pop_studyerp":
            from eegprep.functions.studyfunc.pop_studyerp import pop_studyerp

            study, alleeg, command = pop_studyerp(self.session.ALLEEG)
        else:
            if not self.session.ALLEEG:
                self._warn(parent, "Load at least one dataset before creating a study")
                return
            from eegprep.functions.studyfunc.pop_study import pop_study

            study, alleeg, command = pop_study(None, self.session.ALLEEG)
        self.session.STUDY = study
        self.session.ALLEEG = alleeg
        self.session.CURRENTSTUDY = 1
        self.session.add_history(command)
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
        self.session.add_history(command)
        self._info(parent, "Preferences updated. Reopen the main window to rebuild the menu mode.")
        self._refresh()

    def _save_history(self, variant: str, parent: Any | None) -> None:
        qt_widgets = _require_qt_widgets()
        filename, _filter = qt_widgets.QFileDialog.getSaveFileName(parent, "Save history script", "eegprephist.m", "MATLAB scripts (*.m);;All files (*)")
        if not filename:
            return
        from eegprep.functions.popfunc.pop_saveh import pop_saveh

        path = Path(filename)
        history = self.session.EEG.get("history", "") if variant == "dataset" and isinstance(self.session.EEG, dict) else self.session.ALLCOM
        command = pop_saveh(history, path.name, path.parent)
        self.session.add_history(command)
        self._refresh()

    def _run_script(self, parent: Any | None) -> None:
        qt_widgets = _require_qt_widgets()
        filename, _filter = qt_widgets.QFileDialog.getOpenFileName(parent, "Run history script", "", "Scripts (*.py *.m *.txt);;All files (*)")
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
        self.session.STUDY = namespace.get("STUDY", self.session.STUDY)
        self.session.add_history(command)
        self._refresh()

    def _bids_tool_action(self, action: str, parent: Any | None) -> None:
        if action == "validate_bids":
            qt_widgets = _require_qt_widgets()
            directory = qt_widgets.QFileDialog.getExistingDirectory(parent, "Validate BIDS dataset", "")
            if not directory:
                return
            from eegprep.plugins.EEG_BIDS.bids_tools import validate_bids

            report = validate_bids(directory)
            self._info(parent, f"BIDS validation complete: {len(report['errors'])} errors, {len(report['warnings'])} warnings.")
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
            self.session.add_history(command)
        else:
            self.session.store_current(updated, command=command)
        self._refresh()

    def _show_extension_manager(self, parent: Any | None) -> None:
        plugins = ["clean_rawdata", "ICLabel/viewprops", "firfilt", "DIPFIT", "EEG-BIDS/File-IO"]
        self._info(parent, "Available EEGPrep extensions:\n" + "\n".join(f"- {plugin}" for plugin in plugins))

    def _run_pop_function(self, name: str, parent: Any | None) -> None:
        selection = self._current_selection_or_warn(parent, allow_multiple=name in _MULTIPLE_DATASET_ACTIONS)
        if selection is None:
            return
        if name == "pop_adjustevents":
            from eegprep.functions.popfunc.pop_adjustevents import pop_adjustevents

            out = pop_adjustevents(selection, return_com=True)
        elif name == "pop_clean_rawdata":
            from eegprep.plugins.clean_rawdata.pop_clean_rawdata import pop_clean_rawdata

            out = pop_clean_rawdata(selection, return_com=True)
        elif name == "pop_reref":
            from eegprep.functions.popfunc.pop_reref import pop_reref

            out = pop_reref(selection, return_com=True)
        elif name == "pop_interp":
            from eegprep.functions.popfunc.pop_interp import pop_interp

            out = pop_interp(selection, alleeg=self.session.ALLEEG, return_com=True)
        elif name == "pop_iclabel":
            from eegprep.plugins.ICLabel.pop_iclabel import pop_iclabel

            out = pop_iclabel(selection, return_com=True)
        elif name == "pop_resample":
            from eegprep.functions.popfunc.pop_resample import pop_resample

            out = pop_resample(selection, return_com=True)
        elif name == "pop_runica":
            from eegprep.functions.popfunc.pop_runica import pop_runica

            out = pop_runica(selection, return_com=True)
        elif name == "pop_select":
            from eegprep.functions.popfunc.pop_select import pop_select

            out = pop_select(selection, return_com=True)
        else:
            self.show_coming_soon(name, parent)
            return
        if isinstance(out, tuple):
            eeg_out, command = out[0], out[1] if len(out) > 1 else ""
        else:
            eeg_out, command = out, ""
        if command:
            self.session.store_current(eeg_out, command=command)
            self._refresh()

    def _retrieve_dataset(self, index: int) -> None:
        was_study = self.session.CURRENTSTUDY == 1
        self.session.retrieve(index)
        command = f"[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, CURRENTSET, 'retrieve', {index});"
        if was_study:
            self.session.CURRENTSTUDY = 0
            command = f"CURRENTSTUDY = 0;{command}"
        self.session.add_history(command)
        self._refresh()

    def _show_help(self, function_name: str, parent: Any | None) -> None:
        try:
            pophelp(function_name, parent=parent)
        except FileNotFoundError as exc:
            if function_name in HELP_DOC_PATHS:
                webbrowser.open(_docs_url(HELP_DOC_PATHS[function_name]))
                return
            if function_name in HELP_UNAVAILABLE_TOPICS:
                self._show_unavailable_help(function_name, parent)
                return
            raise exc

    def _show_unavailable_help(self, function_name: str, parent: Any | None) -> None:
        message = unavailable_help_message(function_name)
        qt_widgets = _qt_widgets()
        if qt_widgets is None or parent is None:
            raise FileNotFoundError(message)
        qt_widgets.QMessageBox.information(parent, "EEGPrep", message)

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


def unavailable_help_message(function_name: str) -> str:
    """Return user-facing copy for help topics not yet documented in EEGPrep."""
    label = HELP_TOPIC_LABELS.get(function_name, function_name)
    return (
        f"EEGPrep help for {label} is not available yet.\n\n"
        "Track progress or request this documentation at https://github.com/sccn/eegprep/issues."
    )


def action_kind(action: str) -> str:
    """Return ``implemented``, ``placeholder``, or ``unknown`` for an action id."""
    base = action.partition(":")[0]
    if base in IMPLEMENTED_ACTIONS:
        return "implemented"
    if is_placeholder_action(action):
        return "placeholder"
    return "unknown"


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
