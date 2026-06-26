"""Interactive EEGPrep console sharing state with the GUI."""

from __future__ import annotations

import argparse
import ast
import contextvars
import importlib
import inspect
import logging
import re
import sys
import threading
import warnings
from collections.abc import Callable, Iterator, Mapping
from typing import Any

import eegprep
from eegprep.functions.adminfunc.eegh import eegh, eegh_find
from eegprep.extension_runtime import ExtensionRuntime
from eegprep.functions.adminfunc.eeglab import gui
from eegprep.functions.guifunc.session import EEGPrepSession, normalize_dataset_indices
from eegprep.functions.popfunc.pop_eegplot import eegplot_accept_creates_dataset
from eegprep.functions.popfunc.pop_newset import pop_newset


WORKSPACE_NAMES = ("EEG", "ALLEEG", "CURRENTSET", "ALLCOM", "LASTCOM", "STUDY", "CURRENTSTUDY")
POP_RESULT_PREVIEW_LIMIT = 96
ANSI_CLEAR_LINE = "\r\x1b[2K"
_EEG_CORE_FIELDS = ("nbchan", "srate", "pnts", "trials")
_ACTIVE_TERMINAL_BUFFER = contextvars.ContextVar("_ACTIVE_TERMINAL_BUFFER", default=None)
_MATLAB_MULTI_ASSIGN_PATTERN = re.compile(r"^\s*\[([A-Za-z_][A-Za-z0-9_]*(?:\s+[A-Za-z_][A-Za-z0-9_]*)+)\]\s*=")
_MUTATING_METHODS = {"pop", "update", "clear", "setdefault", "insert", "remove", "append", "extend", "fill"}
_TUPLE_ASSIGNMENT_TARGET_PATTERN = re.compile(
    r"(^|;\s*)\(([A-Za-z_][A-Za-z0-9_]*(?:,\s*[A-Za-z_][A-Za-z0-9_]*)+)\)\s*="
)
_POP_INTERP_CHANNELS_PATTERN = re.compile(r"(pop_interp\s*\(\s*EEG\s*,\s*)\[([0-9,\s]+)\]")
_PYTHON_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_CONSOLE_COMMAND_EXPORTS = {"pop_newset": pop_newset}
_BROWSER_ACCEPT_POP_FUNCTIONS = {
    "pop_autorej",
    "pop_eegthresh",
    "pop_jointprob",
    "pop_rejcont",
    "pop_rejkurt",
    "pop_rejspec",
    "pop_rejtrend",
}


class ConsolePopResult:
    """Compact, unpackable result for EEGPrep console ``pop_*`` calls."""

    def __init__(self, eeg: Any, command: str, *, updated: bool = True) -> None:
        self.eeg = eeg
        self.command = command
        self.updated = updated

    def __iter__(self) -> Iterator[Any]:
        yield self.eeg
        yield self.command

    def __getitem__(self, index: int) -> Any:
        return (self.eeg, self.command)[index]

    def __len__(self) -> int:
        return 2

    def __repr__(self) -> str:
        command = self.command or "(no history command)"
        if len(command) > POP_RESULT_PREVIEW_LIMIT:
            command = command[: POP_RESULT_PREVIEW_LIMIT - 3] + "..."
        state = "EEG updated" if self.updated else "no EEG change"
        return f"<EEGPrep pop result: {state}, LASTCOM={command!r}>"


class ConsoleDatasetResult:
    """Compact, unpackable result for dataset-list ``pop_*`` calls."""

    def __init__(self, alleeg: list[dict[str, Any]], eeg: Any, currentset: Any, command: str) -> None:
        self.alleeg = alleeg
        self.eeg = eeg
        self.currentset = currentset
        self.command = command

    def __iter__(self) -> Iterator[Any]:
        yield self.alleeg
        yield self.eeg
        yield self.currentset
        yield self.command

    def __getitem__(self, index: int) -> Any:
        return (self.alleeg, self.eeg, self.currentset, self.command)[index]

    def __len__(self) -> int:
        return 4

    def __repr__(self) -> str:
        command = self.command or "(no history command)"
        if len(command) > POP_RESULT_PREVIEW_LIMIT:
            command = command[: POP_RESULT_PREVIEW_LIMIT - 3] + "..."
        return f"<EEGPrep dataset result: CURRENTSET={self.currentset!r}, LASTCOM={command!r}>"


class ConsoleStudyResult:
    """Compact, unpackable result for STUDY ``pop_*`` calls."""

    def __init__(
        self,
        study: dict[str, Any],
        alleeg: list[dict[str, Any]] | None,
        command: str,
        *,
        includes_alleeg: bool,
    ) -> None:
        self.study = study
        self.alleeg = alleeg
        self.command = command
        self.includes_alleeg = includes_alleeg

    def __iter__(self) -> Iterator[Any]:
        yield self.study
        if self.includes_alleeg:
            yield self.alleeg
        else:
            yield self.command

    def __getitem__(self, index: int) -> Any:
        return tuple(self)[index]

    def __len__(self) -> int:
        return 2

    def __repr__(self) -> str:
        command = self.command or "(no history command)"
        if len(command) > POP_RESULT_PREVIEW_LIMIT:
            command = command[: POP_RESULT_PREVIEW_LIMIT - 3] + "..."
        return f"<EEGPrep STUDY result: CURRENTSTUDY=1, LASTCOM={command!r}>"


class LazyWorkspaceExport:
    """Lazy proxy for public EEGPrep exports in the console namespace."""

    def __init__(
        self,
        name: str,
        value: Any | None = None,
        resolver: Callable[[], Any] | None = None,
    ) -> None:
        self.name = name
        self._value = value
        self._resolver = resolver

    def resolve(self) -> Any:
        if self._value is None:
            self._value = self._resolver() if self._resolver is not None else getattr(eegprep, self.name)
        return self._value

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.resolve()(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.resolve(), name)

    def __repr__(self) -> str:
        return f"<EEGPrep export {self.name}>"


class ConsolePopFunction(LazyWorkspaceExport):
    """Console wrapper that stores ``pop_*`` EEG outputs back into the session."""

    def __init__(
        self,
        name: str,
        bridge: EEGPrepConsoleWorkspace,
        value: Any | None = None,
        resolver: Callable[[], Any] | None = None,
    ) -> None:
        super().__init__(name, value=value, resolver=resolver)
        self.bridge = bridge

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        function = self.resolve()
        call_kwargs = dict(kwargs)
        recorded_commands: set[str] = set()
        if self.name == "pop_eegplot" and "command_callback" not in call_kwargs:
            source_eeg = _first_call_eeg_argument(args, call_kwargs)
            reject = _pop_eegplot_reject_argument(function, args, call_kwargs)
            target_index = list(self.bridge.session.CURRENTSET)

            def accept_eegplot(eeg_out: Any, command: str) -> None:
                if not isinstance(source_eeg, dict) or not isinstance(eeg_out, dict):
                    return
                store_new = eegplot_accept_creates_dataset(source_eeg, eeg_out, reject)
                store_command = "" if command in recorded_commands else command
                recorded_commands.add(command)
                store_index = None if store_new else target_index
                self.bridge._store_eeg(eeg_out, store_command, new=store_new, index=store_index)
                self.bridge.pull_from_session()
                self.bridge._refresh()

            call_kwargs["command_callback"] = accept_eegplot
        elif self.name in _BROWSER_ACCEPT_POP_FUNCTIONS and "command_callback" not in call_kwargs:
            source_eeg = _first_call_eeg_argument(args, call_kwargs)
            target_index = list(self.bridge.session.CURRENTSET)

            def accept_browser_result(eeg_out: Any, command: str) -> None:
                if not isinstance(source_eeg, dict) or not isinstance(eeg_out, dict):
                    return
                store_new = eegplot_accept_creates_dataset(source_eeg, eeg_out, reject=1)
                store_command = "" if command in recorded_commands else command
                recorded_commands.add(command)
                store_index = None if store_new else target_index
                self.bridge._store_eeg(eeg_out, store_command, new=store_new, index=store_index)
                self.bridge.pull_from_session()
                self.bridge._refresh()

            call_kwargs["command_callback"] = accept_browser_result
        if _accepts_return_com(function) and "return_com" not in call_kwargs:
            call_kwargs["return_com"] = True
        result = function(*args, **call_kwargs)
        if self.name == "pop_eegplot" or self.name in _BROWSER_ACCEPT_POP_FUNCTIONS:
            _eeg, command = _extract_pop_eeg_and_command(result)
            if command:
                recorded_commands.add(command)
            if self.name == "pop_eegplot" and _eeg is None and command:
                self.bridge.session.add_history(command)
                self.bridge._pop_updated_session = True
                self.bridge.pull_from_session()
                return ConsolePopResult(self.bridge.session.EEG, command, updated=False)
        return self.bridge.accept_pop_result(result, args, kwargs)

    def __repr__(self) -> str:
        return f"<EEGPrep console pop function {self.name}>"


class ConsoleEEGPrepModule:
    """Console-local proxy that wraps ``eegprep.pop_*`` calls."""

    def __init__(self, bridge: EEGPrepConsoleWorkspace) -> None:
        self._bridge = bridge

    def __getattr__(self, name: str) -> Any:
        if name.startswith("pop_"):
            return self._bridge.pop_wrapper(name)
        if name == "eegh":
            return self._bridge.namespace["eegh"]
        return getattr(eegprep, name)

    def __dir__(self) -> list[str]:
        return sorted(set(dir(eegprep)) | set(self._bridge.namespace))

    def __repr__(self) -> str:
        return "<EEGPrep console module proxy>"


class ConsoleEegh:
    """Session-bound ``eegh`` command for the EEGPrep console."""

    def __init__(self, bridge: EEGPrepConsoleWorkspace) -> None:
        self.bridge = bridge

    def __call__(self, command: Any = None, *args: Any) -> str:
        if command is None:
            return eegh(None, self.bridge.session.ALLCOM)
        if isinstance(command, str) and command.strip().lower() == "find":
            text = "" if not args else str(args[0])
            return eegh_find(self.bridge.session.ALLCOM, text)
        if isinstance(command, (int, float)) and not isinstance(command, bool):
            value = int(command)
            if float(command) != value:
                raise ValueError("eegh numeric command must be an integer")
            session = self.bridge.session
            if value == 0:
                session.clear_history()
                self.bridge.pull_from_session()
                return ""
            if value < 0:
                session.remove_history(abs(value))
                self.bridge.pull_from_session()
                return ""
            history_command = session.history_command_at(value)
            if history_command:
                self.bridge.execute_history_command(history_command)
            return history_command
        session = self.bridge.session
        normalized = str(command).strip()
        if normalized:
            session.add_history(command)
        else:
            session.clear_last_command()
        if args and isinstance(args[0], dict):
            eegh(normalized, args[0])
        self.bridge.pull_from_session()
        return normalized

    def __repr__(self) -> str:
        return "<EEGPrep console eegh>"


class EEGPrepConsoleWorkspace:
    """Synchronize an IPython namespace with an :class:`EEGPrepSession`."""

    def __init__(
        self,
        session: EEGPrepSession,
        *,
        window: Any | None = None,
        refresh: Callable[[], None] | None = None,
        command_echo: Callable[[str], Any] | None = None,
        exports: Mapping[str, Any] | None = None,
        extension_runtime: ExtensionRuntime | None = None,
    ) -> None:
        self.session = session
        self.window = window
        self.refresh = refresh or getattr(window, "refresh", None)
        self.command_echo = command_echo
        self.extension_runtime = extension_runtime or ExtensionRuntime.empty()
        self.namespace: dict[str, Any] = {}
        self._eegprep_proxy = ConsoleEEGPrepModule(self)
        self._wrapped_pop_exports: dict[str, ConsolePopFunction] = {}
        self._syncing = False
        self._pop_updated_session = False
        self._pop_needs_source_history = False
        self._gui_action_depth = 0
        self._terminal_buffer: _TerminalOutputBuffer | None = None
        self._terminal_buffer_token: contextvars.Token | None = None
        self._bind_base_namespace()
        self._bind_exports(exports)
        self._bind_extension_exports()
        self.pull_from_session()
        self.session.add_change_listener(self._session_changed)
        self.session.add_command_echo_listener(self._echo_session_command)
        self.session.add_gui_action_listener(self._gui_action_event)

    def close(self) -> None:
        """Detach this workspace from session notifications."""
        self.session.remove_change_listener(self._session_changed)
        self.session.remove_command_echo_listener(self._echo_session_command)
        self.session.remove_gui_action_listener(self._gui_action_event)
        self._finish_gui_action_output()

    def sync_state(self) -> None:
        """Force the workspace to push its state to the session and refresh the GUI."""
        eeg = self.namespace.get("EEG")
        if _is_eeg_selection(eeg):
            self._store_eeg(eeg, "")
        self.pull_from_session()
        self._refresh()

    def pull_from_session(self) -> None:
        """Mirror session state into the console namespace."""
        self.namespace["eegprep"] = self._eegprep_proxy
        self.namespace.update(self._wrapped_pop_exports)
        self.namespace["EEG"] = self.session.EEG
        self.namespace["ALLEEG"] = self.session.ALLEEG
        self.namespace["CURRENTSET"] = self.session.current_set_value()
        self.namespace["ALLCOM"] = self.session.ALLCOM
        self.namespace["LASTCOM"] = self.session.LASTCOM
        self.namespace["STUDY"] = self.session.STUDY
        self.namespace["CURRENTSTUDY"] = self.session.CURRENTSTUDY
        self.namespace["refresh"] = self.sync_state

    def after_execute(self, source: str, *, success: bool = True) -> None:
        """Push console-side workspace edits back into the session."""
        if not success:
            self.pull_from_session()
            return
        self._restore_imported_wrappers(source)
        if self._pop_updated_session:
            self._pop_updated_session = False
            if self._pop_needs_source_history:
                self._pop_needs_source_history = False
                command = source.strip()
                if command:
                    self.session.add_history(command)
            self.pull_from_session()
            return

        targets = _workspace_assignment_targets(source)
        history_command = self._history_command_for_source(source, targets)
        changed = False

        if "ALLEEG" in targets or "CURRENTSET" in targets:
            alleeg = self.namespace.get("ALLEEG", [])
            if not isinstance(alleeg, list):
                raise ValueError("ALLEEG must be a list of EEG datasets")
            current = (
                _normalize_currentset(self.namespace.get("CURRENTSET"))
                if "CURRENTSET" in targets
                else self.session.CURRENTSET
            )
            self.session.apply_workspace_state(
                alleeg=alleeg, currentset=current, command="", append_dataset_history=False
            )
            changed = True

        if self._namespace_eeg_changed(targets):
            eeg = self.namespace.get("EEG")
            if not _is_eeg_selection(eeg):
                raise ValueError("EEG must be an EEG dataset dictionary or a list of EEG dataset dictionaries")
            self._store_eeg(eeg, history_command)
            changed = True
        elif "LASTCOM" in targets:
            command = str(self.namespace.get("LASTCOM") or "").strip()
            if command and command != self.session.LASTCOM:
                self.session.add_history(command)
                changed = True

        if "STUDY" in targets:
            study_kwargs: dict[str, Any] = {"study": self.namespace.get("STUDY"), "command": ""}
            if "CURRENTSTUDY" in targets:
                study_kwargs["currentstudy"] = self.namespace.get("CURRENTSTUDY")
            self.session.apply_workspace_state(**study_kwargs)
            changed = True
        elif "CURRENTSTUDY" in targets:
            self.session.apply_workspace_state(currentstudy=self.namespace.get("CURRENTSTUDY"), command="")
            changed = True

        if changed:
            if history_command and history_command != self.session.LASTCOM:
                self.session.add_history(history_command)
        self.pull_from_session()
        if changed:
            self._refresh()

    def accept_pop_result(self, result: Any, args: tuple[Any, ...], kwargs: Mapping[str, Any] | None = None) -> Any:
        """Store a ``pop_*`` result in the current session when appropriate."""
        dataset_state = _extract_pop_dataset_state(result)
        if dataset_state is not None:
            alleeg, eeg, currentset, command = dataset_state
            self.session.apply_workspace_state(
                alleeg=alleeg,
                eeg=eeg,
                currentset=currentset,
                command=command,
                append_dataset_history=False,
            )
            self._pop_updated_session = True
            self.pull_from_session()
            self._refresh()
            return ConsoleDatasetResult(
                self.session.ALLEEG, self.session.EEG, self.session.current_set_value(), command
            )

        study_state = _extract_pop_study_state(result)
        if study_state is not None:
            study, alleeg, command, includes_alleeg = study_state
            self.session.set_study(study, alleeg, command=command)
            self._pop_updated_session = True
            self.pull_from_session()
            self._refresh()
            return ConsoleStudyResult(study, alleeg, command, includes_alleeg=includes_alleeg)

        eeg, command = _extract_pop_eeg_and_command(result)
        if eeg is None:
            if command:
                self.session.add_history(command)
                self._pop_updated_session = True
            return result
        should_store = bool(command) or eeg is not self.session.EEG
        if not should_store:
            return ConsolePopResult(eeg, command, updated=False)
        input_eeg = _first_call_eeg_argument(args, kwargs or {})
        new_dataset = not self.session.CURRENTSET or input_eeg is not self.session.EEG
        self._store_eeg(eeg, command, new=new_dataset)
        self._pop_updated_session = True
        self._pop_needs_source_history = not bool(command)
        self.pull_from_session()
        self._refresh()
        return ConsolePopResult(self.session.EEG, command, updated=True)

    def _bind_base_namespace(self) -> None:
        self.namespace.update({"eegprep": self._eegprep_proxy, "session": self.session, "window": self.window})
        self.namespace.update(_CONSOLE_COMMAND_EXPORTS)
        self.namespace["eegh"] = ConsoleEegh(self)

    def _bind_exports(self, exports: Mapping[str, Any] | None) -> None:
        export_names = exports.keys() if exports is not None else eegprep.__all__
        for name in export_names:
            if name == "__version__":
                self.namespace[name] = eegprep.__version__
            elif name == "eegh":
                continue
            elif name.startswith("pop_"):
                wrapped = ConsolePopFunction(name, self, None if exports is None else exports[name])
                self._wrapped_pop_exports[name] = wrapped
                self.namespace[name] = wrapped
            else:
                self.namespace[name] = LazyWorkspaceExport(name, None if exports is None else exports[name])

    def _bind_extension_exports(self) -> None:
        for name, resolver in self.extension_runtime.pop_function_resolvers().items():
            if name in self._wrapped_pop_exports:
                continue
            wrapped = ConsolePopFunction(name, self, resolver=resolver)
            self._wrapped_pop_exports[name] = wrapped
            self.namespace[name] = wrapped

    def pop_wrapper(self, name: str) -> ConsolePopFunction:
        """Return the console-aware wrapper for a public ``pop_*`` function."""
        wrapped = self._wrapped_pop_exports.get(name)
        if wrapped is None:
            pop_function = self.extension_runtime.pop_function(name)
            resolver = pop_function.load if pop_function is not None else None
            wrapped = ConsolePopFunction(name, self, resolver=resolver)
            self._wrapped_pop_exports[name] = wrapped
        return wrapped

    def _session_changed(self, _session: EEGPrepSession) -> None:
        if not self._syncing:
            self.pull_from_session()

    def _namespace_eeg_changed(self, targets: set[str]) -> bool:
        return "EEG" in targets or self.namespace.get("EEG") is not self.session.EEG

    def _history_command_for_source(self, source: str, targets: set[str]) -> str:
        lastcom = str(self.namespace.get("LASTCOM") or "").strip()
        if lastcom and lastcom != self.session.LASTCOM:
            return lastcom
        if targets:
            return source.strip()
        return ""

    def _restore_imported_wrappers(self, source: str) -> None:
        for local_name, export_name in _eegprep_import_aliases(source).items():
            if export_name == "eegprep":
                self.namespace[local_name] = self._eegprep_proxy
            elif export_name.startswith("pop_"):
                self.namespace[local_name] = self.pop_wrapper(export_name)

    def _store_eeg(self, eeg: Any, command: str, *, new: bool = False, index: int | list[int] | None = None) -> None:
        self._syncing = True
        try:
            self.session.store_current(eeg, new=new, command=command, index=index)
        finally:
            self._syncing = False

    def execute_history_command(self, command: str) -> None:
        """Execute an EEGLAB history command through the console namespace."""
        source = _console_python_command(command)
        exec(source, self.namespace)
        self.after_execute(source)

    def _refresh(self) -> None:
        if self.refresh is not None:
            self.refresh()

    def _echo_session_command(self, command: str) -> None:
        if self.command_echo is not None:
            self.command_echo(command)
        self._release_gui_action_output()

    def _gui_action_event(self, event: str, _action: str) -> None:
        if event == "begin":
            self._gui_action_depth += 1
            if self._gui_action_depth == 1:
                self._terminal_buffer = _TerminalOutputBuffer()
                self._terminal_buffer_token = _ACTIVE_TERMINAL_BUFFER.set(self._terminal_buffer)
            return
        if event == "end":
            self._gui_action_depth = max(0, self._gui_action_depth - 1)
            if self._gui_action_depth == 0:
                self._finish_gui_action_output()

    def _release_gui_action_output(self) -> None:
        if self._terminal_buffer is not None:
            self._terminal_buffer.release(sync=True)

    def _finish_gui_action_output(self) -> None:
        buffer = self._terminal_buffer
        token = self._terminal_buffer_token
        self._terminal_buffer = None
        self._terminal_buffer_token = None
        if token is not None and _active_terminal_buffer() is buffer:
            _ACTIVE_TERMINAL_BUFFER.reset(token)
        if buffer is not None:
            buffer.release(sync=False)


def run_console(
    argv: list[str] | None = None,
    *,
    shell_factory: Callable[[dict[str, Any], str], Any] | None = None,
    gui_launcher: Callable[..., Any] = gui,
) -> int:
    """Launch the EEGPrep GUI plus an IPython workspace console."""
    parser = argparse.ArgumentParser(description="Launch the EEGPrep GUI with a synchronized Python console.")
    parser.add_argument("--full", action="store_true", help="Show legacy/advanced menu items")
    parser.add_argument("--no-plugins", action="store_true", help="Hide plugin-contributed menu items")
    parser.add_argument(
        "--window-menu-bar",
        action="store_true",
        help="Keep menus inside the EEGPrep window instead of using the native macOS menu bar",
    )
    parser.add_argument(
        "--native-file-dialogs",
        action="store_true",
        help="Use platform-native file pickers instead of the stable Qt dialogs used by default in the console",
    )
    args = parser.parse_args(argv)

    session = EEGPrepSession()
    try:
        window = gui_launcher(
            "full" if args.full else None,
            session=session,
            block=False,
            include_plugins=not args.no_plugins,
            native_menu_bar=False if args.window_menu_bar else None,
            # Native macOS QFileDialog can close immediately under IPython's Qt
            # input hook; keep the console default stable but allow opt-in.
            native_file_dialogs=args.native_file_dialogs,
        )
    except RuntimeError as exc:
        if "PySide6" in str(exc):
            raise RuntimeError(
                "PySide6 is required for eegprep-console. Install it with "
                "`pip install eegprep[console]` or `uv sync --extra console`."
            ) from exc
        raise

    extension_runtime = getattr(window, "extension_runtime", None)
    if extension_runtime is None:
        extension_runtime = ExtensionRuntime.discover(include_plugins=not args.no_plugins)
    workspace = EEGPrepConsoleWorkspace(session, window=window, extension_runtime=extension_runtime)
    banner = _console_banner()
    shell = (
        _IPythonShellAdapter(shell_factory(workspace.namespace, banner), workspace)
        if shell_factory is not None
        else _IPythonShellAdapter(_ipython_shell_factory(workspace.namespace, banner), workspace)
    )
    workspace.command_echo = shell.echo_gui_command
    try:
        shell()
    finally:
        workspace.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    """Console-script entry point for ``eegprep-console``."""
    return run_console(argv)


def _ipython_shell_factory(namespace: dict[str, Any], banner: str) -> Any:
    try:
        module = importlib.import_module("IPython.terminal.embed")
    except ImportError as exc:
        raise RuntimeError(
            "IPython is required for eegprep-console. Install it with "
            "`pip install eegprep[console]` or `uv sync --extra console`."
        ) from exc
    return module.InteractiveShellEmbed(user_ns=namespace, banner1=banner, exit_msg="Leaving EEGPrep console.")


class _IPythonShellAdapter:
    def __init__(self, shell: Any, workspace: EEGPrepConsoleWorkspace) -> None:
        self.shell = shell
        self.workspace = workspace

    def __call__(self) -> None:
        _make_shell_prompt_dynamic(self.shell)
        restore_logging = _install_prompt_safe_logging()
        restore_progress_logging = _install_console_progress_logging()
        self.shell.enable_gui("qt")

        def post_run_cell(result: Any) -> None:
            raw_cell = getattr(getattr(result, "info", None), "raw_cell", "")
            success = bool(getattr(result, "success", True))
            _safe_after_execute(self.workspace, raw_cell, success=success, write=sys.stderr.write)

        self.shell.events.register("post_run_cell", post_run_cell)
        if hasattr(self.shell, "ast_transformers"):
            self.shell.ast_transformers.append(_PlotSyncInjector())
        try:
            self.shell()
        finally:
            restore_progress_logging()
            restore_logging()

    def echo_gui_command(self, command: str) -> None:
        """Record and display a GUI history command as console input."""
        line_number = int(getattr(self.shell, "execution_count", 1) or 1)
        console_command = _console_python_command(command)
        history_manager = getattr(self.shell, "history_manager", None)
        if history_manager is not None:
            history_manager.store_inputs(line_number, console_command, console_command)
        self.shell.execution_count = line_number + 1
        _terminal_write(_format_ipython_input(console_command, line_number), stream=sys.stderr, sync=True)


def _console_banner() -> str:
    return (
        "EEGPrep interactive console\n"
        "The GUI and these workspace names share one session: EEG, ALLEEG, CURRENTSET, ALLCOM, LASTCOM, STUDY, CURRENTSTUDY.\n"
        "Call pop_* functions directly, for example: pop_reref(EEG, [])\n"
    )


def _format_ipython_input(command: str, line_number: int) -> str:
    return f"In [{line_number}]: {command.rstrip()}\n"


def _console_python_command(command: str) -> str:
    """Convert EEGLAB-style history text to pasteable Python console input."""
    text = command.strip()
    if text.endswith(";"):
        text = text[:-1].rstrip()
    statements = _split_history_statements(text)
    if len(statements) > 1:
        return "; ".join(_console_python_command(statement) for statement in statements)
    text = _matlab_strings_to_python(text)
    text = _matlab_multi_assign_to_python(text)
    text = _matlab_cells_to_python_lists(text)
    text = _matlab_vectors_to_python_lists(text)
    text = _pythonize_known_pop_arguments(text)
    text = _keywordize_console_pop_calls(text)
    return _normalise_python_command(text)


def _split_history_statements(text: str) -> list[str]:
    statements = []
    start = 0
    depth = 0
    index = 0
    while index < len(text):
        char = text[index]
        if char in {"'", '"'}:
            index = _string_end(text, index)
            continue
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(0, depth - 1)
        elif char == ";" and depth == 0:
            statement = text[start:index].strip()
            if statement:
                statements.append(statement)
            start = index + 1
        index += 1
    statement = text[start:].strip()
    if statement:
        statements.append(statement)
    return statements or [text]


def _matlab_strings_to_python(text: str) -> str:
    output = []
    index = 0
    while index < len(text):
        char = text[index]
        if char != "'":
            output.append(char)
            index += 1
            continue
        index += 1
        value = []
        while index < len(text):
            char = text[index]
            if char != "'":
                value.append(char)
                index += 1
                continue
            if index + 1 < len(text) and text[index + 1] == "'":
                value.append("'")
                index += 2
                continue
            index += 1
            break
        output.append(repr("".join(value)))
    return "".join(output)


def _matlab_multi_assign_to_python(text: str) -> str:
    match = _MATLAB_MULTI_ASSIGN_PATTERN.match(text)
    if not match:
        return text
    names = ", ".join(match.group(1).split())
    return f"{names} =" + text[match.end() :]


def _matlab_cells_to_python_lists(text: str) -> str:
    return _replace_matlab_delimited_lists(text, "{", "}")


def _matlab_vectors_to_python_lists(text: str) -> str:
    return _replace_matlab_delimited_lists(text, "[", "]")


def _replace_matlab_delimited_lists(text: str, open_char: str, close_char: str) -> str:
    output = []
    index = 0
    while index < len(text):
        char = text[index]
        if char in {"'", '"'}:
            end = _string_end(text, index)
            output.append(text[index:end])
            index = end
            continue
        if char != open_char:
            output.append(char)
            index += 1
            continue
        end = _delimiter_end(text, index, open_char, close_char)
        if end is None:
            output.append(char)
            index += 1
            continue
        original = text[index : end + 1]
        content = text[index + 1 : end]
        output.append(_format_matlab_list_content(content, original))
        index = end + 1
    return "".join(output)


def _format_matlab_list_content(content: str, original: str) -> str:
    content = content.strip()
    if not content:
        return "[]"
    rows = [row.strip() for row in content.split(";")]
    if not all(_is_matlab_list_row(row) for row in rows):
        return original
    formatted_rows = [_comma_join_matlab_row(row) for row in rows]
    if len(formatted_rows) == 1:
        return f"[{formatted_rows[0]}]"
    return "[" + ", ".join(f"[{row}]" for row in formatted_rows) + "]"


def _delimiter_end(text: str, start: int, open_char: str, close_char: str) -> int | None:
    depth = 0
    index = start
    while index < len(text):
        char = text[index]
        if char in {"'", '"'}:
            index = _string_end(text, index)
            continue
        if char == open_char:
            depth += 1
        elif char == close_char:
            depth -= 1
            if depth == 0:
                return index
        index += 1
    return None


def _string_end(text: str, start: int) -> int:
    quote = text[start]
    index = start + 1
    escaped = False
    while index < len(text):
        char = text[index]
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == quote:
            return index + 1
        index += 1
    return len(text)


def _is_matlab_list_row(row: str) -> bool:
    if not row:
        return True
    return all(_is_python_literal_token(token) for token in row.replace(",", " ").split())


def _comma_join_matlab_row(row: str) -> str:
    return ", ".join(row.replace(",", " ").split())


def _is_python_literal_token(token: str) -> bool:
    if not token:
        return False
    if token in {"True", "False", "None"}:
        return True
    try:
        float(token)
    except ValueError:
        return token.startswith(("'", '"')) and token.endswith(("'", '"'))
    return True


def _replace_outside_strings(text: str, pattern: re.Pattern[str], replacement: Callable[[re.Match[str]], str]) -> str:
    output = []
    cursor = 0
    for start, end in _string_spans(text):
        output.append(pattern.sub(replacement, text[cursor:start]))
        output.append(text[start:end])
        cursor = end
    output.append(pattern.sub(replacement, text[cursor:]))
    return "".join(output)


def _string_spans(text: str) -> list[tuple[int, int]]:
    spans = []
    index = 0
    while index < len(text):
        if text[index] not in {"'", '"'}:
            index += 1
            continue
        start = index
        index = _string_end(text, index)
        spans.append((start, index))
    return spans


def _pythonize_known_pop_arguments(text: str) -> str:
    text = _pythonize_pop_interp_channels(text)
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return text
    converter = _ConsoleCommandArgumentConverter()
    tree = converter.visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree) if converter.changed else text


def _pythonize_pop_interp_channels(text: str) -> str:
    def replacement(match: re.Match[str]) -> str:
        channels = [int(token) - 1 for token in match.group(2).replace(",", " ").split()]
        return f"{match.group(1)}{channels}"

    return _replace_outside_strings(text, _POP_INTERP_CHANNELS_PATTERN, replacement)


def _keywordize_console_pop_calls(text: str) -> str:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return text
    converter = _ConsoleCommandKeywordizer()
    tree = converter.visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree) if converter.changed else text


def _normalise_python_command(text: str) -> str:
    try:
        return _normalise_tuple_assignment_targets(ast.unparse(ast.parse(text)).strip())
    except SyntaxError:
        return text


def _normalise_tuple_assignment_targets(text: str) -> str:
    return _TUPLE_ASSIGNMENT_TARGET_PATTERN.sub(lambda match: f"{match.group(1)}{match.group(2)} =", text)


class _PlotSyncInjector(ast.NodeTransformer):
    def visit_Expr(self, node: ast.Expr) -> Any:
        self.generic_visit(node)
        return self._inject_if_needed(node)

    def visit_Assign(self, node: ast.Assign) -> Any:
        self.generic_visit(node)
        return self._inject_if_needed(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> Any:
        self.generic_visit(node)
        return self._inject_if_needed(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> Any:
        self.generic_visit(node)
        return self._inject_if_needed(node)

    def _inject_if_needed(self, node: ast.stmt) -> list[ast.stmt] | ast.stmt:
        has_ui_call = False
        for child in ast.walk(node):
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                if child.func.id.startswith("pop_") or child.func.id in {"eegplot", "eeg_browser", "eegbrowser"}:
                    has_ui_call = True
                    break
        if has_ui_call:
            refresh_call = ast.Expr(
                value=ast.Call(
                    func=ast.Name(id="refresh", ctx=ast.Load()),
                    args=[],
                    keywords=[],
                )
            )
            ast.copy_location(refresh_call, node)
            return [refresh_call, node]
        return node


class _ConsoleCommandArgumentConverter(ast.NodeTransformer):
    def __init__(self) -> None:
        self.changed = False

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if not isinstance(node.func, ast.Name):
            return node
        if node.func.id == "pop_reref":
            self._convert_pop_reref(node)
        elif node.func.id == "pop_select":
            self._convert_pop_select(node)
        return node

    def _convert_pop_reref(self, node: ast.Call) -> None:
        if len(node.args) >= 2:
            node.args[1] = self._zero_base_channel_arg(node.args[1])
        for index in range(2, len(node.args) - 1, 2):
            key = self._string_constant(node.args[index])
            if key in {"exclude", "interpchan"}:
                node.args[index + 1] = self._zero_base_channel_arg(node.args[index + 1])

    def _convert_pop_select(self, node: ast.Call) -> None:
        # pop_select history is name/value pairs after EEG (no positional
        # selection arg). Channel selections are 1-based in EEGLAB history but
        # the Python API is 0-based, so zero-base the numeric channel lists to
        # match pop_reref; channel-by-name and chantype selections pass through.
        for index in range(1, len(node.args) - 1, 2):
            key = self._string_constant(node.args[index])
            if key in {"channel", "nochannel", "rmchannel"}:
                node.args[index + 1] = self._zero_base_channel_arg(node.args[index + 1])

    def _zero_base_channel_arg(self, node: ast.AST) -> ast.AST:
        if isinstance(node, ast.List):
            values = [_numeric_ast_constant_value(item) for item in node.elts]
            if not all(value is not None for value in values):
                return node
            self.changed = True
            return ast.List(
                elts=[ast.Constant(value=int(float(value)) - 1) for value in values],
                ctx=node.ctx,
            )
        value = _numeric_ast_constant_value(node)
        if value is not None:
            self.changed = True
            return ast.Constant(value=int(float(value)) - 1)
        return node

    @staticmethod
    def _string_constant(node: ast.AST) -> str | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value.lower()
        return None


class _ConsoleCommandKeywordizer(ast.NodeTransformer):
    def __init__(self) -> None:
        self.changed = False

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        function_name = _console_call_name(node.func)
        if function_name is None or not function_name.startswith("pop_"):
            return node
        function = _resolve_eegprep_callable(function_name)
        if function is None:
            return node
        return self._keywordize_call(node, function)

    def _keywordize_call(self, node: ast.Call, function: Callable[..., Any]) -> ast.Call:
        try:
            parameters = list(inspect.signature(function).parameters.values())
        except (TypeError, ValueError):
            return node
        positional_parameters = [
            parameter
            for parameter in parameters
            if parameter.kind in {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}
        ]
        accepts_var_keywords = any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters)
        keyword_parameters = {
            parameter.name
            for parameter in parameters
            if parameter.kind in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
        }
        existing_keywords = {keyword.arg for keyword in node.keywords if keyword.arg is not None}
        kept_args: list[ast.expr] = []
        converted_keywords: list[ast.keyword] = []
        arg_index = 0
        while arg_index < len(node.args) and arg_index < len(positional_parameters):
            arg = node.args[arg_index]
            parameter = positional_parameters[arg_index]
            if (
                parameter.kind == inspect.Parameter.POSITIONAL_ONLY
                or parameter.name in existing_keywords
                or _is_workspace_argument(parameter.name, arg)
            ):
                kept_args.append(arg)
            else:
                converted_keywords.append(ast.keyword(arg=parameter.name, value=arg))
            arg_index += 1

        option_keywords, leftover_args = _keywordize_option_pairs(
            node.args[arg_index:],
            accepts_var_keywords=accepts_var_keywords,
            keyword_parameters=keyword_parameters,
            existing_keywords=existing_keywords | {keyword.arg for keyword in converted_keywords},
        )
        if leftover_args and (converted_keywords or option_keywords):
            return node
        if not converted_keywords and not option_keywords:
            return node
        self.changed = True
        node.args = kept_args + leftover_args
        node.keywords = converted_keywords + option_keywords + node.keywords
        return node


def _keywordize_option_pairs(
    args: list[ast.expr],
    *,
    accepts_var_keywords: bool,
    keyword_parameters: set[str],
    existing_keywords: set[str | None],
) -> tuple[list[ast.keyword], list[ast.expr]]:
    keywords: list[ast.keyword] = []
    index = 0
    while index + 1 < len(args):
        key = _option_pair_key(args[index])
        if (
            key is None
            or key in existing_keywords
            or not _can_pass_keyword(key, keyword_parameters, accepts_var_keywords)
        ):
            break
        keywords.append(ast.keyword(arg=key, value=args[index + 1]))
        existing_keywords.add(key)
        index += 2
    return keywords, args[index:]


def _option_pair_key(node: ast.expr) -> str | None:
    if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
        return None
    key = node.value.strip().lower()
    if not _PYTHON_IDENTIFIER_PATTERN.match(key):
        return None
    return key


def _can_pass_keyword(key: str, keyword_parameters: set[str], accepts_var_keywords: bool) -> bool:
    return accepts_var_keywords or key in keyword_parameters


def _is_workspace_argument(parameter_name: str, arg: ast.expr) -> bool:
    return (
        parameter_name.upper() in WORKSPACE_NAMES
        and isinstance(arg, ast.Name)
        and arg.id.upper() == parameter_name.upper()
    )


def _console_call_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "eegprep":
        return node.attr
    return None


def _resolve_eegprep_callable(name: str) -> Callable[..., Any] | None:
    try:
        value = getattr(eegprep, name)
    except AttributeError:
        value = _CONSOLE_COMMAND_EXPORTS.get(name)
        if value is None:
            return None
    return value if callable(value) else None


def _numeric_ast_constant_value(node: ast.AST) -> int | float | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    return None


def _make_shell_prompt_dynamic(shell: Any) -> None:
    if getattr(shell, "_eegprep_dynamic_prompt", False):
        return
    extra_prompt_options = getattr(shell, "_extra_prompt_options", None)
    prompts = getattr(shell, "prompts", None)
    if not callable(extra_prompt_options) or prompts is None:
        return
    try:
        formatted_text = importlib.import_module("prompt_toolkit.formatted_text")
    except ImportError:
        return

    def dynamic_extra_prompt_options(*args: Any, **kwargs: Any) -> dict[str, Any]:
        options = extra_prompt_options(*args, **kwargs)
        options["message"] = lambda: formatted_text.PygmentsTokens(prompts.in_prompt_tokens())
        return options

    shell._extra_prompt_options = dynamic_extra_prompt_options
    shell._eegprep_dynamic_prompt = True


class _PromptSafeStream:
    def __init__(self, stream: Any) -> None:
        self.stream = stream

    def write(self, message: str) -> int:
        if message:
            _terminal_write(message, stream=self.stream)
        return len(message)

    def flush(self) -> None:
        self.stream.flush()

    def isatty(self) -> bool:
        return bool(getattr(self.stream, "isatty", lambda: False)())

    def fileno(self) -> int:
        return self.stream.fileno()

    @property
    def encoding(self) -> str | None:
        return getattr(self.stream, "encoding", None)

    @property
    def errors(self) -> str | None:
        return getattr(self.stream, "errors", None)


def _install_prompt_safe_logging() -> Callable[[], None]:
    patched: list[tuple[logging.StreamHandler[Any], Any]] = []
    for logger in _logging_tree():
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler.stream, _PromptSafeStream):
                original_stream = handler.stream
                handler.setStream(_PromptSafeStream(original_stream))
                patched.append((handler, original_stream))
    original_emit = logging.StreamHandler.emit

    def emit(handler: logging.StreamHandler[Any], record: logging.LogRecord) -> None:
        if isinstance(handler, logging.FileHandler) or isinstance(handler.stream, _PromptSafeStream):
            original_emit(handler, record)
            return
        try:
            message = handler.format(record)
            _terminal_write(message + handler.terminator, stream=handler.stream)
            handler.flush()
        except RecursionError:
            raise
        except Exception:
            handler.handleError(record)

    logging.StreamHandler.emit = emit
    original_showwarning = warnings.showwarning

    def showwarning(
        message: Warning | str,
        category: type[Warning],
        filename: str,
        lineno: int,
        file: Any | None = None,
        line: str | None = None,
    ) -> None:
        output = file or sys.stderr
        formatted = warnings.formatwarning(message, category, filename, lineno, line)
        _terminal_write(formatted, stream=output)

    warnings.showwarning = showwarning

    def restore() -> None:
        logging.StreamHandler.emit = original_emit
        for handler, original_stream in patched:
            handler.setStream(original_stream)
        warnings.showwarning = original_showwarning

    return restore


def _install_console_progress_logging() -> Callable[[], None]:
    eegprep_logger = logging.getLogger("eegprep")
    original_level = eegprep_logger.level
    handler_levels = [(handler, handler.level) for handler in logging.getLogger().handlers]
    if original_level == logging.NOTSET or original_level > logging.INFO:
        eegprep_logger.setLevel(logging.INFO)
    for handler, _level in handler_levels:
        if handler.level > logging.INFO:
            handler.setLevel(logging.INFO)

    def restore() -> None:
        eegprep_logger.setLevel(original_level)
        for handler, level in handler_levels:
            handler.setLevel(level)

    return restore


def _logging_tree() -> list[logging.Logger]:
    loggers = [logging.getLogger()]
    for logger in logging.root.manager.loggerDict.values():
        if isinstance(logger, logging.Logger):
            loggers.append(logger)
    return loggers


def _terminal_write(message: str, *, stream: Any | None = None, sync: bool = False) -> None:
    output = stream or sys.stdout
    buffer = _active_terminal_buffer()
    if not sync and buffer is not None:
        if buffer.write(message, output):
            return
        _terminal_write(message, stream=output, sync=True)
        return
    if sync:
        # GUI previews have to appear before warnings raised by the same Qt callback.
        _write_terminal_direct(message, output)
        return
    if threading.current_thread() is not threading.main_thread():
        _write_terminal_direct(message, output)
        return
    try:
        run_module = importlib.import_module("prompt_toolkit.application.run_in_terminal")
    except ImportError:
        _write_terminal_direct(message, output)
        return

    def write_message() -> None:
        output.write(message)
        output.flush()

    run_module.run_in_terminal(write_message)


def _write_terminal_direct(message: str, output: Any) -> None:
    prefix = ANSI_CLEAR_LINE if _is_tty(output) else "\n"
    output.write(f"{prefix}{message}")
    output.flush()


def _is_tty(stream: Any) -> bool:
    return bool(getattr(stream, "isatty", lambda: False)())


class _TerminalOutputBuffer:
    def __init__(self) -> None:
        self.messages: list[tuple[str, Any]] = []
        self.released = False

    def write(self, message: str, stream: Any) -> bool:
        if self.released:
            return False
        self.messages.append((message, stream))
        return True

    def release(self, *, sync: bool) -> None:
        if self.released:
            return
        self.released = True
        for message, stream in self.messages:
            _terminal_write(message, stream=stream, sync=sync)
        self.messages.clear()


def _active_terminal_buffer() -> _TerminalOutputBuffer | None:
    return _ACTIVE_TERMINAL_BUFFER.get()


def _safe_after_execute(
    workspace: EEGPrepConsoleWorkspace,
    source: str,
    *,
    success: bool,
    write: Callable[[str], Any],
) -> None:
    try:
        workspace.after_execute(source, success=success)
    except Exception as exc:
        workspace.pull_from_session()
        write(f"EEGPrep workspace sync failed: {exc}\n")


def _accepts_return_com(function: Callable[..., Any]) -> bool:
    try:
        return "return_com" in inspect.signature(function).parameters
    except (TypeError, ValueError):
        return False


def _extract_pop_eeg_and_command(result: Any) -> tuple[Any | None, str]:
    if isinstance(result, tuple):
        command = str(result[1]).strip() if len(result) > 1 and isinstance(result[1], str) else ""
        if _history_only_command(command):
            return None, command
        if result and _is_eeg_selection(result[0]):
            return result[0], command
        return None, command
    if _is_eeg_selection(result):
        return result, ""
    if isinstance(result, str):
        return None, result.strip()
    return None, ""


def _extract_pop_dataset_state(result: Any) -> tuple[list[dict[str, Any]], Any, Any, str] | None:
    if not isinstance(result, tuple) or len(result) < 4:
        return None
    alleeg, eeg, currentset, command = result[0], result[1], result[2], result[3]
    if not isinstance(alleeg, list) or not _is_eeg_selection(eeg) or not isinstance(command, str):
        return None
    return alleeg, eeg, currentset, command.strip()


def _extract_pop_study_state(
    result: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]] | None, str, bool] | None:
    if not isinstance(result, tuple) or not result:
        return None
    study = result[0]
    if not _is_study_selection(study):
        return None
    if len(result) >= 3 and isinstance(result[1], list) and isinstance(result[2], str):
        return study, result[1], result[2].strip(), True
    if len(result) >= 2 and isinstance(result[1], str):
        return study, None, result[1].strip(), False
    return None


def _first_call_eeg_argument(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> Any | None:
    if args:
        return args[0]
    return kwargs.get("EEG")


def _pop_eegplot_reject_argument(function: Callable[..., Any], args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> int:
    bound = inspect.signature(function).bind_partial(*args, **kwargs)
    return int(bound.arguments.get("reject", 1))


def _is_eeg_selection(value: Any) -> bool:
    if isinstance(value, dict):
        return "data" in value and any(key in value for key in _EEG_CORE_FIELDS)
    return isinstance(value, list) and bool(value) and all(_is_eeg_selection(item) for item in value)


def _history_only_command(command: str) -> bool:
    return command.lstrip().startswith("LASTCOM")


def _is_study_selection(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and not _is_eeg_selection(value)
        and any(key in value for key in ("datasetinfo", "design", "cluster", "changrp", "currentdesign"))
    )


def _normalize_currentset(value: Any) -> list[int]:
    try:
        return normalize_dataset_indices(value)
    except ValueError as exc:
        raise ValueError("CURRENTSET must be a 1-based integer or list of integers") from exc


def _workspace_assignment_targets(source: str) -> set[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    targets: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            raw_targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in raw_targets:
                targets.update(root for root in _target_root_names(target) if root in WORKSPACE_NAMES)
        elif isinstance(node, ast.Delete):
            for target in node.targets:
                targets.update(root for root in _target_root_names(target) if root in WORKSPACE_NAMES)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr in _MUTATING_METHODS:
                targets.update(root for root in _target_root_names(node.func.value) if root in WORKSPACE_NAMES)
    return targets


def _eegprep_import_aliases(source: str) -> dict[str, str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "eegprep":
                    aliases[alias.asname or "eegprep"] = "eegprep"
        elif isinstance(node, ast.ImportFrom) and node.module == "eegprep":
            for alias in node.names:
                if alias.name.startswith("pop_"):
                    aliases[alias.asname or alias.name] = alias.name
    return aliases


def _target_root_names(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, (ast.Tuple, ast.List)):
        roots: set[str] = set()
        for item in node.elts:
            roots.update(_target_root_names(item))
        return roots
    if isinstance(node, ast.Subscript):
        return _target_root_names(node.value)
    if isinstance(node, ast.Attribute):
        return _target_root_names(node.value)
    return set()


if __name__ == "__main__":
    raise SystemExit(main())
