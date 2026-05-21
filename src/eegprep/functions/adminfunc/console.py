"""Interactive EEGPrep console sharing state with the GUI."""

from __future__ import annotations

import argparse
import ast
import importlib
import inspect
import sys
from collections.abc import Callable, Iterator, Mapping
from typing import Any

import eegprep
from eegprep.functions.adminfunc.eeglab import gui
from eegprep.functions.guifunc.session import EEGPrepSession


WORKSPACE_NAMES = ("EEG", "ALLEEG", "CURRENTSET", "ALLCOM", "LASTCOM", "STUDY", "CURRENTSTUDY")
POP_RESULT_PREVIEW_LIMIT = 96


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


class LazyWorkspaceExport:
    """Lazy proxy for public EEGPrep exports in the console namespace."""

    def __init__(self, name: str, value: Any | None = None) -> None:
        self.name = name
        self._value = value

    def resolve(self) -> Any:
        if self._value is None:
            self._value = getattr(eegprep, self.name)
        return self._value

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.resolve()(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.resolve(), name)

    def __repr__(self) -> str:
        return f"<EEGPrep export {self.name}>"


class ConsolePopFunction(LazyWorkspaceExport):
    """Console wrapper that stores ``pop_*`` EEG outputs back into the session."""

    def __init__(self, name: str, bridge: EEGPrepConsoleWorkspace, value: Any | None = None) -> None:
        super().__init__(name, value=value)
        self.bridge = bridge

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        function = self.resolve()
        call_kwargs = dict(kwargs)
        if _accepts_return_com(function) and "return_com" not in call_kwargs:
            call_kwargs["return_com"] = True
        result = function(*args, **call_kwargs)
        return self.bridge.accept_pop_result(result, args)

    def __repr__(self) -> str:
        return f"<EEGPrep console pop function {self.name}>"


class ConsoleEEGPrepModule:
    """Console-local proxy that wraps ``eegprep.pop_*`` calls."""

    def __init__(self, bridge: EEGPrepConsoleWorkspace) -> None:
        self._bridge = bridge

    def __getattr__(self, name: str) -> Any:
        if name.startswith("pop_"):
            wrapped = self._bridge.namespace.get(name)
            if wrapped is None:
                wrapped = ConsolePopFunction(name, self._bridge)
            return wrapped
        return getattr(eegprep, name)

    def __dir__(self) -> list[str]:
        return sorted(set(dir(eegprep)) | set(self._bridge.namespace))

    def __repr__(self) -> str:
        return "<EEGPrep console module proxy>"


class EEGPrepConsoleWorkspace:
    """Synchronize an IPython namespace with an :class:`EEGPrepSession`."""

    def __init__(
        self,
        session: EEGPrepSession,
        *,
        window: Any | None = None,
        refresh: Callable[[], None] | None = None,
        exports: Mapping[str, Any] | None = None,
    ) -> None:
        self.session = session
        self.window = window
        self.refresh = refresh or getattr(window, "refresh", None)
        self.namespace: dict[str, Any] = {}
        self._eegprep_proxy = ConsoleEEGPrepModule(self)
        self._wrapped_pop_exports: dict[str, ConsolePopFunction] = {}
        self._syncing = False
        self._pop_updated_session = False
        self._bind_base_namespace()
        self._bind_exports(exports)
        self.pull_from_session()
        self.session.add_change_listener(self._session_changed)

    def close(self) -> None:
        """Detach this workspace from session notifications."""
        self.session.remove_change_listener(self._session_changed)

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

    def after_execute(self, source: str, *, success: bool = True) -> None:
        """Push console-side workspace edits back into the session."""
        if not success:
            self.pull_from_session()
            return
        if self._pop_updated_session:
            self._pop_updated_session = False
            self.pull_from_session()
            return

        targets = _workspace_assignment_targets(source)
        history_command = self._history_command_for_source(source, targets)
        changed = False

        if "ALLEEG" in targets:
            alleeg = self.namespace.get("ALLEEG", [])
            if not isinstance(alleeg, list):
                raise ValueError("ALLEEG must be a list of EEG datasets")
            self.session.ALLEEG = alleeg
            changed = True

        if "CURRENTSET" in targets:
            current = _normalize_currentset(self.namespace.get("CURRENTSET"))
            if current:
                self.session.retrieve(current if len(current) > 1 else current[0])
            else:
                self.session.CURRENTSET = []
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
            self.session.STUDY = self.namespace.get("STUDY")
            changed = True
        if "CURRENTSTUDY" in targets:
            self.session.CURRENTSTUDY = int(self.namespace.get("CURRENTSTUDY") or 0)
            changed = True

        if changed:
            self.session.notify_changed()
            if history_command and history_command != self.session.LASTCOM:
                self.session.add_history(history_command)
        self.pull_from_session()
        if changed:
            self._refresh()

    def accept_pop_result(self, result: Any, args: tuple[Any, ...]) -> Any:
        """Store a ``pop_*`` result in the current session when appropriate."""
        eeg, command = _extract_pop_eeg_and_command(result)
        if eeg is None:
            if command:
                self.session.add_history(command)
                self._pop_updated_session = True
            return result
        should_store = bool(command) or eeg is not self.session.EEG
        if not should_store:
            return ConsolePopResult(eeg, command, updated=False)
        new_dataset = not self.session.CURRENTSET or not args or args[0] is not self.session.EEG
        self._store_eeg(eeg, command, new=new_dataset)
        self._pop_updated_session = True
        self.pull_from_session()
        self._refresh()
        return ConsolePopResult(self.session.EEG, command, updated=True)

    def _bind_base_namespace(self) -> None:
        self.namespace.update({"eegprep": self._eegprep_proxy, "session": self.session, "window": self.window})

    def _bind_exports(self, exports: Mapping[str, Any] | None) -> None:
        export_names = exports.keys() if exports is not None else eegprep.__all__
        for name in export_names:
            if name == "__version__":
                self.namespace[name] = eegprep.__version__
            elif name.startswith("pop_"):
                wrapped = ConsolePopFunction(name, self, None if exports is None else exports[name])
                self._wrapped_pop_exports[name] = wrapped
                self.namespace[name] = wrapped
            else:
                self.namespace[name] = LazyWorkspaceExport(name, None if exports is None else exports[name])

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

    def _store_eeg(self, eeg: Any, command: str, *, new: bool = False) -> None:
        self._syncing = True
        try:
            self.session.store_current(eeg, new=new, command=command)
        finally:
            self._syncing = False

    def _refresh(self) -> None:
        if self.refresh is not None:
            self.refresh()


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
    args = parser.parse_args(argv)

    session = EEGPrepSession()
    try:
        window = gui_launcher(
            "full" if args.full else None,
            session=session,
            block=False,
            include_plugins=not args.no_plugins,
            native_menu_bar=False if args.window_menu_bar else None,
        )
    except RuntimeError as exc:
        if "PySide6" in str(exc):
            raise RuntimeError(
                "PySide6 is required for eegprep-console. Install it with "
                "`pip install eegprep[console]` or `uv sync --extra console`."
            ) from exc
        raise

    workspace = EEGPrepConsoleWorkspace(session, window=window)
    banner = _console_banner()
    shell = (
        _IPythonShellAdapter(shell_factory(workspace.namespace, banner), workspace)
        if shell_factory is not None
        else _IPythonShellAdapter(_ipython_shell_factory(workspace.namespace, banner), workspace)
    )
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
        self.shell.enable_gui("qt")

        def post_run_cell(result: Any) -> None:
            raw_cell = getattr(getattr(result, "info", None), "raw_cell", "")
            success = bool(getattr(result, "success", True))
            _safe_after_execute(self.workspace, raw_cell, success=success, write=sys.stderr.write)

        self.shell.events.register("post_run_cell", post_run_cell)
        self.shell()


def _console_banner() -> str:
    return (
        "EEGPrep interactive console\n"
        "The GUI and these workspace names share one session: EEG, ALLEEG, CURRENTSET, ALLCOM, LASTCOM, STUDY.\n"
        "Call pop_* functions directly, for example: pop_reref(EEG, [])"
    )


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
        if result and _is_eeg_selection(result[0]):
            return result[0], command
        return None, command
    if _is_eeg_selection(result):
        return result, ""
    if isinstance(result, str):
        return None, result.strip()
    return None, ""


def _is_eeg_selection(value: Any) -> bool:
    if isinstance(value, dict):
        return any(key in value for key in ("data", "nbchan", "setname"))
    return isinstance(value, list) and all(isinstance(item, dict) for item in value)


def _normalize_currentset(value: Any) -> list[int]:
    if value in (None, "", 0):
        return []
    if isinstance(value, (int, float)):
        return [int(value)] if int(value) > 0 else []
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        return [int(item) for item in value if int(item) > 0]
    raise ValueError("CURRENTSET must be a 1-based integer or list of integers")


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
    return targets


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
