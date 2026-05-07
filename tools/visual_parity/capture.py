"""Capture EEGLAB and EEGPrep UI screenshots for visual parity review."""

from __future__ import annotations

import argparse
import os
import pathlib
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass

try:
    from .config import DEFAULT_MANIFEST, TargetSpec, VisualCase, format_command, load_manifest
except ImportError:  # pragma: no cover - supports direct script execution
    from config import DEFAULT_MANIFEST, TargetSpec, VisualCase, format_command, load_manifest


DEFAULT_OUTPUT_DIR = pathlib.Path(".visual-parity")
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class CaptureResult:
    """Result from one target capture command."""

    target: str
    output_path: pathlib.Path
    command: list[str]
    exit_code: int
    stdout: str = ""
    stderr: str = ""

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and self.output_path.exists()


def _matlab_string(value: pathlib.Path | str) -> str:
    text = str(value).replace("\\", "/").replace("'", "''")
    return f"'{text}'"


def _matlab_run_expression(script_path: pathlib.Path) -> str:
    script = _matlab_string(script_path.as_posix())
    return (
        f"try, run({script}); "
        "catch ME, disp(getReport(ME, 'extended')); exit(1); "
        "end; exit(0);"
    )


def _matlab_capture_helper() -> list[str]:
    return [
        "function write_figure_capture(fig, output_file)",
        "set(fig, 'Units', 'pixels');",
        "drawnow;",
        "pause(0.2);",
        "try",
        "    frame = getframe(fig);",
        "    imwrite(frame.cdata, output_file);",
        "catch getframe_error",
        "    try",
        "        pos = get(fig, 'Position');",
        "        screen_size = java.awt.Toolkit.getDefaultToolkit().getScreenSize();",
        "        x = max(0, round(pos(1)));",
        "        y = max(0, round(screen_size.getHeight() - pos(2) - pos(4)));",
        "        w = max(1, round(pos(3)));",
        "        h = max(1, round(pos(4)));",
        "        robot = java.awt.Robot;",
        "        rect = java.awt.Rectangle(x, y, w, h);",
        "        img = robot.createScreenCapture(rect);",
        "        javax.imageio.ImageIO.write(img, 'png', java.io.File(output_file));",
        "    catch robot_error",
        "        error('Figure capture failed. getframe: %s Robot: %s', getframe_error.message, robot_error.message);",
        "    end",
        "end",
        "end",
        "",
    ]


def _command_values(case: VisualCase, target_name: str, output_path: pathlib.Path) -> dict[str, str]:
    width, height = case.window_size
    return {
        "case_id": case.id,
        "target": target_name,
        "output": output_path.as_posix(),
        "output_dir": output_path.parent.as_posix(),
        "width": str(width),
        "height": str(height),
        "repo_root": REPO_ROOT.as_posix(),
        "python": sys.executable,
    }


def _output_path(output_dir: pathlib.Path, case_id: str, target_name: str) -> pathlib.Path:
    base_dir = output_dir if output_dir.is_absolute() else REPO_ROOT / output_dir
    return (base_dir / case_id / f"{target_name}.png").resolve()


def _target_env(case: VisualCase, target_name: str, target: TargetSpec, output_path: pathlib.Path) -> dict[str, str]:
    width, height = case.window_size
    env = os.environ.copy()
    src_path = (REPO_ROOT / "src").as_posix()
    env["PYTHONPATH"] = src_path + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.update(
        {
            "EEGPREP_VISUAL_CASE_ID": case.id,
            "EEGPREP_VISUAL_TARGET": target_name,
            "EEGPREP_VISUAL_ACTION": target.action,
            "EEGPREP_VISUAL_OUTPUT": output_path.as_posix(),
            "EEGPREP_VISUAL_OUTPUT_DIR": output_path.parent.as_posix(),
            "EEGPREP_VISUAL_WINDOW_WIDTH": str(width),
            "EEGPREP_VISUAL_WINDOW_HEIGHT": str(height),
            "EEGPREP_REPO_ROOT": REPO_ROOT.as_posix(),
        }
    )
    env.update(target.env)
    return env


def _write_matlab_figure_script(case: VisualCase, target: TargetSpec, output_path: pathlib.Path) -> pathlib.Path:
    width, height = case.window_size
    eeglab_root = REPO_ROOT / "src" / "eegprep" / "eeglab"
    script_path = output_path.parent / f"{case.id}_eeglab_capture.m"
    matlab_command = target.matlab_command.strip() or "eeglab;"
    script_path.write_text(
        "\n".join(
            [
                "function eegprep_visual_capture()",
                "try",
                f"output_file = {_matlab_string(output_path)};",
                f"eeglab_root = {_matlab_string(eeglab_root)};",
                "addpath(eeglab_root);",
                "set(0, 'DefaultFigureVisible', 'on');",
                matlab_command,
                "drawnow;",
                "pause(1);",
                "fig = findobj('tag', 'EEGLAB');",
                "if isempty(fig)",
                "    fig = gcf;",
                "else",
                "    fig = fig(1);",
                "end",
                f"set(fig, 'Units', 'pixels', 'Position', [100 100 {width} {height}]);",
                "drawnow;",
                "pause(0.5);",
                "write_figure_capture(fig, output_file);",
                "exit(0);",
                "catch ME",
                "disp(getReport(ME, 'extended'));",
                "exit(1);",
                "end",
                "end",
                *_matlab_capture_helper(),
                "eegprep_visual_capture();",
                "",
            ]
        )
    )
    return script_path


def _write_matlab_adjustevents_dialog_script(case: VisualCase, output_path: pathlib.Path) -> pathlib.Path:
    eeglab_root = REPO_ROOT / "src" / "eegprep" / "eeglab"
    script_path = output_path.parent / f"{case.id}_eeglab_capture.m"
    script_path.write_text(
        "\n".join(
            [
                "function eegprep_visual_capture()",
                "capture_timer = [];",
                "try",
                f"output_file = {_matlab_string(output_path)};",
                f"eeglab_root = {_matlab_string(eeglab_root)};",
                "addpath(genpath(eeglab_root));",
                "set(0, 'DefaultFigureVisible', 'on');",
                "EEG = struct();",
                "EEG.nbchan = 1;",
                "EEG.pnts = 1000;",
                "EEG.trials = 1;",
                "EEG.srate = 250;",
                "EEG.xmin = 0;",
                "EEG.xmax = (EEG.pnts-1)/EEG.srate;",
                "EEG.data = zeros(1, EEG.pnts);",
                "EEG.event = struct( ...",
                "    'type', {'stim', 'resp', 'boundary'}, ...",
                "    'latency', {100, 350, 500.5}, ...",
                "    'duration', {0, 0, 20});",
                "capture_timer = timer( ...",
                "    'ExecutionMode', 'fixedSpacing', ...",
                "    'Period', 0.5, ...",
                "    'StartDelay', 0.5, ...",
                "    'UserData', output_file, ...",
                "    'TimerFcn', @capture_pop_adjustevents_dialog);",
                "start(capture_timer);",
                "[EEG, com] = pop_adjustevents(EEG);",
                "if exist(output_file, 'file') ~= 2",
                "    error('visual parity capture did not create %s', output_file);",
                "end",
                "try, stop(capture_timer); delete(capture_timer); catch, end",
                "exit(0);",
                "catch ME",
                "try, if ~isempty(capture_timer), stop(capture_timer); delete(capture_timer); end, catch, end",
                "disp(getReport(ME, 'extended'));",
                "exit(1);",
                "end",
                "end",
                "",
                "function capture_pop_adjustevents_dialog(timer_obj, ~)",
                "output_file = get(timer_obj, 'UserData');",
                "fig = findall(0, 'Type', 'figure', 'Name', 'Adjust event latencies - pop_adjustevents()');",
                "if isempty(fig)",
                "    figs = findall(0, 'Type', 'figure');",
                "    for idx = 1:length(figs)",
                "        fig_name = get(figs(idx), 'Name');",
                "        if contains(fig_name, 'pop_adjustevents') || contains(fig_name, 'Adjust event')",
                "            fig = figs(idx);",
                "            break;",
                "        end",
                "    end",
                "end",
                "if isempty(fig), return; end",
                "fig = fig(1);",
                "ok_button = findobj('parent', fig, 'tag', 'ok');",
                "if isempty(ok_button), return; end",
                "set(fig, 'Units', 'pixels');",
                "drawnow;",
                "pause(0.2);",
                "write_figure_capture(fig, output_file);",
                "set(ok_button, 'userdata', 'retuninginputui');",
                "stop(timer_obj);",
                "delete(timer_obj);",
                "end",
                "",
                *_matlab_capture_helper(),
                "eegprep_visual_capture();",
                "",
            ]
        )
    )
    return script_path


def _write_matlab_reref_dialog_script(case: VisualCase, output_path: pathlib.Path) -> pathlib.Path:
    eeglab_root = REPO_ROOT / "src" / "eegprep" / "eeglab"
    script_path = output_path.parent / f"{case.id}_eeglab_capture.m"
    script_path.write_text(
        "\n".join(
            [
                "function eegprep_visual_capture()",
                "capture_timer = [];",
                "try",
                f"output_file = {_matlab_string(output_path)};",
                f"eeglab_root = {_matlab_string(eeglab_root)};",
                "addpath(genpath(eeglab_root));",
                "set(0, 'DefaultFigureVisible', 'on');",
                "EEG = eeg_emptyset;",
                "EEG.setname = 'reref demo';",
                "EEG.nbchan = 4;",
                "EEG.pnts = 1000;",
                "EEG.trials = 1;",
                "EEG.srate = 250;",
                "EEG.xmin = 0;",
                "EEG.xmax = (EEG.pnts-1)/EEG.srate;",
                "EEG.data = zeros(4, EEG.pnts);",
                "EEG.chanlocs = struct( ...",
                "    'labels', {'Fp1', 'Fp2', 'Cz', 'Oz'}, ...",
                "    'ref', {'common', 'common', 'common', 'common'}, ...",
                "    'theta', {-18, 18, 0, 180}, ...",
                "    'radius', {0.42, 0.42, 0, 0.42});",
                "EEG.chaninfo = struct();",
                "EEG.chaninfo.nodatchans = struct('labels', {'M1'}, 'theta', {-90}, 'radius', {0.5});",
                "EEG.ref = 'common';",
                "EEG.icaweights = [];",
                "EEG.icasphere = [];",
                "EEG.icawinv = [];",
                "EEG.icaact = [];",
                "EEG.icachansind = [];",
                "capture_timer = timer( ...",
                "    'ExecutionMode', 'fixedSpacing', ...",
                "    'Period', 0.5, ...",
                "    'StartDelay', 0.5, ...",
                "    'UserData', output_file, ...",
                "    'TimerFcn', @capture_pop_reref_dialog);",
                "start(capture_timer);",
                "[EEG, com] = pop_reref(EEG);",
                "if exist(output_file, 'file') ~= 2",
                "    error('visual parity capture did not create %s', output_file);",
                "end",
                "try, stop(capture_timer); delete(capture_timer); catch, end",
                "exit(0);",
                "catch ME",
                "try, if ~isempty(capture_timer), stop(capture_timer); delete(capture_timer); end, catch, end",
                "disp(getReport(ME, 'extended'));",
                "exit(1);",
                "end",
                "end",
                "",
                "function capture_pop_reref_dialog(timer_obj, ~)",
                "output_file = get(timer_obj, 'UserData');",
                "fig = findall(0, 'Type', 'figure', 'Name', 'pop_reref - average reference or re-reference data');",
                "if isempty(fig)",
                "    figs = findall(0, 'Type', 'figure');",
                "    for idx = 1:length(figs)",
                "        fig_name = get(figs(idx), 'Name');",
                "        if contains(fig_name, 'pop_reref') || contains(fig_name, 'reference')",
                "            fig = figs(idx);",
                "            break;",
                "        end",
                "    end",
                "end",
                "if isempty(fig), return; end",
                "fig = fig(1);",
                "ok_button = findobj('parent', fig, 'tag', 'ok');",
                "if isempty(ok_button), return; end",
                "set(fig, 'Units', 'pixels');",
                "drawnow;",
                "pause(0.2);",
                "write_figure_capture(fig, output_file);",
                "set(ok_button, 'userdata', 'retuninginputui');",
                "stop(timer_obj);",
                "delete(timer_obj);",
                "end",
                "",
                *_matlab_capture_helper(),
                "eegprep_visual_capture();",
                "",
            ]
        )
    )
    return script_path


def _run_subprocess(
    target_name: str,
    output_path: pathlib.Path,
    command: list[str],
    env: dict[str, str],
    timeout_seconds: int,
) -> CaptureResult:
    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except FileNotFoundError as exc:
        return CaptureResult(target_name, output_path, command, 127, stderr=str(exc))
    except subprocess.TimeoutExpired as exc:
        return CaptureResult(
            target_name,
            output_path,
            command,
            124,
            stdout=exc.stdout or "",
            stderr=f"capture timed out after {timeout_seconds} seconds",
        )
    return CaptureResult(
        target=target_name,
        output_path=output_path,
        command=command,
        exit_code=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def capture_target(
    case: VisualCase,
    target_name: str,
    output_dir: pathlib.Path = DEFAULT_OUTPUT_DIR,
    command_override: str | None = None,
    matlab_executable: str = "matlab",
    timeout_seconds: int | None = None,
) -> CaptureResult:
    """Capture one target for a visual parity case."""
    if target_name not in case.targets:
        raise ValueError(f"{case.id}: no {target_name} target is configured")

    target = case.targets[target_name]
    output_path = _output_path(output_dir, case.id, target_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    timeout = timeout_seconds or case.timeout_seconds
    env = _target_env(case, target_name, target, output_path)
    values = _command_values(case, target_name, output_path)
    values["action"] = target.action

    if command_override:
        command = [part.format(**values) for part in shlex.split(command_override)]
    elif target.type == "matlab_figure" and target_name == "eeglab":
        if shutil.which(matlab_executable) is None:
            return CaptureResult(
                target_name,
                output_path,
                [matlab_executable, "-nosplash", "-nodesktop", "-r", "<generated capture script>"],
                127,
                stderr=f"MATLAB executable not found: {matlab_executable}",
            )
        script_path = _write_matlab_figure_script(case, target, output_path)
        command = [matlab_executable, "-nosplash", "-nodesktop", "-r", _matlab_run_expression(script_path)]
    elif target.type == "matlab_dialog" and target_name == "eeglab":
        if target.action not in {"pop_adjustevents", "pop_reref"}:
            return CaptureResult(
                target_name,
                output_path,
                [],
                2,
                stderr=f"unsupported MATLAB dialog action: {target.action}",
            )
        if shutil.which(matlab_executable) is None:
            return CaptureResult(
                target_name,
                output_path,
                [matlab_executable, "-nosplash", "-nodesktop", "-r", "<generated dialog capture script>"],
                127,
                stderr=f"MATLAB executable not found: {matlab_executable}",
            )
        if target.action == "pop_adjustevents":
            script_path = _write_matlab_adjustevents_dialog_script(case, output_path)
        else:
            script_path = _write_matlab_reref_dialog_script(case, output_path)
        command = [matlab_executable, "-nosplash", "-nodesktop", "-r", _matlab_run_expression(script_path)]
    else:
        if not target.command:
            message = (
                f"{case.id}/{target_name} has no capture command. Add one to "
                f"{DEFAULT_MANIFEST} or pass --{target_name}-command."
            )
            return CaptureResult(target_name, output_path, [], 2, stderr=message)
        command = format_command(target.command, values)

    return _run_subprocess(target_name, output_path, command, env, timeout)


def capture_case(
    case: VisualCase,
    target: str,
    output_dir: pathlib.Path = DEFAULT_OUTPUT_DIR,
    eeglab_command: str | None = None,
    eegprep_command: str | None = None,
    matlab_executable: str = "matlab",
    timeout_seconds: int | None = None,
) -> list[CaptureResult]:
    """Capture one or both targets for a case."""
    target_names = ["eeglab", "eegprep"] if target == "both" else [target]
    overrides = {"eeglab": eeglab_command, "eegprep": eegprep_command}
    return [
        capture_target(
            case,
            target_name,
            output_dir=output_dir,
            command_override=overrides[target_name],
            matlab_executable=matlab_executable,
            timeout_seconds=timeout_seconds,
        )
        for target_name in target_names
    ]


def _print_result(result: CaptureResult) -> None:
    status = "ok" if result.ok else "FAIL"
    print(f"{result.target}: {status} -> {result.output_path}")
    if result.command:
        print("  command:", shlex.join(result.command))
    if result.stdout.strip():
        print("  stdout:")
        print(result.stdout.rstrip())
    if result.stderr.strip():
        print("  stderr:")
        print(result.stderr.rstrip())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=pathlib.Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--case", help="Visual parity case id")
    parser.add_argument("--target", choices=["eeglab", "eegprep", "both"], default="both")
    parser.add_argument("--list", action="store_true", help="List configured cases")
    parser.add_argument("--eeglab-command", help="Override command for the EEGLAB target")
    parser.add_argument("--eegprep-command", help="Override command for the EEGPrep target")
    parser.add_argument("--matlab-executable", default="matlab")
    parser.add_argument("--timeout", type=int, help="Override capture timeout in seconds")
    args = parser.parse_args(argv)

    cases = load_manifest(args.manifest)
    if args.list:
        for case in cases.values():
            print(f"{case.id}: {case.description}")
        return 0

    if not args.case:
        parser.error("--case is required unless --list is used")
    if args.case not in cases:
        parser.error(f"unknown case: {args.case}")

    results = capture_case(
        cases[args.case],
        args.target,
        output_dir=args.output_dir,
        eeglab_command=args.eeglab_command,
        eegprep_command=args.eegprep_command,
        matlab_executable=args.matlab_executable,
        timeout_seconds=args.timeout,
    )
    for result in results:
        _print_result(result)
    return 0 if all(result.ok for result in results) else 1


if __name__ == "__main__":
    sys.exit(main())
