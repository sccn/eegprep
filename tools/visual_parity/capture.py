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
EEGLAB_REFERENCE_ROOT = (
    REPO_ROOT.parent / "eeglab"
    if (REPO_ROOT.parent / "eeglab" / "eeglab.m").exists()
    else REPO_ROOT / "src" / "eegprep" / "eeglab"
)


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


def _split_action(action: str) -> tuple[str, str]:
    base, _separator, variant = action.partition(":")
    return base, variant or "default"


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
        "function write_figure_screen_capture(fig, output_file)",
        "set(fig, 'Units', 'pixels');",
        "drawnow;",
        "pause(0.2);",
        "try",
        "    pos = get(fig, 'Position');",
        "    screen_size = java.awt.Toolkit.getDefaultToolkit().getScreenSize();",
        "    x = max(0, round(pos(1)));",
        "    y = max(0, round(screen_size.getHeight() - pos(2) - pos(4)));",
        "    w = max(1, round(pos(3)));",
        "    h = max(1, round(pos(4)));",
        "    robot = java.awt.Robot;",
        "    rect = java.awt.Rectangle(x, y, w, h);",
        "    img = robot.createScreenCapture(rect);",
        "    javax.imageio.ImageIO.write(img, 'png', java.io.File(output_file));",
        "catch robot_error",
        "    error('Screen capture failed. Robot: %s', robot_error.message);",
        "end",
        "end",
        "",
        "function open_figure_menu(fig, menu_label)",
        "menu_handle = findobj(fig, 'Type', 'uimenu', 'Label', menu_label);",
        "if isempty(menu_handle)",
        "    error('Could not find menu label: %s', menu_label);",
        "end",
        "drawnow;",
        "pause(0.5);",
        "opened = false;",
        "try",
        "    frames = java.awt.Frame.getFrames();",
        "    for frame_idx = 1:length(frames)",
        "        frame = frames(frame_idx);",
        "        if ~frame.isShowing(), continue; end",
        "        menubar = find_java_menubar(frame);",
        "        if isempty(menubar), continue; end",
        "        for menu_idx = 0:menubar.getMenuCount()-1",
        "            java_menu = menubar.getMenu(menu_idx);",
        "            if isempty(java_menu), continue; end",
        "            if strcmp(char(java_menu.getText()), menu_label)",
        "                java_menu.doClick();",
        "                opened = true;",
        "                pause(0.6);",
        "                return;",
        "            end",
        "        end",
        "    end",
        "catch menu_error",
        "    warning('EEGPrepVisualParity:MenuOpen', 'Java menu open failed: %s', menu_error.message);",
        "end",
        "if ~opened && ismac",
        "    opened = open_macos_menu(menu_label);",
        "    if opened",
        "        pause(0.6);",
        "        return;",
        "    end",
        "end",
        "if ~opened",
        "    warning('EEGPrepVisualParity:MenuOpen', 'Menu was found but could not be opened: %s', menu_label);",
        "end",
        "end",
        "",
        "function menubar = find_java_menubar(component)",
        "menubar = [];",
        "try",
        "    if isa(component, 'javax.swing.JMenuBar')",
        "        menubar = component;",
        "        return;",
        "    end",
        "catch",
        "end",
        "try",
        "    candidate = component.getJMenuBar();",
        "    if ~isempty(candidate)",
        "        menubar = candidate;",
        "        return;",
        "    end",
        "catch",
        "end",
        "try",
        "    children = component.getComponents();",
        "catch",
        "    children = [];",
        "end",
        "for child_idx = 1:length(children)",
        "    menubar = find_java_menubar(children(child_idx));",
        "    if ~isempty(menubar), return; end",
        "end",
        "end",
        "",
        "function opened = open_macos_menu(menu_label)",
        "opened = false;",
        "script_file = [tempname '.applescript'];",
        "safe_label = strrep(menu_label, '\\', '\\\\');",
        "safe_label = strrep(safe_label, '\"', '\\\"');",
        "fid = fopen(script_file, 'w');",
        "if fid == -1, return; end",
        "fprintf(fid, 'set menuLabel to \"%s\"\\n', safe_label);",
        "fprintf(fid, 'tell application \"System Events\"\\n');",
        "fprintf(fid, '  set matlabProcesses to every process whose name contains \"MATLAB\"\\n');",
        "fprintf(fid, '  repeat with matlabProcess in matlabProcesses\\n');",
        "fprintf(fid, '    try\\n');",
        "fprintf(fid, '      tell matlabProcess\\n');",
        "fprintf(fid, '        set frontmost to true\\n');",
        "fprintf(fid, '        delay 0.2\\n');",
        "fprintf(fid, '        click menu bar item menuLabel of menu bar 1\\n');",
        "fprintf(fid, '      end tell\\n');",
        "fprintf(fid, '      return \"opened\"\\n');",
        "fprintf(fid, '    end try\\n');",
        "fprintf(fid, '  end repeat\\n');",
        "fprintf(fid, 'end tell\\n');",
        "fclose(fid);",
        "[status, result] = system(['osascript ' shell_quote(script_file)]);",
        "try, delete(script_file); catch, end",
        "if status == 0",
        "    opened = true;",
        "else",
        "    warning('EEGPrepVisualParity:MenuOpen', 'AppleScript menu open failed: %s', result);",
        "end",
        "end",
        "",
        "function write_macos_screen_capture(output_file)",
        "[status, result] = system(['screencapture -x ' shell_quote(output_file)]);",
        "if status ~= 0",
        "    error('macOS screen capture failed: %s', result);",
        "end",
        "end",
        "",
        "function quoted = shell_quote(value)",
        "quoted = ['''' strrep(value, '''', '''\"''\"''') ''''];",
        "end",
        "",
    ]


def _matlab_main_window_demo_helper() -> list[str]:
    return [
        "function make_menu_demo_dataset()",
        "global EEG ALLEEG CURRENTSET;",
        "EEG = eeg_emptyset;",
        "EEG.setname = 'menu demo';",
        "EEG.filename = 'menu_demo.set';",
        "EEG.filepath = tempdir;",
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
        "    'radius', {0.42, 0.42, 0, 0.42}, ...",
        "    'X', {-0.25, 0.25, 0, 0}, ...",
        "    'Y', {0.75, 0.75, 0, -0.8}, ...",
        "    'Z', {0.55, 0.55, 1, 0.55}, ...",
        "    'type', {'EEG', 'EEG', 'EEG', 'EEG'});",
        "EEG.chaninfo = struct();",
        "EEG.event = struct( ...",
        "    'type', {'stim', 'resp'}, ...",
        "    'latency', {100, 350}, ...",
        "    'duration', {0, 0});",
        "EEG.urevent = [];",
        "EEG.epoch = [];",
        "EEG.history = '';",
        "EEG.icaweights = eye(4);",
        "EEG.icasphere = eye(4);",
        "EEG.icawinv = eye(4);",
        "EEG.icachansind = 1:4;",
        "EEG.icaact = zeros(4, EEG.pnts);",
        "EEG = eeg_checkset(EEG, 'eventconsistency');",
        "[ALLEEG, EEG, CURRENTSET] = eeg_store(ALLEEG, EEG, 0);",
        "eeglab redraw;",
        "drawnow;",
        "pause(0.5);",
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
    eeglab_root = EEGLAB_REFERENCE_ROOT
    script_path = output_path.parent / f"{case.id}_eeglab_capture.m"
    matlab_command = target.matlab_command.strip() or "eeglab;"
    action, variant = _split_action(target.action)
    menu_label = variant if action == "open_menu" else ""
    script_path.write_text(
        "\n".join(
            [
                "function eegprep_visual_capture()",
                "try",
                f"output_file = {_matlab_string(output_path)};",
                f"menu_label = {_matlab_string(menu_label)};",
                f"eeglab_root = {_matlab_string(eeglab_root)};",
                "addpath(eeglab_root);",
                "set(0, 'DefaultFigureVisible', 'on');",
                matlab_command,
                "if ~isempty(menu_label)",
                "    make_menu_demo_dataset();",
                "end",
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
                "if ~isempty(menu_label)",
                "    open_figure_menu(fig, menu_label);",
                "    write_figure_screen_capture(fig, output_file);",
                "else",
                "    write_figure_capture(fig, output_file);",
                "end",
                "exit(0);",
                "catch ME",
                "disp(getReport(ME, 'extended'));",
                "exit(1);",
                "end",
                "end",
                *_matlab_capture_helper(),
                *_matlab_main_window_demo_helper(),
                "eegprep_visual_capture();",
                "",
            ]
        )
    )
    return script_path


def _write_matlab_adjustevents_dialog_script(case: VisualCase, output_path: pathlib.Path) -> pathlib.Path:
    eeglab_root = EEGLAB_REFERENCE_ROOT
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


def _write_matlab_reref_dialog_script(
    case: VisualCase,
    output_path: pathlib.Path,
    variant: str = "default",
) -> pathlib.Path:
    eeglab_root = EEGLAB_REFERENCE_ROOT
    script_path = output_path.parent / f"{case.id}_eeglab_capture.m"
    script_path.write_text(
        "\n".join(
            [
                "function eegprep_visual_capture()",
                "capture_timer = [];",
                "try",
                f"output_file = {_matlab_string(output_path)};",
                f"capture_variant = {_matlab_string(variant)};",
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
                "    'radius', {0.42, 0.42, 0, 0.42}, ...",
                "    'X', {-0.25, 0.25, 0, 0}, ...",
                "    'Y', {0.75, 0.75, 0, -0.8}, ...",
                "    'Z', {0.55, 0.55, 1, 0.55}, ...",
                "    'type', {'EEG', 'EEG', 'EEG', 'EEG'});",
                "EEG.chaninfo = struct();",
                "EEG.chaninfo.nodatchans = struct('labels', {'M1'}, 'theta', {-90}, 'radius', {0.5}, 'type', {'REF'});",
                "EEG.chaninfo.removedchans = struct( ...",
                "    'labels', {'Pz'}, ...",
                "    'theta', {180}, ...",
                "    'radius', {0.25}, ...",
                "    'X', {0}, ...",
                "    'Y', {-0.4}, ...",
                "    'Z', {0.8}, ...",
                "    'type', {'EEG'});",
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
                "    'UserData', struct('output_file', output_file, 'variant', capture_variant), ...",
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
                "payload = get(timer_obj, 'UserData');",
                "output_file = payload.output_file;",
                "variant = payload.variant;",
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
                "apply_pop_reref_variant(fig, variant);",
                "drawnow;",
                "pause(0.2);",
                "write_figure_capture(fig, output_file);",
                "set(ok_button, 'userdata', 'retuninginputui');",
                "stop(timer_obj);",
                "delete(timer_obj);",
                "end",
                "",
                "function apply_pop_reref_variant(fig, variant)",
                "if strcmpi(variant, 'channels')",
                "    set(findobj('parent', fig, 'tag', 'ave'), 'Value', 0);",
                "    set(findobj('parent', fig, 'tag', 'huberef'), 'Value', 0);",
                "    set(findobj('parent', fig, 'tag', 'rerefstr'), 'Value', 1);",
                "    set(findobj('parent', fig, 'tag', 'reref'), 'Enable', 'on', 'String', 'Fp1');",
                "    set(findobj('parent', fig, 'tag', 'refbr'), 'Enable', 'on');",
                "    set(findobj('parent', fig, 'tag', 'keepref'), 'Enable', 'on', 'Value', 1);",
                "elseif strcmpi(variant, 'huber')",
                "    set(findobj('parent', fig, 'tag', 'ave'), 'Value', 0);",
                "    set(findobj('parent', fig, 'tag', 'rerefstr'), 'Value', 0);",
                "    set(findobj('parent', fig, 'tag', 'huberef'), 'Value', 1);",
                "    set(findobj('parent', fig, 'tag', 'reref'), 'Enable', 'off');",
                "    set(findobj('parent', fig, 'tag', 'refbr'), 'Enable', 'off');",
                "    set(findobj('parent', fig, 'tag', 'keepref'), 'Enable', 'off', 'Value', 0);",
                "elseif strcmpi(variant, 'interp_removed')",
                "    set(findobj('parent', fig, 'tag', 'interp'), 'Value', 1);",
                "end",
                "end",
                "",
                *_matlab_capture_helper(),
                "eegprep_visual_capture();",
                "",
            ]
        )
    )
    return script_path


def _write_matlab_interp_dialog_script(
    case: VisualCase,
    output_path: pathlib.Path,
    variant: str = "continuous",
) -> pathlib.Path:
    eeglab_root = EEGLAB_REFERENCE_ROOT
    script_path = output_path.parent / f"{case.id}_eeglab_capture.m"
    script_path.write_text(
        "\n".join(
            [
                "function eegprep_visual_capture()",
                "capture_timer = [];",
                "try",
                f"output_file = {_matlab_string(output_path)};",
                f"capture_variant = {_matlab_string(variant)};",
                f"eeglab_root = {_matlab_string(eeglab_root)};",
                "addpath(genpath(eeglab_root));",
                "set(0, 'DefaultFigureVisible', 'on');",
                "global EEG ALLEEG;",
                "[EEG, ALLEEG] = make_pop_interp_demo(capture_variant);",
                "capture_timer = timer( ...",
                "    'ExecutionMode', 'fixedSpacing', ...",
                "    'Period', 0.5, ...",
                "    'StartDelay', 0.5, ...",
                "    'UserData', output_file, ...",
                "    'TimerFcn', @capture_pop_interp_dialog);",
                "start(capture_timer);",
                "[EEG, com] = pop_interp(EEG);",
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
                "function [EEG, ALLEEG] = make_pop_interp_demo(variant)",
                "EEG = eeg_emptyset;",
                "EEG.setname = 'interp demo';",
                "EEG.nbchan = 4;",
                "EEG.srate = 250;",
                "EEG.xmin = 0;",
                "EEG.chanlocs = struct( ...",
                "    'labels', {'Fp1', 'Fp2', 'Cz', 'Oz'}, ...",
                "    'theta', {-18, 18, 0, 180}, ...",
                "    'radius', {0.42, 0.42, 0, 0.42}, ...",
                "    'X', {-0.25, 0.25, 0, 0}, ...",
                "    'Y', {0.75, 0.75, 0, -0.8}, ...",
                "    'Z', {0.55, 0.55, 1, 0.55}, ...",
                "    'type', {'EEG', 'EEG', 'EEG', 'EEG'});",
                "if strcmpi(variant, 'epoched') || strcmpi(variant, 'epoched_removed')",
                "    EEG.pnts = 500;",
                "    EEG.trials = 2;",
                "    EEG.data = zeros(4, EEG.pnts, EEG.trials);",
                "    EEG.xmax = (EEG.pnts-1)/EEG.srate;",
                "    EEG.epoch = struct('event', {1, 2});",
                "else",
                "    EEG.pnts = 1000;",
                "    EEG.trials = 1;",
                "    EEG.data = zeros(4, EEG.pnts);",
                "    EEG.xmax = (EEG.pnts-1)/EEG.srate;",
                "    EEG.epoch = [];",
                "end",
                "EEG.chaninfo = struct();",
                "if strcmpi(variant, 'removed') || strcmpi(variant, 'epoched_removed')",
                "    EEG.chaninfo.removedchans = struct( ...",
                "        'labels', {'M1', 'M2'}, ...",
                "        'theta', {-90, 90}, ...",
                "        'radius', {0.5, 0.5}, ...",
                "        'X', {-0.8, 0.8}, ...",
                "        'Y', {0, 0}, ...",
                "        'Z', {0.4, 0.4}, ...",
                "        'type', {'EEG', 'EEG'});",
                "end",
                "ALLEEG = EEG;",
                "ALLEEG(2) = EEG;",
                "ALLEEG(2).setname = 'other dataset';",
                "ALLEEG(2).chanlocs(end+1) = struct('labels', 'Pz', 'theta', 180, 'radius', 0.25, 'X', 0, 'Y', -0.4, 'Z', 0.8, 'type', 'EEG');",
                "end",
                "",
                "function capture_pop_interp_dialog(timer_obj, ~)",
                "output_file = get(timer_obj, 'UserData');",
                "fig = findall(0, 'Type', 'figure', 'Name', 'Interpolate channel(s) -- pop_interp()');",
                "if isempty(fig)",
                "    figs = findall(0, 'Type', 'figure');",
                "    for idx = 1:length(figs)",
                "        fig_name = get(figs(idx), 'Name');",
                "        if contains(fig_name, 'pop_interp') || contains(fig_name, 'Interpolate channel')",
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


def _write_matlab_dataset_index_dialog_script(case: VisualCase, output_path: pathlib.Path) -> pathlib.Path:
    eeglab_root = EEGLAB_REFERENCE_ROOT
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
                "capture_timer = timer( ...",
                "    'ExecutionMode', 'fixedSpacing', ...",
                "    'Period', 0.5, ...",
                "    'StartDelay', 1.0, ...",
                "    'UserData', output_file, ...",
                "    'TimerFcn', @capture_dataset_index_dialog);",
                "start(capture_timer);",
                "answer = inputdlg2({ 'Dataset index' }, 'Choose dataset', 1, { '' });",
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
                "function capture_dataset_index_dialog(timer_obj, ~)",
                "output_file = get(timer_obj, 'UserData');",
                "figs = findall(0, 'Type', 'figure');",
                "fig = [];",
                "for idx = 1:length(figs)",
                "    fig_name = get(figs(idx), 'Name');",
                "    if contains(fig_name, 'Choose dataset')",
                "        fig = figs(idx);",
                "        break;",
                "    end",
                "end",
                "if isempty(fig), return; end",
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


def _write_matlab_pophelp_dialog_script(
    case: VisualCase,
    output_path: pathlib.Path,
    function_name: str,
) -> pathlib.Path:
    eeglab_root = EEGLAB_REFERENCE_ROOT
    script_path = output_path.parent / f"{case.id}_eeglab_capture.m"
    script_path.write_text(
        "\n".join(
            [
                "function eegprep_visual_capture()",
                "capture_timer = [];",
                "try",
                f"output_file = {_matlab_string(output_path)};",
                f"function_name = {_matlab_string(function_name)};",
                f"eeglab_root = {_matlab_string(eeglab_root)};",
                "addpath(genpath(eeglab_root));",
                "set(0, 'DefaultFigureVisible', 'on');",
                "capture_timer = timer( ...",
                "    'ExecutionMode', 'fixedSpacing', ...",
                "    'Period', 0.5, ...",
                "    'StartDelay', 1.0, ...",
                "    'UserData', struct('output_file', output_file, 'function_name', function_name), ...",
                "    'TimerFcn', @capture_pophelp_dialog);",
                "start(capture_timer);",
                "pophelp(function_name);",
                "pause(2);",
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
                "function capture_pophelp_dialog(timer_obj, ~)",
                "payload = get(timer_obj, 'UserData');",
                "output_file = payload.output_file;",
                "function_name = payload.function_name;",
                "figs = findall(0, 'Type', 'figure');",
                "fig = [];",
                "for idx = 1:length(figs)",
                "    fig_name = get(figs(idx), 'Name');",
                "    if contains(lower(fig_name), lower(function_name)) || contains(lower(fig_name), 'function help')",
                "        fig = figs(idx);",
                "        break;",
                "    end",
                "end",
                "if isempty(fig)",
                "    if write_java_window_capture(function_name, output_file)",
                "        stop(timer_obj);",
                "        delete(timer_obj);",
                "        return;",
                "    end",
                "    write_pophelp_text_capture(function_name, output_file);",
                "    if exist(output_file, 'file') == 2",
                "        stop(timer_obj);",
                "        delete(timer_obj);",
                "    end",
                "    return;",
                "end",
                "set(fig, 'Units', 'pixels', 'Position', [100 100 720 520]);",
                "drawnow;",
                "pause(0.5);",
                "write_figure_capture(fig, output_file);",
                "try, close(fig); catch, end",
                "stop(timer_obj);",
                "delete(timer_obj);",
                "end",
                "",
                "function captured = write_java_window_capture(function_name, output_file)",
                "captured = false;",
                "try",
                "    frames = java.awt.Frame.getFrames();",
                "    fallback = [];",
                "    for idx = 1:length(frames)",
                "        frame = frames(idx);",
                "        if ~frame.isShowing(), continue; end",
                "        bounds = frame.getBounds();",
                "        if bounds.getWidth() < 300 || bounds.getHeight() < 200, continue; end",
                "        title = char(frame.getTitle());",
                "        if isempty(fallback), fallback = frame; end",
                "        if contains(lower(title), lower(function_name)) || contains(lower(title), 'help') || contains(lower(title), 'web')",
                "            fallback = frame;",
                "            break;",
                "        end",
                "    end",
                "    if isempty(fallback), return; end",
                "    bounds = fallback.getBounds();",
                "    rect = java.awt.Rectangle(bounds.getX(), bounds.getY(), bounds.getWidth(), bounds.getHeight());",
                "    robot = java.awt.Robot;",
                "    img = robot.createScreenCapture(rect);",
                "    javax.imageio.ImageIO.write(img, 'png', java.io.File(output_file));",
                "    captured = true;",
                "catch",
                "    captured = false;",
                "end",
                "end",
                "",
                "function write_pophelp_text_capture(function_name, output_file)",
                "source_path = which(function_name);",
                "help_text = help(function_name);",
                "if length(function_name) > 4 && strcmpi(function_name(1:4), 'pop_')",
                "    called_name = function_name(5:end);",
                "    try",
                "        called_text = help(called_name);",
                "        if ~isempty(called_text)",
                "            divider = sprintf(['\\n___________________________________________________________________\\n\\n' ...",
                "                ' The ''pop'' function above calls the eponymous Matlab function below\\n' ...",
                "                ' and could use some of its optional parameters\\n' ...",
                "                '___________________________________________________________________\\n\\n']);",
                "            help_text = [help_text divider called_text];",
                "        end",
                "    catch",
                "    end",
                "end",
                "fig = figure('Name', [function_name ' - ' upper(function_name)], ...",
                "    'Units', 'pixels', 'Position', [100 100 720 520], ...",
                "    'MenuBar', 'none', 'ToolBar', 'none', 'Color', [1 1 1], ...",
                "    'NumberTitle', 'off');",
                "title_text = sprintf('%s\\n%s\\n\\n%s', upper(function_name), source_path, help_text);",
                "uicontrol('Parent', fig, 'Style', 'edit', 'String', title_text, ...",
                "    'Units', 'pixels', 'Position', [16 16 688 488], ...",
                "    'HorizontalAlignment', 'left', 'Max', 2, 'Min', 0, ...",
                "    'BackgroundColor', [1 1 1], 'ForegroundColor', [0 0 0], ...",
                "    'FontName', 'Monospaced', 'FontSize', 10);",
                "drawnow;",
                "pause(0.2);",
                "write_figure_capture(fig, output_file);",
                "try, close(fig); catch, end",
                "end",
                "",
                *_matlab_capture_helper(),
                "eegprep_visual_capture();",
                "",
            ]
        )
    )
    return script_path


def _write_matlab_pop_chansel_dialog_script(case: VisualCase, output_path: pathlib.Path) -> pathlib.Path:
    eeglab_root = EEGLAB_REFERENCE_ROOT
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
                "capture_timer = timer( ...",
                "    'ExecutionMode', 'fixedSpacing', ...",
                "    'Period', 0.5, ...",
                "    'StartDelay', 0.5, ...",
                "    'UserData', output_file, ...",
                "    'TimerFcn', @capture_pop_chansel_dialog);",
                "start(capture_timer);",
                "[chanlist, chanliststr, allchanstr] = pop_chansel({'Fp1', 'Fp2', 'Cz', 'Oz'}, 'withindex', 'on');",
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
                "function capture_pop_chansel_dialog(timer_obj, ~)",
                "output_file = get(timer_obj, 'UserData');",
                "listbox = findobj(0, 'tag', 'listboxvals');",
                "if isempty(listbox), return; end",
                "fig = ancestor(listbox(1), 'figure');",
                "if isempty(fig), return; end",
                "set(fig, 'Units', 'pixels');",
                "drawnow;",
                "pause(0.2);",
                "write_figure_capture(fig, output_file);",
                "set(fig, 'userdata', 'cancel');",
                "drawnow;",
                "pause(0.1);",
                "try, close(fig); catch, end",
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
        action, variant = _split_action(target.action)
        if action not in {"pop_adjustevents", "pop_chansel", "pop_reref", "pop_interp", "inputdlg2", "pophelp"}:
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
        if action == "pop_adjustevents":
            script_path = _write_matlab_adjustevents_dialog_script(case, output_path)
        elif action == "pop_chansel":
            script_path = _write_matlab_pop_chansel_dialog_script(case, output_path)
        elif action == "pop_interp":
            script_path = _write_matlab_interp_dialog_script(case, output_path, variant)
        elif action == "inputdlg2":
            script_path = _write_matlab_dataset_index_dialog_script(case, output_path)
        elif action == "pophelp":
            script_path = _write_matlab_pophelp_dialog_script(case, output_path, variant)
        else:
            script_path = _write_matlab_reref_dialog_script(case, output_path, variant)
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
