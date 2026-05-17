---
name: eeglab-gui-visual-parity
description: Build, port, or iterate on EEGPrep GUI features so they visually and behaviorally match EEGLAB MATLAB UI. Use when implementing a pop_ function GUI, adding an EEGPrep Qt dialog/window, creating or updating visual parity cases under tools/visual_parity, comparing EEGLAB and EEGPrep screenshots, debugging MATLAB GUI capture under X11, or tuning layout/style from an end-user screenshot feedback loop.
---

# EEGLAB GUI Visual Parity

Use this skill to implement an EEGPrep GUI component with an EEGLAB reference
screenshot in the loop. The goal is not pixel perfection; it is end-user parity:
same controls, labels, order, enabled state, layout, and obvious hierarchy.

## Work From The Repo Root

Run commands from the EEGPrep repo root:

```bash
git rev-parse --show-toplevel
```

Use the uv-managed project environment by default:

```bash
uv run pytest tests/test_visual_parity.py
```

For repeated capture runs after the environment is synced, skip dependency
resolution:

```bash
uv run --no-sync python tools/visual_parity/capture.py --list
uv run --no-sync python tools/visual_parity/compare.py --case adjust_events_dialog
```

Install optional GUI dependencies before Python dialog capture:

```bash
uv sync --extra gui --group dev
```

## Choose The Capture Path First

Before starting MATLAB or any display server, identify the machine class and
then use the matching section below. Do not start X11 tooling on a Mac, and do
not use the macOS desktop workflow on the SCCN Linux server.

```bash
uname -s
uname -m
hostname
command -v matlab || true
command -v uv || true
command -v conda || true
command -v Xvnc || true
command -v Xvfb || true
command -v xdpyinfo || true
command -v screencapture || true
command -v osascript || true
```

Use this decision rule:

- If `uname -s` is `Linux` and the machine has the SCCN server setup
  (`conda`, `eegprep-dev`, and usually `Xvnc`), use **SCCN Server Fast Path**.
- If `uname -s` is `Darwin`, use **MacBook Fast Path**.
- If neither path fits, probe dependencies first and then adapt the closest
  path without hard-coding checkout-specific absolute paths.

Always probe the Python used by capture before judging screenshots:

```bash
uv run --no-sync python - <<'PY'
import importlib.util, platform, sys
print(platform.platform())
print(sys.executable)
for name in ("numpy", "PySide6", "PIL"):
    print(name, importlib.util.find_spec(name) is not None)
PY
```

## SCCN Server Fast Path

This is the tested path on the shared server environment used for EEGPrep GUI
parity work. Run it from the repo root; do not hard-code a checkout path.

Probe tools and choose the Python environment first:

```bash
command -v matlab || true
command -v Xvfb || true
command -v openbox || true
command -v Xvnc || true
command -v xdpyinfo || true
command -v identify || true

# On this server, use the EEGPrep dev env, not the default Python, unless this
# dependency probe passes with another interpreter.
conda activate eegprep-dev 2>/dev/null || true
PYTHON=${PYTHON:-$(command -v python)}

"$PYTHON" - <<'PY'
import importlib.util, sys
print(sys.executable)
for name in ("numpy", "PySide6"):
    print(name, importlib.util.find_spec(name) is not None)
PY
```

If `conda activate` is not available in the non-interactive shell, use this
path-free fallback instead of guessing a checkout-specific interpreter path:

```bash
PYTHON=$(conda run -n eegprep-dev python -c 'import sys; print(sys.executable)')
```

If `numpy` or `PySide6` is missing, set `PYTHON` to an interpreter from the
`eegprep-dev` environment or install `.[gui]` into the active environment. The
visual capture command uses `sys.executable`, so the capture subprocess will use
the interpreter running `tools/visual_parity/capture.py`.

Start a detached X11 display. This server may not have `Xvfb` or `openbox`, so
the reliable path is `Xvnc`. Use `setsid` so the display survives after the
shell command returns, and use `>|` because some shells enable `noclobber`.

```bash
if ! DISPLAY=:99 xdpyinfo >/tmp/xdpyinfo99.log 2>&1; then
  setsid Xvnc :99 -geometry 1920x1080 -depth 24 \
    -SecurityTypes None -localhost -ac -noreset +extension GLX +render \
    >| /tmp/xvnc99-eegprep.log 2>&1 < /dev/null &
  sleep 2
fi

export DISPLAY=:99
if locale -a | grep -qi '^en_US\.utf8$'; then
  export LANG=en_US.utf8
  export LC_ALL=en_US.utf8
fi
xdpyinfo | head
```

Then run the capture and compare loop:

```bash
PYTHONPATH=src "$PYTHON" tools/visual_parity/capture.py \
  --case adjust_events_dialog \
  --target both \
  --timeout 180

PYTHONPATH=src "$PYTHON" tools/visual_parity/compare.py \
  --case adjust_events_dialog

identify .visual-parity/adjust_events_dialog/eeglab.png \
  .visual-parity/adjust_events_dialog/eegprep.png \
  .visual-parity/adjust_events_dialog/side_by_side.png
sed -n '1,120p' .visual-parity/adjust_events_dialog/report.md
```

Open `.visual-parity/adjust_events_dialog/side_by_side.png` and judge the UI
like a user. In the successful server run, both screenshots were `858x169`; the
remaining differences after tuning were font rendering and native widget bevels.

## MacBook Fast Path

Use this path on a logged-in macOS desktop. Do not start `Xvfb`, `Xvnc`, or
`openbox` on macOS. MATLAB and Qt can render on the physical desktop, and the
capture helpers should be run from the uv environment.

Probe the Mac desktop tools first:

```bash
uname -s
sw_vers
command -v matlab || true
command -v screencapture || true
command -v osascript || true
uv run --no-sync python - <<'PY'
import importlib.util, platform, sys
print(platform.platform())
print(sys.executable)
for name in ("numpy", "PySide6", "PIL"):
    print(name, importlib.util.find_spec(name) is not None)
PY
```

If `matlab` is missing, add the MATLAB command-line launcher to `PATH` before
running visual parity. If Qt or NumPy is missing, run:

```bash
uv sync --extra gui --group dev
```

For screenshots of main windows and modal dialogs, use the normal generated
capture scripts. They use MATLAB `-nosplash -nodesktop -r` and the Python
capture entrypoint; no `DISPLAY` variable is needed.

```bash
uv run --no-sync python tools/visual_parity/capture.py --list
uv run --no-sync python tools/visual_parity/capture.py \
  --case main_window \
  --target both \
  --timeout 180
uv run --no-sync python tools/visual_parity/compare.py --case main_window
sed -n '1,120p' .visual-parity/main_window/report.md
```

For iteration, keep the EEGLAB screenshot fixed and recapture only EEGPrep after
each patch:

```bash
uv run --no-sync python tools/visual_parity/capture.py \
  --case main_window \
  --target eegprep \
  --timeout 60
uv run --no-sync python tools/visual_parity/compare.py --case main_window
```

On macOS Retina displays, small numeric deltas can come from device-pixel-ratio
scaling, MATLAB font rendering, and Qt antialiasing. Treat `compare.py` metrics
as a smoke signal and inspect `side_by_side.png` at original size.

For main-window menu parity on macOS, prefer inventory parity over open-menu
screenshots. MATLAB figure menus can be hard to capture reliably on macOS
because the native application menu bar may receive focus instead of the EEGLAB
figure menu. Use screenshots for the main window and modal dialogs; use menu
inventories for menu labels, order, separators, and enabled state.

Export the EEGLAB inventories from the sibling checkout when present:

```bash
mkdir -p .visual-parity/menu_inventory
matlab -nosplash -nodesktop -r "try, addpath('tools/visual_parity/matlab'); states = {'startup','continuous','epoched','multiple'}; for i = 1:numel(states), export_eeglab_menu_inventory(['.visual-parity/menu_inventory/eeglab_' states{i} '_default.json'], '../eeglab', states{i}); end; catch ME, disp(getReport(ME, 'extended')); exit(1); end; exit(0);"
```

Then export and compare EEGPrep:

```bash
for state in startup continuous epoched multiple; do
  uv run --no-sync python tools/visual_parity/export_eegprep_menu_inventory.py \
    --state "$state" \
    --output ".visual-parity/menu_inventory/eegprep_${state}_default.json"
  uv run --no-sync python tools/visual_parity/menu_inventory.py \
    --reference ".visual-parity/menu_inventory/eeglab_${state}_default.json" \
    --candidate ".visual-parity/menu_inventory/eegprep_${state}_default.json" \
    --report ".visual-parity/menu_inventory/${state}_default_report.md"
done
```

Inspect any differences:

```bash
for state in startup continuous epoched multiple; do
  echo "== $state =="
  sed -n '1,160p' ".visual-parity/menu_inventory/${state}_default_report.md"
done
```

If you need a quick local screenshot of an EEGPrep-opened menu on macOS, capture
the EEGPrep target only:

```bash
uv run --no-sync python tools/visual_parity/capture.py \
  --case file_menu \
  --target eegprep \
  --timeout 60
```

Do not treat a failed or blank EEGLAB open-menu screenshot on macOS as a menu
implementation failure. Verify the menu tree with the inventory exporter.

## Physical Desktop UX Check

Use this when the user wants to try the flow manually on a logged-in desktop,
not just inspect screenshots. Run from the repo root in a desktop terminal so
MATLAB and Qt can open real windows.

For EEGLAB, launch MATLAB desktop with the bundled EEGLAB on the path:

```bash
matlab -desktop
```

In MATLAB:

```matlab
addpath(genpath(fullfile(pwd, 'src/eegprep/eeglab')));
close(findobj('tag','EEGLAB'));
eeglab full

EEG = eeg_emptyset;
EEG.setname = 'adjustevents demo';
EEG.data = zeros(1,1000);
EEG.nbchan = 1;
EEG.pnts = 1000;
EEG.trials = 1;
EEG.srate = 250;
EEG.xmin = 0;
EEG.xmax = (EEG.pnts-1)/EEG.srate;
EEG.chanlocs = struct('labels', {'Cz'});
EEG.event = struct( ...
    'type', {'stim', 'resp', 'stim'}, ...
    'latency', {100, 350, 700}, ...
    'duration', {0, 0, 0});

EEG = eeg_checkset(EEG, 'eventconsistency');
[ALLEEG, EEG, CURRENTSET] = eeg_store(ALLEEG, EEG, 0);
eeglab redraw
```

Then use the real EEGLAB user path. For `pop_adjustevents`, this is:

```text
Edit > Adjust event latencies
```

Try the same values you will try in EEGPrep, for example event type `stim`,
milliseconds `20`, then OK.

If the menu item is missing, it is usually because EEGLAB is in simplified menu
mode. Use `eeglab full`, or enable all/advanced menu items in EEGLAB
preferences, then restart EEGLAB. Verify path and menu state with:

```matlab
which eeglab -all
which pop_adjustevents -all
findall(findobj('tag','EEGLAB'), 'Label', 'Adjust event latencies')
```

For EEGPrep, the current user-facing flow for the first pop-function GUI is the
pop function call with no processing arguments. There is not yet an EEGPrep main
window menu path unless that shell/menu layer has been built for the feature.

```bash
PYTHON=${PYTHON:-python}
PYTHONPATH=src "$PYTHON" - <<'PY'
import numpy as np
from eegprep import pop_adjustevents

EEG = {
    "data": np.zeros((1, 1000), dtype=np.float32),
    "nbchan": 1,
    "pnts": 1000,
    "trials": 1,
    "srate": 250.0,
    "xmin": 0.0,
    "xmax": 3.996,
    "event": [
        {"type": "stim", "latency": 100.0, "duration": 0.0},
        {"type": "resp", "latency": 350.0, "duration": 0.0},
        {"type": "stim", "latency": 700.0, "duration": 0.0},
    ],
}

out, com = pop_adjustevents(EEG, return_com=True)
print(com)
print([event["latency"] for event in out["event"]])
PY
```

Enter the same values as the EEGLAB run. With `stim` and `20` ms at 250 Hz, the
printed stim latencies should shift by 5 samples. For another pop function,
replace the demo data, menu item, and `pop_adjustevents(EEG)` call with that
feature's real EEGLAB path and EEGPrep pop-function entrypoint.

## Implementation Pattern

For an EEGLAB pop function such as `pop_adjustevents`:

1. Read the MATLAB source first:

   ```bash
   sed -n '1,260p' src/eegprep/eeglab/functions/popfunc/pop_adjustevents.m
   ```

2. Keep the Python files simple and parallel to EEGLAB:

   - Backend/API and dialog spec: `src/eegprep/functions/popfunc/pop_<name>.py`
   - Export: `src/eegprep/__init__.py`
   - Shared GUI primitives: `src/eegprep/functions/guifunc/spec.py`, `src/eegprep/functions/guifunc/inputgui.py`, `src/eegprep/functions/guifunc/qt.py`
   - Visual capture entrypoint: `src/eegprep/functions/guifunc/visual_capture.py`
   - Tests: `tests/test_pop_<name>.py`, `tests/test_gui_pop_<name>.py`, `tests/test_visual_parity.py`

3. Make the dialog spec mirror EEGLAB's `uilist` and `uigeom`.

   Keep labels, control order, `tag` values, dialog title, and callback intent
   close to MATLAB. Store original MATLAB callback strings as metadata when it
   helps future agents maintain parity, but implement callbacks in explicit
   Python functions.

4. Put toolkit-specific tuning in the renderer, not the pop-function spec.

   The spec should describe the EEGLAB-like dialog. The Qt renderer can own
   colors, margins, button order, checkbox rendering, and fixed widget sizing
   needed to look like MATLAB.

5. Test backend behavior separately from screenshots.

   Use ordinary unit tests for argument parsing, data mutation, errors, command
   history, GUI cancel, and renderer-returned values. Do not rely on screenshots
   to prove numerical behavior.

## Start A Virtual X11 Display

MATLAB GUI capture needs a real X11 display. Do not use `matlab -batch` for GUI
screenshots.

Preferred setup when available:

```bash
pkill -f "Xvfb :99" || true
Xvfb :99 -screen 0 1920x1080x24 -ac +extension GLX +render -noreset \
  > /tmp/xvfb99.log 2>&1 &
export DISPLAY=:99
openbox > /tmp/openbox99.log 2>&1 &
xdpyinfo | head
```

Fallback on servers that have TigerVNC but not `Xvfb`/`openbox`:

```bash
# Keep this running in a long-lived process.
Xvnc :99 -geometry 1920x1080 -depth 24 -SecurityTypes None -localhost \
  -ac -noreset +extension GLX +render
```

Then in the capture shell:

```bash
export DISPLAY=:99
xdpyinfo | head
```

Clean up after the loop:

```bash
pkill -u "$USER" -x Xvnc || true
pkill -f "Xvfb :99" || true
pkill -u "$USER" -x openbox || true
pgrep -a -u "$USER" 'Xvnc|Xvfb|openbox' || true
```

## Add Or Update A Visual Parity Case

Add a case to `tools/visual_parity/cases.json`:

```json
{
  "id": "adjust_events_dialog",
  "description": "Adjust event latencies dialog opened from pop_adjustevents.",
  "window_size": [858, 169],
  "timeout_seconds": 120,
  "targets": {
    "eeglab": {
      "type": "matlab_dialog",
      "action": "pop_adjustevents"
    },
    "eegprep": {
      "type": "command",
      "action": "adjust_events_dialog",
      "command": [
        "{python}",
        "-m",
        "eegprep.functions.guifunc.visual_capture",
        "--case",
        "{case_id}",
        "--output",
        "{output}"
      ]
    }
  }
}
```

For MATLAB modal dialogs, generate a temporary MATLAB script from
`tools/visual_parity/capture.py`. Use:

```bash
matlab -nosplash -nodesktop -r \
  "try, run('/absolute/path/to/generated_capture.m'); catch ME, disp(getReport(ME, 'extended')); exit(1); end; exit(0);"
```

Use a MATLAB timer to capture modal dialogs while `inputgui` is open. Capture
only after the dialog's OK button exists, then set its userdata to unblock
`inputgui`. Capturing too early can delete or close the figure while EEGLAB is
still constructing it.

## Screenshot Feedback Loop

Run the loop from one shell with `DISPLAY` set:

```bash
export DISPLAY=:99
PYTHON=${PYTHON:-python}

PYTHONPATH=src "$PYTHON" tools/visual_parity/capture.py --list
PYTHONPATH=src "$PYTHON" tools/visual_parity/capture.py \
  --case adjust_events_dialog \
  --target eeglab \
  --timeout 180
PYTHONPATH=src "$PYTHON" tools/visual_parity/capture.py \
  --case adjust_events_dialog \
  --target eegprep \
  --timeout 60
PYTHONPATH=src "$PYTHON" tools/visual_parity/compare.py \
  --case adjust_events_dialog
```

For iteration, keep the EEGLAB screenshot fixed and recapture only EEGPrep after
each Python-side patch:

```bash
PYTHONPATH=src "$PYTHON" tools/visual_parity/capture.py \
  --case adjust_events_dialog \
  --target eegprep \
  --timeout 60
PYTHONPATH=src "$PYTHON" tools/visual_parity/compare.py \
  --case adjust_events_dialog
```

Inspect:

```bash
identify .visual-parity/adjust_events_dialog/eeglab.png
identify .visual-parity/adjust_events_dialog/eegprep.png
sed -n '1,120p' .visual-parity/adjust_events_dialog/report.md
```

Open or view these artifacts:

- `.visual-parity/<case>/eeglab.png`
- `.visual-parity/<case>/eegprep.png`
- `.visual-parity/<case>/side_by_side.png`
- `.visual-parity/<case>/diff.png`
- `.visual-parity/<case>/report.md`

When iterating, patch the smallest relevant layer:

- Wrong labels/order/tags: patch the pop-function dialog spec.
- Wrong behavior after clicking/editing: patch callbacks or backend parsing.
- Wrong spacing/colors/button order/native widget shape: patch the Qt renderer.
- MATLAB capture blank or missing: patch the generated MATLAB capture script.
- Python capture blank or wrong state: patch `eegprep.functions.guifunc.visual_capture`.
- Import errors after file moves: fix the moved module imports before judging
  screenshots. For example, after moving pop functions into `functions/popfunc`, relative
  imports such as `.utils` should use the current EEGPrep package path for that helper.

Treat pixel metrics as a smoke signal, not the final judge. A good dialog can
still have differences from font rendering, antialiasing, native bevels, or OS
theme. Prioritize user-visible structure.

Do not commit generated `.visual-parity/` screenshots unless explicitly asked
for a durable reference. For PRs involving GUI work, attach side-by-side images
as GitHub user attachments in a PR comment instead of committing them.

Actionable PR attachment workflow:

- Keep comparison artifacts local, usually under
  `.visual-parity/<case>/side_by_side.png` or `/tmp`.
- If `gh image` is unavailable, install it once with
  `gh extension install drogers0/gh-image`.
- Upload images with `gh image --repo sccn/eegprep <path-to-png> ...`; use the
  returned `https://github.com/user-attachments/assets/...` Markdown in the PR
  comment.
- Put all GUI cases in one concise PR comment, starting with `🤖`, and wrap the
  image list in `<details>` when there are many screenshots.
- Post or refresh the comment with `gh pr comment <pr-number> --body-file
  /tmp/gui-parity-comment.md`.
- Label each image with the feature/state it covers. If an image is UX-only
  evidence rather than strict EEGLAB parity, say that explicitly.

## Iterative Development Checklist

1. Read the EEGLAB MATLAB source and confirm the dialog title, labels, `uilist`,
   `uigeom`, tags, callbacks, and default values.
2. Capture EEGLAB and EEGPrep once with `--target both`; if either capture fails,
   fix the capture environment before changing GUI code.
3. Inspect `side_by_side.png` at original size. Prefer structural parity over
   exact pixel metrics: labels, order, enabled state, control sizes, alignment,
   and button placement matter most.
4. Patch the smallest layer: spec for structure, renderer for toolkit styling,
   visual capture for deterministic screenshot state, capture script for MATLAB.
5. Recapture only EEGPrep and rerun `compare.py`; repeat until user-visible
   differences are down to font rendering/native bevels.
6. Recapture both targets once at the end so the final artifacts came from the
   same display session.
7. Run focused tests and pre-commit:

   ```bash
   PYTHONPATH=src "$PYTHON" -m unittest \
     tests.test_pop_adjustevents \
     tests.test_gui_pop_adjustevents \
     tests.test_visual_parity
   ./pre-commit.py <changed files>
   ```
8. Stop the display you started:

   ```bash
   pgrep -a -u "$USER" 'Xvnc|Xvfb|openbox|MATLAB|matlab' || true
   pkill -u "$USER" -x Xvnc || true
   pkill -f "Xvfb :99" || true
   pkill -u "$USER" -x openbox || true
   ```

## Validation Commands

Run focused tests during development:

```bash
PYTHONPATH=src "$PYTHON" -m unittest \
  tests.test_pop_adjustevents \
  tests.test_gui_pop_adjustevents \
  tests.test_visual_parity
```

Run compile checks after editing GUI/capture code:

```bash
PYTHONPATH=src "$PYTHON" -m compileall -q \
  src/eegprep/functions/popfunc/pop_adjustevents.py \
  src/eegprep/functions/guifunc \
  tools/visual_parity \
  tests/test_pop_adjustevents.py \
  tests/test_gui_pop_adjustevents.py \
  tests/test_visual_parity.py
```

Before finishing, verify no display sessions were left behind:

```bash
pgrep -a -u "$USER" 'Xvnc|Xvfb|openbox' || true
```

## Troubleshooting

- `xdpyinfo` cannot open `:99`: the virtual display is not running or `DISPLAY`
  was not exported in the current shell.
- `Xvnc` starts and then disappears: start it with `setsid ... < /dev/null &`.
  A foreground X server tied to a short-lived shell may exit before capture.
- `/tmp/xvnc99-eegprep.log: cannot overwrite existing file`: use `>|` instead
  of `>` or delete the old log. Some shells enable `noclobber`.
- Qt says it cannot load `xcb`: the display may have died, or system packages
  such as `libxcb-cursor0` may be missing.
- Qt warns about locale `C`: set `LANG=en_US.utf8` and `LC_ALL=en_US.utf8` if
  `locale -a` lists `en_US.utf8`. The warning is usually non-fatal.
- EEGPrep capture fails with `ModuleNotFoundError` before opening a window:
  fix import breakage first. Do not tune screenshots against a failing import.
- MATLAB capture hangs: use `-nosplash -nodesktop -r`, not `-batch`; ensure the
  timer starts before opening the modal dialog.
- MATLAB capture creates a blank image: verify the dialog is visible on
  `DISPLAY=:99`; the capture helper should try `getframe` first and Java Robot
  second. If both fail, use an external X11 screenshot tool such as `import` or
  `scrot`.
- MATLAB prints `Not enough parameters selected` after dialog capture: expected
  for `pop_adjustevents` when the timer presses OK on an empty dialog. The PNG is
  still valid if the capture command exits successfully.
- MATLAB prints a software OpenGL warning: expected under virtual X11 and not a
  visual parity failure by itself.
- Default `python` lacks `numpy` or `PySide6`: activate `eegprep-dev` or set
  `PYTHON` to that environment's interpreter before running capture.
- Wayland desktop attach fails: use Xvfb, Xvnc, VNC, or an Xorg desktop. Wayland
  blocks arbitrary screenshot/window automation by design.
