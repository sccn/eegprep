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

Use the current environment's Python, or set `PYTHON` to the interpreter for
the active EEGPrep development environment:

```bash
PYTHON=${PYTHON:-python}
PYTHONPATH=src "$PYTHON" -m unittest tests.test_visual_parity
```

If `uv` is available and the environment is already synced, these are equivalent:

```bash
uv run --no-sync python tools/visual_parity/capture.py --list
uv run --no-sync python tools/visual_parity/compare.py --case adjust_events_dialog
```

Install optional GUI dependencies before Python dialog capture:

```bash
python -m pip install -e '.[gui]'
```

## Implementation Pattern

For an EEGLAB pop function such as `pop_adjustevents`:

1. Read the MATLAB source first:

   ```bash
   sed -n '1,260p' src/eegprep/eeglab/functions/popfunc/pop_adjustevents.m
   ```

2. Keep the Python files simple and parallel to EEGLAB:

   - Backend/API: `src/eegprep/pop_<name>.py`
   - Export: `src/eegprep/__init__.py`
   - Dialog spec: `src/eegprep/gui/specs/pop_<name>.py`
   - Shared GUI primitives: `src/eegprep/gui/spec.py`, `src/eegprep/gui/inputgui.py`, `src/eegprep/gui/qt.py`
   - Visual capture entrypoint: `src/eegprep/gui/visual_capture.py`
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
        "eegprep.gui.visual_capture",
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
- Python capture blank or wrong state: patch `eegprep.gui.visual_capture`.

Treat pixel metrics as a smoke signal, not the final judge. A good dialog can
still have differences from font rendering, antialiasing, native bevels, or OS
theme. Prioritize user-visible structure.

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
  src/eegprep/pop_adjustevents.py \
  src/eegprep/gui \
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
- Qt says it cannot load `xcb`: the display may have died, or system packages
  such as `libxcb-cursor0` may be missing.
- Qt warns about locale `C`: usually non-fatal for screenshots, but prefer a
  UTF-8 locale when available.
- MATLAB capture hangs: use `-nosplash -nodesktop -r`, not `-batch`; ensure the
  timer starts before opening the modal dialog.
- MATLAB capture creates a blank image: verify the dialog is visible on
  `DISPLAY=:99`; if `getframe` fails, try Java Robot or external screenshot
  tools such as `scrot`.
- Wayland desktop attach fails: use Xvfb, Xvnc, VNC, or an Xorg desktop. Wayland
  blocks arbitrary screenshot/window automation by design.
