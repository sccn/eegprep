.. _visual_parity:

=====================
Visual Parity Testing
=====================

EEGPREP aims to keep future desktop UI behavior familiar to EEGLAB users. The
visual parity tools capture the same UI state in EEGLAB and EEGPREP, generate a
side-by-side image, and write a short review prompt that an agent or human can
use to give concrete UI feedback.

The workflow is local and advisory. It is not part of required CI because it
depends on a stable desktop session, MATLAB Desktop, fixed fonts, and the
optional EEGPREP Qt UI.

Headless MATLAB batch sessions can run structural checks, but they cannot
capture EEGLAB UI controls with ``getframe`` unless figure windows are actually
displayed.

Install the optional EEGPREP GUI dependencies before capturing Python dialogs:

.. code-block:: bash

   python -m pip install -e '.[gui]'

Start a Virtual Desktop
=======================

MATLAB GUI capture needs an X11 display. Prefer a reusable virtual desktop so
each capture command inherits the same ``DISPLAY``:

.. code-block:: bash

   Xvfb :99 -screen 0 1920x1080x24 -ac +extension GLX +render -noreset \
       > /tmp/xvfb99.log 2>&1 &
   export DISPLAY=:99
   openbox > /tmp/openbox99.log 2>&1 &
   xdpyinfo | head

Some shared servers provide TigerVNC instead of ``Xvfb``. In that case, this is
equivalent for local screenshot capture:

.. code-block:: bash

   # Terminal 1: keep the virtual X server running.
   Xvnc :99 -geometry 1920x1080 -depth 24 -SecurityTypes None -localhost \
       -ac -noreset +extension GLX +render

.. code-block:: bash

   # Terminal 2: run captures against that display.
   export DISPLAY=:99
   xdpyinfo | head

List Available Cases
====================

.. code-block:: bash

   uv run --no-sync python tools/visual_parity/capture.py --list

Capture Screenshots
===================

Capture both targets for a configured case:

.. code-block:: bash

   export DISPLAY=:99
   uv run --no-sync python tools/visual_parity/capture.py --case main_window --target both

Capture the first pop-function dialog parity case:

.. code-block:: bash

   export DISPLAY=:99
   uv run --no-sync python tools/visual_parity/capture.py \
       --case adjust_events_dialog \
       --target both

Capture only EEGPREP using a command supplied by the caller:

.. code-block:: bash

   uv run --no-sync python tools/visual_parity/capture.py \
       --case file_menu \
       --target eegprep \
       --eegprep-command "python -m eegprep.functions.guifunc.visual_capture --case {case_id} --output {output}"

Capture commands receive these environment variables:

- ``EEGPREP_VISUAL_CASE_ID``
- ``EEGPREP_VISUAL_TARGET``
- ``EEGPREP_VISUAL_ACTION``
- ``EEGPREP_VISUAL_OUTPUT``
- ``EEGPREP_VISUAL_OUTPUT_DIR``
- ``EEGPREP_VISUAL_WINDOW_WIDTH``
- ``EEGPREP_VISUAL_WINDOW_HEIGHT``
- ``EEGPREP_REPO_ROOT``

Compare Screenshots
===================

.. code-block:: bash

   uv run --no-sync python tools/visual_parity/compare.py --case adjust_events_dialog

The comparison writes artifacts under ``.visual-parity/<case>/``:

- ``eeglab.png`` and ``eegprep.png``
- ``side_by_side.png``
- ``diff.png``
- ``report.md``

Use ``report.md`` as the high-signal prompt for a visual review model. The
pixel metrics are useful for spotting large differences, but menu labels,
ordering, enabled state, and dialog layout are more important than exact pixels.
Keep generated screenshots and side-by-side images out of committed docs unless
the project explicitly asks for a durable reference image. For PR reviews, attach
or link the generated ``side_by_side.png`` in the PR comment instead.

Compare Menu Structure
======================

The MATLAB helper exports EEGLAB's menu tree:

.. code-block:: matlab

   export_eeglab_menu_inventory('.visual-parity/eeglab_menus.json', 'src/eegprep/eeglab')

Compare the exported EEGLAB menu JSON with an EEGPREP menu model:

.. code-block:: bash

   python tools/visual_parity/menu_inventory.py \
       --reference .visual-parity/eeglab_menus.json \
       --candidate .visual-parity/eegprep_menus.json

Prefer this structural comparison before pixel review. It catches missing menu
items, wrong order, and disabled/enabled mismatches without font or operating
system noise.
