"""
From History to a Replayable Script
===================================

This example shows the EEGPrep scripting loop:

1. run ``pop_*`` steps with ``return_com=True`` to capture command strings,
2. accumulate them in an :class:`~eegprep.EEGPrepSession` so ``ALLCOM``,
   ``LASTCOM`` and ``EEG["history"]`` match what the GUI would record,
3. write the recorded history with :func:`~eegprep.pop_saveh`,
4. write the *executable Python* version of the same pipeline and replay it with
   :func:`~eegprep.pop_runscript`,
5. verify the replay reproduces the interactive result bit-for-bit.

Recorded command strings are EEGLAB-syntax for parity and portability, so the
replayable artifact is a Python script you keep alongside them.
"""

# %%
# Imports and headless setup
# --------------------------

import matplotlib

matplotlib.use("Agg")

import tempfile
from pathlib import Path

import numpy as np

import eegprep
from eegprep import (
    EEGPrepSession,
    eegh,
    pop_eegfiltnew,
    pop_epoch,
    pop_loadset,
    pop_rmbase,
    pop_reref,
    pop_runscript,
    pop_saveh,
)

REPO_ROOT = Path(eegprep.__file__).resolve().parents[2]  # sphinx-gallery defines no __file__
SAMPLE = REPO_ROOT / "sample_data" / "eeglab_data.set"

_MASKS: list[tuple[str, str]] = [(str(REPO_ROOT), "<repo>")]


def mask(text: str) -> str:
    """Replace machine-specific paths so the rendered page is reproducible."""
    for actual, label in _MASKS:
        text = text.replace(actual, label)
    return text


# %%
# Interactive-style pass, recording history
# -----------------------------------------
# Every step returns ``(EEG, com)``. ``session.add_history`` appends to
# ``ALLCOM`` and updates ``LASTCOM`` exactly as a GUI menu action would.

session = EEGPrepSession()
EEG = pop_loadset(str(SAMPLE))
EEG["history"] = ""  # start from a clean dataset history for this example
session.store_current(EEG, new=True)
session.add_history(f"EEG = pop_loadset({str(SAMPLE)!r});")

EEG, com = pop_eegfiltnew(session.EEG, locutoff=1, hicutoff=40, plotfreqz=False, return_com=True)
session.store_current(EEG, command=com)

EEG, com = pop_reref(session.EEG, [], return_com=True)
session.store_current(EEG, command=com)

EEG, com = pop_epoch(session.EEG, ["square"], [-1, 2], return_com=True)
session.store_current(EEG, command=com)

EEG, com = pop_rmbase(session.EEG, [-1000, 0], return_com=True)
session.store_current(EEG, command=com)

interactive = session.EEG
print(f"epoched: nbchan={interactive['nbchan']} pnts={interactive['pnts']} trials={interactive['trials']}")
print(mask(f"LASTCOM: {session.LASTCOM}"))

# %%
# Session history vs dataset history
# ----------------------------------
# ``ALLCOM`` is the whole session. ``eegh()`` renders it newest-first, the way
# the console ``eegh`` magic does. ``EEG["history"]`` is per-dataset and, as in
# EEGLAB, load/save commands are not part of it.

print("--- eegh() session history ---")
print(mask(eegh(None, session.ALLCOM)))
print("--- EEG['history'] dataset history ---")
print(mask(str(interactive["history"]).strip()))

# %%
# Saving the recorded history
# ---------------------------
# ``pop_saveh`` writes the recorded command strings. They are EEGLAB-syntax, so
# treat this file as an audit trail rather than something EEGPrep can re-run.

workdir = Path(tempfile.mkdtemp(prefix="eegprep_history_"))
_MASKS.append((str(workdir), "<tmpdir>"))
saveh_com = pop_saveh(session.ALLCOM, "eegprephist.m", workdir)
print(mask(saveh_com))
# Skip the "generated on <date>" banner: it would change on every docs build.
saved = (workdir / "eegprephist.m").read_text().strip().splitlines()
print(mask("\n".join(line for line in saved if "generated on" not in line)))

# %%
# The replayable Python script
# ----------------------------
# This is the artifact you keep under version control: the same steps as plain
# Python, with explicit assignment of every return value.

script = f'''from eegprep import pop_eegfiltnew, pop_epoch, pop_reref, pop_rmbase, pop_loadset

EEG = pop_loadset({str(SAMPLE)!r})
EEG, com_filter = pop_eegfiltnew(EEG, locutoff=1, hicutoff=40, plotfreqz=False, return_com=True)
EEG, com_reref = pop_reref(EEG, [], return_com=True)
EEG, com_epoch = pop_epoch(EEG, ["square"], [-1, 2], return_com=True)
EEG, com_rmbase = pop_rmbase(EEG, [-1000, 0], return_com=True)
ALLCOM = [com_filter, com_reref, com_epoch, com_rmbase]
'''
script_path = workdir / "replay_pipeline.py"
script_path.write_text(script, encoding="utf-8")

# %%
# Replay and verify
# -----------------
# ``pop_runscript`` executes a ``.py`` history script in a namespace you supply,
# which is how the GUI's ``File > History scripts > Run script`` feeds results
# back into the shared session.

namespace: dict = {}
run_com = pop_runscript(script_path, namespace)
print(mask(str(run_com)))

replayed = namespace["EEG"]
assert np.array_equal(np.asarray(replayed["data"]), np.asarray(interactive["data"]))
assert replayed["trials"] == interactive["trials"]
print(f"replay matches interactive run: {np.asarray(replayed['data']).shape}")
print("replayed ALLCOM:")
for line in namespace["ALLCOM"]:
    print(mask(f"  {line}"))
