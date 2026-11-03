"""``dir(eegprep)`` and ``eegprep.help()`` list the public names without importing them."""

from __future__ import annotations

import os
import re
import subprocess
import sys

import eegprep

HEAVY_MODULES = ("numpy", "scipy", "mne", "matplotlib", "h5py", "oct2py")


def _run_fresh(code: str) -> str:
    """Run ``code`` in a new interpreter and fail if it imported a heavy library.

    A subprocess, because this process may already have imported those libraries.
    """
    script = (
        "import sys, eegprep\n"
        f"{code}\n"
        f"loaded = [name for name in {HEAVY_MODULES!r} if name in sys.modules]\n"
        "assert not loaded, loaded\n"
    )
    env = dict(os.environ)
    env["EEGPREP_SKIP_MATLAB"] = "1"
    result = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    return result.stdout


def _line_mentions(text: str, name: str) -> bool:
    return re.search(rf"(?m)^  (?:await )?{re.escape(name)}\b", text) is not None


def test_help_is_not_in_dunder_all() -> None:
    assert "help" not in eegprep.__all__
    assert callable(eegprep.help)


def test_dir_lists_every_lazy_name_without_importing_the_stack() -> None:
    names = _run_fresh("print('\\n'.join(dir(eegprep)))").split()

    missing = [name for name in eegprep.__all__ if name not in names]
    assert not missing, f"dir() omits {missing[:12]}"
    assert "help" in names
    assert "pop_loadset" in names
    assert "pop_writeeeg" in names


def test_help_describes_every_public_name_without_importing_the_stack() -> None:
    text = _run_fresh("eegprep.help()")

    absent = [name for name in eegprep.__all__ if not _line_mentions(text, name)]
    assert not absent, f"help() does not mention {absent[:12]}"
    assert "Load an EEGLAB dataset" in text
    assert "Plot a 2D topographic map" in text
    assert "Write continuous EEG data to EDF, BDF, or GDF" in text
    for extra in ("eeglab", "sys", "edf", "full"):
        assert f'pip install "eegprep[{extra}]"' in text
    assert "EDF and BDF output" in text
    described = [line for line in text.splitlines() if line.startswith("  ") and "  " in line[2:]]
    summaries = [line for line in described if line.split("  ", 2)[-1].strip()]
    assert len(summaries) > 400
