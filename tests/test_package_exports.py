"""Tests for the public eegprep package export surface."""

from __future__ import annotations

import subprocess
import sys
import textwrap


import eegprep


def test_eegrej_export_matches_eeglab_low_level_function() -> None:
    from eegprep.functions.popfunc.eeg_eegrej import eeg_eegrej
    from eegprep.functions.popfunc.eeg_multieegplot import eeg_multieegplot
    from eegprep.functions.sigprocfunc.eegplot import eegplot as sigproc_eegplot
    from eegprep.functions.sigprocfunc.eegrej import eegrej as sigproc_eegrej
    from eegprep.functions.sigprocfunc.rmbase import rmbase as sigproc_rmbase

    assert eegprep.eegplot is sigproc_eegplot
    assert eegprep.eeg_multieegplot is eeg_multieegplot
    assert eegprep.eegrej is sigproc_eegrej
    assert eegprep.eeg_eegrej is eeg_eegrej
    assert eegprep.eegrej is not eegprep.eeg_eegrej
    assert eegprep.rmbase is sigproc_rmbase


def test_import_eegprep_is_lightweight() -> None:
    code = textwrap.dedent(
        """
        import sys
        import eegprep

        blocked = ["PySide6", "torch", "mne"]
        print(",".join(name for name in blocked if name in sys.modules))
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == ""
