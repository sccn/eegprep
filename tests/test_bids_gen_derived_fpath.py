"""Unit tests for gen_derived_fpath path construction (no MATLAB/pybids needed)."""

import os
import unittest

from eegprep.plugins.EEG_BIDS.bids import gen_derived_fpath


def _raw_fpath():
    # A BIDS-style raw EEG file path inside a dataset rooted at <root>.
    return os.path.join(os.sep, 'data', 'ds001', 'sub-01', 'eeg', 'sub-01_task-rest_eeg.set')


class TestGenDerivedFpath(unittest.TestCase):
    def test_default_root_placeholder_substituted(self):
        """The default outputdir's {root} placeholder is replaced with the dataset root.

        Reproduces the bug where the documented default '${root}/...' left a literal
        placeholder ('$/data/...') in the output path.
        """
        out = gen_derived_fpath(_raw_fpath(), keyword='desc-cleaned')
        expected = os.path.join(
            os.sep,
            'data',
            'ds001',
            'derivatives',
            'clean_artifacts',
            'sub-01',
            'eeg',
            'sub-01_task-rest_desc-cleaned_eeg.set',
        )
        self.assertEqual(out, expected)
        self.assertNotIn('{root}', out)
        self.assertNotIn('$', out)

    def test_explicit_root_placeholder_substituted(self):
        """An explicit '{root}/...' outputdir is substituted with the dataset root."""
        out = gen_derived_fpath(_raw_fpath(), outputdir='{root}/derivatives/eegprep')
        expected = os.path.join(
            os.sep, 'data', 'ds001', 'derivatives', 'eegprep', 'sub-01', 'eeg', 'sub-01_task-rest_eeg.set'
        )
        self.assertEqual(out, expected)

    def test_path_assembly_uses_os_sep(self):
        """The assembled path uses the OS separator throughout (no hardcoded '/')."""
        out = gen_derived_fpath(_raw_fpath(), keyword='desc-cleaned')
        # Every separator must be the platform separator produced by os.path.join.
        self.assertEqual(out, os.path.normpath(out))


if __name__ == '__main__':
    unittest.main()
