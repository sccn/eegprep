"""
Test suite for clean_artifacts.py - All-in-one artifact removal.

This module tests the clean_artifacts function that provides comprehensive
artifact removal including flatline channels, drifts, noisy channels, bursts, and windows.
"""

import logging
import unittest
import sys
import socket
import numpy as np
import tempfile
import os
import shutil

logger = logging.getLogger(__name__)

# Add src to path for imports
sys.path.insert(0, 'src')
from eegprep.clean_artifacts import clean_artifacts
from eegprep.utils.testing import DebuggableTestCase

curhost = socket.gethostname()
slow_tests_hosts_only = [] # ['ck-carbon']
reservation = '8GB' if curhost in ['ck-carbon'] else ''


class TestBidsPreproc(DebuggableTestCase):
    """Basic test cases for clean_artifacts function."""

    def setUp(self):
        """Set up test fixtures."""
        # root path of all OpenNeuro datasets on this host
        if curhost == 'ck-carbon':
            self.root_path = os.path.expanduser('~/data/OpenNeuro')
        else:
            self.root_path = None
            logger.warning(f"Skipping test TestBidsPreproc on unknown test host {curhost}; "
                           f"please add support for your hostname to the above list to "
                           f"enable this test.")

        # the study that we want to analyze
        self.test_study = 'ds003061'

    def test_end2end(self):
        """End-to-end test vs MATLAB."""
        from eegprep import bids_preproc
        from eegprep.eeglabcompat import get_eeglab
        from eegprep.eeg_compare import eeg_compare

        if self.root_path is None:
            self.skipTest("Skipping test_end2end on unknown host")

        candidates = [self.test_study, self.test_study + '-download']
        retain = [d for d in os.listdir(self.root_path) if d in candidates]
        if len(retain) == 0:
            self.skipTest(f"Skipping test_end2end because neither {candidates} exist in {self.root_path}")
        study_path = os.path.join(self.root_path, retain[0])

        print(f"Running bids_pipeline() on {study_path}...")
        eeglab = get_eeglab('MATLAB')
        ALLEEG_mat = eeglab.bids_pipeline(study_path)

        print(f"Running bids_preproc() on {study_path}...")
        ALLEEG_py = bids_preproc(
            study_path,
            ReservePerJob=reservation,
            # just the first 2 subjects of the main task
            subjects=[0,1], runs=[1],
            SkipIfPresent=True, # <- for quicker re-runs
            bidsevent=True,
            SamplingRate=128,
            WithInterp=True, EpochEvents=[], EpochLimits=[-0.2, 0.5], EpochBaseline=[None, 0],
            WithPicard=True, WithICLabel=True,
            MinimizeDiskUsage=False,
            ReturnData=True)

        print("Comparing Python vs MATLAB results...")
        for k in range(min(len(ALLEEG_py), len(ALLEEG_mat))):
            EEG_py = ALLEEG_py[k]
            EEG_mat = ALLEEG_mat[k]
            print(f"Comparing subject #{k}: {EEG_py['setname']}...")
            eeg_compare(EEG_py, EEG_mat)

    @unittest.skipIf(curhost not in slow_tests_hosts_only, f"Slow stress test skipped by default on hosts other than {slow_tests_hosts_only}")
    def test_crashability_slow(self):
        """Test basic preproc, first k recordings in all OpenNeuro folders."""
        from eegprep import bids_preproc
        if self.root_path is None:
            self.skipTest("Skipping test_crashability_slow on unknown host")

        all_paths = sorted([d for d in os.listdir(self.root_path) if d.endswith('-download')])
        print(f"Found {len(all_paths)} BIDS datasets in {self.root_path}:")
        for p in all_paths:
            print(f" - {p}")

        for p in all_paths:
            bids_preproc(
                os.path.join(self.root_path, p),
                ReservePerJob=reservation,
                # process just the first few subjects/sessions/runs, across all tasks
                subjects=[0,1], sessions=[0,1], runs=[0,1,2],
                SkipIfPresent=True, # <- for quicker re-runs
                # maximal settings enabled to test everything that could go wrong
                # (except ICA/IClabel, which are too slow for a test)
                bidsevent=True,
                SamplingRate=128,
                WithInterp=True, EpochEvents=[], EpochLimits=[-0.2, 0.5], EpochBaseline=[None, 0],
                MinimizeDiskUsage=False)
