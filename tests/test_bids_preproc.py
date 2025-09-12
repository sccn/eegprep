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
run_on_hosts = ['ck-carbon']

@unittest.skipIf(curhost not in run_on_hosts, f"Slow stress test skipped by default on hosts other than {run_on_hosts}")
class TestBidsPreprocSlow(DebuggableTestCase):
    """Basic test cases for clean_artifacts function."""

    def setUp(self):
        """Set up test fixtures."""
        if curhost == 'ck-carbon':
            self.root_path = os.path.expanduser('~/data/OpenNeuro')
        else:
            self.root_path = None
            logger.warning(f"Skipping test TestBidsPreprocSlow on unknown test host {host}; "
                           f"please add support for your hostname to the above list to "
                           f"enable this test.")

    def test_crashability(self):
        """Test basic preproc, first k recordings in all OpenNeuro folders."""
        from eegprep import bids_preproc
        all_paths = sorted([d for d in os.listdir(self.root_path) if d.endswith('-download')])
        print(f"Found {len(all_paths)} BIDS datasets in {self.root_path}:")
        for p in all_paths:
            print(f" - {p}")

        for p in all_paths:
            bids_preproc(
                os.path.join(self.root_path, p),
                # process just the first 2 subjects/sessions, across all runs/tasks
                subjects=[0,1], sessions=[0,1],
                SkipIfPresent=True, # <- for quicker re-runs
                # maximal settings enabled to test everything that could go wrong
                # (except ICA/IClabel, which are too slow for a test)
                bidsevent=True,
                SamplingRate=128,
                WithInterp=True, EpochEvents=[], EpochLimits=[-0.2, 0.5], EpochBaseline=[None, 0],
                MinimizeDiskUsage=False)
