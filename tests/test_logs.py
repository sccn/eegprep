import logging
import unittest
import contextlib
import io
from unittest.mock import patch

from eegprep.functions.adminfunc.logs import setup_logging


class TestLogs(unittest.TestCase):
    @contextlib.contextmanager
    def _preserve_root_logger(self):
        root_logger = logging.getLogger()
        old_level = root_logger.level
        old_handlers = list(root_logger.handlers)
        try:
            # Remove existing handlers to isolate tests
            for h in list(root_logger.handlers):
                root_logger.removeHandler(h)
            yield root_logger
        finally:
            # Restore original handlers and level
            for h in list(root_logger.handlers):
                root_logger.removeHandler(h)
            for h in old_handlers:
                root_logger.addHandler(h)
            root_logger.setLevel(old_level)

    def test_setup_logging_idempotent_no_duplicate_handlers(self):
        with self._preserve_root_logger() as root:
            self.assertEqual(len(root.handlers), 0)
            setup_logging()
            self.assertEqual(len(root.handlers), 1)
            # Call again; should skip due to only_if_unset=True default
            setup_logging()
            self.assertEqual(len(root.handlers), 1)

    def test_setup_logging_level_switch_and_formatting(self):
        with self._preserve_root_logger():
            # Capture stderr
            captured_err = io.StringIO()
            with patch('sys.stderr', captured_err):
                setup_logging(level=logging.INFO)
                logging.debug("dbg")
                logging.info("hello")
                err = captured_err.getvalue()
                # Debug should not appear at INFO level
                self.assertNotIn("dbg", err)
                # Format should contain level, name, and message
                self.assertIn("INFO (root) hello", err)

        with self._preserve_root_logger():
            captured_err = io.StringIO()
            with patch('sys.stderr', captured_err):
                setup_logging(level=logging.DEBUG)
                logging.debug("dbg2")
                err = captured_err.getvalue()
                self.assertIn("DEBUG (root) dbg2", err)


if __name__ == '__main__':
    unittest.main()
