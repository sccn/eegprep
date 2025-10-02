import logging
import unittest
import contextlib
import io
import sys
import warnings
from unittest.mock import patch

from eegprep.utils.logs import setup_logging, ColoredWarningFormatter


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
    
    def _make_record(self, name: str, level: int, msg: str) -> logging.LogRecord:
        return logging.LogRecord(name=name, level=level, pathname=__file__, lineno=1, msg=msg, args=(), exc_info=None)
    
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
    
    def test_setup_logging_only_if_unset_true_skips_reconfig(self):
        with self._preserve_root_logger() as root:
            setup_logging(level=logging.INFO)
            self.assertEqual(root.level, logging.INFO)
            # Try to change level, but should skip due to existing handlers
            setup_logging(level=logging.DEBUG, only_if_unset=True)
            self.assertEqual(root.level, logging.INFO)
            self.assertEqual(len(root.handlers), 1)
    
    def test_setup_logging_only_if_unset_false_adds_handler_and_changes_level(self):
        with self._preserve_root_logger() as root:
            setup_logging(level=logging.INFO)
            self.assertEqual(len(root.handlers), 1)
            setup_logging(level=logging.DEBUG, only_if_unset=False)
            # A second handler should have been added and level updated
            self.assertEqual(len(root.handlers), 2)
            self.assertEqual(root.level, logging.DEBUG)
    
    def test_colored_warning_formatter_branches(self):
        # Non-color branch
        with patch('eegprep.utils.logs._COLORAMA_AVAILABLE', False):
            fmt = ColoredWarningFormatter()
            out_warn = fmt.format(self._make_record("nm", logging.WARNING, "warnmsg"))
            out_info = fmt.format(self._make_record("nm", logging.INFO, "infomsg"))
            self.assertEqual(out_warn, "WARNING (nm) warnmsg")
            self.assertEqual(out_info, "INFO (nm) infomsg")
        
        # Color branch uses specialized format for WARNING/ERROR and default for INFO
        with patch('eegprep.utils.logs._COLORAMA_AVAILABLE', True):
            fmt2 = ColoredWarningFormatter()
            out_warn2 = fmt2.format(self._make_record("nm2", logging.WARNING, "warn2"))
            out_err2 = fmt2.format(self._make_record("nm2", logging.ERROR, "err2"))
            out_info2 = fmt2.format(self._make_record("nm2", logging.INFO, "info2"))
            # Should contain the message and name; colored content also includes reset codes
            self.assertIn("warn2", out_warn2)
            self.assertIn("(nm2)", out_warn2)
            self.assertIn("err2", out_err2)
            self.assertIn("(nm2)", out_err2)
            self.assertEqual(out_info2, "INFO (nm2) info2")
    
    def test_setup_logging_warns_without_colorama(self):
        with self._preserve_root_logger():
            with patch('eegprep.utils.logs._COLORAMA_AVAILABLE', False):
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    captured_err = io.StringIO()
                    with patch('sys.stderr', captured_err):
                        setup_logging(level=logging.WARNING)
                        logging.warning("wmsg")
                        err = captured_err.getvalue()
                        # Standard formatter used and message printed
                        self.assertIn("WARNING (root) wmsg", err)
                    # Check that ImportWarning was raised
                    self.assertTrue(any(issubclass(warning.category, ImportWarning) for warning in w))


if __name__ == '__main__':
    unittest.main()


