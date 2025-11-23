"""Logging utilities."""

import logging
import sys
import warnings

# Try importing colorama, handle if missing
try:
    import colorama
    _COLORAMA_AVAILABLE = True
    # Initialize colorama for Windows compatibility
    colorama.init()
except ImportError:
    _COLORAMA_AVAILABLE = False


logger = logging.getLogger(__name__)


class ColoredWarningFormatter(logging.Formatter):
    """A custom logging formatter that colors WARNING and ERROR/CRITICAL messages when
    outputting to a TTY.

    Uses colorama if available.
    """

    # ANSI color codes
    # Use colorama's constants if available, otherwise use raw ANSI codes
    # (these might not work on Windows without colorama)
    YELLOW = colorama.Fore.YELLOW if _COLORAMA_AVAILABLE else '\033[93m'
    RED = colorama.Fore.RED if _COLORAMA_AVAILABLE else '\033[91m'
    RESET = colorama.Style.RESET_ALL if _COLORAMA_AVAILABLE else '\033[0m'

    # Define log format strings
    log_format = '%(levelname)s (%(name)s) %(message)s'
    # Wrap the entire format string with color codes
    log_format_warn = f'{YELLOW}%(levelname)s (%(name)s) %(message)s{RESET}'
    log_format_error = f'{RED}%(levelname)s (%(name)s) %(message)s{RESET}'

    def __init__(self, fmt=log_format, datefmt=None, style='%'):
        """Initialize the formatter."""
        super().__init__(fmt=fmt, datefmt=datefmt, style=style)
        # Store formatters for different levels only if colorama is used
        if _COLORAMA_AVAILABLE:
            self.formats = {
                logging.WARNING: logging.Formatter(self.log_format_warn, datefmt=datefmt, style=style),
                logging.ERROR: logging.Formatter(self.log_format_error, datefmt=datefmt, style=style),
                logging.CRITICAL: logging.Formatter(self.log_format_error, datefmt=datefmt, style=style),
            }
        else:
            self.formats = {} # No special formats if no colorama


    def format(self, record):
        """Format the record with color if applicable."""
        if _COLORAMA_AVAILABLE:
            # Get the specialized formatter if one exists for this level
            formatter = self.formats.get(record.levelno)
            if formatter:
                # Use the level-specific formatter
                return formatter.format(record)
            else:
                # No specific formatter for this level, use the default
                # formatting by calling the base class's format method.
                return super().format(record)
        else:
            # If colorama is not available, just use the base class format method
            return super().format(record)


def setup_logging(level=logging.INFO, only_if_unset=True):
    """Configure logging for the application.

    Sets up a handler that writes to stderr. If running in a TTY and
    'colorama' is installed, it uses ColoredWarningFormatter to colorize
    warnings (yellow) and errors/criticals (red). Otherwise, uses standard
    formatting.

    Parameters
    ----------
    level : int
        The minimum logging level to output (e.g., logging.INFO, logging.DEBUG).
    only_if_unset : bool
        If True (default), configuration is skipped if the
        root logger already has handlers configured.
    """
    root_logger = logging.getLogger() # Get the root logger

    # Check if handlers already exist and if we should skip configuration
    if only_if_unset and root_logger.hasHandlers():
        logging.getLogger(__name__).debug(
            "Root logger already has handlers, skipping setup_logging."            
        )
        return # Skip configuration

    # --- Proceed with configuration ---
    # Set level *after* checking handlers, so we don't change level if skipping.
    root_logger.setLevel(level) # Set the minimum level for the logger itself

    # Configure console handler
    console_handler = logging.StreamHandler(sys.stderr)

    # Determine formatter based on TTY and colorama availability
    if _COLORAMA_AVAILABLE and console_handler.stream.isatty():
        formatter = ColoredWarningFormatter()
        console_handler.setFormatter(formatter)
    elif not _COLORAMA_AVAILABLE:
         warnings.warn(
             "Optional dependency 'colorama' not found. Log colors will be disabled. "
             "Install with: pip install colorama",
             ImportWarning
         )
         # Fallback to standard formatter if colorama is missing
         formatter = logging.Formatter(ColoredWarningFormatter.log_format)
         console_handler.setFormatter(formatter)
    else:
        # Use standard formatter if not a TTY (e.g., redirecting to a file)
        formatter = logging.Formatter(ColoredWarningFormatter.log_format)
        console_handler.setFormatter(formatter)


    # Set the level for the handler (controls what messages the handler *outputs*)
    console_handler.setLevel(level)

    # Add the handler to the root logger
    root_logger.addHandler(console_handler)
