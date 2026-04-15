"""
Central logger for VibeFinder 2.0

Every module calls get_logger(__name__) to get a pre-configured logger.
Logs are written to:
  - Console   : INFO and above (human-readable)
  - File      : DEBUG and above (full detail for debugging)

Log file location: logs/vibefinder.log  (relative to project root)
"""

import logging
import os
from logging.handlers import RotatingFileHandler

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LOG_DIR  = os.path.join(_BASE_DIR, "logs")
_LOG_FILE = os.path.join(_LOG_DIR, "vibefinder.log")

_CONSOLE_FORMAT = "%(levelname)-8s %(name)s — %(message)s"
_FILE_FORMAT    = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT    = "%Y-%m-%d %H:%M:%S"

# Track whether root setup has already run so we don't duplicate handlers
_configured = False


def _setup() -> None:
    global _configured
    if _configured:
        return

    os.makedirs(_LOG_DIR, exist_ok=True)

    root = logging.getLogger("vibefinder")
    root.setLevel(logging.DEBUG)          # capture everything; handlers filter

    # Console handler — INFO and above, clean format
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(_CONSOLE_FORMAT))
    root.addHandler(console)

    # Rotating file handler — DEBUG and above, full timestamps
    # Max 1 MB per file, keep 3 backups
    fh = RotatingFileHandler(
        _LOG_FILE, maxBytes=1_000_000, backupCount=3, encoding="utf-8"
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(_FILE_FORMAT, datefmt=_DATE_FORMAT))
    root.addHandler(fh)

    # Silence noisy third-party loggers
    for noisy in ("sentence_transformers", "transformers", "chromadb",
                  "httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """
    Return a child logger under the 'vibefinder' namespace.

    Usage:
        from src.logger import get_logger
        log = get_logger(__name__)
        log.info("Something happened")
        log.warning("Something suspicious")
        log.error("Something failed", exc_info=True)
    """
    _setup()
    # Strip 'src.' prefix so log names are short: 'agent', 'retriever', etc.
    short_name = name.removeprefix("src.")
    return logging.getLogger(f"vibefinder.{short_name}")
