#!/usr/bin/env python3

import os
import sys
import shutil
import logging
import pathlib
from dotenv import load_dotenv

# ==========================================
# 1. DIRECTORY CONFIGURATION & ENV LOADING
# ==========================================

SRC_DIR = pathlib.Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent 

def ensure_env_file() -> None:
    """Ensure a ``.env`` file exists in the project root directory.

    Checks for a ``.env`` file at ``ROOT_DIR``.  If it is missing, the
    function attempts to copy ``.env.example`` to ``.env`` automatically.
    If neither file is present a warning is printed to stdout.

    This function is called once at module import time so that all
    subsequent :func:`os.getenv` calls can rely on the environment being
    populated.
    """
    env_path = ROOT_DIR / ".env"
    env_example_path = ROOT_DIR / ".env.example"
    
    if not env_path.exists():
        if env_example_path.exists():
            shutil.copyfile(env_example_path, env_path)
            print(f"--- INFO: .env file automatically created from .env.example ---")
        else:
            print(f"--- WARNING: No .env or .env.example found in {ROOT_DIR} ---")

ensure_env_file()
load_dotenv(dotenv_path=ROOT_DIR / ".env")

# Environment & App Constants
API_URL = os.getenv("API_URL", "https://rsm/ane.gov.co/api")
VERBOSE = os.getenv("VERBOSE", "false").lower() == "true"
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
DEVELOPMENT = os.getenv("DEVELOPMENT", "false").lower() == "true"
COUNTRY = os.getenv("COUNTRY", "America/Bogota")
APP_NAME = os.getenv("APP_NAME", "ANE2-Calibration-SDR")
APP_VERSION = os.getenv("APP_VERSION", "0.1.0")

# ==========================================
# 2. LOGGER CONFIGURATION
# ==========================================

class SimpleFormatter(logging.Formatter):
    """A :class:`logging.Formatter` subclass with fixed-width level names.

    Pads the ``levelname`` field to nine characters so that log lines from
    different levels remain visually aligned.  Exception records receive the
    custom label ``EXCEPTION`` instead of ``ERROR`` to make them easier to
    spot in the console output.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format *record* with a fixed-width, human-readable level name.

        Parameters
        ----------
        record : logging.LogRecord
            The log record to format.

        Returns
        -------
        str
            The formatted log string produced by the parent formatter after
            the level name has been normalised.
        """
        if record.exc_info: 
            record.levelname = "EXCEPTION"
        record.levelname = f"{record.levelname:<9}"
        return super().format(record)

def set_logger() -> logging.Logger:
    """Create and configure the application-wide logger.

    The logger name is derived from the calling script's filename stem
    (upper-cased), falling back to ``"APP"`` when that cannot be determined.

    Console output level is determined by the environment flags loaded
    earlier in this module:

    * ``DEBUG=true``   → :data:`logging.DEBUG` (all messages).
    * ``VERBOSE=true`` → :data:`logging.INFO` (informational and above).
    * Otherwise        → :data:`logging.ERROR` (errors and exceptions only).

    A :class:`logging.StreamHandler` pointing to ``stdout`` is attached only
    if no ``StreamHandler`` already exists on the logger, preventing
    duplicate handlers when the module is reloaded.

    Returns
    -------
    logging.Logger
        Configured logger instance ready for use throughout the application.
    """
    try: 
        name = pathlib.Path(sys.argv[0]).stem.upper()
    except Exception: 
        name = "APP"

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG) 
    
    fmt = SimpleFormatter("[%(name)s]%(levelname)s %(message)s")

    if DEBUG:
        console_level = logging.DEBUG
    else:
        console_level = logging.INFO if VERBOSE else logging.ERROR

    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        c_handler = logging.StreamHandler(sys.stdout)
        c_handler.setLevel(console_level)
        c_handler.setFormatter(fmt)
        logger.addHandler(c_handler)

    return logger

log = set_logger()

#Always exec when cfg is imported:
log.info(f"Starting {APP_NAME} v{APP_VERSION} in {COUNTRY}...")

# ==========================================
# 3. SYSTEM DIAGNOSTICS (Main Log)
# ==========================================

if __name__ == "__main__":
    log.info("="*50)
    log.info(f"{APP_NAME} v{APP_VERSION}")
    log.info("="*50)
    
    # Paths
    log.info(f"DIRECTORY SETUP:")
    log.info(f"  - SRC_DIR:   {SRC_DIR}")
    log.info(f"  - ROOT_DIR:  {ROOT_DIR}")
    
    # Environment Flags
    log.info(f"ENVIRONMENT FLAGS:")
    log.info(f"  - DEBUG:       {DEBUG}")
    log.info(f"  - VERBOSE:     {VERBOSE}")
    log.info(f"  - DEVELOPMENT: {DEVELOPMENT}")
    
    # Regional & Network
    log.info(f"NETWORK & LOCALIZATION:")
    log.info(f"  - API_URL:     {API_URL}")
    log.info(f"  - COUNTRY:     {COUNTRY}")
    
    log.info("="*50)
    log.debug("System initialized in DEBUG mode.")