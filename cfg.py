"""
cfg.py - Centralised configuration using python-dotenv.

Copy .env.example to .env and adjust values before running.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Environment
DEVELOPMENT: bool = os.getenv("DEVELOPMENT", "true").lower() == "true"
DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

# Application
APP_NAME: str = os.getenv("APP_NAME", "ANE2-Calibration-SDR")
APP_VERSION: str = os.getenv("APP_VERSION", "0.1.0")
SECRET_KEY: str = os.getenv("SECRET_KEY", "")

if not DEVELOPMENT and not SECRET_KEY:
    raise ValueError("SECRET_KEY must be set in production. Copy .env.example to .env and set a strong value.")

# Paths
DATA_DIR: str = os.getenv("DATA_DIR", "data/")
OUTPUT_DIR: str = os.getenv("OUTPUT_DIR", "output/")
