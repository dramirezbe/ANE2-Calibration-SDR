# ANE2-Calibration-SDR

> Python toolkit for calibrating and monitoring a distributed network of Software Defined Radio (SDR) sensors managed by **ANE** (Agencia Nacional del Espectro – Colombia). It pulls real-time and historical RF measurement data from the ANE Remote Spectrum Monitoring (RSM) REST API, organises it in pandas DataFrames, and exposes it through Jupyter notebooks for interactive analysis.

**Version:** 0.1.0

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Installation](#installation)
4. [Environment Configuration (`.env`)](#environment-configuration-env)
5. [Configuration Module (`cfg.py`)](#configuration-module-cfgpy)
6. [Architecture & Data Flow](#architecture--data-flow)
7. [Source Code Walkthrough](#source-code-walkthrough)
8. [Developing New Libraries](#developing-new-libraries)
9. [Jupyter Notebook Examples](#jupyter-notebook-examples)
10. [Building the Documentation](#building-the-documentation)

---

## Project Overview

| Term | Meaning |
|------|---------|
| **ANE** | *Agencia Nacional del Espectro* – the Colombian national spectrum authority. |
| **SDR** | *Software Defined Radio* – radio systems whose components (filters, modulators, etc.) are implemented in software. |
| **Calibration** | The process of comparing sensor readings under different SDR configurations (e.g. DC-offset on/off, IQ correction on/off) to find optimal settings. |
| **RSM API** | The REST API at `https://rsm.ane.gov.co:12443/api` that serves campaign and real-time data from a network of 10 sensor nodes. |

The project allows you to:

* Fetch **real-time** power-spectrum data (`pxx`) from any sensor node.
* Retrieve **campaign** configurations (schedule, frequency, gain, bandwidth, etc.).
* Download **bulk historical signals** with automatic pagination.
* Load data for multiple campaigns and nodes at once into **pandas DataFrames** for analysis.

---

## Repository Structure

```
ANE2-Calibration-SDR/
├── .env.example                       # Template for environment variables
├── .gitignore                         # Git ignore rules
├── .readthedocs.yaml                  # Read the Docs build configuration
├── README.md                          # This file
├── install.sh                         # Linux / macOS setup script
├── install.ps1                        # Windows PowerShell setup script
├── requirements.txt                   # Python runtime dependencies
│
├── docs/                              # Sphinx documentation sources
│   ├── conf.py                        # Sphinx configuration
│   ├── index.rst                      # Documentation index page
│   ├── Makefile                       # Build docs on Unix
│   ├── make.bat                       # Build docs on Windows
│   ├── requirements.txt              # Docs-only dependencies
│   ├── _static/                       # Static assets for docs
│   └── _templates/                    # Custom Sphinx templates
│
└── src/                               # Main source code
    ├── cfg.py                         # Configuration & logging setup
    ├── main.py                        # Application entry point
    ├── example-realtime.ipynb         # Notebook: real-time signal monitoring
    ├── example-campaign_nodes.ipynb   # Notebook: campaign data analysis
    └── libs/                          # Reusable library modules
        ├── __init__.py                # Package marker
        └── data_request.py            # API client & data structures
```

---

## Installation

### Prerequisites

* **Python 3.11+**
* **pip** (comes with Python)

### Linux / macOS

```bash
git clone https://github.com/dramirezbe/ANE2-Calibration-SDR.git
cd ANE2-Calibration-SDR
bash install.sh                 # creates venv/, installs deps
source venv/bin/activate        # activate the virtual environment
```

### Windows (PowerShell)

```powershell
git clone https://github.com/dramirezbe/ANE2-Calibration-SDR.git
cd ANE2-Calibration-SDR
.\install.ps1                   # creates .venv\, installs deps
.\.venv\Scripts\Activate.ps1    # activate the virtual environment
```

Both scripts perform the same steps:

1. Create a Python virtual environment (`venv/` on Linux, `.venv/` on Windows).
2. Upgrade `pip`.
3. Install the packages listed in `requirements.txt`.

### Running the app

```bash
cd src
python main.py
```

### Running notebooks

```bash
cd src
jupyter notebook
# Open example-realtime.ipynb or example-campaign_nodes.ipynb
```

---

## Environment Configuration (`.env`)

Runtime settings are managed through a **`.env`** file in the project root. The file is loaded automatically by `cfg.py` using the [`python-dotenv`](https://pypi.org/project/python-dotenv/) library.

### How it works

1. When any module imports `cfg`, the function `ensure_env_file()` runs automatically.
2. If `.env` does **not** exist yet, it is copied from `.env.example`.
3. `python-dotenv` then loads every `KEY=value` pair from `.env` into `os.environ`.
4. Each variable is read with `os.getenv("KEY", default)`, so even without a `.env` file the application falls back to sensible defaults.

> **Important:** `.env` is listed in `.gitignore` – it is never committed. Only `.env.example` is tracked in version control.

### Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `API_URL` | `https://rsm.ane.gov.co:12443/api` | Base URL of the ANE RSM REST API. |
| `VERBOSE` | `false` | When `true`, console log level is set to **INFO**. |
| `DEBUG` | `false` | When `true`, console log level is set to **DEBUG** (overrides `VERBOSE`). |
| `DEVELOPMENT` | `false` | Flag for development-specific behaviour (reserved for future use). |
| `COUNTRY` | `America/Bogota` | Timezone identifier used for date/time operations. |
| `APP_NAME` | `ANE2-Calibration-SDR` | Application display name. |
| `APP_VERSION` | `0.1.0` | Current version string. |

### Example `.env` file

```dotenv
API_URL=https://rsm.ane.gov.co:12443/api
VERBOSE=true
DEBUG=true
DEVELOPMENT=true
COUNTRY=America/Bogota
APP_NAME=ANE2-Calibration-SDR
APP_VERSION=0.1.0
```

---

## Configuration Module (`cfg.py`)

`src/cfg.py` is the **central configuration hub**. Every other module imports it to access environment settings and a pre-configured logger.

### What `cfg.py` does on import

```
1. Compute paths
      SRC_DIR  → directory containing cfg.py  (src/)
      ROOT_DIR → project root                 (one level above src/)

2. Call ensure_env_file()
      If .env is missing, copy .env.example → .env

3. Load .env into os.environ
      Uses python-dotenv

4. Expose configuration constants
      API_URL, VERBOSE, DEBUG, DEVELOPMENT, COUNTRY, APP_NAME, APP_VERSION

5. Define the logger factory  →  set_logger()
```

### Logger

`set_logger()` creates a `logging.Logger` whose console level depends on the environment flags:

| `DEBUG` | `VERBOSE` | Console Level |
|---------|-----------|---------------|
| `true`  | any       | `DEBUG`       |
| `false` | `true`    | `INFO`        |
| `false` | `false`   | `ERROR`       |

Log lines follow the format:

```
[SCRIPT_NAME]LEVEL     message
```

For example:

```
[MAIN]    INFO      Starting ANE2-Calibration-SDR v0.1.0 in America/Bogota...
```

### Using `cfg.py` in your code

```python
import cfg

log = cfg.set_logger()

log.info(f"API URL is {cfg.API_URL}")
log.debug(f"Debug mode: {cfg.DEBUG}")
```

### Running cfg.py directly (diagnostics)

```bash
cd src
python cfg.py
```

This prints a diagnostics report showing paths, flags, and network settings – useful when debugging environment issues.

---

## Architecture & Data Flow

```
┌─────────────────────────────────────────────────────┐
│                   .env  file                        │
│  API_URL, DEBUG, VERBOSE, COUNTRY, APP_*, …         │
└──────────────────────┬──────────────────────────────┘
                       │  loaded by python-dotenv
                       ▼
┌─────────────────────────────────────────────────────┐
│                 cfg.py  module                       │
│  • Exposes configuration constants                  │
│  • Provides set_logger() for per-module logging     │
└──────────────────────┬──────────────────────────────┘
                       │  imported by
          ┌────────────┴────────────┐
          ▼                         ▼
   ┌────────────┐         ┌──────────────────┐
   │  main.py   │         │ Jupyter notebooks│
   └─────┬──────┘         └────────┬─────────┘
         │                         │
         └────────────┬────────────┘
                      │  use
                      ▼
   ┌──────────────────────────────────────────────────┐
   │         DataRequest  (data_request.py)           │
   │  • Node ↔ MAC address mapping (10 nodes)        │
   │  • get_realtime_signal(node_id)                  │
   │  • get_campaign_params(campaign_id)              │
   │  • get_api_signals(mac, camp_id)                 │
   │  • load_campaigns_and_nodes(campaigns, node_ids) │
   └──────────────────────┬───────────────────────────┘
                          │  HTTP (requests, verify=False)
                          ▼
   ┌──────────────────────────────────────────────────┐
   │            ANE RSM REST API                      │
   │  https://rsm.ane.gov.co:12443/api                │
   │                                                  │
   │  /campaigns/sensor/{MAC}/realtime                │
   │  /campaigns/{id}/parameters                      │
   │  /campaigns/sensor/{MAC}/signals                 │
   └──────────────────────────────────────────────────┘
```

---

## Source Code Walkthrough

### `src/main.py` – Entry point

```python
import cfg
log = cfg.set_logger()
from libs.data_request import DataRequest

def main():
    log.info(f"Starting {cfg.APP_NAME} v{cfg.APP_VERSION} in {cfg.COUNTRY}...")
    dr = DataRequest(log=log, base_url=cfg.API_URL)
    log.info(f"Object class DataRequest print = {dr}")

if __name__ == "__main__":
    main()
```

* Imports `cfg` first so that the environment is loaded before anything else.
* Creates a `DataRequest` instance wired to the configured API URL and logger.
* Currently acts as a skeleton – extend it with your own logic.

### `src/libs/data_request.py` – API client

#### Dataclasses

| Class | Fields | Purpose |
|-------|--------|---------|
| `ScheduleParams` | `start_date`, `end_date`, `start_time`, `end_time`, `interval_seconds` | Measurement schedule for a campaign. |
| `ConfigParams` | `rbw`, `span`, `antenna`, `lna_gain`, `vga_gain`, `antenna_amp`, `center_freq_hz`, `sample_rate_hz`, `centerFrequency` | SDR hardware configuration for a campaign. |
| `CampaignParams` | `name`, `schedule` (ScheduleParams), `config` (ConfigParams) | Combined campaign metadata. |

#### `DataRequest` class

| Method | Endpoint | Returns | Description |
|--------|----------|---------|-------------|
| `get_realtime_signal(node_id)` | `GET /campaigns/sensor/{MAC}/realtime` | `RealtimeSignal` | Fetches the latest power spectrum (`pxx`) and frequency range for a sensor node. |
| `get_campaign_params(campaign_id)` | `GET /campaigns/{id}/parameters` | `CampaignParams` | Returns the schedule and SDR config for a campaign. |
| `get_api_signals(mac, camp_id)` | `GET /campaigns/sensor/{MAC}/signals` | `list[dict]` | Downloads **all** signal measurements for a sensor/campaign, handling pagination automatically (page size 5000). |
| `load_campaigns_and_nodes(campaigns, node_ids)` | *(wraps `get_api_signals`)* | `dict[str, dict[str, DataFrame]]` | Bulk-loads signals for multiple campaigns × nodes. Returns a nested dict: `{ campaign_label: { "NodeN": DataFrame } }`. |

**Node mapping** — the constructor defines a dictionary mapping node IDs (1–10) to their MAC addresses:

```python
self.node_macs = {
    1: 'd8:3a:dd:f7:1d:f2',  2: 'd8:3a:dd:f4:4e:26',  ...
    10: 'd8:3a:dd:f7:1d:90'
}
```

---

## Developing New Libraries

All reusable modules live under **`src/libs/`**. To add a new library:

### 1. Create the module

```bash
touch src/libs/my_module.py
```

### 2. Import configuration and logging

```python
# src/libs/my_module.py
import cfg

log = cfg.set_logger()

class MyProcessor:
    def __init__(self):
        log.info("MyProcessor initialised")
        self.api_url = cfg.API_URL
        # ...
```

### 3. Use it from `main.py` or a notebook

```python
# src/main.py
from libs.my_module import MyProcessor

proc = MyProcessor()
```

### 4. Guidelines

* **Always import `cfg` first** in every module that needs configuration or logging – it ensures `.env` is loaded before any other code runs.
* Keep your module in `src/libs/` so that Sphinx autodoc can discover it.
* Use the existing `DataRequest` class to interact with the API; extend it or compose it rather than duplicating HTTP logic.
* Follow the existing pattern of dataclasses (`ScheduleParams`, `ConfigParams`, `CampaignParams`) to define structured objects for new API responses.
* Add docstrings to your functions and classes – Sphinx with the Napoleon extension supports both Google and NumPy docstring styles.

---

## Jupyter Notebook Examples

Two notebooks in `src/` demonstrate typical workflows:

### `example-realtime.ipynb` – Real-time Signal Monitoring

* Creates a `DataRequest` instance.
* Polls `get_realtime_signal()` in a loop.
* Plots the power spectrum only when new data arrives (avoids redundant redraws).
* Useful for monitoring a single sensor node in real time.

### `example-campaign_nodes.ipynb` – Campaign Data Analysis

* Defines a set of campaign IDs that represent different SDR configurations:

  ```python
  camp_ids = {
      'no dc, no iq': 202,
      'no dc, yes iq': 203,
      'yes dc, yes iq': 204,
      'FM original': 176,
  }
  ```

* Selects target nodes: `[1, 2, 3, 5, 9]`.
* Uses `load_campaigns_and_nodes()` to bulk-download all measurements into DataFrames.
* Enables comparison of signal quality across configurations to support calibration decisions.

---

## Building the Documentation

The project uses [Sphinx](https://www.sphinx-doc.org/) with the **Read the Docs** theme. Documentation is auto-built on [Read the Docs](https://readthedocs.org/) via `.readthedocs.yaml`.

### Build locally

```bash
# Install docs dependencies
pip install -r docs/requirements.txt

# Build HTML
cd docs
make html          # Linux / macOS
.\make.bat html    # Windows
```

The output will be in `docs/_build/html/`.

### Sphinx extensions in use

| Extension | Purpose |
|-----------|---------|
| `sphinx.ext.autodoc` | Generates API docs from Python docstrings. |
| `sphinx.ext.viewcode` | Adds links to highlighted source code. |
| `sphinx.ext.napoleon` | Supports Google-style and NumPy-style docstrings. |