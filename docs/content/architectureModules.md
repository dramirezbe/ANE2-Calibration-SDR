

## **DOCUMENTO 3: MÓDULOS & ARQUITECTURA**


# MODULE ARCHITECTURE & API REFERENCE

### 0. MODULE DEPENDENCY MAP 


```bash


IMPORTS ONLY FROM:

data/loader.py
    ├─ pandas
    ├─ pathlib
    └─ logging

data/validator.py
    ├─ pandas
    ├─ numpy
    └─ logger.py

processing/noise_floor.py
    ├─ numpy
    └─ logger.py

processing/normalization.py
    ├─ numpy
    └─ logger.py

processing/correlation.py
    ├─ numpy
    ├─ scipy.stats
    ├─ logger.py
    └─ itertools

pipeline.py
    ├─ data/loader.py
    ├─ data/validator.py
    ├─ processing/*
    ├─ config.py
    ├─ logger.py
    └─ visualization/*

visualization/*
    ├─ matplotlib
    ├─ numpy
    └─ logger.py

NO CIRCULAR IMPORTS!
```

### 1. FLOW DIAGRAM (Módulos Interconectados)

```bash

┌─────────────────────────────────────────────────────────┐
│ USER ENTRY POINT │
│ (Notebook / Python Script / CLI) │
└──────────────────┬──────────────────────────────────────┘
│
┌──────────▼──────────┐
│ MAIN PIPELINE │
│ (pipeline.py) │
└──┬────────┬────────┬┘
│ │ │
┌────────▼──┐ ┌──▼───────┐ ┌──▼─────────┐
│ DATA │ │PROCESSING│ │ VISUAL │
│ LAYER │ │ LAYER │ │ LAYER │
└────────┬──┘ └──┬───────┘ └──┬─────────┘
│ │ │
┌────┴───┬───┴───┐ ┌────┴─────┐
▼ ▼ ▼ ▼ ▼
data/ process config visual logger
loader core mgmt output
```


### 2. MÓDULO: DATA LAYER

#### **Archivo: `src/rf_spectrum/data/loader.py`**

```python
"""Data loading and management."""

from typing import Dict, Optional, Set
import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class CSVDataLoader:
    """Load and manage RF spectrum CSV data."""
    
    def __init__(self, exclude_nodes: Optional[Set[str]] = None):
        """Initialize loader with optional exclusion list."""
        self.exclude = exclude_nodes or set()
    
    def load_node(self, filepath: str) -> pd.DataFrame:
        """Load single CSV file with validation.
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            DataFrame with columns: timestamp, pxx, latitude, longitude
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If CSV structure invalid
        """
        ...
    
    def load_all_nodes(
        self,
        directory: str,
        pattern: str = "Node*.csv"
    ) -> Dict[str, pd.DataFrame]:
        """Load all node CSVs in directory.
        
        Args:
            directory: Path containing CSV files
            pattern: Glob pattern for file matching
            
        Returns:
            Dictionary {node_name: DataFrame}
        """
        ...

def validate_dataframe(df: pd.DataFrame) -> bool:
    """Validate DataFrame structure and content."""
    required_cols = {'timestamp', 'pxx', 'latitude', 'longitude'}
    return required_cols.issubset(df.columns)


```

#### Archivo: src/rf_spectrum/data/validator.py

```py

"""Data validation and quality checks."""

class DataValidator:
    """Validate spectrum data integrity."""
    
    @staticmethod
    def check_pxx_format(pxx_str: str) -> bool:
        """Validate pxx string is valid Python list."""
        ...
    
    @staticmethod
    def check_value_ranges(df: pd.DataFrame) -> Dict[str, bool]:
        """Check PSD values in realistic range [-120, -40] dB."""
        ...
    
    @staticmethod
    def report_anomalies(
        df: pd.DataFrame,
        nodes: Dict[str, pd.DataFrame]
    ) -> Dict[str, list]:
        """Generate anomaly report across all nodes."""
        ...

```



### 3. MODULO PROCESSING LAYER 

#### Archivo: src/rf_spectrum/processing/noise_floor.py

```py

"""Noise floor estimation algorithms."""

class NoiseFloorEstimator:
    """Estimate baseline noise from PSD."""
    
    @staticmethod
    def histogram_mode(
        pxx: np.ndarray,
        nbins: Union[int, str] = 'sturges'
    ) -> float:
        """Estimate noise floor using histogram mode.
        
        Uses Sturges' rule for adaptive bin sizing:
        nbins = ceil(log2(n)) + 1
        
        Args:
            pxx: Power array [shape: (n,)]
            nbins: Number of histogram bins or 'sturges'
            
        Returns:
            Noise floor estimate in dB
        """
        if isinstance(nbins, str) and nbins == 'sturges':
            nbins = int(np.ceil(np.log2(len(pxx)) + 1))
        
        counts, edges = np.histogram(pxx, bins=nbins)
        modal_idx = np.argmax(counts)
        return 0.5 * (edges[modal_idx] + edges[modal_idx + 1])
    
    @staticmethod
    def percentile_method(pxx: np.ndarray, p: float = 5) -> float:
        """Alternative: percentile-based noise floor."""
        ...


```

#### Archivo: src/rf_spectrum/processing/normalization.py


```python

"""Spectral correlation analysis."""

from scipy.stats import pearsonr
from itertools import combinations

class CorrelationAnalyzer:
    """Compute spectral correlations."""
    
    @staticmethod
    def pairwise_correlations(
        pxx_dict: Dict[str, np.ndarray],
        normalize: bool = True
    ) -> Tuple[np.ndarray, List[str]]:
        """Compute pairwise Pearson correlation matrix.
        
        Args:
            pxx_dict: {node_name: spectrum_array}
            normalize: Apply z-score normalization
            
        Returns:
            (correlation_matrix, sorted_node_names)
        """
        node_names = sorted(pxx_dict.keys())
        n = len(node_names)
        corr_matrix = np.eye(n)
        
        for i, j in combinations(range(n), 2):
            xi = pxx_dict[node_names[i]]
            xj = pxx_dict[node_names[j]]
            
            if normalize:
                xi = (xi - np.mean(xi)) / (np.std(xi) + 1e-10)
                xj = (xj - np.mean(xj)) / (np.std(xj) + 1e-10)
            
            r, _ = pearsonr(xi, xj)
            corr_matrix[i, j] = r
            corr_matrix[j, i] = r
        
        return corr_matrix, node_names
    
    @staticmethod
    def cumulative_scores(corr_matrix: np.ndarray) -> np.ndarray:
        """Compute cumulative correlation per node."""
        return np.sum(corr_matrix, axis=1) - 1.0
    
    @staticmethod
    def rank_nodes(
        cumulative_scores: np.ndarray,
        node_names: List[str]
    ) -> Dict[str, float]:
        """Rank nodes by correlation score."""
        scores_dict = dict(zip(node_names, cumulative_scores))
        return dict(sorted(
            scores_dict.items(),
            key=lambda x: x[1],
            reverse=True
        ))
```
#### 4. MÓDULO MAIN PIPELINE

#### Archivo: src/rf_spectrum/pipeline.py

```python 
"""Main analysis pipeline orchestrator."""

from .data.loader import CSVDataLoader
from .processing.correlation import CorrelationAnalyzer
from .config import Config

class SpectrumAnalysisPipeline:
    """Orchestrate complete RF analysis pipeline."""
    
    def __init__(self, config: Config):
        self.config = config
        self.loader = CSVDataLoader(exclude_nodes=config.excluded_nodes)
        self.analyzer = CorrelationAnalyzer()
    
    def run(
        self,
        data_dir: str,
        output_dir: str
    ) -> Dict[str, any]:
        """Execute complete pipeline.
        
        Pipeline stages:
        1. Load data from CSVs
        2. Estimate noise floors
        3. Normalize spectra
        4. Compute correlations
        5. Rank nodes
        6. Generate reports
        
        Args:
            data_dir: Directory with CSV files
            output_dir: Where to save results
            
        Returns:
            Dictionary with all results
        """
        logger.info("Starting RF spectrum analysis pipeline")
        
        # Stage 1: Load
        nodes_data = self.loader.load_all_nodes(data_dir)
        logger.info(f"Loaded {len(nodes_data)} nodes")
        
        # Stage 2-5: Process
        results = self._analyze_all_records(nodes_data)
        
        # Stage 6: Export
        self._save_results(results, output_dir)
        logger.info("Pipeline complete")
        
        return results
    
    def _analyze_all_records(
        self,
        nodes_data: Dict[str, pd.DataFrame]
    ) -> Dict:
        """Process all records and return aggregated results."""
        ...

```

