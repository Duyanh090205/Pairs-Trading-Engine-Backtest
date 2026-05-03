"""
Plan 0: Data Gateway
Ingests orderbook data, cleans anomalies without breaking time alignment, 
and extracts structural spread features.
"""

from .ingest import process_orderbook
from .features import compute_microstructure_features
from .seasonality import compute_intraday_seasonality
from .rolling import compute_rolling_instability

__all__ = [
    'process_orderbook',
    'compute_microstructure_features',
    'compute_intraday_seasonality',
    'compute_rolling_instability'
]
