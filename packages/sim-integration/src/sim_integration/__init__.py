"""Core simulation integration library."""

__version__ = "0.1.0"

from sim_integration.dataloader import ForecastingDataset, load_parquet
from sim_integration.evaluator import ForecastMetrics, compute_metrics
from sim_integration.methods.base import ForecastMethod
from sim_integration.methods.const_vel import ConstantVelocityMethod
from sim_integration.runner import Pipeline

__all__ = [
    "ForecastingDataset",
    "load_parquet",
    "ForecastMetrics",
    "compute_metrics",
    "ForecastMethod",
    "ConstantVelocityMethod",
    "Pipeline",
]
