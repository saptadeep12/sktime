"""Simons: small, open time series foundation models for probabilistic forecasting.

Source: https://github.com/Abel-ai-lab/Simons
Vendored at commit 8e338e915d26c8960713889ef12ab1cf5c08d591 under Apache-2.0.
Model weights are not bundled; see the Simons model license.
sktime migration: 2026-11-13.
Original authors: Abel AI Lab (https://github.com/Abel-ai-lab),
Shu-Wan (https://github.com/Shu-Wan), and ruizhu-lgtm
(https://github.com/ruizhu-lgtm).
"""

# ruff: noqa

from .config import SimonsConfig
from .covariates import encode_categorical, prepare_covariates
from .forecast import QUANTILES, forecast, forecast_items, forecast_series
from .model import Simons

__all__ = [
    "QUANTILES",
    "Simons",
    "SimonsConfig",
    "encode_categorical",
    "forecast",
    "forecast_items",
    "forecast_series",
    "prepare_covariates",
]
