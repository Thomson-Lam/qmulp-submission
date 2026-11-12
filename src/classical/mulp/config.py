"""
This file contains helper functions that handles parameter values and edge cases for the MultiLayerRanker class in mulp.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

VALID_INFLUENCE_TYPES = ("inter", "intra", "combined")  # from mulp


"""
Checks whether the parquet dataset path that MuLP is initialized with actually exists and is a .parquet file.
"""


def validate_dataset_path(dataset: str | Path) -> Path:
    parquet_path = Path(dataset)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset}")
    if parquet_path.suffix.lower() != ".parquet":
        raise ValueError("Dataset is not a parquet file")
    return parquet_path


"""
Function to check that the values of stickiness are valid and accepted values, and handle edge cases.
"""


def validate_stickiness(stickiness: Any) -> float:
    if isinstance(stickiness, (bool, np.bool_)):
        raise ValueError(f"stickiness must be numeric and > 0; got boolean {stickiness!r}")
    try:
        value = float(stickiness)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"stickiness must be numeric and > 0; got {stickiness!r}") from exc
    if np.isnan(value):
        raise ValueError("stickiness must not be NaN")
    if np.isneginf(value):
        raise ValueError("stickiness must not be negative infinity")
    if value <= 0.0:
        raise ValueError(f"stickiness must be > 0 or positive infinity; got {value}")
    return value


"""
Check influence type values and convert to lower case for the class to use.
"""


def normalize_influence_type(value: str) -> str:
    """Lowercase and validate a MuLP influence type."""
    normalized = value.lower()
    if normalized not in VALID_INFLUENCE_TYPES:
        raise ValueError("Invalid influence type. Must be inter, intra or combined.")
    return normalized
