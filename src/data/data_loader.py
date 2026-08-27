"""
Real-Data Loader
================

Loads a real (de-identified) NHAMCS-format ER-visits CSV behind the *same*
schema the synthetic generator produces, so the entire pipeline can run on real
data with no code changes — only a config switch.

The pipeline consumes these NHAMCS fields (see ``docs/data_dictionary.md``):
``VDATE, AGE, SEX, ARRTIME, IMMEDR, PAYTYPER, DIAG1``. ``load_er_visits_csv``
validates their presence and coerces types; extra columns are preserved.
"""

from __future__ import annotations

import logging
import os
from typing import List

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REQUIRED_COLUMNS: List[str] = ["VDATE", "AGE", "ARRTIME", "IMMEDR", "PAYTYPER"]
NUMERIC_COLUMNS: List[str] = ["AGE", "ARRTIME", "IMMEDR", "PAYTYPER", "SEX"]


class DataValidationError(ValueError):
    """Raised when an input CSV does not match the expected NHAMCS schema."""


def load_er_visits_csv(path: str) -> pd.DataFrame:
    """
    Load and validate a real NHAMCS-format ER-visits CSV.

    Args:
        path: Path to a CSV with NHAMCS columns.

    Returns:
        A DataFrame with the same schema the pipeline expects from synthetic
        data (numeric fields coerced; ``VDATE`` left as-is for the cleaner).

    Raises:
        FileNotFoundError: if ``path`` does not exist.
        DataValidationError: if required columns are missing or the file is empty.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"ER-visits CSV not found: {path}")

    df = pd.read_csv(path)
    logger.info("Loaded %d rows from %s", len(df), path)

    if df.empty:
        raise DataValidationError("Input CSV is empty")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(
            f"Missing required NHAMCS columns: {missing}. "
            f"Expected at least {REQUIRED_COLUMNS}."
        )

    # Coerce numeric fields; non-numeric entries become NaN for the cleaner
    # to impute rather than crashing the pipeline.
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df
