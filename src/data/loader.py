"""Data loading module for housing prices dataset."""

import pandas as pd
from pathlib import Path

# Import shared utilities
from src.utils import get_logger

logger = get_logger(__name__)


def load_housing_data(data_path: str) -> pd.DataFrame:
    """Load housing prices dataset from CSV."""
    try:
        df = pd.read_csv(data_path)
        logger.info(f"Loaded housing data from {data_path}. Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data from {data_path}: {e}")
        raise


def save_processed_data(df: pd.DataFrame, output_path: str) -> None:
    """Save processed data to CSV."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved processed data to {output_path}")


def check_data_quality(df: pd.DataFrame) -> dict:
    """Perform basic data quality checks."""
    report = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_rows": df.duplicated().sum(),
    }
    logger.info(f"Data Quality Report: {report}")
    return report
