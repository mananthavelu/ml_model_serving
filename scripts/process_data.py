"""Data pipeline for housing prices dataset."""

from pathlib import Path
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from src.utils import setup_logging, load_config, get_config_path, get_logger
from src.data import (
    load_housing_data,
    save_processed_data,
    check_data_quality,
    HousingDataValidator,
    HousingPreprocessor,
)

setup_logging()
logger = get_logger(__name__)


def process_housing_data(
    raw_data_path: str, config_path: str, output_dir: str = "data/processed"
) -> tuple:
    """Complete data processing pipeline."""
    config = load_config(config_path)

    df = load_housing_data(raw_data_path)
    check_data_quality(df)

    validator = HousingDataValidator()
    results = validator.validate(df)
    if not results["passed"]:
        raise ValueError("Validation failed: " + ", ".join(results["errors"]))

    preprocessor = HousingPreprocessor(
        scaling=config["features"].get("scaling", "standard")
    )
    df_processed = preprocessor.preprocess(df, fit=True)

    target_col = config["features"].get("target_column", "Price")
    X, y = df_processed.drop(columns=[target_col]), df_processed[[target_col]]
    test_size = config["data"]["split"].get("test_size", 0.2)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    save_processed_data(
        pd.concat([X_train, y_train], axis=1), f"{output_dir}/train.csv"
    )
    save_processed_data(pd.concat([X_test, y_test], axis=1), f"{output_dir}/test.csv")

    Path("models/artifacts").mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, "models/artifacts/preprocessor.pkl")
    logger.info(f"Pipeline complete: train {X_train.shape}, test {X_test.shape}")
    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    try:
        process_housing_data("data/raw/housing.csv", get_config_path())
        logger.info("Data processing successful!")
    except Exception as e:
        logger.error(f"Data processing failed: {e}")
        raise
