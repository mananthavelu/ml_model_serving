"""Model training script for housing prices prediction."""

from pathlib import Path
import pandas as pd
import mlflow
from src.utils import setup_logging, load_config, get_config_path, get_logger
from src.data import load_housing_data
from src.models import HousingModelTrainer

setup_logging()
logger = get_logger(__name__)

# Set MLflow tracking directory
mlflow.set_tracking_uri("file:./mlruns")


def train_housing_model(
    train_data_path: str,
    config_path: str,
    model_output_path: str = "models/artifacts/housing_model.pkl"
) -> dict:
    """Train housing prices prediction model."""
    config = load_config(config_path)
    train_df = load_housing_data(train_data_path)

    target_col = config['features'].get('target_column', 'Price')
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[[target_col]]

    model_type = config['model'].get('type', 'random_forest')
    params = config['model'].get('hyperparameters', {})
    mlflow_run_name = config['model']['registry'].get('run_name', 'default_training_run')

    mlflow.start_run(run_name=mlflow_run_name) 
    mlflow.log_param("model_type", model_type)
    for key, value in params.items():
        mlflow.log_param(key, value)

    trainer = HousingModelTrainer(model_type=model_type, **params)
    metrics = trainer.train(X_train, y_train)

    if metrics:
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, float(metric_value))

    Path(model_output_path).parent.mkdir(parents=True, exist_ok=True)
    trainer.save_model(model_output_path)

    # Log the model to MLflow artifacts
    mlflow.sklearn.log_model(sk_model=trainer.model, artifact_path="model") # Log the actual scikit-learn model
    mlflow.end_run()

    logger.info(f"Model saved to {model_output_path}")
    return {
        "model_type": model_type,
        "metrics": metrics,
        "shape": train_df.shape
    }


if __name__ == "__main__":
    try:
        train_housing_model("data/processed/train.csv", get_config_path())
        logger.info("Model training successful!")
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        raise
