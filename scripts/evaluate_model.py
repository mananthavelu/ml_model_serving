"""Model evaluation script for housing prices prediction."""

import mlflow
from src.utils import setup_logging, load_config, get_config_path, get_logger
from src.data import load_housing_data
from src.models import HousingModelTrainer

setup_logging()
logger = get_logger(__name__)

# Set MLflow tracking directory
mlflow.set_tracking_uri("file:./mlruns")


def evaluate_housing_model(
    test_data_path: str,
    config_path: str,
    model_path: str = "models/artifacts/housing_model.pkl",
) -> dict:
    """Evaluate a trained housing prices prediction model."""
    config = load_config(config_path)
    test_df = load_housing_data(test_data_path)

    target_col = config["features"].get("target_column", "Price")
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[[target_col]]

    mlflow.start_run(
        run_name=f"{config['model']['registry'].get('run_name', 'default_evaluation_run')}_eval"
    )
    trainer = HousingModelTrainer.load_model(model_path)
    metrics = trainer.evaluate(X_test, y_test)

    if metrics:
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, float(metric_value))
    mlflow.end_run()

    logger.info(f"Evaluation metrics: {metrics}")
    return metrics


if __name__ == "__main__":
    try:
        evaluate_housing_model("data/processed/test.csv", get_config_path())
        logger.info("Model evaluation successful!")
    except Exception as e:
        logger.error(f"Model evaluation failed: {e}")
        raise
