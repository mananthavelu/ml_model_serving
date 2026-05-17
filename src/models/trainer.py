"""Model training module for housing prices prediction."""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
from pathlib import Path
from src.utils import get_logger

logger = get_logger(__name__)


class HousingModelTrainer:
    """Trainer for housing prices prediction models."""

    def __init__(self, model_type: str = "random_forest", **model_params):
        self.model_type = model_type
        self.model_params = model_params
        self.model = self._create_model()
        self.is_trained = False

    def _create_model(self):
        if self.model_type == "random_forest":
            defaults = {'n_estimators': 100, 'max_depth': 15, 'min_samples_split': 5,
                       'min_samples_leaf': 2, 'random_state': 42, 'n_jobs': -1}
            defaults.update(self.model_params)
            return RandomForestRegressor(**defaults)
        elif self.model_type == "linear_regression":
            return LinearRegression(**self.model_params)
        raise ValueError(f"Unknown model type: {self.model_type}")

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> dict:
        logger.info(f"Training {self.model_type} on {X_train.shape}")
        self.model.fit(X_train, y_train.values.ravel())
        self.is_trained = True
        y_pred = self.model.predict(X_train)
        metrics = self._calculate_metrics(y_train, y_pred)
        logger.info(f"Training R²: {metrics['r2_score']:.4f}")
        return metrics

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        logger.info(f"Evaluating on {X_test.shape}")
        y_pred = self.model.predict(X_test)
        metrics = self._calculate_metrics(y_test, y_pred)
        logger.info(f"Test R²: {metrics['r2_score']:.4f}")
        return metrics

    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> dict:
        """Calculate regression metrics."""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        return {
            "mse": round(mse, 4),
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "r2_score": round(r2, 4)
        }

    def save_model(self, model_path: str) -> None:
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, model_path)
        logger.info(f"Model saved to: {model_path}")

    @classmethod
    def load_model(cls, model_path: str):
        model = joblib.load(model_path)
        trainer = cls()
        trainer.model = model
        trainer.is_trained = True
        logger.info(f"Model loaded from: {model_path}")
        return trainer
