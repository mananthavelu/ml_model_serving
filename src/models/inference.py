"""Model inference module for housing prices prediction."""

import pandas as pd
from typing import Union, Dict, Any
from src.utils import get_logger
from .trainer import HousingModelTrainer

logger = get_logger(__name__)


class HousingPredictor:
    """Predictor for housing prices using trained models."""

    def __init__(self, model_path: str = None, preprocessor=None):
        self.model = None
        self.preprocessor = preprocessor
        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: str) -> None:
        self.model = HousingModelTrainer.load_model(model_path).model
        logger.info(f"Model loaded from {model_path}")

    def preprocess_input(self, input_data: Union[pd.DataFrame, Dict]) -> pd.DataFrame:
        if not self.preprocessor:
            raise ValueError("Preprocessor not set")
        if isinstance(input_data, dict):
            input_data = [input_data]
        if isinstance(input_data, list):
            input_data = pd.DataFrame(input_data)
        return self.preprocessor.preprocess(input_data, fit=False)

    def predict(self, input_data: Union[pd.DataFrame, Dict]) -> Dict[str, Any]:
        if not self.model:
            raise ValueError("Model not loaded")
        processed = self.preprocess_input(input_data)
        predictions = self.model.predict(processed)
        preds = (
            predictions.tolist()
            if hasattr(predictions, "tolist")
            else [float(predictions[0])]
        )
        logger.info(f"Predictions: {len(preds)} samples")
        return {"predictions": preds, "count": len(preds)}

    def predict_single(self, house_data: Dict) -> Dict[str, Any]:
        result = self.predict(house_data)
        price = result["predictions"][0]
        return {
            "predicted_price": round(price, 2),
            "confidence_range": {
                "lower": round(price * 0.9, 2),
                "upper": round(price * 1.1, 2),
            },
            "input_features": house_data,
        }


def create_predictor(model_path: str, preprocessor) -> HousingPredictor:
    """Factory function to create a predictor."""
    predictor = HousingPredictor(preprocessor=preprocessor)
    predictor.load_model(model_path)
    return predictor
