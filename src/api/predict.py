"""REST API for housing prices prediction."""

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict
import mlflow
from mlflow.tracking import MlflowClient

from src.utils import setup_logging, load_config, get_config_path, get_logger
from src.data import HousingPreprocessor
from src.models import create_predictor

setup_logging("INFO")
logger = get_logger(__name__)

config = load_config(get_config_path())
PREPROCESSOR_PATH = "models/artifacts/preprocessor.pkl"

# Configure MLflow client
client = MlflowClient()
MODEL_NAME = config["model"]["registry"].get("model_name", "HousingPricePredictor")
MODEL_STAGE = config["model"]["registry"].get("model_stage", "Production")

try:
    # Load preprocessor
    preprocessor = joblib.load(PREPROCESSOR_PATH)
except Exception:
    logger.warning(
        f"Preprocessor not found at {PREPROCESSOR_PATH}. Initializing default preprocessor."
    )
    preprocessor = HousingPreprocessor(
        scaling=config["features"].get("scaling", "standard")
    )

try:
    # Load model from MLflow Model Registry
    model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    predictor = mlflow.pyfunc.load_model(model_uri)
    logger.info(
        f"Successfully loaded model '{MODEL_NAME}' in stage '{MODEL_STAGE}' from MLflow Registry."
    )
except Exception as e:
    logger.error(f"Failed to load model from MLflow Registry: {e}")
    logger.error(
        "Ensure an MLflow tracking server is running and the model is registered."
    )
    predictor = None
    # Fallback to local model if MLflow fails (optional, for dev)
    try:
        MODEL_PATH = "models/artifacts/housing_model.pkl"
        predictor = create_predictor(MODEL_PATH, preprocessor)
        logger.warning("Falling back to local model artifact.")
    except Exception as e_fallback:
        logger.error(f"Failed to load local model artifact either: {e_fallback}")
        raise RuntimeError("No model could be loaded for serving.")

app = FastAPI(title="Housing Prices Prediction API", version="1.0.0")


class HouseFeatures(BaseModel):
    MedInc: float = Field(..., gt=0)
    HouseAge: float = Field(..., ge=0)
    AveRooms: float = Field(..., gt=0)
    AveBedrms: float = Field(..., gt=0)
    Population: float = Field(..., gt=0)
    AveOccup: float = Field(..., gt=0)
    Latitude: float = Field(..., ge=-90, le=90)
    Longitude: float = Field(..., ge=-180, le=180)


class PredictionResponse(BaseModel):
    predicted_price: float
    confidence_range: Dict[str, float]
    input_features: Dict[str, float]


class BatchPredictionRequest(BaseModel):
    houses: List[HouseFeatures]


class BatchPredictionResponse(BaseModel):
    predictions: List[float]
    count: int


@app.get("/")
async def root():
    return {"message": "Housing Prices Prediction API", "status": "healthy"}


@app.get("/health")
async def health_check():
    if not predictor:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict", response_model=PredictionResponse)
async def predict_price(house: HouseFeatures):
    if not predictor:
        raise HTTPException(status_code=503, detail="Model not available")
    try:
        result = predictor.predict_single(house.dict())
        logger.info(f"Prediction: {result['predicted_price']}")
        return result
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    if not predictor:
        raise HTTPException(status_code=503, detail="Model not available")
    try:
        result = predictor.predict([h.dict() for h in request.houses])
        return BatchPredictionResponse(
            predictions=result["predictions"], count=result["count"]
        )
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info")
async def model_info():
    if not predictor:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "model_type": type(predictor.model).__name__,
        "model_path": MODEL_PATH,
        "preprocessor_config": {
            "scaling": preprocessor.scaling,
            "numerical_features": preprocessor.numerical_features,
            "target_column": preprocessor.target_column,
        },
    }
