"""
API serving script for housing prices prediction.
Runs the FastAPI server for model inference.
"""

import logging
import uvicorn

from src.utils import setup_logging, get_logger

# Setup logging
setup_logging("INFO")
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    from src.api.predict import app

    logger.info("Starting Housing Prices Prediction API server...")
    logger.info("API will be available at: http://localhost:8000")
    logger.info("Documentation at: http://localhost:8000/docs")

    # Run the FastAPI server
    uvicorn.run(
        "src.api.predict:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Auto-reload on code changes (development only)
        log_level="info"
    )
