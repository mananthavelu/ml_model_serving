"""Test the housing prices prediction API."""

import requests
import logging
from src.utils import setup_logging

setup_logging("INFO")
logger = logging.getLogger(__name__)

API_BASE_URL = "http://localhost:8000"


def _make_request(method: str, endpoint: str, json_data=None) -> tuple:
    """Helper to make HTTP requests and return status and data."""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        if method == "GET":
            response = requests.get(url)
        else:
            response = requests.post(url, json=json_data)
        return (
            response.status_code,
            response.json() if response.status_code == 200 else response.text,
        )
    except Exception as e:
        logger.error(f"Request error: {e}")
        return None, str(e)


def test_api_health():
    """Test API health endpoint."""
    status, _ = _make_request("GET", "/health")
    if status == 200:
        logger.info("API health check passed")
        assert True  # Replaced return True with assert True
    else:
        logger.error(f"✗ API health check failed: {status}")
        assert False


def test_model_info():
    """Test model info endpoint."""
    status, data = _make_request("GET", "/model/info")
    if status == 200:
        logger.info(f"Model info retrieved: {data['model_type']}")
        assert True  # Replaced return True with assert True
    else:
        logger.error(f"✗ Model info failed: {status}")
        assert False


def test_single_prediction():
    """Test single house prediction."""
    house_data = {
        "MedInc": 8.5,
        "HouseAge": 25.0,
        "AveRooms": 6.2,
        "AveBedrms": 1.1,
        "Population": 1200.0,
        "AveOccup": 3.2,
        "Latitude": 37.5,
        "Longitude": -122.2,
    }
    status, data = _make_request("POST", "/predict", house_data)
    if status == 200:
        logger.info(f"Single prediction: ${data['predicted_price']:.2f}k")
        assert True  # Replaced return True with assert True
    else:
        logger.error(f"✗ Single prediction failed: {status}")
        assert False


def test_batch_prediction():
    """Test batch prediction."""
    batch_data = {
        "houses": [
            {
                "MedInc": 8.5,
                "HouseAge": 25.0,
                "AveRooms": 6.2,
                "AveBedrms": 1.1,
                "Population": 1200.0,
                "AveOccup": 3.2,
                "Latitude": 37.5,
                "Longitude": -122.2,
            },
            {
                "MedInc": 4.2,
                "HouseAge": 35.0,
                "AveRooms": 5.1,
                "AveBedrms": 1.0,
                "Population": 800.0,
                "AveOccup": 2.8,
                "Latitude": 34.0,
                "Longitude": -118.2,
            },
        ]
    }
    status, data = _make_request("POST", "/predict/batch", batch_data)
    if status == 200:
        logger.info(f"Batch prediction: {data['count']} houses predicted")
        assert True  # Replaced return True with assert True
    else:
        logger.error(f"✗ Batch prediction failed: {status}")
        assert False


def run_api_tests():
    """Run all API tests."""
    tests = [
        test_api_health,
        test_model_info,
        test_single_prediction,
        test_batch_prediction,
    ]

    passed = sum(1 for test in tests if test())
    logger.info(f"API Tests: {passed}/{len(tests)} passed")
    return passed == len(tests)


if __name__ == "__main__":
    success = run_api_tests()
    exit(0 if success else 1)
