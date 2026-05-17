# Quick Start Guide - Housing Prices Data Pipeline

## 1. Setup (First Time)

```bash
# Install dependencies
pip install -r requirements.txt

# Create project directories
mkdir -p data/raw data/processed models/artifacts logs
```

## 2. Generate Sample Data

To test the pipeline, generate synthetic housing data:

```bash
python -m scripts.generate_sample_data
```

This creates a sample dataset at `data/raw/housing.csv` with 1000 housing records.

## 3. Run Data Processing Pipeline

Process the raw data through validation and preprocessing:

```bash
python -m scripts.process_data
```

This will:
- ✓ Load raw data
- ✓ Validate data quality
- ✓ Handle missing values
- ✓ Remove outliers
- ✓ Scale features
- ✓ Split train/test data
- ✓ Save processed data to `data/processed/`

**Output:**
- `data/processed/train.csv` - Training dataset (80%)
- `data/processed/test.csv` - Test dataset (20%)

## 4. Run Tests

Verify the data processing pipeline works correctly:

```bash
pytest tests/unit/test_data.py -v
```

## 5. Train the Model

Train a machine learning model on the processed data:

```bash
python -m scripts.train_model
```

This will:
- ✓ Load processed training data
- ✓ Train Random Forest model (configured in `config.yaml`)
- ✓ Calculate training metrics (R², RMSE, MAE)
- ✓ Save trained model to `models/artifacts/housing_model.pkl`

## 6. Evaluate the Model

Evaluate the trained model on the unseen test dataset:

```bash
python -m scripts.evaluate_model
```

This will:
- ✓ Load the trained model from `models/artifacts/housing_model.pkl`
- ✓ Load the processed test data from `data/processed/test.csv`
- ✓ Calculate and report evaluation metrics (R², RMSE, MAE) on the test set

## 7. Test the Complete Pipeline

Run the entire MLOps pipeline from data to trained model:

```bash
bash scripts/run_pipeline.sh
```

This runs steps 1-5 automatically and tests API startup.

## 8. Serve the Model via API

Start the REST API server for predictions:

```bash
python -m scripts.serve_api
```

The API will be available at:
- **Main API**: http://localhost:8000
- **Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 9. Test the API

In a separate terminal, test the running API:

```bash
python -m scripts.test_api
```

This will test:
- ✓ API health check
- ✓ Model information
- ✓ Single house prediction
- ✓ Batch predictions

## 10. Make Manual API Requests

Test the API manually:

```bash
# Single prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "MedInc": 8.5,
    "HouseAge": 25.0,
    "AveRooms": 6.2,
    "AveBedrms": 1.1,
    "Population": 1200.0,
    "AveOccup": 3.2,
    "Latitude": 37.5,
    "Longitude": -122.2
  }'

# Batch prediction
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "houses": [
      {
        "MedInc": 8.5,
        "HouseAge": 25.0,
        "AveRooms": 6.2,
        "AveBedrms": 1.1,
        "Population": 1200.0,
        "AveOccup": 3.2,
        "Latitude": 37.5,
        "Longitude": -122.2
      }
    ]
  }'

## Project Structure

```
ml_model_serving/
├── data/
│   ├── raw/                    # Original housing data
│   ├── processed/              # Cleaned, split data
│   └── features/               # Feature sets for experiments
│
├── src/
│   ├── data/
│   │   ├── loader.py          # Data loading functions
│   │   ├── validator.py       # Data validation rules
│   │   ├── preprocessor.py    # Data preprocessing
│   │   └── __init__.py
│   ├── features/              # Feature engineering (empty - add as needed)
│   ├── models/
│   │   ├── trainer.py         # Model training and saving
│   │   ├── inference.py       # Model loading and prediction
│   │   └── __init__.py
│   ├── api/
│   │   └── predict.py         # REST API endpoints
│   ├── utils/
│   │   └── __init__.py        # Shared utilities (logging, config)
│   └── monitoring/            # Monitoring (empty - add as needed)
│
├── scripts/
│   ├── generate_sample_data.py     # Generate synthetic data
│   ├── process_data.py             # Data processing pipeline
│   ├── train_model.py              # Model training
│   ├── evaluate_model.py           # Model evaluation
│   ├── serve_api.py                # API server startup
│   ├── test_api.py                 # API tests
│   └── run_pipeline.py             # Full MLOps pipeline orchestrator
│   ├── evaluate_model.py           # Model evaluation
│   ├── serve_api.py                # API server startup
│   ├── test_api.py                 # API tests
│   └── run_pipeline.py             # Full MLOps pipeline orchestrator
│
├── tests/
│   ├── unit/test_data.py           # Data pipeline tests
│   └── integration/                # Integration tests
│
├── notebooks/                       # Jupyter notebooks (to be added)
├── config/config.yaml              # Configuration
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

## Key Files

| `scripts/train_model.py` | Model training pipeline |
| `scripts/evaluate_model.py` | Model evaluation pipeline |
| File | Purpose |
|------|---------|
| `scripts/generate_sample_data.py` | Create synthetic housing dataset |
| `scripts/process_data.py` | Main data processing pipeline |
| `src/data/loader.py` | Data loading utilities |
| `src/data/validator.py` | Data validation logic |
| `src/data/preprocessor.py` | Feature preprocessing |
| `config/config.yaml` | Pipeline configuration |
| `tests/unit/test_data.py` | Unit tests |

## Configuration

Edit `config/config.yaml` to customize:

```yaml
data:
  test_size: 0.2                # Train/test split ratio
  split:
    random_state: 42            # Reproducibility seed

features:
  scaling: "standard"           # standard, minmax, robust
  outlier_removal: true         # Remove outliers?
  iqr_multiplier: 1.5           # Outlier threshold
```

## Troubleshooting

### Issue: File not found error
```bash
# Ensure raw data exists
python -m scripts.generate_sample_data
```

### Issue: Import errors
```bash
# Install/reinstall dependencies
pip install -r requirements.txt
```

### Issue: Tests fail
```bash
# Run with verbose output
pytest tests/unit/test_data.py -v -s
```

## Documentation

- [Data Processing Module](DATA_PROCESSING.md) - Detailed documentation
- [README.md](../README.md) - Full project overview
- `config/config.yaml` - Configuration options
