# Housing Prices Data Processing Module

## Overview

This module provides a complete data processing pipeline for the housing prices prediction dataset. It includes data loading, validation, and preprocessing components designed specifically for housing market data.

## Dataset Features

The housing prices dataset includes the following features:

| Feature | Description | Type |
|---------|-------------|------|
| **MedInc** | Median income in block group | Numerical |
| **HouseAge** | Median house age in block group (years) | Numerical |
| **AveRooms** | Average number of rooms | Numerical |
| **AveBedrms** | Average number of bedrooms | Numerical |
| **Population** | Block group population | Numerical |
| **AveOccup** | Average occupancy | Numerical |
| **Latitude** | Geographic latitude coordinate | Numerical |
| **Longitude** | Geographic longitude coordinate | Numerical |
| **Price** | Median house price (target variable) | Numerical |

## Components

### 1. Data Loader (`loader.py`)

Handles loading and saving housing data.

**Key Functions:**
- `load_housing_data(data_path)` - Load CSV data
- `save_processed_data(df, output_path)` - Save processed data
- `check_data_quality(df)` - Generate quality metrics

**Example:**
```python
from src.data import load_housing_data, check_data_quality

# Load data
df = load_housing_data("data/raw/housing.csv")

# Check quality
quality = check_data_quality(df)
print(f"Shape: {quality['total_rows']} rows, {quality['total_columns']} columns")
print(f"Missing values: {quality['missing_values']}")
```

### 2. Data Validator (`validator.py`)

Ensures data meets quality requirements using Great Expectations.

**Validation Checks:**
- ✓ Minimum sample size (100+ rows)
- ✓ All required columns present
- ✓ No duplicate rows
- ✓ No negative prices
- ✓ No missing values in critical columns

**Example:**
```python
from src.data import HousingDataValidator

validator = HousingDataValidator()
results = validator.validate(df)

if not results['passed']:
    for error in results['errors']:
        print(f"❌ {error}")
else:
    for validation in results['validations']:
        print(f"✓ {validation}")
```

### 3. Data Preprocessor (`preprocessor.py`)

Performs data cleaning and feature transformation.

**Processing Steps:**
1. **Missing Value Handling** - Fill with median values
2. **Outlier Removal** - IQR-based outlier detection
3. **Feature Scaling** - Standard, MinMax, or Robust scaling

**Example:**
```python
from src.data import HousingPreprocessor

# Initialize with standard scaling
preprocessor = HousingPreprocessor(scaling="standard")

# Preprocess training data (fit transformers)
df_train_processed = preprocessor.preprocess(df_train, fit=True)

# Preprocess test data (use fitted transformers)
df_test_processed = preprocessor.preprocess(df_test, fit=False)
```

## Usage

### Complete Pipeline

Run the full data processing pipeline:

```bash
python scripts/process_data.py
```

This script:
1. Loads raw data from `data/raw/housing.csv`
2. Validates data quality
3. Preprocesses features
4. Splits into train/test sets
5. Saves processed data to `data/processed/`

### Programmatic Usage

```python
from src.data import (
    load_housing_data,
    HousingDataValidator,
    HousingPreprocessor,
    save_processed_data
)
from sklearn.model_selection import train_test_split
import pandas as pd

# 1. Load raw data
df = load_housing_data("data/raw/housing.csv")

# 2. Validate
validator = HousingDataValidator()
results = validator.validate(df)
if not results['passed']:
    raise ValueError("Validation failed")

# 3. Preprocess
preprocessor = HousingPreprocessor(scaling="standard")
df_clean = preprocessor.preprocess(df, fit=True)

# 4. Split
X = df_clean.drop('Price', axis=1)
y = df_clean[['Price']]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Save
train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)
save_processed_data(train_df, "data/processed/train.csv")
save_processed_data(test_df, "data/processed/test.csv")
```

## Configuration

Settings are defined in `config/config.yaml`:

```yaml
features:
  scaling: "standard"        # standard, minmax, robust
  outlier_removal: true      # Enable/disable outlier removal
  iqr_multiplier: 1.5        # IQR multiplier for outlier detection

data:
  split:
    test_size: 0.2           # Train/test split ratio
    random_state: 42         # For reproducibility
```

## Testing

Run unit tests:

```bash
pytest tests/unit/test_data.py -v
```

Tests cover:
- Data loading and saving
- Validation logic
- Preprocessing transformations
- Data quality checks

## Logging

All components use Python's `logging` module. Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

Output includes:
- Data loading info
- Validation results
- Preprocessing steps
- Quality metrics
