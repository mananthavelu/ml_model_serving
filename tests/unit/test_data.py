"""
Unit tests for data loading, validation, and preprocessing.
Tests the housing prices data processing pipeline.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

from src.data import (
    load_housing_data,
    save_processed_data,
    check_data_quality,
    HousingDataValidator,
    HousingPreprocessor
)


@pytest.fixture
def sample_housing_data():
    """Create sample housing data for testing."""
    np.random.seed(42)
    data = {
        'MedInc': np.random.uniform(0.5, 15, 150),
        'HouseAge': np.random.uniform(1, 52, 150),
        'AveRooms': np.random.uniform(1, 15, 150),
        'AveBedrms': np.random.uniform(1, 5, 150),
        'Population': np.random.uniform(10, 5000, 150),
        'AveOccup': np.random.uniform(1, 10, 150),
        'Latitude': np.random.uniform(32, 42, 150),
        'Longitude': np.random.uniform(-125, -114, 150),
        'Price': np.random.uniform(0.15, 5, 150)
    }
    return pd.DataFrame(data)


@pytest.fixture
def temp_data_file(sample_housing_data):
    """Create temporary CSV file with sample data."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
        sample_housing_data.to_csv(f.name, index=False)
        yield f.name
    Path(f.name).unlink()


class TestDataLoader:
    """Test data loading functionality."""
    
    def test_load_housing_data(self, temp_data_file, sample_housing_data):
        """Test loading housing data from CSV."""
        df = load_housing_data(temp_data_file)
        assert df.shape[0] == 150
        assert df.shape[1] == 9
        assert all(df.columns == sample_housing_data.columns)
    
    def test_load_nonexistent_file(self):
        """Test loading from non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_housing_data("nonexistent_file.csv")
    
    def test_save_processed_data(self, sample_housing_data):
        """Test saving processed data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_output.csv"
            save_processed_data(sample_housing_data, str(output_path))
            
            assert output_path.exists()
            loaded_df = pd.read_csv(output_path)
            assert loaded_df.shape == sample_housing_data.shape


class TestDataValidator:
    """Test data validation functionality."""
    
    def test_valid_data(self, sample_housing_data):
        """Test validation passes for valid data."""
        validator = HousingDataValidator()
        results = validator.validate(sample_housing_data)
        assert results['passed'] is True
        assert len(results['errors']) == 0
    
    def test_insufficient_samples(self):
        """Test validation fails with too few samples."""
        df = pd.DataFrame({
            'MedInc': [1, 2],
            'HouseAge': [10, 20],
            'AveRooms': [5, 6],
            'AveBedrms': [2, 2],
            'Population': [100, 200],
            'AveOccup': [3, 3],
            'Latitude': [35, 36],
            'Longitude': [-120, -121],
            'Price': [1.5, 2.0]
        })
        validator = HousingDataValidator()
        results = validator.validate(df)
        assert results['passed'] is False
    
    def test_missing_columns(self):
        """Test validation fails with missing columns."""
        df = pd.DataFrame({'MedInc': [1, 2, 3]})
        validator = HousingDataValidator()
        results = validator.validate(df)
        assert results['passed'] is False
    
    def test_negative_prices(self):
        """Test validation fails with negative prices."""
        df = pd.DataFrame({
            'MedInc': [1]*200,
            'HouseAge': [10]*200,
            'AveRooms': [5]*200,
            'AveBedrms': [2]*200,
            'Population': [100]*200,
            'AveOccup': [3]*200,
            'Latitude': [35]*200,
            'Longitude': [-120]*200,
            'Price': [-1] * 100 + [2] * 100
        })
        validator = HousingDataValidator()
        results = validator.validate(df)
        assert results['passed'] is False


class TestDataPreprocessor:
    """Test data preprocessing functionality."""
    
    def test_preprocessor_initialization(self):
        """Test preprocessor initializes correctly."""
        preprocessor = HousingPreprocessor(scaling="standard")
        assert preprocessor.scaling == "standard"
        assert preprocessor.target_column == "Price"
    
    def test_handle_missing_values(self, sample_housing_data):
        """Test missing value handling."""
        df = sample_housing_data.copy()
        df.loc[0, 'MedInc'] = np.nan
        
        preprocessor = HousingPreprocessor()
        df_filled = preprocessor.handle_missing_values(df)
        assert df_filled['MedInc'].isnull().sum() == 0
    
    def test_remove_outliers(self, sample_housing_data):
        """Test outlier removal."""
        df = sample_housing_data.copy()
        initial_len = len(df)
        
        preprocessor = HousingPreprocessor()
        df_clean = preprocessor.remove_outliers(df, iqr_multiplier=1.5)
        
        # Some rows should be removed (or dataset is clean)
        assert len(df_clean) <= initial_len
    
    def test_scale_features_standard(self, sample_housing_data):
        """Test standard scaling."""
        df = sample_housing_data.copy()
        X = df.drop(columns=['Price'])
        
        preprocessor = HousingPreprocessor(scaling="standard")
        X_scaled = preprocessor.scale_features(X, fit=True)
        
        # Check that features are scaled (approximately mean 0, std 1)
        for col in preprocessor.numerical_features:
            if col in X_scaled.columns:
                assert abs(X_scaled[col].mean()) < 0.01
                assert abs(X_scaled[col].std() - 1) < 0.01
    
    def test_full_preprocessing_pipeline(self, sample_housing_data):
        """Test complete preprocessing pipeline."""
        preprocessor = HousingPreprocessor()
        df_processed = preprocessor.preprocess(sample_housing_data.copy(), fit=True)
        
        assert df_processed.shape[0] > 0
        assert 'Price' in df_processed.columns
        assert df_processed.isnull().sum().sum() == 0


class TestDataQuality:
    """Test data quality checks."""
    
    def test_check_data_quality(self, sample_housing_data):
        """Test data quality report generation."""
        report = check_data_quality(sample_housing_data)
        
        assert report['total_rows'] == 150
        assert report['total_columns'] == 9
        assert report['duplicate_rows'] == 0
        assert isinstance(report['missing_values'], dict)
