"""Data validation module for housing prices dataset."""

import pandas as pd
from src.utils import get_logger

logger = get_logger(__name__)


class HousingDataValidator:
    """Validator for housing prices dataset."""

    def validate(self, df: pd.DataFrame) -> dict:
        """Validate housing dataset against expectations."""
        required_cols = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population',
                        'AveOccup', 'Latitude', 'Longitude', 'Price']
        results = {"passed": True, "validations": [], "errors": []}

        checks = [
            (len(df) >= 100, f"Minimum {len(df)} rows >= 100"),
            (all(col in df.columns for col in required_cols), "All required columns present"),
            (df.duplicated().sum() == 0, "No duplicate rows found"),
            ((df['Price'] >= 0).all() if 'Price' in df.columns else True, "All prices non-negative"),
            (df[required_cols].isnull().sum().sum() == 0, "No missing values in critical columns"),
        ]

        for passed, msg in checks:
            if passed:
                results["validations"].append(f"✓ {msg}")
            else:
                results["errors"].append(msg)
                results["passed"] = False
                logger.error(msg)

        logger.info(f"Validation: {'PASSED' if results['passed'] else 'FAILED'}")
        return results
