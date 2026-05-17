"""Data validation module for housing prices dataset."""

from typing import Dict, Any
import pandas as pd
from src.utils import get_logger

logger = get_logger(__name__)


class HousingDataValidator:
    """Validator for housing prices dataset."""

    def validate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate housing dataset against expectations."""
        required_cols = [
            "MedInc",
            "HouseAge",
            "AveRooms",
            "AveBedrms",
            "Population",
            "AveOccup",
            "Latitude",
            "Longitude",
            "Price",
        ]
        results: Dict[str, Any] = {
            "passed": True,
            "validations": [],
            "errors": [],
        }

        # Ensure all required columns are present before proceeding
        # with other column-specific checks
        has_all_required_cols = all(
            col in df.columns for col in required_cols
        )
        if not has_all_required_cols:
            results["errors"].append("Missing required columns")
            results["passed"] = False
            logger.error("Missing required columns")
            return results  # Early exit if fundamental columns are missing

        checks = [
            (len(df) >= 100, f"Minimum {len(df)} rows >= 100"),
            (df.duplicated().sum() == 0, "No duplicate rows found"),
            ((df["Price"] >= 0).all(), "All prices non-negative"),
            (
                df[required_cols].isnull().sum().sum() == 0,
                "No missing values in critical columns",
            ),
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
