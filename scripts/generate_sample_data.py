"""Generate sample housing prices dataset."""

from pathlib import Path
import numpy as np
import pandas as pd
from src.utils import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)


def generate_housing_data(
    n_samples: int = 1000, random_state: int = 42
) -> pd.DataFrame:
    """Generate synthetic housing prices dataset."""
    np.random.seed(random_state)
    data = {
        "MedInc": np.random.lognormal(mean=1.5, sigma=0.8, size=n_samples),
        "HouseAge": np.random.gamma(shape=2, scale=12, size=n_samples) + 1,
        "AveRooms": np.random.gamma(shape=2, scale=3, size=n_samples) + 1,
        "AveBedrms": np.random.gamma(shape=2, scale=1.5, size=n_samples) + 1,
        "Population": np.random.lognormal(mean=5, sigma=1.5, size=n_samples),
        "AveOccup": np.random.gamma(shape=2, scale=2, size=n_samples) + 1,
        "Latitude": np.random.uniform(32, 42, n_samples),
        "Longitude": np.random.uniform(-125, -114, n_samples),
    }
    df = pd.DataFrame(data)
    price = (
        0.4 * df["MedInc"]
        + 0.01 * np.maximum(52 - df["HouseAge"], 0)
        + 0.05 * df["AveRooms"]
        + 0.005 * (40 - df["Latitude"])
        + 0.01 * (-115 - df["Longitude"])
        + np.random.normal(0, 0.3, n_samples)
    )
    df["Price"] = np.clip(price, 0.5, 5.0)
    price_min = f"${df['Price'].min():.2f}k"
    price_max = f"${df['Price'].max():.2f}k"
    logger.info(
        f"Generated {df.shape[0]} samples. "
        f"Price range: {price_min} - {price_max}"
    )
    return df


def save_sample_data(output_path: str = "data/raw/housing.csv") -> None:
    """Generate and save sample housing data."""
    df = generate_housing_data(n_samples=1000)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved to {output_path}")


if __name__ == "__main__":
    save_sample_data("data/raw/housing.csv")
