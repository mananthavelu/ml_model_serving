"""Data preprocessing module for housing prices dataset."""

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from src.utils import get_logger

logger = get_logger(__name__)


class HousingPreprocessor:
    """Preprocessor for housing prices dataset."""

    def __init__(self, scaling: str = "standard"):
        self.scaling = scaling
        self.scaler = {"standard": StandardScaler, "minmax": MinMaxScaler,
                      "robust": RobustScaler}.get(scaling, StandardScaler)()
        self.numerical_features = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
                                   'Population', 'AveOccup', 'Latitude', 'Longitude']
        self.target_column = 'Price'

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in self.numerical_features:
            if col in df.columns and df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
        return df

    def remove_outliers(self, df: pd.DataFrame, iqr_multiplier: float = 1.5) -> pd.DataFrame:
        df = df.copy()
        for col in self.numerical_features:
            if col in df.columns:
                Q1, Q3 = df[col].quantile([0.25, 0.75])
                IQR = Q3 - Q1
                bounds = (Q1 - iqr_multiplier * IQR, Q3 + iqr_multiplier * IQR)
                df = df[(df[col] >= bounds[0]) & (df[col] <= bounds[1])]
        logger.info(f"Removed outliers. Shape: {df.shape}")
        return df

    def scale_features(self, X: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        X = X.copy()
        cols = [col for col in self.numerical_features if col in X.columns]
        X[cols] = self.scaler.fit_transform(X[cols]) if fit else self.scaler.transform(X[cols])
        return X

    def preprocess(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        df = self.handle_missing_values(df)
        df = self.remove_outliers(df)
        if self.target_column in df.columns:
            X, y = df.drop(columns=[self.target_column]), df[[self.target_column]]
            df = pd.concat([self.scale_features(X, fit=fit), y], axis=1)
        else:
            df = self.scale_features(df, fit=fit)
        logger.info(f"Preprocessing complete: {df.shape}")
        return df
