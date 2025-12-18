import pandas as pd
from zenml import step
from sklearn.preprocessing import LabelEncoder
from typing_extensions import Annotated
from typing import Tuple

@step
def categorical_encode(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes categorical columns.
    
    Args:
        df: Raw DataFrame.
    """
    try:
        cols_to_encode = ["product_id", "product_category_name", "product_score"]
        for col in cols_to_encode:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
        return df
    except Exception as e:
        raise e

@step
def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs feature engineering (date extraction).
    
    Args:
        df: DataFrame with encoded categories.
    """
    try:
        # Check if month_year needs to be processed
        if "month_year" in df.columns:
            df["month_year"] = pd.to_datetime(df["month_year"])
            df["month"] = df["month_year"].dt.month
            df["year"] = df["month_year"].dt.year
            df["is_weekend"] = df["month_year"].dt.dayofweek > 4
            
            # Drop original date extraction if redundant or keep for reference
            # For this model, we likely want to drop the timestamp itself for regression
            df.drop(columns=["month_year"], inplace=True)
            
        # Ensure numeric types
        df.fillna(0, inplace=True) # Simple imputation
        return df
    except Exception as e:
        raise e
