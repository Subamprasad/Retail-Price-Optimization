from typing import List, Tuple
import pandas as pd
from typing_extensions import Annotated
from zenml import step
from zenml.logger import get_logger
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression

from steps.model_building import LinearRegressionModel

logger = get_logger(__name__)

@step(experiment_tracker="mlflow_tracker",
      settings={"experiment_tracker.mlflow": {"experiment_name": "retail_price_optimization"}})
def train_model(
    X_train: Annotated[pd.DataFrame, "X_train"],
    y_train: Annotated[pd.Series, "y_train"]
) -> Tuple[
    Annotated[LinearRegression, "model"],
    Annotated[List[str], "predictors"],
]:
    """Trains a linear regression model."""
    try:
        mlflow.sklearn.autolog()
        
        # Using the class from model_building, or directly sklearn if simple
        # The reference uses LinearRegressionModel wrapper which uses statsmodels
        # BUT the reference `sklearn_train` step uses `LinearRegression()` from sklearn directly!
        # The reference `re_train` uses `LinearRegressionModel` (wrapper).
        
        # My pipeline calls `train_model` which seems to map to `sklearn_train` in reference.
        # So I will implement a simple sklearn training here.
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Save model locally for app.py
        import joblib
        joblib.dump(model, "model.pkl")
        joblib.dump(list(X_train.columns), "predictors.pkl")
        logger.info("Model saved to model.pkl")
        
        predictors = X_train.columns.tolist()
        
        return model, predictors
        
    except Exception as e:
        logger.error(e)
        raise e
