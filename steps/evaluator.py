import pandas as pd
from typing import Annotated, Tuple
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
from zenml import step
from zenml.logger import get_logger
import mlflow
from zenml.client import Client
from zenml.integrations.mlflow.experiment_trackers import MLFlowExperimentTracker

logger = get_logger(__name__)

# Verify experiment tracker
experiment_tracker = Client().active_stack.experiment_tracker
tracker_name = experiment_tracker.name if experiment_tracker and isinstance(experiment_tracker, MLFlowExperimentTracker) else None

@step(experiment_tracker=tracker_name)
def evaluation(
    model: LinearRegression,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[
    Annotated[float, "mse"],
    Annotated[float, "rmse"],
]:
    """Validates the model using separate X and y test sets."""
    try:
        prediction = model.predict(X_test)
        mse = mean_squared_error(y_test, prediction)
        rmse = np.sqrt(mse)
        
        logger.info(f"MSE: {mse}")
        logger.info(f"RMSE: {rmse}")
        
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)
        
        return mse, rmse
    except Exception as e:
        logger.error(e)
        raise e
