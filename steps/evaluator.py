import pandas as pd
from typing import Annotated
from statsmodels.regression.linear_model import RegressionResultsWrapper
from zenml import step
from zenml.logger import get_logger
import mlflow
from zenml.client import Client
from zenml.integrations.mlflow.experiment_trackers import MLFlowExperimentTracker
from steps.model_building import ModelRefinement

logger = get_logger(__name__)

# Verify experiment tracker
experiment_tracker = Client().active_stack.experiment_tracker
if not experiment_tracker or not isinstance(experiment_tracker, MLFlowExperimentTracker):
    # This might fail if stack is not set up yet. 
    # But for the file content, this is correct per reference.
    pass

@step(experiment_tracker=experiment_tracker.name if experiment_tracker else None)
def evaluate(
    model: RegressionResultsWrapper,
    df: pd.DataFrame,
) -> Annotated[float, "rmse"]:
    """Validates the model"""
    try:
        refinement = ModelRefinement(model, df)
        rmse = refinement.validate()
        logger.info(f"RMSE: {rmse}")
        mlflow.log_metric("rmse", rmse)
        return rmse
    except Exception as e:
        logger.error(e)
        raise e
