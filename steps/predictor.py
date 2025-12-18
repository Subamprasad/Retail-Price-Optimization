from zenml import step
import pandas as pd
import numpy as np
from zenml.integrations.mlflow.services import MLFlowDeploymentService

@step(enable_cache=False)
def predictor(
    service: MLFlowDeploymentService,
    data: pd.DataFrame,
) -> np.ndarray:
    """Run an inference request against a prediction service"""

    service.start(timeout=10)  # should be a NOP if already started
    data = data.to_json(orient="split")
    prediction = service.predict(data)
    return prediction
