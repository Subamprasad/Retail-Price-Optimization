from __future__ import annotations

import typing as t
from typing import TYPE_CHECKING

import bentoml
import numpy as np
from bentoml.io import NumpyNdarray

from constants import MODEL_NAME, SERVICE_NAME

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Get the model from the model store
reg_runner = bentoml.sklearn.get(MODEL_NAME).to_runner()

# Create the service
svc = bentoml.Service(name=SERVICE_NAME, runners=[reg_runner])

@svc.api(
    input=NumpyNdarray(dtype="float32", enforce_dtype=True),
    output=NumpyNdarray(dtype="int64"),
)
async def predict_ndarray(inp: NDArray[t.Any]) -> NDArray[t.Any]:
    """
    Predicts the target using the deployed model.
    """
    return await reg_runner.predict.async_run(inp)
