from zenml.integrations.bentoml.steps import bentoml_model_deployer_step

# We need a constant for MODEL_NAME. 
# Attempting to import from constants, but if not present, defining it here or handling it later.
# For now, I'll rely on constants.py being present or created.
from constants import MODEL_NAME

bentoml_model_deployer = bentoml_model_deployer_step.with_options(
    parameters=dict(
        model_name=MODEL_NAME,  # Name of the model
        port=3001,  # Port to be used by the http server
        production=False,  # Deploy the model in production mode
        timeout=1000,
    )
)
