from typing import cast
from zenml import step
from zenml.integrations.bentoml.model_deployers.bentoml_model_deployer import BentoMLModelDeployer
from zenml.integrations.bentoml.services.bentoml_deployment import BentoMLDeploymentService

@step(enable_cache=False)
def bentoml_prediction_service_loader(
    pipeline_name: str, step_name: str, model_name: str
) -> BentoMLDeploymentService:
    """Get the BentoML prediction service started by the deployment pipeline."""
    model_deployer = BentoMLModelDeployer.get_active_model_deployer()

    services = model_deployer.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name=step_name,
        model_name=model_name,
    )
    if not services:
        raise RuntimeError(
            f"No BentoML prediction server deployed by the '{step_name}' step "
            f"in the '{pipeline_name}' pipeline for the '{model_name}' model."
        )

    if not services[0].is_running:
        raise RuntimeError(
            f"The BentoML prediction server deployed by the '{step_name}' step "
            f"in the '{pipeline_name}' pipeline for the '{model_name}' model is not running."
        )

    return cast(BentoMLDeploymentService, services[0])
