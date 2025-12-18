from zenml import pipeline
from zenml.config import DockerSettings
from zenml.integrations.constants import MLFLOW

from steps.ingest_data import ingest_data_for_inference
from steps.prediction_service_loader import prediction_service_loader
from steps.predictor import predictor

docker_settings = DockerSettings(required_integrations=[MLFLOW])

@pipeline(enable_cache=False, settings={"docker": docker_settings})
def inference_pipeline(
    pipeline_name: str,
    pipeline_step_name: str,
):
    # Link all the steps together
    # Data ingestion for inference
    inference_data = ingest_data_for_inference()
    
    # Load the deployed model service
    model_deployment_service = prediction_service_loader(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        running=False,
    )
    
    # Run predictions
    predictor(service=model_deployment_service, data=inference_data)
