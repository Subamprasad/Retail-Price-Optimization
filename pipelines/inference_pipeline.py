from zenml import pipeline
from zenml.config import DockerSettings
from zenml.integrations.constants import BENTOML, PYTORCH

from steps.ingest_data import ingest_data
from steps.predict_step import predictor
from steps.prediction_service_loader_step import bentoml_prediction_service_loader
from steps.process_data import categorical_encode, feature_engineer

docker_settings = DockerSettings(required_integrations=[PYTORCH, BENTOML])

@pipeline(settings={"docker": docker_settings})
def inference_pipeline(
    model_name: str, pipeline_name: str, step_name: str
):
    # Link: ingest_data needs to support a flag or we use a different step? 
    # Reference used `ingest(table_name="retail_prices", for_predict=True)`
    # My ingest_data step doesn't have args yet. I might need to update ingest_data.
    
    inference_data = ingest_data(for_predict=True) 
    df_processed = categorical_encode(inference_data)
    df_transformed = feature_engineer(df_processed)
    
    prediction_service = bentoml_prediction_service_loader(
        model_name=model_name, pipeline_name=pipeline_name, step_name=step_name
    )
    predictor(inference_data=df_transformed, service=prediction_service)
