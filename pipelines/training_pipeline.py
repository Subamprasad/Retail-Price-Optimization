from zenml import pipeline
from zenml.config import DockerSettings
from zenml.integrations.constants import BENTOML, DEEPCHECKS, MLFLOW

from steps.bento_builder import bento_builder
from steps.data_splitter import split_data
from steps.deployer import bentoml_model_deployer
from steps.deployment_trigger_step import deployment_trigger
from steps.evaluator import evaluate
from steps.ingest_data import ingest_data
from steps.process_data import categorical_encode, feature_engineer
from steps.model_building import LinearRegressionModel # Keep if needed, or remove
from steps.train_model import train_model

docker_settings = DockerSettings(required_integrations=[BENTOML, MLFLOW])

@pipeline(enable_cache=False, settings={"docker": docker_settings})
def training_retail():
    """Train a model and deploy it with BentoML."""
    df = ingest_data(table_name="retail_prices") 
    df_processed = categorical_encode(df)
    df_transformed = feature_engineer(df_processed)  
    X_train, X_test, y_train, y_test = split_data(df_transformed)  
    
    # Train model
    model = train_model(X_train, y_train)         
    
    # Evaluate model
    rmse = evaluate(model=model, df=df_transformed)
 
    
    decision = deployment_trigger(accuracy=rmse, min_accuracy=0.80)
    
    # Bento builder expects model. My train_model returns model.
    # Bento builder expects model. My train_model returns model.
    # bento = bento_builder(model=model)
    
    # bentoml_model_deployer(bento=bento, deploy_decision=decision)
