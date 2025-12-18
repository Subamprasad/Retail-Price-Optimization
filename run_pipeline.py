import argparse
from typing import cast

from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from zenml.client import Client

from pipelines.deployment_pipeline import deployment_pipeline
from pipelines.inference_pipeline import inference_pipeline
from constants import MODEL_NAME, PIPELINE_NAME, PIPELINE_STEP_NAME

def main(
    config: str = "deploy",
    min_accuracy: float = 0.5,
):
    print(f"MLflow tracking URI: {get_tracking_uri()}")
    
    if config == "deploy":
        deployment_pipeline(
            min_accuracy=min_accuracy,
            workers=3,
            timeout=60,
        )
        print(
            "You can run the prediction pipeline with the "
            "`python run_pipeline.py --config predict` command "
            "to perform inference using the deployed model."
        )

    elif config == "predict":
        inference_pipeline(
            pipeline_name=PIPELINE_NAME,
            pipeline_step_name=PIPELINE_STEP_NAME,
        )
    
    elif config == "train":
        # Fallback to just training without deployment if needed, 
        # or we assume deployment_pipeline handles training too.
        # For strictly training, we can import training_pipeline if it exists separately
        from pipelines.training_pipeline import training_pipeline
        training_pipeline()

    else:
        print(f"Unknown config: {config}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", 
        type=str, 
        default="deploy", 
        help="Pipeline configuration: 'deploy', 'predict', or 'train'"
    )
    parser.add_argument(
        "--min-accuracy", 
        type=float, 
        default=0.5, 
        help="Minimum accuracy required to deploy the model"
    )
    args = parser.parse_args()
    
    main(config=args.config, min_accuracy=args.min_accuracy)
