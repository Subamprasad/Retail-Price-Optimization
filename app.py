import pandas as pd
import pandas as pd
from zenml.client import Client
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)
from flask import Flask, render_template, request
import numpy as np

app = Flask(__name__)

# Load deployed model service
try:
    print("Connecting to MLflow Model Deployer...")
    deployer = MLFlowModelDeployer.get_active_model_deployer()
    services = deployer.find_model_server(
        pipeline_name="deployment_pipeline",
        pipeline_step_name="mlflow_model_deployer_step",
        model_name="retail_price_model",
        running=True,
    )
    
    if services:
        service = services[0]
        print(f"Connected to service: {service.prediction_url}")
        # Identify predictors - for now we hardcode or fetch from a known artifact if possible.
        # Ideally, we should fetch the 'predictors' artifact from the training run associated with this service.
        # For simplicity in this step, we will use a fixed list or try to inspect the model signature if available.
        # But wait, looking at training_pipeline, we don't easily expose predictors artifact ID here.
        # Let's fallback to a predefined list or the one from the dataframe columns.
        
        # NOTE: To truly make this dynamic without local pickles, we'd need to fetch the artifact from ZenML.
        # For now, to suffice user request, we can assume all numeric columns from schema are predictors 
        # OR keep predictors.pkl? User said "remove this all".
        # Let's hardcode the predictors based on EDA knowledge for robustness or fetch from metadata store.
        
        # Let's use a standard list based on the dataset we know.
        predictors = ['qty', 'total_price', 'freight_price', 'unit_price', 'product_name_lenght', 'product_description_lenght', 'product_photos_qty', 'product_weight_g', 'product_score', 'customers', 'weekday', 'weekend', 'holiday', 'month', 'year', 's', 'volume', 'comp_1', 'ps1', 'fp1', 'comp_2', 'ps2', 'fp2', 'comp_3', 'ps3', 'fp3', 'lag_price']
        # Wait, 'qty' is target.
        predictors = ['total_price', 'freight_price', 'unit_price', 'product_name_lenght', 'product_description_lenght', 'product_photos_qty', 'product_weight_g', 'product_score', 'customers', 'weekday', 'weekend', 'holiday', 'month', 'year', 's', 'volume', 'comp_1', 'ps1', 'fp1', 'comp_2', 'ps2', 'fp2', 'comp_3', 'ps3', 'fp3', 'lag_price']

    else:
        service = None
        predictors = []
        print("No running service found. Please run: python run_pipeline.py --config deploy")

except Exception as e:
    service = None
    predictors = []
    print(f"Error loading service: {e}")


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        if not service:
            return render_template("index.html", prediction="Model service not active. Run deployment pipeline.", predictors=predictors)
            
        try:
            # Dynamically build input dataframe based on predictors
            input_data = {}
            for col in predictors:
                val = request.form.get(col)
                if val:
                    input_data[col] = float(val)
                else:
                    input_data[col] = 0.0 # Default handling
            
            df = pd.DataFrame([input_data])
            
            # If the model expects specific structure (like constant added by statsmodels), 
            # sklearn LinearRegression usually handles just X.
            # However, my training used sklearn LinearRegression.
            
            df = pd.DataFrame([input_data])
            json_data = df.to_json(orient="split")
            
            # Predict using the service
            pred_response = service.predict(json_data)
            # Response might be a numpy array or list inside
            pred = pred_response[0] if len(pred_response) > 0 else 0
            
            prediction = f"Predicted Sales Quantity: {pred:.2f}"
            
        except Exception as e:
            prediction = f"Error: {e}"

    return render_template("index.html", prediction=prediction, predictors=predictors)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
