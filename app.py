import pandas as pd
import joblib
import numpy as np
from zenml.client import Client
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)
from flask import Flask, render_template, request

"""
Flask Application for Retail Price Optimization.
Serves a web interface for predicting sales quantity based on product features.
Connects to either a running MLflow service or a local model artifact.
"""
from flask import Flask, render_template, request

# Initialize Flask application
app = Flask(__name__)

# Global variables
model_service = None
local_model = None
predictors = []
mode = "unknown"

def load_resources():
    global model_service, local_model, predictors, mode
    
    # 1. Try MLflow Service
    try:
        print("Attempting to connect to MLflow Model Deployer...")
        deployer = MLFlowModelDeployer.get_active_model_deployer()
        services = deployer.find_model_server(
            pipeline_name="deployment_pipeline",
            pipeline_step_name="mlflow_model_deployer_step",
            model_name="retail_price_model",
            running=True,
        )
        
        if services:
            model_service = services[0]
            mode = "mlflow"
            print(f"Connected to MLflow service: {model_service.prediction_url}")
            # Try to infer predictors or use fallback list
            # For simplicity, we use the hardcoded list or load from file if exists
            if not predictors:
                 try:
                     predictors = joblib.load("predictors.pkl")
                 except:
                     pass
        else:
            print("No running MLflow service found.")
            
    except Exception as e:
        print(f"MLflow connection error: {e}")

    # 2. Fallback to Local Model
    if mode != "mlflow":
        try:
            print("Attempting to load local model.pkl...")
            local_model = joblib.load("model.pkl")
            predictors = joblib.load("predictors.pkl")
            mode = "local"
            print("Local model loaded successfully.")
        except Exception as e:
            print(f"Local model loading error: {e}")
            mode = "none"

    # Default predictors if still missing
    if not predictors:
        # Based on dataset schema
        predictors = ['total_price', 'freight_price', 'unit_price', 'product_name_lenght', 
                      'product_description_lenght', 'product_photos_qty', 'product_weight_g', 
                      'product_score', 'customers', 'weekday', 'weekend', 'holiday', 
                      'month', 'year', 'volume', 'comp_1', 'ps1', 'fp1', 'comp_2', 
                      'ps2', 'fp2', 'comp_3', 'ps3', 'fp3', 'lag_price']

load_resources()

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    
    if request.method == "POST":
        if mode == "none":
            return render_template("index.html", prediction="Error: No model available (MLflow or Local).", predictors=predictors)
            
        try:
            # Build input
            input_data = {}
            for col in predictors:
                val = request.form.get(col)
                if val:
                    input_data[col] = float(val)
                else:
                    input_data[col] = 0.0 # Default
            
            df = pd.DataFrame([input_data])
            
            pred = 0.0
            if mode == "mlflow":
                json_data = df.to_json(orient="split")
                response = model_service.predict(json_data)
                # response typically numpy array
                if hasattr(response, "flatten"):
                     pred = response.flatten()[0]
                elif isinstance(response, list):
                     pred = response[0]
                else:
                     pred = float(response)
                     
            elif mode == "local":
                # Ensure input structure matches exactly what model expects?
                # Sklearn is lenient usually if columns match
                resp = local_model.predict(df)
                pred = resp[0]
                
            prediction = f"Predicted Sales Quantity: {pred:.2f} (Mode: {mode})"
            
        except Exception as e:
            prediction = f"Prediction Error: {e}"

    return render_template("index.html", prediction=prediction, predictors=predictors)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
