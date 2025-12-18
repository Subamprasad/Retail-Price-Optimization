import pandas as pd
import joblib
from flask import Flask, render_template, request
import numpy as np

app = Flask(__name__)

# Load model and predictors
try:
    model = joblib.load("model.pkl")
    predictors = joblib.load("predictors.pkl")
    print("Model loaded successfully.")
except Exception as e:
    model = None
    predictors = []
    print(f"Error loading model: {e}")
    print("Please run 'python run_pipeline.py' to generate the model.")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        if not model:
            return render_template("index.html", prediction="Model not found. Run pipeline first.", predictors=predictors)
            
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
            
            pred = model.predict(df)[0]
            prediction = f"Predicted Sales Quantity: {pred:.2f}"
            
        except Exception as e:
            prediction = f"Error: {e}"

    return render_template("index.html", prediction=prediction, predictors=predictors)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
