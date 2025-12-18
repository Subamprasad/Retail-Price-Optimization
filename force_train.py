from data.managament.retreiver import get_latest_data
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
import pandas as pd
import joblib

def force_train():
    print("Forcing training...")
    
    # 1. Ingest
    print("Ingesting data...")
    df = get_latest_data()
    if df.empty:
        print("Data empty!")
        return
        
    if "qty" in df.columns:
        y = df["qty"]
        X = df.drop(columns=["qty"])
    else:
        print("Target 'qty' not found.")
        return

    # 2. Process
    print("Processing data...")
    # Encode
    cols_to_encode = ["product_id", "product_category_name", "product_score"]
    for col in cols_to_encode:
        if col in X.columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            
    # Date features
    if "month_year" in X.columns:
        X["month_year"] = pd.to_datetime(X["month_year"])
        X["month"] = X["month_year"].dt.month
        X["year"] = X["month_year"].dt.year
        X["is_weekend"] = X["month_year"].dt.dayofweek > 4
        X.drop(columns=["month_year"], inplace=True)
        
    X.fillna(0, inplace=True)
    
    # 3. Train
    print("Training model...")
    model = LinearRegression()
    model.fit(X, y)
    
    # 4. Save
    print("Saving model...")
    joblib.dump(model, "model.pkl")
    joblib.dump(list(X.columns), "predictors.pkl")
    print("Done! Model saved to model.pkl")

if __name__ == "__main__":
    force_train()
