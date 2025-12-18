# Retail Price Optimization - End-to-End MLOps Project


Welcome to my Retail Price Optimization project. I built this end-to-end MLOps solution to demonstrate how to build, deploy, and manage machine learning pipelines in a production-like environment.

In this project, I tackled the challenge of optimizing retail prices to maximize sales quantity using real-world data concepts. I moved beyond simple notebooks to build a robust, modular system using **ZenML**, **MLflow**, and **BentoML**.

---

## 🚀 How I Built This Project

### 1. The Architecture
I designed this project with modularity in mind. Instead of a monolithic script, I broke down the workflow into distinct steps and pipelines:
- **Ingestion**: I created a custom data retrieval mechanism (`data/managament/retreiver.py`) that pulls data from a PostgreSQL database, ensuring my model always trains on the latest data.
- **Processing**: I implemented feature engineering `steps/process_data.py` to extracting temporal features (month, year, weekend) and encoding categorical variables.
- **Training**: I used a Linear Regression model as a baseline, focusing on the *pipeline infrastructure* rather than just model complexity.
- **Evaluation**: I integrated MLflow to track every run. My pipeline automatically evaluates the model's RMSE and decides whether to deploy it.
- **Deployment**: I used BentoML to containerize the best model as a high-performance API service.

### 2. Key Technologies I Used
- **ZenML**: To orchestrate the entire workflow. It helps me ensure reproducibility.
- **MLflow**: For experiment tracking. I can see exactly how my model performed in every run.
- **BentoML**: To serve the model. It automatically builds a production-ready API endpoint.
- **PostgreSQL**: As my primary data store.
- **Python**: The core language, using `pandas`, `scikit-learn`, and strict type hinting.

---

## 📂 Code Structure Explained

Here is how I organized the codebase:

- `pipelines/`: Contains the logic for connecting steps.
    - `training_pipeline.py`: Defines the flow from data ingestion to model deployment.
    - `inference_pipeline.py`: Handles making predictions with the deployed model.
- `steps/`: The building blocks of my pipelines.
    - `ingest_data.py`: Connects to my database.
    - `process_data.py`: Handles data cleaning and feature engineering.
    - `train_model.py`: Contains the model training logic.
    - `evaluator.py`: Computes performance metrics.
- `data/managament/`: My custom scripts for database operations.
    - `fill_table.py`: Scripts I wrote to populate the initial database.
    - `retreiver.py`: The interface I built to fetch data for the pipeline.

---

## 🛠️ How to Run My Project

### Prerequisites
- Python 3.8+
- PostgreSQL
- ZenML, BentoML, MLflow

### Installation
1.  **Clone my repo**:
    ```bash
    git clone https://github.com/Subamprasad/Retail-Price-Optimization-MLOPS.git
    cd Retail-Price-Optimization-MLOPS
    ```

2.  **Install the dependencies**:
    ```bash
    pip install -r requirements.txt
    zenml integration install bentoml mlflow -y
    ```

3.  **Setup the Database**:
    I've included a script to help you get started quickly:
    ```bash
    python data/managament/fill_table.py
    ```

### Execution
To run the full training and deployment pipeline, simply run:
```bash
python run_pipeline.py --config deploy_and_predict
```

If you encounter issues with the pipeline, you can run the standalone training script to generate the model:
```bash
python force_train.py
```

### Run the App
Launch the prediction interface:
```bash
python app.py
```
Open your browser at `http://localhost:5000`.

### Dashboard
You can visualize the pipelines using:
```bash
zenml show
```

### 📓 Interactive Analysis
I have also included a Jupyter Notebook for exploratory data analysis (EDA) and interactive model training.
- Check `notebooks/Analysis.ipynb` to see how I visualize price distributions and run experiments interactively.
- To run it:
  ```bash
  jupyter notebook notebooks/Analysis.ipynb
  ```

---

## 📈 Future Improvements
- I plan to implement more complex models like XGBoost.
- I will add implementing drift detection to monitor data quality over time.
- I want to build a frontend using Streamlit to interact with the API visually.


