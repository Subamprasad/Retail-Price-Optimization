# Retail Price Optimization - MLOps Project

Welcome to my Retail Price Optimization project! In this repository, I have built an end-to-end Machine Learning pipeline designed to solve a real-world business problem: determining the optimal pricing for retail products to maximize sales and revenue.

I created this project to demonstrate an "industry-level" approach to ML, moving beyond simple static notebooks and into robust, scalable, and automated pipelines using **ZenML**, **MLflow**, and **PostgreSQL**.

## 🎯 Project Objective

The core goal of this project is to predict the sales quantity of products based on various features (like price, product score, competition data, etc.). By understanding the relationship between price and demand, businesses can optimize their pricing strategies.

I focused on building a system that is:
*   **Modular**: Code is organized into reusable steps and pipelines.
*   **Reproducible**: Experiments are tracked, and data is versioned.
*   **Deployable**: The model is served via a real-time web interface.

## 🛠️ Tech Stack

I selected a modern MLOps stack to build this solution:

*   **Python**: The core programming language.
*   **ZenML**: To create reproducible and portable ML pipelines.
*   **MLflow**: For experiment tracking (logging parameters, metrics) and model deployment.
*   **PostgreSQL**: As the source of truth for our data ingestion.
*   **Flask**: To build a user-friendly prediction web interface.
*   **Scikit-Learn**: For the machine learning modeling (Linear Regression).

## 📂 Project Structure

I've organized the codebase to follow best practices:

*   `pipelines/`: Contains the definitions of my Training/Deployment and Inference pipelines.
*   `steps/`: Individual logic blocks (Ingestion, Processing, Training, Evaluation) that make up the pipelines.
*   `notebooks/`: Interactive notebooks (like `pipeline_orchestration.ipynb`) where I orchestrate the runs.
*   `app.py`: The Flask application serving the web UI.
*   `data/`: Where local artifact data or CSVs are stored.
*   `run_pipeline.py`: A CLI entry point to trigger pipelines.

## 🚀 Key Features

### 1. Automated Pipelines
Instead of running loose scripts, I defined clear pipelines:
*   **Deployment Pipeline**: Ingests data from SQL -> Cleans it -> Trains a Model -> Evaluates it -> Deploys it if it meets accuracy standards.
*   **Inference Pipeline**: Loads the deployed model -> Ingests new data -> Runs predictions.

### 2. SQL Data Ingestion
I integrated **PostgreSQL** to simulate a real production environment. The system connects to a local database (`cs002test`) to fetch the `retail_prices` dataset securely using environment variables.

### 3. Usage of ZenML & MLflow
I used ZenML to orchestrate the flow and MLflow to log every single run. This means I can go back and see exactly what parameters produced the best model.

### 4. Interactive Web Application
I built a clean Front-End using **HTML/CSS** and **Flask**. This allows non-technical users to input product details and get an instant sales prediction.

## � How to Run

### Prerequisites
*   Python 3.8+
*   PostgreSQL installed and running.

### Installation
1.  Clone this repository.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Set up your `.env` file with your database credentials:
    ```
    DB_URL=postgresql+psycopg2://postgres:PASSWORD@localhost:5432/DATABASE_NAME
    ```

### Running the Project
You can run the project in two ways:

**Option A: Using the Jupyter Notebook (Recommended)**
Open `notebooks/pipeline_orchestration.ipynb` and run the cells to execute the training and inference pipelines interactively.

**Option B: Using the CLI**
Train and deploy the model:
```bash
python run_pipeline.py --config deploy
```
Run inference:
```bash
python run_pipeline.py --config predict
```

### Launching the Web App
To see the model in action:
```bash
python app.py
```
Then open [http://localhost:5000](http://localhost:5000) in your browser.

---
*Created by Subamprasad*
