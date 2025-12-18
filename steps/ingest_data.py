import pandas as pd
from zenml import step
import logging
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine

# Load env variables from CWD (Project Root)
load_dotenv(os.path.join(os.getcwd(), ".env"))

@step
def ingest_data(
    table_name: str = "retail_prices",
    for_predict: bool = False,
) -> pd.DataFrame:
    """
    Ingests data directly from DB to avoid module path issues.
    """
    try:
        db_url = os.getenv("DB_URL")
        if not db_url:
            raise ValueError("DB_URL environment variable is not set. Please set it in .env file.")

        engine = create_engine(db_url)
        
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql(query, engine)
        
        logging.info(f"Ingested {len(df)} rows from database.")

        if for_predict:
            if "qty" in df.columns:
                df.drop(columns=["qty"], inplace=True)
            logging.info("Dropped 'qty' column for prediction.")
                
        return df
        
    except Exception as e:
        logging.error(f"Error while ingesting data: {e}")
        raise e

@step
def ingest_data_for_inference(table_name: str = "retail_prices") -> pd.DataFrame:
    """
    Ingests data specifically for inference.
    """
    return ingest_data.entrypoint(table_name=table_name, for_predict=True)
