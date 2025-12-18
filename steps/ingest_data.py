import pandas as pd
from zenml import step
from data.managament.retreiver import get_latest_data
from data.managament.fill_table import fill_table
import logging

@step
def ingest_data(
    table_name: str = "retail_prices",
    for_predict: bool = False,
) -> pd.DataFrame:
    """
    Ingests data from the database. 
    Triggers data population if the table is empty.
    
    Args:
        table_name: Name of the table to read from.
        for_predict: If True, drops certain columns for prediction (e.g. qty).
    """
    try:
        # Check if we need to initialize DB
        # This check might need to be more robust, but kept simple for now
        df = get_latest_data()
        if df.empty:
            logging.info("Database empty. Populating...")
            fill_table()
            df = get_latest_data()
        
        logging.info(f"Ingested {len(df)} rows from database.")

        if for_predict:
            # Drop target column for prediction if present
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
    Ingests data specifically for inference, dropping target column.
    """
    return ingest_data.entrypoint(table_name=table_name, for_predict=True)
