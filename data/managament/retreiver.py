import sys
import os
# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd
from sqlalchemy.orm import Session
from data.managament.index import engine, RetailPrices

def get_latest_data():
    """
    Fetches all data from the retail_prices table.

    Returns:
        pd.DataFrame: DataFrame containing the retail prices data.
    """
    try:
        # Use pandas read_sql for efficient DataFrame creation
        # We query the entire table as per the user's apparent requirement for "filling the table" context
        query = "SELECT * FROM retail_prices"
        df = pd.read_sql(query, engine)
        return df
    except Exception as e:
        print(f"Error retrieving data: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    df = get_latest_data()
    print(f"Retrieved {len(df)} records.")
    print(df.head())