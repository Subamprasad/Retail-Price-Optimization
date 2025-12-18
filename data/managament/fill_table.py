import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from data.managament.index import Base, RetailPrices
# We need to construct the engine dynamically
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "../../.env"))

def create_database_if_not_exists(db_url):
    """Creates the database if it doesn't exist."""
    # Assuming standard postgres url: postgresql://user:pass@host:port/dbname
    from sqlalchemy.engine.url import make_url
    
    url = make_url(db_url)
    db_name = url.database
    
    # Connect to default 'postgres' database to create the new one
    # Replace dbname with 'postgres' in the URL
    postgres_url = url.set(database='postgres')
    
    engine = create_engine(postgres_url, isolation_level="AUTOCOMMIT")
    
    with engine.connect() as conn:
        result = conn.execute(text(f"SELECT 1 FROM pg_database WHERE datname = '{db_name}'"))
        if not result.scalar():
            print(f"Database '{db_name}' does not exist. Creating...")
            conn.execute(text(f"CREATE DATABASE {db_name}"))
            print(f"Database '{db_name}' created.")
        else:
            print(f"Database '{db_name}' already exists.")

def fill_table():
    """Reads CSV and populates the retail_prices table."""
    db_url = os.getenv("DB_URL")
    if not db_url:
        print("Error: DB_URL not found in environment.")
        return

    create_database_if_not_exists(db_url)
    
    # Now connect to the actual DB
    engine = create_engine(db_url)
    
    csv_path = os.path.join(os.path.dirname(__file__), "../retail_price.csv")
    
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return

    print(f"Reading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Standardize column names
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    
    # Create tables
    Base.metadata.create_all(engine)
    
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        # Check if data exists
        if session.query(RetailPrices).first():
            print("Table already has data. Skipping insertion.")
            session.close()
            return

        print("Inserting records...")
        
        # Convert DataFrame to list of dicts for bulk insert
        model_columns = [c.key for c in RetailPrices.__table__.columns if c.key != 'id']
        
        # Filter df to available columns
        available_cols = [c for c in model_columns if c in df.columns]
        df_subset = df[available_cols]
        
        records = df_subset.to_dict(orient='records')
        
        session.bulk_insert_mappings(RetailPrices, records)
        session.commit()
        print(f"Successfully inserted {len(records)} records.")
        
    except Exception as e:
        session.rollback()
        print(f"Error occurred: {e}")
    finally:
        session.close()

if __name__ == "__main__":
    fill_table()
