import os
import sys
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

load_dotenv(".env")
db_url = os.getenv("DB_URL")

try:
    engine = create_engine(db_url)
    with engine.connect() as conn:
        result = conn.execute(text("SELECT count(*) FROM retail_prices"))
        count = result.scalar()
        print(f"VERIFICATION SUCCESS: Found {count} records in retail_prices table.")
except Exception as e:
    print(f"VERIFICATION FAILED: {e}")
