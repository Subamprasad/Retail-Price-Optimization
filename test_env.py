import os
import sys
from dotenv import load_dotenv

print(f"CWD: {os.getcwd()}")
env_path = os.path.abspath(".env")
print(f"Target .env: {env_path}")
print(f"Exists: {os.path.exists(env_path)}")

load_dotenv(env_path)
print(f"DB_URL: {os.getenv('DB_URL')}")
