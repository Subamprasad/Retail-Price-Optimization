import os

content = "DB_URL=postgresql://postgres:8055@localhost:5432/retail_db\n"
with open(".env", "w", encoding="utf-8") as f:
    f.write(content)

print(f"Wrote .env with content: {content.strip()}")
